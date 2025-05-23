import whisperx
import gc
import torch
import logging
import os
import time
import numpy as np
import sounddevice as sd
import collections
import threading
import queue
import soundfile as sf # For saving audio segments
import requests # For Google Translate API
import json # For parsing Google Translate API response
from stream_system_audio_ffmpeg import FFmpegStreamer
from config import get_app_config, DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS
from transcription_processor import transcription_thread_func as process_transcription
from audio_handler import list_audio_devices, sounddevice_audio_callback # Import audio handling functions

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Shared queues and buffers
# DEFAULT_SAMPLE_RATE and DEFAULT_CHANNELS are now imported from config.py
GOOGLE_TRANSLATE_API_KEY = "" # Hardcoded API Key
audio_queue = collections.deque() # For raw audio data from callback
transcription_output_queue = queue.Queue() # For transcribed text segments

# Log available host APIs at the start
try:
    logger.info(f"Sounddevice available host APIs: {sd.query_hostapis()}")
except Exception as e:
    logger.error(f"Could not query host APIs: {e}")
audio_buffer_lock = threading.Lock()
stop_event = threading.Event()

# list_audio_devices is now imported from audio_handler.py
# audio_callback (now sounddevice_audio_callback) is now imported from audio_handler.py

def translate_text_google(text_to_translate: str, source_language: str, api_key: str) -> str | None:
    """Translates text to English using Google Translate API v2."""
    if not text_to_translate:
        return "" # Return empty if input is empty to avoid API call
        
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        'key': api_key,
        'q': text_to_translate,
        'target': "en", # Hardcoded target language
        'source': source_language,
        'format': 'text' # Ensure we get plain text
    }
    try:
        response = requests.get(url, params=params, timeout=10) # Added timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        result = response.json()
        
        if 'data' in result and 'translations' in result['data'] and \
           len(result['data']['translations']) > 0 and 'translatedText' in result['data']['translations'][0]:
            return result['data']['translations'][0]['translatedText']
        else:
            logger.error(f"Google Translate API response did not contain translated text. Response: {result}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Translate API request failed: {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from Google Translate API: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}", exc_info=True)
        return None
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during translation: {e}", exc_info=True)
        return None

def application_event_loop(args, transcription_output_queue, stop_event, trans_thread):
    """Manages the main event loop for the application, handling transcription output and translation."""
    logger.info("Application event loop started.")
    
    translation_buffer = []
    last_translation_send_time = time.time()
    # How long to wait for more segments before force translating the buffer
    TRANSLATION_BUFFER_TIMEOUT_SECONDS = getattr(args, 'translation_buffer_timeout_seconds', 2.0)
    # (Consider adding translation_buffer_timeout_seconds to config.py later if more control is needed)

    # Ensure update_interval_seconds is respected for console output,
    # but process queue items more frequently for buffering.
    last_console_update_time = time.time()

    while not stop_event.is_set():
        try:
            output_item = None
            try:
                # Get item from queue without blocking indefinitely, to allow stop_event check
                output_item = transcription_output_queue.get(timeout=0.05)
            except queue.Empty:
                pass # No item, continue loop to check stop_event or other conditions

            text_to_display_on_console = None

            if output_item:
                if output_item.get("type") == "raw_segment":
                    raw_text = output_item["text"]
                    is_final_in_vad_chunk = output_item.get("is_final_in_vad_chunk", False)
                    
                    # Display raw segment text immediately (or based on update_interval_seconds)
                    # For now, let's print raw text immediately for responsiveness feedback
                    print(f"RAW: {raw_text}")
                    text_to_display_on_console = raw_text # Default to raw text

                    if args.translate and args.language:
                        translation_buffer.append(raw_text)
                        
                        # Check if it's time to translate the buffer
                        should_translate_now = False
                        if is_final_in_vad_chunk and translation_buffer:
                            should_translate_now = True
                            logger.debug("Translating buffer due to 'is_final_in_vad_chunk'.")
                        elif translation_buffer and (time.time() - last_translation_send_time > TRANSLATION_BUFFER_TIMEOUT_SECONDS):
                            should_translate_now = True
                            logger.debug(f"Translating buffer due to timeout ({TRANSLATION_BUFFER_TIMEOUT_SECONDS}s).")

                        if should_translate_now:
                            text_to_translate_chunk = " ".join(translation_buffer).strip()
                            if text_to_translate_chunk:
                                if GOOGLE_TRANSLATE_API_KEY:
                                    logger.info(f"Attempting to translate buffered: '{text_to_translate_chunk}' from '{args.language}' to 'en'")
                                    translated_chunk = translate_text_google(
                                        text_to_translate_chunk,
                                        source_language=args.language,
                                        api_key=GOOGLE_TRANSLATE_API_KEY
                                    )
                                    if translated_chunk is not None:
                                        logger.info(f"Successfully translated to: '{translated_chunk}'")
                                        # This translated text is what we'll primarily display and write
                                        text_to_display_on_console = translated_chunk
                                        print(f"TRANSLATED: {translated_chunk}")
                                    else:
                                        logger.warning(f"Failed to translate buffered segment: '{text_to_translate_chunk}'. Using original.")
                                        # If translation fails, text_to_display_on_console remains the last raw segment
                                else:
                                    logger.warning("Translation enabled but GOOGLE_TRANSLATE_API_KEY missing. Skipping translation of buffered text.")
                            translation_buffer = [] # Clear buffer
                            last_translation_send_time = time.time()
                    # If not translating, text_to_display_on_console is already the raw_text
                else:
                    # Handle older string-only messages if any, or log unexpected item
                    logger.warning(f"Received unexpected item from transcription_output_queue: {output_item}")
                    text_to_display_on_console = str(output_item) # Best effort

            # Console output and file writing based on update_interval_seconds
            # This part needs to be careful not to re-print if already printed above.
            # For simplicity now, we'll write text_to_display_on_console if it was set in this iteration.
            # A more robust approach might queue display lines and flush them periodically.
            if text_to_display_on_console and (time.time() - last_console_update_time >= args.update_interval_seconds):
                # The immediate print of RAW and TRANSLATED above gives faster feedback.
                # This section is more for the file writing and potentially a consolidated console update.
                # Let's adjust to only write to file here, assuming console output is handled above.
                try:
                    with open("live-audio/transcription.txt", "a", encoding="utf-8") as f:
                        f.write(text_to_display_on_console + "\n")
                    logger.debug(f"Wrote to transcription.txt: {text_to_display_on_console}")
                except IOError as e:
                    logger.error(f"Failed to write transcription to file: {e}")
                last_console_update_time = time.time()
            
            # Check if transcription thread is still alive
            if not trans_thread.is_alive() and not stop_event.is_set():
                logger.error("Transcription thread unexpectedly died. Stopping.")
                stop_event.set()
                break
            
            time.sleep(0.1) # Main loop sleep

        except KeyboardInterrupt:
            logger.info("Ctrl+C received in event loop. Shutting down...")
            stop_event.set()
            break
        except Exception as e:
            logger.error(f"Error in application event loop: {e}", exc_info=True)
            stop_event.set()
            break
    logger.info("Application event loop finished.")

def main():
    # Get application configuration from config.py
    args, selected_device, selected_compute_type = get_app_config()

    # The logger level for the root logger is now set in config.py
    # Individual loggers can still have their levels set if needed.
    # logger.setLevel(args.log_level.upper()) # This specific logger for live_system_transcriber
    logger.info(f"Main application logger ({__name__}) is active.")


    if args.list_devices:
        list_audio_devices()
        return

    # Directory creation for audio_segments_dir is now handled in config.py
    # We can check args.save_audio_segments directly.
    if args.save_audio_segments:
        logger.info(f"Audio segment saving is enabled. Segments will be saved to: {args.audio_segments_dir}")
    else:
        logger.info("Audio segment saving is disabled.")

    # Device and compute_type selection is now handled in config.py
    # We directly use selected_device and selected_compute_type
    logger.info(f"Using device: {selected_device} with compute_type: {selected_compute_type}")

    if selected_device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA was selected or auto-detected, but it's not available. Exiting.")
        return

    # Start transcription thread
    trans_thread = threading.Thread(
        target=process_transcription, # Use the imported function
        args=(
            args.model, args.language, selected_device, selected_compute_type,
            args.batch_size, args.sample_rate, # segment_duration_seconds removed, max_segment_duration_seconds added below
            args.save_audio_segments, args.audio_segments_dir,
            audio_queue, transcription_output_queue, audio_buffer_lock, stop_event,
            # VAD related args
            args.vad_enabled,
            args.max_segment_duration_seconds, # Renamed from segment_duration_seconds
            args.vad_threshold,
            args.vad_min_silence_duration_ms,
            args.vad_min_speech_duration_ms,
            args.vad_speech_pad_ms
        ),
        daemon=True # Allows main program to exit even if this thread is still running
    )
    trans_thread.start()

    # Setup and start audio stream
    stream = None
    ffmpeg_streamer = None

    try:
        if args.audio_capture_method == "sounddevice":
            logger.info("Using sounddevice for audio capture.")
            # Attempt to find a loopback/monitor device if no specific device is given
            selected_input_device = None
            if args.input_device_name:
                selected_input_device = args.input_device_name
                logger.info(f"Using specified input device name: {selected_input_device}")
            elif args.input_device_index is not None:
                selected_input_device = args.input_device_index
                logger.info(f"Using specified input device index: {selected_input_device}")
            else:
                logger.info(f"No input_device_index or input_device_name specified. Using default system input device for sounddevice. This might not be system audio output.")
                pass # selected_input_device remains None, sounddevice will use default input

            logger.info(f"Attempting to open sounddevice stream with: Device: {selected_input_device if selected_input_device is not None else 'Default System Input'}, Sample Rate: {args.sample_rate}, Channels: {args.channels}")
            
            # Use a lambda to pass shared objects to the sounddevice_audio_callback
            stream_callback = lambda indata, frames, time, status: sounddevice_audio_callback(
                indata, frames, time, status, audio_queue, audio_buffer_lock
            )
            
            stream = sd.InputStream(
                device=selected_input_device,
                channels=args.channels,
                samplerate=args.sample_rate,
                callback=stream_callback, # Use the lambda callback
                dtype='float32' # Whisper expects float32
            )
            stream.start()
            logger.info("Sounddevice stream started. Capturing audio...")

        elif args.audio_capture_method == "ffmpeg":
            logger.info("Using ffmpeg for audio capture.")
            bytes_per_sample = 4 # float32
            chunk_size_bytes = args.ffmpeg_chunk_samples * bytes_per_sample * args.channels

            ffmpeg_streamer = FFmpegStreamer(
                audio_queue=audio_queue, # The same queue used by transcription_thread_func
                stop_event=stop_event,
                monitor_source=args.ffmpeg_source,
                sample_rate=args.sample_rate,
                channels=args.channels,
                chunk_size_bytes=chunk_size_bytes,
                ffmpeg_path=args.ffmpeg_path
            )
            if not ffmpeg_streamer.start_stream():
                logger.error("Failed to start FFmpeg stream. Exiting.")
                # stop_event is set by FFmpegStreamer on failure
                return # Exit main if ffmpeg fails to start
            logger.info("FFmpeg stream started. Capturing audio...")
        
        print("Listening... (Press Ctrl+C to stop)")
        # Call the application event loop
        application_event_loop(args, transcription_output_queue, stop_event, trans_thread)
                
    except Exception as e: # This will catch exceptions from stream setup
        logger.error(f"Failed to start audio stream or critical error: {e}", exc_info=True)
        stop_event.set()
    finally:
        logger.info("Stopping audio capture and cleaning up...")
        stop_event.set() # Signal all threads to stop first

        if args.audio_capture_method == "sounddevice" and stream is not None:
            if stream.active:
                stream.stop()
                stream.close()
                logger.info("Sounddevice stream stopped and closed.")
        elif args.audio_capture_method == "ffmpeg" and ffmpeg_streamer is not None:
            ffmpeg_streamer.stop_stream() # This will also wait for its thread
            logger.info("FFmpeg streamer stopped.")
        
        if trans_thread.is_alive():
            logger.info("Waiting for transcription thread to finish...")
            trans_thread.join(timeout=5.0) # Wait for thread to finish
            if trans_thread.is_alive():
                logger.warning("Transcription thread did not finish in time.")

        # Final cleanup (especially for CUDA, adapted from transcribe_file.py)
        if selected_device == "cuda":
            logger.info("Performing final garbage collection and emptying CUDA cache.")
            gc.collect()
            try:
                torch.cuda.empty_cache()
                logger.info("Successfully emptied CUDA cache.")
            except Exception as e:
                logger.warning(f"Could not empty CUDA cache during final cleanup: {e}", exc_info=True)
        else:
            gc.collect()
            logger.info("Performed final garbage collection.")
        
        logger.info("Application finished.")

if __name__ == "__main__":
    main()