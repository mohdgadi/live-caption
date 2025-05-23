import whisperx
import gc
import torch
import logging
import os
import time
import numpy as np
import soundfile as sf # For saving audio segments

# Configure logger for this module
logger = logging.getLogger(__name__)

def transcription_thread_func(
    model_name,
    language,
    device,
    compute_type,
    batch_size,
    sample_rate,
    save_segments,
    segments_dir,
    audio_queue, # Shared deque for raw audio data
    transcription_output_queue, # Shared queue for transcribed text
    audio_buffer_lock, # Shared lock for audio_queue
    stop_event, # Shared event to signal stopping
    # VAD and related parameters
    vad_enabled,
    max_segment_duration_seconds, # Max duration for VAD processing window or fixed chunk
    vad_threshold,
    vad_min_silence_duration_ms,
    vad_min_speech_duration_ms,
    vad_speech_pad_ms
):
    """
    Thread function to load the model and continuously transcribe audio.
    Uses VAD for dynamic segmentation if enabled.
    Optionally saves processed audio segments to disk.
    """
    logger.info(f"Transcription processor thread started. Whisper model: {model_name} on {device} with compute_type {compute_type}. VAD enabled: {vad_enabled}")
    if save_segments:
        logger.info(f"Audio segments will be saved to: {segments_dir}")
        os.makedirs(segments_dir, exist_ok=True)

    whisper_model = None
    vad_model = None
    vad_utils = {} # To store VAD utility functions like get_speech_timestamps

    segment_counter = 0 # For saving audio files

    try:
        # Load Whisper model
        whisper_model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
        logger.info(f"Successfully loaded Whisper model: {model_name}")

        if vad_enabled:
            try:
                # Load Silero VAD model and utils
                torch.set_num_threads(1) # Recommended for Silero VAD
                vad_model, vad_utils_funcs = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                            model='silero_vad',
                                                            force_reload=False, # Set to True if you always want the latest
                                                            onnx=False) # Set to True if you have onnxruntime and prefer it
                
                # The VAD utilities are returned as a tuple. We need to unpack them.
                # The typical order is: get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks
                # We are primarily interested in get_speech_timestamps.
                try:
                    (get_speech_timestamps_func,
                     save_audio_func,
                     read_audio_func,
                     vad_iterator_func,
                     collect_chunks_func) = vad_utils_funcs
                    
                    vad_utils['get_speech_timestamps'] = get_speech_timestamps_func
                    # Optionally store others if needed:
                    # vad_utils['save_audio'] = save_audio_func
                    # vad_utils['read_audio'] = read_audio_func
                    # ... etc.
                    logger.info("Successfully loaded and unpacked Silero VAD model and utilities.")
                except ValueError:
                    logger.error("Could not unpack expected VAD utility functions. The returned tuple structure might have changed.", exc_info=True)
                    vad_enabled = False # Disable VAD if utils can't be unpacked
                except TypeError: # If vad_utils_funcs is not iterable
                    logger.error("VAD utility functions were not returned as an iterable. VAD will be disabled.", exc_info=True)
                    vad_enabled = False


                # Check if get_speech_timestamps was successfully assigned
                if not vad_utils.get('get_speech_timestamps'):
                    logger.warning("'get_speech_timestamps' function not available after loading VAD. VAD will be disabled.")
                    vad_enabled = False
            except Exception as e_vad:
                logger.error(f"Failed to load Silero VAD model: {e_vad}. Disabling VAD.", exc_info=True)
                vad_enabled = False # Fallback to non-VAD mode

        # Main processing loop
        if vad_enabled and vad_model and vad_utils.get('get_speech_timestamps'):
            # VAD-based processing
            collected_audio_for_vad_list = []
            total_samples_collected_for_vad = 0
            vad_processing_window_samples = int(max_segment_duration_seconds * sample_rate)

            logger.info(f"VAD enabled. Processing window: {max_segment_duration_seconds}s ({vad_processing_window_samples} samples).")
            logger.info(f"VAD params: threshold={vad_threshold}, min_silence={vad_min_silence_duration_ms}ms, min_speech={vad_min_speech_duration_ms}ms, pad={vad_speech_pad_ms}ms")

            while not stop_event.is_set():
                chunk = None
                with audio_buffer_lock:
                    if audio_queue:
                        chunk = audio_queue.popleft()
                
                if chunk is not None:
                    collected_audio_for_vad_list.append(chunk)
                    total_samples_collected_for_vad += len(chunk)

                if total_samples_collected_for_vad >= vad_processing_window_samples or \
                   (stop_event.is_set() and total_samples_collected_for_vad > 0) or \
                   (chunk is None and total_samples_collected_for_vad > 0 and not audio_queue): # Process remaining if no new audio

                    current_vad_window_audio_np = np.concatenate(collected_audio_for_vad_list)
                    collected_audio_for_vad_list = []
                    total_samples_collected_for_vad = 0
                    
                    logger.debug(f"Processing VAD window of {current_vad_window_audio_np.shape[0] / sample_rate:.2f}s")
                    audio_tensor = torch.from_numpy(current_vad_window_audio_np).float()

                    try:
                        speech_timestamps = vad_utils['get_speech_timestamps'](
                            audio_tensor,
                            vad_model,
                            threshold=vad_threshold,
                            sampling_rate=sample_rate,
                            min_silence_duration_ms=vad_min_silence_duration_ms,
                            speech_pad_ms=vad_speech_pad_ms,
                            min_speech_duration_ms=vad_min_speech_duration_ms,
                            return_seconds=False # Get samples
                        )
                        
                        if not speech_timestamps:
                            logger.debug("VAD: No speech detected in the current window.")
                        
                        for i, ts in enumerate(speech_timestamps):
                            start_sample, end_sample = ts['start'], ts['end']
                            vad_segment_audio = current_vad_window_audio_np[start_sample:end_sample]
                            
                            if vad_segment_audio.size == 0:
                                logger.debug("VAD segment is empty, skipping.")
                                continue
                            
                            logger.debug(f"VAD segment {i+1}/{len(speech_timestamps)}: Duration {vad_segment_audio.shape[0]/sample_rate:.2f}s")

                            if save_segments:
                                segment_counter += 1
                                seg_filename = os.path.join(segments_dir, f"vad_segment_{time.strftime('%Y%m%d-%H%M%S')}_{segment_counter:04d}.wav")
                                try:
                                    sf.write(seg_filename, vad_segment_audio, sample_rate)
                                    logger.debug(f"Saved VAD audio segment to {seg_filename}")
                                except Exception as e_save:
                                    logger.error(f"Failed to save VAD audio segment {seg_filename}: {e_save}")

                            # Transcribe VAD segment with Whisper
                            result = whisper_model.transcribe(vad_segment_audio, batch_size=batch_size, language=language)
                            
                            if result and "segments" in result and result["segments"]:
                                num_whisper_segments = len(result["segments"])
                                for idx, whisper_seg_data in enumerate(result["segments"]):
                                    segment_text = whisper_seg_data["text"].strip()
                                    if segment_text:
                                        is_final_in_vad = (idx == num_whisper_segments - 1)
                                        transcription_output_queue.put({
                                            "type": "raw_segment",
                                            "text": segment_text,
                                            "is_final_in_vad_chunk": is_final_in_vad
                                        })
                                        logger.debug(f"Whisper segment (from VAD): '{segment_text}' (final_in_vad: {is_final_in_vad})")
                            else:
                                logger.debug("No Whisper segments from VAD chunk or result is empty.")
                    except Exception as e_vad_proc:
                        logger.error(f"Error during VAD processing or subsequent transcription: {e_vad_proc}", exc_info=True)

                    if stop_event.is_set() and not audio_queue: # Ensure we break if stopping and no more data
                        break
                else: # No chunk from queue
                    if stop_event.is_set() and not collected_audio_for_vad_list: # Break if stopping and buffer is empty
                        break
                    time.sleep(0.05) # Wait if queue is empty but not stopping or buffer has data

        else: # VAD disabled or failed to load - use fixed duration chunking
            logger.info(f"VAD disabled or failed. Using fixed duration chunking: {max_segment_duration_seconds}s")
            min_audio_samples_for_transcription = int(max_segment_duration_seconds * sample_rate)
            
            while not stop_event.is_set():
                current_audio_data_list = []
                current_samples = 0
                
                with audio_buffer_lock:
                    while audio_queue and current_samples < min_audio_samples_for_transcription:
                        chunk = audio_queue.popleft()
                        current_audio_data_list.append(chunk)
                        current_samples += len(chunk)
                    
                    if current_samples < min_audio_samples_for_transcription and current_audio_data_list:
                        # Not enough data, put back unless stopping and it's all we have
                        if not (stop_event.is_set() and not audio_queue):
                            for chunk_to_put_back in reversed(current_audio_data_list):
                                audio_queue.appendleft(chunk_to_put_back)
                            current_audio_data_list = [] # Clear
                
                if not current_audio_data_list:
                    if stop_event.is_set() and not audio_queue: # Break if stopping and no more data
                        break
                    time.sleep(0.1)
                    continue

                concatenated_audio = np.concatenate(current_audio_data_list)
                duration_collected = concatenated_audio.shape[0] / sample_rate
                logger.debug(f"Collected {duration_collected:.2f}s of audio for fixed-duration transcription.")

                if save_segments and concatenated_audio.size > 0:
                    segment_counter += 1
                    seg_filename = os.path.join(segments_dir, f"fixed_segment_{time.strftime('%Y%m%d-%H%M%S')}_{segment_counter:04d}.wav")
                    try:
                        sf.write(seg_filename, concatenated_audio, sample_rate)
                        logger.debug(f"Saved fixed audio segment to {seg_filename}")
                    except Exception as e_save:
                        logger.error(f"Failed to save fixed audio segment {seg_filename}: {e_save}")
                
                # Transcribe fixed segment
                result = whisper_model.transcribe(concatenated_audio, batch_size=batch_size, language=language)
                
                if result and "segments" in result and result["segments"]:
                    num_whisper_segments = len(result["segments"])
                    for idx, whisper_seg_data in enumerate(result["segments"]):
                        segment_text = whisper_seg_data["text"].strip()
                        if segment_text:
                            is_final = (idx == num_whisper_segments - 1)
                            transcription_output_queue.put({
                                "type": "raw_segment", # Consistent type
                                "text": segment_text,
                                "is_final_in_vad_chunk": is_final # For fixed, this means final in the fixed chunk
                            })
                            logger.debug(f"Whisper segment (fixed chunk): '{segment_text}' (final: {is_final})")
                else:
                    logger.debug("No Whisper segments from fixed chunk or result is empty.")
                
                if stop_event.is_set() and not audio_queue: # Ensure we break if stopping and no more data
                    break


    except Exception as e:
        logger.error(f"Critical error in transcription thread: {e}", exc_info=True)
        stop_event.set()
    finally:
        if whisper_model is not None:
            del whisper_model
            logger.info("Whisper model unloaded.")
        if vad_model is not None:
            del vad_model
            logger.info("Silero VAD model unloaded.")
        if device == "cuda":
            gc.collect()
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA cache emptied in transcription_processor thread.")
            except Exception as e_cuda:
                logger.warning(f"Could not empty CUDA cache in transcription_processor thread: {e_cuda}")
        logger.info("Transcription processor thread finished.")