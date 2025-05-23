import argparse
import logging
import torch
import os
from stream_system_audio_ffmpeg import DEFAULT_PULSE_MONITOR_SOURCE as FFMPEG_DEFAULT_SOURCE

# Configure logger for this module
logger = logging.getLogger(__name__)

# Whisper models typically expect 16kHz mono audio
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1

def get_app_config():
    """Parses command-line arguments and determines device/compute configurations."""
    logger.info("Loading application configuration...")

    parser = argparse.ArgumentParser(description="Live transcribe system audio using whisperX.")
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model name (e.g., tiny.en, base, small, medium, large-v3).")
    parser.add_argument("--language", type=str, default="en", help="Language code for transcription (e.g., 'en', 'ja'). Defaults to 'en' (English).")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to use (cpu or cuda). Default: auto-detect CUDA, fallback to CPU if not available or specified.")
    parser.add_argument("--compute_type", type=str, default=None, help="Compute type (e.g., float16, int8 for GPU; int8, float32 for CPU). Auto-selects if None.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for transcription.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    
    parser.add_argument("--update_interval_seconds", type=float, default=2.0, help="How often to print updates to the terminal.")

    # VAD (Voice Activity Detection) Settings
    vad_group = parser.add_argument_group("VAD Settings")
    vad_group.add_argument("--vad_enabled", action="store_true", help="Enable Voice Activity Detection. If not set, uses fixed duration segments via --max_segment_duration_seconds.")
    vad_group.add_argument("--max_segment_duration_seconds", type=float, default=10.0, help="Maximum duration of an audio segment (in seconds) when VAD is enabled, or the fixed segment duration if VAD is disabled.")
    vad_group.add_argument("--vad_threshold", type=float, default=0.5, help="Speech probability threshold for Silero VAD (0.0 to 1.0). Higher values are more conservative in detecting speech.")
    vad_group.add_argument("--vad_min_silence_duration_ms", type=int, default=700, help="Minimum duration of silence (in ms) after speech to consider a segment complete.")
    vad_group.add_argument("--vad_min_speech_duration_ms", type=int, default=250, help="Minimum duration of speech (in ms) to be considered a valid speech segment.")
    vad_group.add_argument("--vad_speech_pad_ms", type=int, default=200, help="Milliseconds of audio padding to add to the start and end of detected speech segments.")

    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help=f"Sample rate for audio capture. Whisper models typically expect {DEFAULT_SAMPLE_RATE}Hz.")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help=f"Number of audio channels. Whisper models typically expect {DEFAULT_CHANNELS} (mono).")
    parser.add_argument("--input_device_index", type=int, default=None, help="Specific input device index. Use --list_devices to see available devices.")
    parser.add_argument("--input_device_name", type=str, default=None, help="Specific input device name (e.g., from 'pactl list sources'). Overrides --input_device_index if provided.")
    parser.add_argument("--list_devices", action="store_true", help="List available audio devices and exit (sounddevice only).")

    # Audio Capture Method
    parser.add_argument("--audio_capture_method", type=str, default="sounddevice", choices=["sounddevice", "ffmpeg"], help="Method to capture audio ('sounddevice' or 'ffmpeg').")
    # FFmpeg specific arguments
    parser.add_argument("--ffmpeg_source", type=str, default=FFMPEG_DEFAULT_SOURCE, help="FFmpeg PulseAudio monitor source (if --audio_capture_method is ffmpeg).")
    parser.add_argument("--ffmpeg_path", type=str, default="ffmpeg", help="Path to the ffmpeg executable.")
    parser.add_argument("--ffmpeg_chunk_samples", type=int, default=1024, help="Number of samples per chunk for ffmpeg streaming (affects latency and buffer size).")

    # Audio Segment Saving
    parser.add_argument("--save_audio_segments", action="store_true", help="Save processed audio segments to WAV files.")
    parser.add_argument("--audio_segments_dir", type=str, default="segments", help="Directory to save audio segments if --save_audio_segments is enabled.")

    # Google Translate arguments
    parser.add_argument("--translate", action="store_true", help="Enable translation to English using Google Translate API. Requires --language to be set to the source language. API key is hardcoded.")

    args = parser.parse_args()

    # Set root logger level based on parsed args.
    # This affects all loggers unless they have their own level set.
    logging.getLogger().setLevel(args.log_level.upper())
    logger.info(f"Root log level set to: {args.log_level.upper()} from config module")

    if args.save_audio_segments:
        try:
            os.makedirs(args.audio_segments_dir, exist_ok=True)
            logger.info(f"Audio segments will be saved in: {os.path.abspath(args.audio_segments_dir)}")
        except OSError as e:
            logger.error(f"Could not create directory for audio segments '{args.audio_segments_dir}': {e}. Disabling segment saving.")
            args.save_audio_segments = False # Disable if directory creation fails

    # Determine device for computation
    selected_device = args.device
    if selected_device is None:
        if torch.cuda.is_available():
            selected_device = "cuda"
            logger.info("CUDA is available. Automatically selecting 'cuda' device.")
        else:
            selected_device = "cpu"
            logger.info("CUDA not available. Automatically selecting 'cpu' device.")
    elif selected_device == "cuda":
        if not torch.cuda.is_available():
            logger.error("User specified device 'cuda', but CUDA is not available. Application might exit or fail in main.")
            # Main script should handle exit if this is critical
        else:
            logger.info("User specified device 'cuda'. CUDA is available.")
    elif selected_device == "cpu":
        logger.info("User specified device 'cpu'.")
    
    # Auto-select compute_type if not specified
    selected_compute_type = args.compute_type
    if selected_compute_type is None:
        if selected_device == "cuda":
            selected_compute_type = "float16" 
            logger.info(f"Device is 'cuda', auto-selecting compute_type: {selected_compute_type}")
        else: # cpu
            selected_compute_type = "int8" 
            logger.info(f"Device is 'cpu', auto-selecting compute_type: {selected_compute_type}")
    else:
        logger.info(f"User specified compute_type: {selected_compute_type}")
    
    logger.info("Application configuration loaded successfully.")
    return args, selected_device, selected_compute_type

if __name__ == '__main__':
    # This part is for testing the config module independently
    print("Testing configuration parsing...")
    parsed_args, dev, comp_type = get_app_config()
    print("\nParsed Arguments:")
    for arg, value in vars(parsed_args).items():
        print(f"  {arg}: {value}")
    print(f"\nSelected Device: {dev}")
    print(f"Selected Compute Type: {comp_type}")
    if parsed_args.list_devices:
        print("\nNote: --list_devices was specified. The main application would typically call list_audio_devices() and exit.")