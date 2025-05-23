# Live Audio Transcription CLI Options

This document outlines the command-line options available for the live audio transcription scripts. These options are primarily configured in `config.py`.

## Main Application Options

These options control the core behavior of the transcription application.

| Argument                    | Type    | Default      | Description                                                                                                |
| --------------------------- | ------- | ------------ | ---------------------------------------------------------------------------------------------------------- |
| `--model`                   | str     | `large-v3`   | Whisper model name (e.g., `tiny.en`, `base`, `small`, `medium`, `large-v3`).                               |
| `--language`                | str     | `en`         | Language code for transcription (e.g., 'en', 'ja'). Defaults to 'en' (English).                            |
| `--device`                  | str     | `None`       | Device to use (`cpu` or `cuda`). Default: auto-detect CUDA, fallback to CPU if not available or specified. |
| `--compute_type`            | str     | `None`       | Compute type (e.g., `float16`, `int8` for GPU; `int8`, `float32` for CPU). Auto-selects if None.          |
| `--batch_size`              | int     | `16`         | Batch size for transcription.                                                                              |
| `--log_level`               | str     | `INFO`       | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).                                           |
| `--update_interval_seconds` | float   | `2.0`        | How often to print updates to the terminal.                                                                |
| `--segment_duration_seconds`| float   | `10.0`       | Duration of audio (in seconds) to process for each transcription call. This is the audio recording time per segment. |

## Audio Configuration Options

These options relate to audio input and processing.

| Argument                    | Type    | Default                          | Description                                                                                                |
| --------------------------- | ------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `--sample_rate`             | int     | `16000`                          | Sample rate for audio capture. Whisper models typically expect 16000Hz.                                    |
| `--channels`                | int     | `1`                              | Number of audio channels. Whisper models typically expect 1 (mono).                                        |
| `--input_device_index`      | int     | `None`                           | Specific input device index. Use `--list_devices` to see available devices.                                |
| `--input_device_name`       | str     | `None`                           | Specific input device name (e.g., from 'pactl list sources'). Overrides `--input_device_index` if provided. |
| `--list_devices`            | action  | `store_true`                     | List available audio devices and exit (sounddevice only).                                                  |
| `--audio_capture_method`    | str     | `sounddevice`                    | Method to capture audio ('sounddevice' or 'ffmpeg').                                                       |

## FFmpeg Specific Options

These options are applicable if `--audio_capture_method` is set to `ffmpeg`.

| Argument                 | Type    | Default                          | Description                                                                                                |
| ------------------------ | ------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `--ffmpeg_source`        | str     | (OS dependent, e.g., `alsa_output.pci-0000_00_1f.3.analog-stereo.monitor`) | FFmpeg PulseAudio monitor source (if --audio_capture_method is ffmpeg). The default is system-dependent. |
| `--ffmpeg_path`          | str     | `ffmpeg`                         | Path to the ffmpeg executable.                                                                             |
| `--ffmpeg_chunk_samples` | int     | `1024`                           | Number of samples per chunk for ffmpeg streaming (affects latency and buffer size).                        |

## Audio Segment Saving Options

These options control the saving of processed audio segments.

| Argument                 | Type    | Default      | Description                                                                              |
| ------------------------ | ------- | ------------ | ---------------------------------------------------------------------------------------- |
| `--save_audio_segments`  | action  | `store_true` | Save processed audio segments to WAV files.                                                |
| `--audio_segments_dir`   | str     | `segments`   | Directory to save audio segments if `--save_audio_segments` is enabled.                    |

## How to Use

You can pass these arguments when running the main transcription script (e.g., `live_system_transcriber.py`).

Example:
```bash
python live-audio/live_system_transcriber.py --model small --language en --segment_duration_seconds 5 --log_level DEBUG
```
This command would run the transcriber using the 'small' model, for English language, with 5-second audio segments, and set the logging level to DEBUG.