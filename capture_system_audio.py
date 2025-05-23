import subprocess
import time
import argparse
import os # For deleting the temporary file

# Default Parameters
DEFAULT_DURATION = 10  # seconds
DEFAULT_OUTPUT_FILENAME = "captured_audio_last_10s.wav"
DEFAULT_PULSE_MONITOR_SOURCE = "alsa_output.pci-0000_75_00.6.analog-stereo.monitor"
DEFAULT_VOLUME_FACTOR = 2.0 # 1.0 means no change, 2.0 is double amplitude

TEMP_CAPTURE_FILENAME = "temp_captured_audio.wav"

def process_audio_with_ffmpeg(duration, monitor_source, output_filename, volume_factor):
    """
    Captures audio using ffmpeg, then processes it to adjust volume,
    and saves it to a WAV file.
    """
    print(f"Starting {duration}-second audio capture using ffmpeg...")
    print(f"Source: {monitor_source}")
    print(f"Temporary capture file: {TEMP_CAPTURE_FILENAME}")

    capture_command = [
        "ffmpeg",
        "-y",  # Overwrite output file without asking
        "-f", "pulse",
        "-i", monitor_source,
        "-t", str(duration),
        TEMP_CAPTURE_FILENAME # Save to temporary file first
    ]

    try:
        # Step 1: Capture audio to temporary file
        print(f"Executing capture command: {' '.join(capture_command)}")
        capture_result = subprocess.run(capture_command, capture_output=True, text=True, check=True)
        print("ffmpeg capture command executed successfully.")
        if capture_result.stderr:
            print("ffmpeg capture output:\n", capture_result.stderr)

        # Step 2: Apply volume filter and save to final output file
        if volume_factor != 1.0:
            print(f"\nApplying volume factor: {volume_factor}...")
            print(f"Input: {TEMP_CAPTURE_FILENAME}, Output: {output_filename}")
            volume_command = [
                "ffmpeg",
                "-y",
                "-i", TEMP_CAPTURE_FILENAME,
                "-filter:a", f"volume={volume_factor}",
                output_filename
            ]
            print(f"Executing volume adjustment command: {' '.join(volume_command)}")
            volume_result = subprocess.run(volume_command, capture_output=True, text=True, check=True)
            print("ffmpeg volume adjustment command executed successfully.")
            if volume_result.stderr:
                print("ffmpeg volume output:\n", volume_result.stderr)
            print(f"Volume adjusted recording saved to {output_filename}")
        else:
            # If volume_factor is 1.0, just rename the temp file to the final output
            print("\nVolume factor is 1.0, no volume adjustment needed. Renaming temp file.")
            os.rename(TEMP_CAPTURE_FILENAME, output_filename)
            print(f"Recording saved to {output_filename}")


    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e}")
        if e.stdout: # check if stdout has content
            print("ffmpeg stdout:\n", e.stdout)
        if e.stderr: # check if stderr has content
            print("ffmpeg stderr:\n", e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Step 3: Clean up temporary file
        if os.path.exists(TEMP_CAPTURE_FILENAME):
            try:
                os.remove(TEMP_CAPTURE_FILENAME)
                print(f"\nTemporary file {TEMP_CAPTURE_FILENAME} deleted.")
            except OSError as e:
                print(f"Error deleting temporary file {TEMP_CAPTURE_FILENAME}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture system audio using ffmpeg and optionally adjust volume.")
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_PULSE_MONITOR_SOURCE,
        help=f"The PulseAudio/PipeWire monitor source to record from. Defaults to '{DEFAULT_PULSE_MONITOR_SOURCE}'."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Duration of the recording in seconds. Defaults to {DEFAULT_DURATION}."
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output filename for the WAV file. Defaults to '{DEFAULT_OUTPUT_FILENAME}'."
    )
    parser.add_argument(
        "--volume_factor",
        type=float,
        default=DEFAULT_VOLUME_FACTOR,
        help=f"Volume multiplication factor (e.g., 1.5 for 50%% louder, 2.0 for 100%% louder). Defaults to {DEFAULT_VOLUME_FACTOR} (no change)."
    )

    args = parser.parse_args()

    print(f"Script will capture {args.duration} seconds of system audio.")
    print(f"Using audio source: {args.source}")
    if args.volume_factor != 1.0:
        print(f"Volume will be multiplied by: {args.volume_factor}")
    print(f"Final output will be saved to: {args.outfile}")

    process_audio_with_ffmpeg(args.duration, args.source, args.outfile, args.volume_factor)