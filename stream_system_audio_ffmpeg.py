import subprocess
import threading
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

DEFAULT_PULSE_MONITOR_SOURCE = "alsa_output.pci-0000_75_00.6.analog-stereo.monitor"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
# For float32, each sample is 4 bytes.
# A chunk of 1024 samples would be 4096 bytes. This is a common buffer size.
DEFAULT_CHUNK_SAMPLES = 1024 # Number of samples per chunk
DEFAULT_BYTES_PER_SAMPLE = 4 # For float32 (4 bytes)
DEFAULT_CHUNK_SIZE_BYTES = DEFAULT_CHUNK_SAMPLES * DEFAULT_BYTES_PER_SAMPLE

class FFmpegStreamer:
    def __init__(self,
                 audio_queue,
                 stop_event,
                 monitor_source=DEFAULT_PULSE_MONITOR_SOURCE,
                 sample_rate=DEFAULT_SAMPLE_RATE,
                 channels=DEFAULT_CHANNELS,
                 chunk_size_bytes=DEFAULT_CHUNK_SIZE_BYTES,
                 ffmpeg_path="ffmpeg"):
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.monitor_source = monitor_source
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size_bytes = chunk_size_bytes
        self.ffmpeg_path = ffmpeg_path
        self.ffmpeg_process = None
        self.thread = None
        self.bytes_per_sample = 4 # float32

    def _reader_thread(self):
        try:
            logger.info(f"FFmpeg reader thread started for source: {self.monitor_source}")
            while not self.stop_event.is_set() and self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                raw_audio_chunk = self.ffmpeg_process.stdout.read(self.chunk_size_bytes)
                if not raw_audio_chunk:
                    logger.info("FFmpeg stdout stream ended.")
                    break
                
                # Convert raw bytes to numpy array of float32
                # Ensure the number of bytes is a multiple of bytes_per_sample
                if len(raw_audio_chunk) % self.bytes_per_sample != 0:
                    logger.warning(f"Received incomplete audio chunk of {len(raw_audio_chunk)} bytes. Discarding.")
                    continue

                audio_array = np.frombuffer(raw_audio_chunk, dtype=np.float32)
                
                if audio_array.size > 0:
                    # Double the volume
                    amplified_audio_array = audio_array * 2.0
                    # Clip to prevent values outside the valid range [-1.0, 1.0]
                    amplified_audio_array = np.clip(amplified_audio_array, -1.0, 1.0)
                    logger.debug(f"FFmpeg audio chunk volume doubled. Original max_amp: {np.max(np.abs(audio_array)):.4f}, New max_amp: {np.max(np.abs(amplified_audio_array)):.4f}")
                    
                    self.audio_queue.append(amplified_audio_array)
                    logger.debug(f"Queued {amplified_audio_array.shape[0]} amplified audio samples from FFmpeg.")
                else:
                    logger.debug("Empty audio array after conversion, not queuing.")

        except Exception as e:
            logger.error(f"Error in FFmpeg reader thread: {e}", exc_info=True)
        finally:
            logger.info("FFmpeg reader thread finished.")
            if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                logger.info("Terminating FFmpeg process from reader thread finalization.")
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg process did not terminate gracefully, killing.")
                    self.ffmpeg_process.kill()
            self.stop_event.set() # Ensure other parts of the application know to stop

    def start_stream(self):
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            logger.warning("FFmpeg process already running.")
            return

        ffmpeg_command = [
            self.ffmpeg_path,
            '-loglevel', 'error', # Reduce ffmpeg's own console output
            '-f', 'pulse',       # Input format: PulseAudio
            '-i', self.monitor_source, # Input device
            '-f', 'f32le',       # Output format: 32-bit float, little-endian
            '-ar', str(self.sample_rate), # Output sample rate
            '-ac', str(self.channels),    # Output channels (mono)
            '-nostdin',          # Disable interaction on stdin
            '-'                  # Output to stdout
        ]

        try:
            logger.info(f"Starting FFmpeg stream with command: {' '.join(ffmpeg_command)}")
            self.ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Start the reader thread
            self.thread = threading.Thread(target=self._reader_thread, daemon=True)
            self.thread.start()
            logger.info("FFmpeg stream started and reader thread launched.")
            return True
        except FileNotFoundError:
            logger.error(f"Error: {self.ffmpeg_path} command not found. Please ensure ffmpeg is installed and in your PATH.")
            self.stop_event.set()
            return False
        except Exception as e:
            logger.error(f"Failed to start FFmpeg stream: {e}", exc_info=True)
            self.stop_event.set()
            return False

    def stop_stream(self):
        logger.info("Attempting to stop FFmpeg stream...")
        self.stop_event.set() # Signal the reader thread to stop

        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            logger.info("Terminating FFmpeg process...")
            self.ffmpeg_process.terminate() # Send SIGTERM
            try:
                self.ffmpeg_process.wait(timeout=5.0) # Wait for graceful shutdown
                logger.info("FFmpeg process terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg process did not terminate in time, killing.")
                self.ffmpeg_process.kill() # Force kill
                self.ffmpeg_process.wait() # Ensure it's dead
                logger.info("FFmpeg process killed.")
            except Exception as e:
                logger.error(f"Exception during FFmpeg process termination: {e}")
        else:
            logger.info("FFmpeg process was not running or already stopped.")

        if self.thread and self.thread.is_alive():
            logger.info("Waiting for FFmpeg reader thread to join...")
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("FFmpeg reader thread did not join in time.")
        logger.info("FFmpeg stream stop sequence complete.")

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
    
    # Use collections.deque for the audio_queue for efficient appends and pops
    import collections
    test_audio_queue = collections.deque()
    test_stop_event = threading.Event()

    # Create and start the streamer
    streamer = FFmpegStreamer(test_audio_queue, test_stop_event)
    
    if not streamer.start_stream():
        logger.error("Failed to start ffmpeg stream. Exiting.")
        exit(1)

    logger.info("Streaming started. Press Ctrl+C to stop.")
    
    try:
        processed_chunks = 0
        while not test_stop_event.is_set():
            if test_audio_queue:
                audio_chunk = test_audio_queue.popleft()
                processed_chunks += 1
                logger.info(f"Main: Popped audio chunk of {len(audio_chunk)} samples. Total chunks processed: {processed_chunks}")
                # In a real application, this chunk would be passed to a transcription model
            time.sleep(0.05) # Simulate work or prevent tight loop
            if processed_chunks > 50 and processed_chunks % 50 == 0: # Stop after some chunks for testing
                 logger.info("Test limit reached, stopping stream.")
                 # test_stop_event.set() # This would be set by external signal like Ctrl+C
    except KeyboardInterrupt:
        logger.info("Ctrl+C received, stopping stream...")
    finally:
        logger.info("Main: Initiating shutdown of streamer.")
        streamer.stop_stream()
        logger.info("Main: Streamer shutdown complete. Exiting.")