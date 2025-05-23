import logging
import sounddevice as sd
import numpy as np

# Configure logger for this module
logger = logging.getLogger(__name__)

def list_audio_devices():
    """Lists available audio devices with more details."""
    logger.info("Listing available audio devices from audio_handler module:")
    try:
        devices = sd.query_devices()
        host_apis = sd.query_hostapis() # This is a list of host API dicts
        for i, device in enumerate(devices):
            device_info_str = f"  {i}: {device['name']}"
            device_info_str += f" (Input Channels: {device['max_input_channels']}, Output Channels: {device['max_output_channels']})"
            device_info_str += f" (Default Sample Rate: {device['default_samplerate']})"
            
            host_api_index = device['hostapi']
            if 0 <= host_api_index < len(host_apis):
                host_api_info = host_apis[host_api_index]
                device_info_str += f" (Host API: {host_api_info['name']})"
                if host_api_info['name'] == 'ALSA':
                     device_info_str += f" (ALSA device name in sounddevice: '{device['name']}')"
            else:
                device_info_str += f" (Host API Index: {host_api_index} - Invalid)"

            logger.info(device_info_str)
            
    except Exception as e:
        logger.error(f"Could not query audio devices from audio_handler: {e}", exc_info=True)

def sounddevice_audio_callback(indata, frames, time, status, audio_queue, audio_buffer_lock):
    """
    This is called (from a separate thread) for each audio block by sounddevice.
    It adds the incoming audio data to the shared audio_queue.
    """
    if status:
        logger.warning(f"Sounddevice audio callback status: {status} (from audio_handler)")
    
    max_amplitude_chunk = np.max(np.abs(indata))
    logger.debug(f"Audio chunk received via sounddevice: frames={frames}, max_amplitude={max_amplitude_chunk:.4f}, status={status} (from audio_handler)")
    
    # Double the volume
    amplified_indata = indata * 2.0
    # Clip to prevent values outside the valid range [-1.0, 1.0] after amplification
    amplified_indata = np.clip(amplified_indata, -1.0, 1.0)
    logger.debug(f"Audio chunk volume doubled. Original max_amp: {max_amplitude_chunk:.4f}, New max_amp: {np.max(np.abs(amplified_indata)):.4f}")

    with audio_buffer_lock:
        audio_queue.append(amplified_indata.copy())

if __name__ == '__main__':
    # Basic test for this module
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing audio_handler.py...")
    list_audio_devices()
    # Note: Testing the callback would require setting up a stream,
    # which is more involved and better tested in the main application.
    logger.info("audio_handler.py test complete.")