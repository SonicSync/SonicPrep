import numpy as np
import librosa


def resample(audio_data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """
    Resamples audio data from the original sampling rate to the
    target sampling rate.

    Args:
    -----
    - `audio_data` (`np.ndarray`): The input audio data.
    - `original_sr` (`int`): The original sampling rate of the
    audio data.
    - `target_sr` (`int`): The target sampling rate to resample
    the audio data to.

    Returns:
    --------
    `np.ndarray`: The resampled audio data.
    """
    if not (40000 <= target_sr <= 120000):
        raise ValueError("Target sampling rate is out of range.")
    return librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr, scale=True)
