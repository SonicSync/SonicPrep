import numpy as np
import librosa


def trim(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Trims the given audio data by removing the silent parts
    based on the threshold.

    Args:
    -----
    - `audio_data` (`np.ndarray`): The input audio data as a
    NumPy array.
    - `threshold` (`float, optional`): The threshold value for
    determining silence. Defaults to `0.01`.

    Returns:
    --------
    - `np.ndarray`: The trimmed audio data as a NumPy array.
    """
    if not isinstance(audio_data, np.ndarray):
        raise TypeError('audio_data must be an instance of np.ndarray')
    if len(audio_data) == 0:
        raise ValueError('Audio data is empty.')

    envelope = np.abs(librosa.effects.preemphasis(audio_data))
    non_silent_frames = np.argwhere(envelope > threshold).flatten()

    if non_silent_frames.size > 0:
        start_frame, end_frame = non_silent_frames[0], non_silent_frames[-1]
        audio_data = audio_data[start_frame:end_frame]

    return audio_data
