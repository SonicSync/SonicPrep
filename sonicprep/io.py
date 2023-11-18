import os
from typing import Tuple
import librosa
import numpy as np
from .exceptions import *


def validate_audio_type(file_path):
    """
    Validates the audio type of a given file.

    Args:
    -----
    - `file_path` (`str`): The path of the file to be validated.

    Raises:
    -------
    - `AudioTypeError`: If the file type is not supported.
    """
    accepted_extensions = {'.wav', '.mp3', '.flac'}
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in accepted_extensions:
        raise AudioTypeError(f"File type {extension} not supported.")


def load_audio_file(file_path: str, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file from the specified `file_path` and
    returns the audio data as a tuple of the audio waveform and
    the sample rate.

    Args:
    -----
    - `file_path` (`str`): The path to the audio file.
    - `sample_rate` (`int, optional`): The desired sample rate of
    the audio data. Defaults to 44100.

    Returns:
    --------
    - `tuple`: A tuple containing the audio data as a numpy array
    and the sample rate as an integer.

    Raises:
    -------
    - `AudioTypeError`: If the file type is not supported.
    """
    validate_audio_type(file_path)
    audio_data = librosa.load(file_path, sr=sample_rate)
    return audio_data
