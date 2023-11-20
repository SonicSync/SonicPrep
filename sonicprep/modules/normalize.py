from typing import Union
import numpy as np
import torch
from torch import Tensor


def _validate_args(audio_data, target_dB):
    if not isinstance(audio_data, np.ndarray):
        raise TypeError('audio must be a numpy array')
    if not isinstance(target_dB, (int, float)):
        raise TypeError('target_db must be an int or float')
    if target_dB > 0:
        raise ValueError('target_db must be negative')
    if audio_data.size == 0:
        raise ValueError("Input audio_data is empty.")
    if np.any((audio_data == 0) | np.isinf(audio_data)):
        raise ValueError(
            "Input audio_data contains NaN or infinite values.")


def normalize(audio: np.ndarray, target_db: float = -10.0) -> np.ndarray:
    """
    Normalizes the input audio to a target decibel level.

    Args:
        audio (Union[Tensor, np.ndarray]): The audio to be normalized.
        target_db (float, optional): The target decibel level. 
        Defaults to -10.0.

    Returns:
        np.ndarray: The normalized audio.

    Raises:
        TypeError: If audio is not a numpy array or target_db 
        is not an int or float.
        ValueError: If target_db is positive.
    """
    _validate_args(audio, target_db)
    current_db = 10 * np.log10(np.mean(audio**2))

    scaling_factor = 10 ** ((target_db - current_db) / 20)
    audio = audio * scaling_factor

    return audio
