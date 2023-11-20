import logging
from typing import List, Union
import numpy as np
import torch
from torch import Tensor
# import demucs.api as demucs


def _remodel(audio: Union[np.ndarray, Tensor], channels: int) -> Tensor:

    if not isinstance(audio, (np.ndarray, Tensor)):
        raise TypeError('audio must be a numpy array or Tensor.')
    if channels < 1:
        raise ValueError('channels must be a positive integer.')

    audio = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
    audio = audio.to(torch.float32)

    return audio.view(channels, -1)


def split_sources(audio: Union[Tensor, np.ndarray], **kwargs) -> Union[List[Tensor], List[np.ndarray]]:
    """
    Splits the given audio sources into individual sources.

    Parameters:
    ----------
        audio (Union[Tensor, np.ndarray]): The audio data to be split.
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        Union[List[Tensor], List[np.ndarray]]: A list of individual audio sources.

    Raises:
    -------
        TypeError: If audio is not a numpy array or Tensor.
        ValueError: If channels is not an integer.
    """
    # Initialize persist_np flag
    persist_np = False

    # Get the number of channels from kwargs
    channels = kwargs.get('channels', 2)

    # Check if audio is a numpy array or Tensor
    if not isinstance(audio, (np.ndarray, Tensor)):
        raise TypeError('audio must be a numpy array or Tensor.')

    # Check if channels is an integer
    if not isinstance(channels, int):
        raise ValueError('channels must be an integer.')

    # Check if audio is a numpy array
    if isinstance(audio, np.ndarray):
        # Set persist_np flag to True
        persist_np = kwargs.get('persist_np', True)

    # Remodel the audio based on the number of channels
    audio = _remodel(audio, channels)

    # Get the model from kwargs
    model = kwargs.get('model', 'htdemucs_6s')

    # Initialize the separator with the model
    separator = demucs.Separator(model=model)

    # Separate the audio into sources
    sources = list(separator.separate_tensor(audio))

    # Convert sources to numpy arrays if persist_np is True
    if persist_np:
        sources = [s.numpy() for s in sources]

    # Return the list of sources
    return sources
