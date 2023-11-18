from itertools import islice
from typing import List
import os
import glob
from typing import Tuple
import librosa
import pydub
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
    accepted_ext = {'.wav', '.mp3', '.flac'}
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in accepted_ext:
        raise AudioTypeError(f"File type {extension} not supported.")


def load_mp3_audio(file_path: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Loads an MP3 audio file from the specified `file_path` and
    returns the audio data as a tuple of the audio waveform and
    the sample rate.

    Args:
    -----
    - `file_path` (`str`): The path to the audio file.
    - `sr` (`int, optional`): The desired sample rate of
    the audio data. Defaults to 44100.

    Returns:
    --------
    - `tuple`: A tuple containing the audio data as a numpy array
    and the sample rate as an integer.

    """
    a: pydub.AudioSegment = pydub.AudioSegment.from_file(
        file_path, format="mp3")
    a.get_array_of_samples()
    y = np.array(a.get_array_of_samples())
    y = y.astype(np.float32)
    if a.channels == 2:
        y = y.reshape((-1, 2))
    return y, a.frame_rate


def load_audio_file(file_path: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file from the specified `file_path` and
    returns the audio data as a tuple of the audio waveform and
    the sample rate.

    Args:
    -----
    - `file_path` (`str`): The path to the audio file.
    - `sr` (`int, optional`): The desired sample rate of
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
    try:
        print('LOADING NON-MP3')
        audio_data = librosa.load(file_path, sr=sr)
        print('NON-MP3 SHAPE', audio_data[0].shape)
    except AttributeError:
        print('ATTRIBUTE ERROR')
        audio_data = load_mp3_audio(file_path, sr=sr)
    return audio_data


def find_all_files(dir: str, ext=['mp3', 'flac', 'wav']):
    """
    Iterate over audio files in a directory with specified ext.

    Args:
    -----
    - `dir`: Path to the directory containing audio files.
    - `ext`: List of file ext to consider.

    Returns:
    --------
    - `files`: List of paths to audio files.

    Raises:
    -------
    - `NoFilesError`: If no audio files are found in the directory.
    """
    files = []

    for extension in ext:
        pattern = os.path.join(dir, f'*.{extension}')
        files.extend(glob.glob(pattern))

    if files == []:
        raise NoFilesError("No audio files found in the specified directory.")

    return files


def generate_file_batches(files: List[str], batch_size: int) -> List[List]:
    """
    Generates batches of files from a given list of files.

    Args:
    -----
    - `files` (`list`): A list of files to be processed.
    - `batch_size` (`int`): The number of files in each batch.

    Yields:
    -------
    - `list`: A batch of files.

    """
    it = iter(files)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            return
        yield batch


def audio_batch_generator(files: List[str], batch_size: int = 10) -> List[List]:
    """
    Generate batches of audio files.

    Args:
    - `files` (List[str]): A list of audio file paths.
    - `batch_size` (int, optional): The number of audio files per
    batch. Defaults to 10.

    Yields:
    - `List[List]`: A batch of loaded audio files.
    """
    batches = generate_file_batches(files, batch_size)
    for batch in batches:
        yield [load_audio_file(file) for file in batch]
