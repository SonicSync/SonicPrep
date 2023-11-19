from typing import List
from typing import Literal
import numpy as np
import librosa
import pyloudnorm as pyln
from .modules.filter import Filter
from .models import Variation


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


class DynamicsNormalizer:
    metrics = ['rms', 'itu']

    def __init__(self, sr, target_lvl, metric: Literal['rms', 'itu'] = 'rms'):
        """
        Initializes a new instance of `DynamicsNormalizer`.

        Args:
        -----
        - `sr` (`int`): The sample rate of the audio.
        - `target_lvl` (`str`): The target of the audio.
        - `metric` (`Literal['rms', 'itu']`): The metric by which
        to normalize. rms normalizes the audio by the root mean
        square, and itu normalizes the audio by the integrated
        loudness. Defaults to `'rms'`.
        """
        self._validate_init_args(metric, target_lvl)
        self.sr = sr
        self.target = target_lvl
        self.metric = metric
        self.meter = pyln.Meter(self.sr) if self.metric == "itu" else None

    def normalize(self, audio_data: np.ndarray):
        """
        Normalize the given audio data dynamically based on the
        specified metric.

        Args:
        -----
        - `audio_data` (`np.ndarray`): The input audio data as a
        NumPy array.

        Returns:
        --------
        - `np.ndarray`: The normalized audio data as a NumPy
        array.
        """
        self._validate_audio_data(audio_data)

        current_level = {
            "rms": self._calculate_current_rms,
            "itu": self._calculate_current_il,
        }.get(self.metric)(audio_data)

        return self._scale_audio(audio_data, current_level)

    def _scale_audio(self, audio_data: np.ndarray, current_level: float) -> np.ndarray:
        return audio_data * 10 ** ((self.target - current_level) / 20)

    def _calculate_current_rms(self, audio_data: np.ndarray) -> np.ndarray:
        return 10 * np.log10(np.mean(audio_data ** 2))

    def _calculate_current_il(self, audio_data: np.ndarray) -> np.ndarray:
        return self.meter.integrated_loudness(audio_data)

    def _validate_audio_data(self, audio_data: np.ndarray):
        if audio_data.size == 0:
            raise ValueError("Input audio_data is empty.")
        if np.any((audio_data == 0) | np.isinf(audio_data)):
            raise ValueError(
                "Input audio_data contains NaN or infinite values.")

    def _validate_init_args(self, metric, target):
        valid_metrics = ", ".join(self.metrics)
        if metric not in self.metrics:
            raise ValueError(
                "Invalid metric: {}. Must be one of {}.".format(metric, valid_metrics))

        if target > 0:
            raise ValueError(
                "Invalid target: {}. Must be less than or equal to 0.".format(target))


class AudioChunker:
    def __init__(self, duration: int = 30, sr: int = 44100):
        """
        Initializes a new instance of `AudioChunker`.

        Args:
        -----
        - `duration` (`int`): The duration of each
        segment in seconds. Defaults to 30.
        - `sr` (`int`): The sample rate in Hz. Defaults
        to 44100.

        Raises:
        -------
        - `ValueError`: If the duration is less than or
        equal to 0.
        """
        self.duration = duration
        self.sr = sr

        if duration <= 0:
            raise ValueError('duration must be greater than 0')

    def chunk(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """
        Splits an audio signal into smaller chunks.

        If the duration of the audio signal is not evenly
        divisible by the `duration`, the last chunk will
        be shorter than the specified length.

        Args:
        -----
        - `audio_data` (`np.ndarray`): The audio signal to be
        chunked.

        Returns:
        --------
        - `List[np.ndarray]`: A list of chunked audio signals.
        """
        if not isinstance(audio_data, np.ndarray):
            raise ValueError('audio must be an instance of ndarray')

        return self._chunk_audio(audio_data, self.sr)

    def _calculate_segments(self, audio_data, sample_rate):
        duration_seconds = self.duration
        samples_per_segment = sample_rate * duration_seconds
        num_segments, remaining_samples = divmod(
            len(audio_data), samples_per_segment)
        return num_segments, remaining_samples, int(samples_per_segment)

    def _get_chunk(self, i, samples_per_segment, audio_data):
        return audio_data[i * samples_per_segment: (i + 1) * samples_per_segment]

    def _get_last_chunk(self, num_segments, samples_per_segment, remaining_samples, audio_data):
        start_index = num_segments * samples_per_segment
        end_index = start_index + remaining_samples
        return audio_data[start_index:end_index]

    def _chunk_audio(self, audio_data, sample_rate):
        num_segments, remaining_samples, samples_per_segment = self._calculate_segments(
            audio_data, sample_rate)

        chunks = [self._get_chunk(i, samples_per_segment, audio_data)
                  for i in range(num_segments)]
        chunks.append(self._get_last_chunk(
            num_segments, samples_per_segment, remaining_samples, audio_data)) if remaining_samples > 0 else None

        return chunks
