from typing import Literal
import numpy as np
import pyloudnorm as pyln


class DynamicsNormalizer:
    metrics = ['rms', 'itu']

    def __init__(self, sr, target, metric: Literal['rms', 'itu'] = 'rms'):
        """
        Initializes a new instance of `DynamicsNormalizer`.

        Args:
        -----
        - `sr` (`int`): The sample rate of the audio.
        - `target` (`str`): The target of the audio.
        - `metric` (`Literal['rms', 'itu']`): The metric by which
        to normalize. rms normalizes the audio by the root mean
        square, and itu normalizes the audio by the integrated
        loudness. Defaults to `'rms'`.
        """
        self._validate_init_args(metric, target)
        self.sr = sr
        self.target = target
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
