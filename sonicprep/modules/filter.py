
from typing import List, Tuple, Optional, Union
import scipy.signal as signal
import numpy as np


class Filter:
    def __init__(self, order=4, sample_rate=44100, low=20, high=20000, **kwargs):
        self.order = order
        self.sample_rate = sample_rate
        self.low = low
        self.high = high
        self.btype = kwargs.get('btype', 'band')
        self.a, self.b = self._design_filter()

    def filter(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Filter the audio data using a given frequency range.

        Args:
            audio_data (Union[Audio, np.ndarray]): The audio data to filter.
            sample_rate (Optional[int]): The sample rate of the audio data.
            low (Union[int, float], optional): The low cutoff frequency. Defaults to 20.
            high (Union[int, float], optional): The high cutoff frequency. Defaults to 20000.

        Returns:
            np.ndarray: The filtered audio data or an Audio object with the filtered audio data.
        """
        if not isinstance(audio_data, np.ndarray):
            raise TypeError('audio must be an instance of Audio or np.ndarray')

        # Apply the filter to the audio data
        filtered_audio = signal.lfilter(self.b, self.a, audio_data, axis=0)
        return filtered_audio

    def _design_filter(self) -> signal.butter:
        """
        Designs a filter using the Butterworth method.

        Returns:
            signal.butter: The designed filter.
        """
        nyquist_freq = 0.5 * self.sample_rate
        low_normalized = self.low / nyquist_freq
        high_normalized = self.high / nyquist_freq
        return signal.butter(
            self.order, [low_normalized, high_normalized], btype=self.btype)
