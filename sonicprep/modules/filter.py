from typing import List, Tuple, Optional, Union
import scipy.signal as signal
from numpy import ndarray
from torch import Tensor


class BandpassFilter:
    def __init__(self, **kwargs):
        self.roll_off = kwargs.get('roll_off', 4)
        if not isinstance(self.roll_off, int):
            raise TypeError('roll_off must be an integer')
        self.sample_rate = kwargs.get('sample_rate', 44100)
        if not isinstance(self.sample_rate, int):
            raise TypeError('sample_rate must be an integer')
        self.nf = self.sample_rate / 2

    def filter(self, audio: Union[Tensor, ndarray], low_cutoff: Union[int, float] = 20,
               high_cutoff: Union[int, float] = 20000) -> ndarray | Tensor:
        """
        Apply a digital filter to the input audio signal.

        Args:
            audio (Union[Tensor, ndarray]): The input audio signal to be filtered.
            low_cutoff (Union[int, float]): The lower cutoff frequency for the filter in Hz. 
            Defaults to 20.
            high_cutoff (Union[int, float]): The higher cutoff frequency for the filter in Hz. 
            Defaults to 20000.

        Returns:
            ndarray | Tensor: The filtered audio signal.

        Raises:
            TypeError: If the input audio signal is not a tensor or ndarray.
            TypeError: If the low_cutoff or high_cutoff values are not integers or floats.
            ValueError: If the low_cutoff value is greater than the high_cutoff value.

        """

        self._validate_inputs(audio, low_cutoff, high_cutoff)
        b, a = self._design_filter(low_cutoff, high_cutoff)
        return signal.lfilter(b, a, audio)

    def _validate_inputs(self, audio, low_cutoff, high_cutoff):
        if not isinstance(audio, (Tensor, ndarray)):
            raise TypeError('audio_data must be Tensor or ndarray')
        if not isinstance(high_cutoff, (int, float)) or not isinstance(low_cutoff, (int, float)):
            raise TypeError('high and low cutoff values must be int or float')
        if low_cutoff >= high_cutoff:
            raise ValueError(
                'low cutoff value must be less than high cutoff value')

    def _design_filter(self, low: int, high: int) -> signal.butter:
        b, a = signal.butter(
            self.roll_off, [low/self.nf, high/self.nf], btype='band')
        return b, a


class MultibandFilter(BandpassFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def multiband_filter(self, audio: Union[Tensor, ndarray], bands: int) -> List[ndarray] | List[Tensor]:
        """
        Applies a multiband filter to the input audio signal.

        Args:
            audio (Union[Tensor, ndarray]): The input audio signal.
            bands (int): The number of bands for the multiband filter.
        Returns:
            List[ndarray] | List[Tensor]: A list of filtered audio signals.
        Raises:
            TypeError: If bands is not an integer.
            ValueError: If bands is less than or equal to 0.
        """
        if not isinstance(bands, int):
            raise TypeError('bands must be an integer')
        if bands <= 0:
            raise ValueError('bands must be greater than 0')

        bandwidths = self._calculate_bandwidths(bands)
        filtered_audio = [
            self.filter(audio, *self._calculate_band_cutoffs(i, bandwidths))
            for i in range(bands)
        ]
        return filtered_audio

    def _calculate_bandwidths(self, bands):
        return (20000.0 - 20.0) / bands

    def _calculate_band_cutoffs(self, bi, bandwidth):
        return 20.0 + bi * bandwidth, 20.0 + (bi + 1) * bandwidth


class CustomMultibandFilter(BandpassFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def multiband_filter(self, audio: Union[Tensor, ndarray],
                         ranges: List[Tuple[float or int, float or int]]) -> List[ndarray] | List[Tensor]:
        """
        Apply a multiband filter to the given audio signal.

        Args:
            audio (Union[Tensor, ndarray]): The audio signal to be filtered.
            ranges (List[Tuple[float or int, float or int]]): A list of frequency ranges 
                to apply the filter. Each range is represented as a tuple of low and high 
                frequency values.

        Returns:
            List[ndarray] | List[Tensor]: A list of filtered audio signals, one for each 
                frequency range.

        Raises:
            TypeError: If `ranges` is not a list or if `low` or `high` in `ranges` is not 
                an integer or a float.
        """
        print(type(ranges))
        if not isinstance(ranges, list):
            raise TypeError('ranges must be a list')
        return [self.filter(audio, low, high) for low, high in ranges
                if all([isinstance(low, (int, float)), isinstance(high, (int, float))])]
