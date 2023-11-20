
import numpy as np
from .models import Variation
from .modules import normalize, trim, BandpassFilter, Chunker, resample


class Standardizer:

    def __init__(self, **kwargs):
        self._get_kwargs(**kwargs)
        self.chunker = Chunker()
        self.filter = BandpassFilter()

    def standardize(self, name, audio_data, sr):
        """
        Standardize the given audio by resampling, normalizing,
        trimming, filtering, and chunking it.

        Args:
        -----
        - `name` (`str`): The name of the audio.
        - `audio_data` (`np.ndarray`): The audio data to be
        standardized.
        - `sr` (`int`): The sample rate of the audio data.

        Returns:
        - `variants`: A list of standardized audio `Variation`
        objects.
        """
        resampled_audio = resample(audio_data, sr, self.target_sr)
        normalized_audio = normalize(resampled_audio, self.target_lvl)
        trimmed_audio = trim(normalized_audio, self.trim_thresh)
        filtered_audio = self.filter.filter(trimmed_audio)

        root_audio = Variation(name, name, filtered_audio)

        chunked_audio = self.chunker.chunk(filtered_audio)
        chunk_variants = [
            Variation(name, f'{name}_chunk_{i}', chunk)
            for i, chunk in enumerate(chunked_audio)]
        variants = chunk_variants + [root_audio]
        return variants

    def _get_kwargs(self, **kwargs):
        self.target_sr = kwargs.get('target_sr', 44100)
        self.target_lvl = kwargs.get('target_lvl', -0.1)
        self.normalizer_metric = kwargs.get('normalizer_metric', 'rms')
        self.trim_thresh = kwargs.get('trim_thresh', 0.1)
        self.chunk_duration = kwargs.get('chunk_duration', 30)
        self.filter_order = kwargs.get('filter_order', 4)
        self.filter_low = kwargs.get('filter_low', 20)
        self.filter_high = kwargs.get('filter_high', 20000)
