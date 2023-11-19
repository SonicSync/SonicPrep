from .standardize import *
from .models import *
from .augment import *
from .extract import *
from .io import *


class Standardizer:

    def __init__(self, **kwargs):
        self._get_kwargs(**kwargs)
        self.normalizer = DynamicsNormalizer(
            sr=self.target_sr, target_lvl=self.target_lvl, metric=self.normalizer_metric)
        self.chunker = AudioChunker(
            duration=self.chunk_duration, sr=self.target_sr)
        self.filter = Filter(order=self.filter_order, sr=self.target_sr,
                             low=self.filter_low, high=self.filter_high)

    def standardize(self, name, audio_data, sr):
        """
        Standardize the given audio by resampling, normalizing,
        trimming, filtering, and chunking it.

        Args:
        -----
        - `name` (`str`): The name of the audio.
        - `audio_data` (`np.ndarray`): The audio data to be
        standardized.s
        - `sr` (`int`): The sample rate of the audio data.

        Returns:
        - `variants`: A list of standardized audio `Variation`
        objects.
        """
        resampled_audio = resample(audio_data, sr, self.target_sr)
        normalized_audio = self.normalizer.normalize(resampled_audio)
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


class Prepper:
    def __init__(self, output_dir, output_type, **kwargs):
        self.output_dir = output_dir
        self.output_type = output_type
        self.batch_prep = kwargs.get('batch_prep', True)
        parsed_kwargs = self._parse_kwargs(**kwargs)
        self.standardizer = Standardizer(**parsed_kwargs[0])

    def prep(self, path):
        if self.batch_prep:
            all_files = find_all_files(path)
            for batch in audio_batch_generator(all_files):
                for i, name, audio_data, sr in enumerate(batch):
                    self._run_processes(name, audio_data, sr)

        else:
            pass

    def _run_processes(self, i, name, audio_data, sr):
        batch_num = i + 1
        variations = self.standardizer.standardize(name, audio_data, sr)

    def _parse_kwargs(self, **kwargs):
        standardizer_kwargs = {
            'target_sr': kwargs.get('target_sr', 44100),
            'target_lvl': kwargs.get('target_lvl', -0.1),
            'normalizer_metric': kwargs.get('normalizer_metric', 'rms'),
            'trim_thresh': kwargs.get('trim_thresh', 0.1),
            'chunk_duration': kwargs.get('chunk_duration', 30),
            'filter_order': kwargs.get('filter_order', 4),
            'filter_low': kwargs.get('filter_low', 20),
            'filter_high': kwargs.get('filter_high', 20000),
        }

        augmenter_kwargs = {

        }

        extractor_kwargs = {

        }

        return standardizer_kwargs, augmenter_kwargs, extractor_kwargs
