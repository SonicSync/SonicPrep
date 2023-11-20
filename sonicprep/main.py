from .augment import Augmenter
from .standardize import *
from .models import *
from .standardize import *
from .extract import *
from .io import *


class Prepper:
    def __init__(self, output_dir, output_type, **kwargs):
        self.output_dir = output_dir
        self.output_type = output_type
        self.batch_prep = kwargs.get('batch_prep', True)
        parsed_kwargs = self._parse_kwargs(**kwargs)
        self.standardizer = Standardizer(**parsed_kwargs[0])
        self.augmenter = Augmenter(**parsed_kwargs[1])

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
        variations = [self.augmenter.apply_combinations(v) for v in variations]
        variations = [self.standardizer.standardize(
            name, v, sr) for v in variations]
        return variations

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
            'vars': kwargs.get('vars', 3),
            'ir_dir': kwargs.get('ir_dir', os.path.abspath('impulses')),
        }

        extractor_kwargs = {

        }

        return standardizer_kwargs, augmenter_kwargs, extractor_kwargs
