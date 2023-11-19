import unittest
import numpy as np
from sonicprep.modules.filter import Filter
from sonicprep.utils import analyze_frequency
from datafactory.datafactory import generate_audio_with_freqs, generate_frequency_response


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.sr = 44100
        self.dur = 5

    def test_happy_path(self):
        pass

    def test_edge_case_broad_bandwidth(self):
        pass

    def test_edge_case_narrow_bandwidth(self):
        pass

    def test_raises_invalid_audio_data(self):
        pass


if __name__ == "__main__":
    unittest.main()
