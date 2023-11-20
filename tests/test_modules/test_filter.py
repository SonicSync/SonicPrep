import unittest
import scipy.signal as signal

from sonicprep.modules import BandpassFilter, MultibandFilter, CustomMultibandFilter


def mock_filter(mock_audio, low_cutoff, high_cutoff, sample_rate):
    nyquist_freq = sample_rate * 0.5
    low_normalized = low_cutoff / nyquist_freq
    high_normalizer = high_cutoff / nyquist_freq
    b, a = signal.butter(4, [low_normalized, high_normalizer], btype='band')
    mock_filtered = signal.lfilter(b, a, mock_audio)
    return mock_filtered


class TestBandpassFilter(unittest.TestCase):
    def setUp(self):
        self.filter = BandpassFilter()

    def tearDown(self) -> None:
        self.filter = BandpassFilter()


class TestMultibandFilter(unittest.TestCase):

    def setUp(self):
        self.filter = MultibandFilter()
        self.sample_rate = 44100
        self.min_freq = 20.0
        self.max_freq = 20000.0
        self.nyquist_freq = self.sample_rate * 0.5


class TestCustomMultibandFilter(unittest.TestCase):

    def setUp(self):
        self.filter = CustomMultibandFilter()


if __name__ == "__main__":
    unittest.main()
