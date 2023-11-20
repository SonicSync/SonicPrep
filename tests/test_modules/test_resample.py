import unittest
import librosa
import soundfile as sf
import tempfile
import os
from sonicprep.modules import resample
from sonicprep.utils import Calculator
from datafactory import generate_random_audio


class TestResample(unittest.TestCase):

    def test_happy_path(self):
        sample_rate = 48000
        targe_sample_rate = 44100
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = Calculator.sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_happy_path_no_change(self):
        sample_rate = 44100
        targe_sample_rate = 44100
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = Calculator.sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_boundary_40k(self):
        sample_rate = 44100
        targe_sample_rate = 40000
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = Calculator.sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_boundary_120k(self):
        sample_rate = 44100
        targe_sample_rate = 120000
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = Calculator.sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_edge_case_high_original(self):
        sample_rate = 441000
        targe_sample_rate = 44100
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = Calculator.sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_edge_case_low_original(self):
        sample_rate = 4410
        targe_sample_rate = 44100
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = Calculator.sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_raises_too_low_target(self):
        sample_rate = 44100
        targe_sample_rate = 4000
        audio_data = generate_random_audio(5, sample_rate)

        with self.assertRaises(ValueError):
            resample(audio_data, sample_rate, targe_sample_rate)

    def test_raises_too_high_target(self):
        sample_rate = 44100
        targe_sample_rate = 120001
        audio_data = generate_random_audio(5, sample_rate)

        with self.assertRaises(ValueError):
            resample(audio_data, sample_rate, targe_sample_rate)


if __name__ == "__main__":
    unittest.main()
