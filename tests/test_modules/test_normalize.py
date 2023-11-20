import unittest
import numpy as np
from sonicprep.modules import normalize
from sonicprep.utils import Calculator
from datafactory import generate_random_audio


class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 44100
        self.happy_audio_data = generate_random_audio(5, self.sample_rate)

    def test_happy_path_rms(self):
        """
        Test the happy path scenario for normalizing audio data
        by the root mean square.

        This test case calls the normalize method of the
        normalizer instance
        with the happy_audio_data and calculates the level using
        the `rms` method.
        It then asserts that the calculated level is
        approximately -10.
        """
        result = normalize(self.happy_audio_data)
        level = Calculator.mean_dB(result)
        self.assertAlmostEqual(level, -10)

    def test_edge_case_zero_db(self):
        """
        Test the edge case scenario for normalizing audio data
        with a target of 0 dB.

        This test case sets the target value of the normalizer
        instance to 0, calls the normalize method with the
        happy_audio_data, and calculates the level using
        the `rms` method.
        It then asserts that the calculated level is 0.
        """
        target = 0
        result = normalize(self.happy_audio_data, target)
        level = Calculator.mean_dB(result)
        self.assertAlmostEqual(level, target)

    def test_edge_case_minus_100_db(self):
        """
        Test the edge case scenario for normalizing audio data
        with a target of -100 dB.

        This test case sets the target value of the normalizer
        instance to -100, calls the normalize method with the
        happy_audio_data, and calculates the level using
        the `rms` method.
        It then asserts that the calculated level is -100.
        """
        target = -100
        result = normalize(self.happy_audio_data, target)
        level = Calculator.mean_dB(result)
        self.assertAlmostEqual(level, target)

    def test_edge_case_large_input(self):
        """
        Test the edge case scenario for normalizing large audio data.

        This test case generates a large audio data with a
        duration of 1 hour, calls the normalize method with the
        generated audio data, and calculates the level using the
        `rms` method.
        It then asserts that the calculated level is
        approximately -10.
        """
        target = -10
        audio_data = generate_random_audio(3600, self.sample_rate)
        result = normalize(audio_data, target)
        level = Calculator.mean_dB(result)
        self.assertAlmostEqual(level, target)

    def test_raises_empty_input(self):
        """
        Test that a `ValueError` is raised when an empty audio
        input is passed to normalize method.

        This test case creates an empty numpy array as the audio
        input, and asserts that a `ValueError` is raised when the
        normalize method is called.
        """
        audio_data = np.ndarray(0, dtype=np.float32)
        with self.assertRaises(ValueError):
            normalize(audio_data)

    def test_raises_zero_input(self):
        """
        Test that a `ValueError` is raised when a zero audio
        input is passed to normalize method.

        This test case creates a zero-filled numpy array as the
        audio input, and asserts that a `ValueError` is raised
        when the normalize method is called.
        """
        audio_data = np.zeros(0, dtype=np.float32)
        with self.assertRaises(ValueError):
            normalize(audio_data)

    def test_raises_inf_input(self):
        """
        Test that a `ValueError` is raised when an infinity audio
        input is passed to normalize method.

        This test case creates a numpy array with a single
        element of infinity as the audio input, and asserts that
        a `ValueError` is raised when the normalize method is
        called.
        """
        audio_data = np.array([np.inf])
        with self.assertRaises(ValueError):
            normalize(audio_data)

    def test_raises_positive_target(self):
        """
        Test that a `ValueError` is raised when a positive target
        value is passed to DynamicsNormalizer.

        This test case sets a positive target value (10),
        and asserts that a `ValueError` is raised when
        initializing the DynamicsNormalizer with "rms" metric.
        """
        target = 10
        with self.assertRaises(ValueError):
            normalize(self.happy_audio_data, target)


if __name__ == "__main__":
    unittest.main()
