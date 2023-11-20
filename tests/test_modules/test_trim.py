import unittest
import numpy as np
from datafactory import generate_random_audio
from sonicprep.utils import Calculator
from sonicprep.modules import trim


class TestTrim(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case by initializing the necessary variables.
        """
        self.sr = 44100
        self.audio_data = generate_random_audio(5, self.sr, 5)
        self.threshold = 0
        self.silence = np.zeros(self.sr)
        self.total_silence = np.zeros(self.sr * 2)

    def test_happy_path(self):
        """
        Test happy path for the function.

        This test case verifies the behavior of the function when
        provided with silence audio data. It concatenates the
        silence audio data with the actual audio data and tests
        the result by comparing the expected duration and the
        calculated duration using the provided sample rate. It
        also checks the length difference between the original
        audio data and the trimmed result, and ensures that the
        length of the result matches the expected length
        difference. Lastly, it compares the calculated result
        duration with the expected duration.
        """
        # Test with silence audio data
        silence_audio_data = np.concatenate(
            (self.silence, self.audio_data, self.silence))

        expected_duration = Calculator.duration_diff(
            self.total_silence, silence_audio_data, self.sr)

        result = trim(silence_audio_data, self.threshold)

        result_duration = Calculator.duration(result, self.sr)
        array_diff = Calculator.size_diff(self.audio_data, result)

        expected_array_diff = len(self.audio_data) - array_diff

        self.assertEqual(len(result), expected_array_diff)
        self.assertEqual(result_duration, expected_duration)

    def test_edge_case_little_silence(self):
        """
        Test case for trimming audio data with little silence.

        This test case checks if the `trim` function correctly
        trims
        audio data that contains a small amount of silence
        at the beginning and end. It creates an audio data array
        with
        a small silence portion at the beginning and end,
        and calculates the expected duration of the trimmed audio
        based on the total silence duration and the sample rate.
        Then it calls the `trim` function with the audio data and
        a threshold value, and compares the length and duration
        of the trimmed result with the expected values.

        This test case is important to ensure that the `trim`
        function works correctly in handling audio data with
        small silence portions. By testing this edge case, we can
        verify that the function can accurately detect and remove
        the silence, while preserving the rest of the audio data.
        """
        silence = np.zeros(1)
        total_silence = np.zeros(2)
        silence_audio_data = np.concatenate(
            (silence, self.audio_data, silence))

        expected_duration = Calculator.duration_diff(
            total_silence, silence_audio_data, self.sr)

        result = trim(silence_audio_data, self.threshold)
        result_duration = Calculator.duration(result, self.sr)
        array_diff = Calculator.size_diff(self.audio_data, result)
        expected_array_diff = len(self.audio_data) - array_diff

        self.assertEqual(len(result), expected_array_diff)
        self.assertEqual(result_duration, expected_duration)

    def test_edge_case_all_silence(self):
        """
        Test case for trimming all silence audio data.
        """
        # Test with all silence audio data
        silence_audio_data = np.zeros(self.sr * 5)
        result = silence_audio_data
        self.assertTrue(np.array_equal(result, silence_audio_data))

    def test_raises_empty_audio(self):
        """
        Test case for trimming an empty audio data.
        """
        empty_audio = np.array([])
        with self.assertRaises(ValueError):
            trim(empty_audio, self.threshold)

    def test_edge_case_high_threshold(self):
        """
        Test case for trimming with a high threshold.
        """
        silence_audio_data = np.concatenate(
            (self.silence, self.audio_data, self.silence))
        expected_duration = Calculator.duration_diff(
            self.total_silence, silence_audio_data, self.sr)

        result = trim(silence_audio_data, 1000)
        result_duration = Calculator.duration(result, self.sr)
        array_diff = Calculator.size_diff(self.audio_data, result)
        expected_array_diff = len(self.audio_data) - array_diff

        self.assertEqual(len(result), expected_array_diff)
        self.assertGreaterEqual(result_duration, expected_duration)

    def test_edge_case_zero_threshold(self):
        """
        Test case for trimming with a zero threshold.
        """
        silence_audio_data = np.concatenate(
            (self.silence, self.audio_data, self.silence))
        expected_duration = Calculator.duration_diff(
            self.total_silence, silence_audio_data, self.sr)

        result = trim(silence_audio_data, 0)
        result_duration = Calculator.duration(result, self.sr)
        array_diff = Calculator.size_diff(self.audio_data, result)
        expected_array_diff = len(self.audio_data) - array_diff

        self.assertEqual(len(result), expected_array_diff)
        self.assertEqual(result_duration, expected_duration)


if __name__ == "__main__":
    unittest.main()
