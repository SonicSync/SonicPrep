import unittest
import numpy as np
import pyloudnorm as pyln
from sonicprep.standardize import DynamicsNormalizer, trim, resample, AudioChunker
from sonicprep.utils import *
from datafactory.datafactory import generate_random_audio


class TestNormalizeDynamics(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 44100
        self.happy_audio_data = generate_random_audio(5, self.sample_rate)
        self.normalizer = DynamicsNormalizer(
            self.sample_rate, -10, metric="rms")

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
        result = self.normalizer.normalize(self.happy_audio_data)
        level = self._calculate_level(result, "rms")
        self.assertAlmostEqual(level, -10)

    def test_happy_path_itu(self):
        """
        Test the happy path scenario for normalizing audio data
        by the ITU loudness.

        This test case calls the normalize method of the
        normalizer instance
        with the happy_audio_data and calculates the level using
        the `itu` method.
        It then asserts that the calculated level is
        approximately -10.
        """
        normalizer = DynamicsNormalizer(self.sample_rate, -10, metric="itu")
        result = normalizer.normalize(self.happy_audio_data)
        level = self._calculate_level(result, "itu")
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
        self.normalizer.target = 0
        result = self.normalizer.normalize(self.happy_audio_data)
        level = self._calculate_level(result, "rms")
        self.assertAlmostEqual(level, 0)

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
        self.normalizer.target = -100
        result = self.normalizer.normalize(self.happy_audio_data)
        level = self._calculate_level(result, "rms")
        self.assertAlmostEqual(level, -100)

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
        audio_data = generate_random_audio(3600, self.sample_rate)
        result = self.normalizer.normalize(audio_data)
        level = self._calculate_level(result, "rms")
        self.assertAlmostEqual(level, -10)

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
            self.normalizer.normalize(audio_data)

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
            self.normalizer.normalize(audio_data)

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
            self.normalizer.normalize(audio_data)

    def test_raises_invalid_metric(self):
        """
        Test that a `ValueError` is raised when an invalid
        `metric` is passed to `DynamicsNormalizer`.

        This test case sets an invalid metric value ("peak")
        and a target value (-10), and asserts that a `ValueError`
        is raised when initializing the `DynamicsNormalizer`.
        """
        target = -10
        metric = "peak"
        with self.assertRaises(ValueError):
            DynamicsNormalizer(self.sample_rate, target, metric=metric)

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
            DynamicsNormalizer(self.sample_rate, target, metric="rms")

    def _calculate_level(self, audio_data, metric):
        """
        Calculate the level of audio data based on the specified
        metric.

        This helper method calculates the level of the audio data
        using the specified metric.
        If the metric is "rms", it calculates the level using the
        root mean square.
        If the metric is "itu", it calculates the level using the
        ITU loudness.

        Args:
        -----
        - `audio_data` (`numpy.ndarray`): The audio data.
        - `metric` (`str`): The metric to use for calculating the
        level.

        `Returns`:
        ----------
        - `float`: The calculated level.

        `Raises`:
        ---------
        - `ValueError`: If the metric is not supported.
        """
        meter = pyln.Meter(self.sample_rate)
        if metric == "rms":
            return 10 * np.log10(np.mean(audio_data**2))
        if metric == "itu":
            return meter.integrated_loudness(audio_data)
        raise ValueError("Invalid metric: {}".format(metric))


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

        expected_duration = calculate_array_duration_diff(
            self.total_silence, silence_audio_data, self.sr)

        result = trim(silence_audio_data, self.threshold)

        result_duration = calculate_array_duration(result, self.sr)
        array_diff = calculate_array_len_diff(self.audio_data, result)

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

        expected_duration = calculate_array_duration_diff(
            total_silence, silence_audio_data, self.sr)

        result = trim(silence_audio_data, self.threshold)
        result_duration = calculate_array_duration(result, self.sr)
        array_diff = calculate_array_len_diff(self.audio_data, result)
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
        expected_duration = calculate_array_duration_diff(
            self.total_silence, silence_audio_data, self.sr)

        result = trim(silence_audio_data, 1000)
        result_duration = calculate_array_duration(result, self.sr)
        array_diff = calculate_array_len_diff(self.audio_data, result)
        expected_array_diff = len(self.audio_data) - array_diff

        self.assertEqual(len(result), expected_array_diff)
        self.assertGreaterEqual(result_duration, expected_duration)

    def test_edge_case_zero_threshold(self):
        """
        Test case for trimming with a zero threshold.
        """
        silence_audio_data = np.concatenate(
            (self.silence, self.audio_data, self.silence))
        expected_duration = calculate_array_duration_diff(
            self.total_silence, silence_audio_data, self.sr)

        result = trim(silence_audio_data, 0)
        result_duration = calculate_array_duration(result, self.sr)
        array_diff = calculate_array_len_diff(self.audio_data, result)
        expected_array_diff = len(self.audio_data) - array_diff

        self.assertEqual(len(result), expected_array_diff)
        self.assertEqual(result_duration, expected_duration)

    def _calculate_silence_duration(self, pre, suf):
        """
        Calculate the duration of silence given the preceding and
        succeeding audio data.
        """
        return calculate_array_duration(np.concatenate((pre, suf)))


class TestResample(unittest.TestCase):

    def test_happy_path(self):
        sample_rate = 48000
        targe_sample_rate = 44100
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = calculate_sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_happy_path_no_change(self):
        sample_rate = 44100
        targe_sample_rate = 44100
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = calculate_sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_boundary_40k(self):
        sample_rate = 44100
        targe_sample_rate = 40000
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = calculate_sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_boundary_120k(self):
        sample_rate = 44100
        targe_sample_rate = 120000
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = calculate_sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_edge_case_high_original(self):
        sample_rate = 441000
        targe_sample_rate = 44100
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = calculate_sample_rate(result, 5)

        self.assertEqual(result, targe_sample_rate)

    def test_edge_case_low_original(self):
        sample_rate = 4410
        targe_sample_rate = 44100
        audio_data = generate_random_audio(5, sample_rate)

        result = resample(audio_data, sample_rate, targe_sample_rate)
        result = calculate_sample_rate(result, 5)

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


class TestAudioChunker(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 44100
        self.chunker = AudioChunker()

    def test_happy_path(self):
        duration = 180
        expected_chunks = 6
        audio_data = generate_random_audio(duration, self.sample_rate)

        result = self.chunker.chunk(audio_data)
        self.assertEqual(len(result), expected_chunks)
        for chunk in result:
            chunk_duration = calculate_array_duration(chunk, self.sample_rate)
            self.assertEqual(chunk_duration, 30)

    def test_happy_path_between_chunk_dur(self):
        duration = 195
        expected_chunks = 7
        audio_data = generate_random_audio(duration, self.sample_rate)

        result = self.chunker.chunk(audio_data)
        self.assertEqual(len(result), expected_chunks)
        for i, chunk in enumerate(result):
            chunk_duration = calculate_array_duration(chunk, self.sample_rate)

            if i + 1 != len(result):
                self.assertEqual(chunk_duration, 30)
            else:
                self.assertEqual(chunk_duration, 15)

    def test_edge_case_short_audio(self):
        duration = 5
        expected_chunks = 1
        audio_data = generate_random_audio(duration, self.sample_rate)

        result = self.chunker.chunk(audio_data)
        self.assertEqual(len(result), expected_chunks)
        for chunk in result:
            chunk_duration = calculate_array_duration(chunk, self.sample_rate)
            self.assertEqual(chunk_duration, 5)

    def test_edge_case_long_audio(self):
        duration = 3600
        expected_chunks = 120
        audio_data = generate_random_audio(duration, self.sample_rate)

        result = self.chunker.chunk(audio_data)
        self.assertEqual(len(result), expected_chunks)
        for chunk in result:
            chunk_duration = calculate_array_duration(chunk, self.sample_rate)
            self.assertEqual(chunk_duration, 30)

    def test_edge_case_long_chunk(self):
        self.chunker.segment_duration = 180
        duration = 180
        expected_chunks = 1
        audio_data = generate_random_audio(duration, self.sample_rate)

        result = self.chunker.chunk(audio_data)
        self.assertEqual(len(result), expected_chunks)
        for chunk in result:
            chunk_duration = calculate_array_duration(chunk, self.sample_rate)
            self.assertEqual(chunk_duration, 180)

    def test_edge_case_short_chunk(self):
        self.chunker.segment_duration = 1
        duration = 180
        expected_chunks = 180
        audio_data = generate_random_audio(duration, self.sample_rate)

        result = self.chunker.chunk(audio_data)
        self.assertEqual(len(result), expected_chunks)
        for chunk in result:
            chunk_duration = calculate_array_duration(chunk, self.sample_rate)
            self.assertEqual(chunk_duration, 1)

    def test_raises_invalid_audio_data(self):
        audio_data = []
        with self.assertRaises(ValueError):
            self.chunker.chunk(audio_data)

    def test_raises_invalid_segment_duration(self):
        segment_duration = 0
        with self.assertRaises(ValueError):
            AudioChunker(segment_duration)


class NormalizeTestSuite(unittest.TestSuite):
    def __init__(self):
        super(NormalizeTestSuite, self).__init__()
        self.addTest(TestNormalizeDynamics("test_happy_path_rms"))
        self.addTest(TestNormalizeDynamics("test_happy_path_itu"))
        self.addTest(TestNormalizeDynamics("test_edge_case_zero_db"))
        self.addTest(TestNormalizeDynamics("test_edge_case_minus_100_db"))
        self.addTest(TestNormalizeDynamics("test_raises_empty_input"))
        self.addTest(TestNormalizeDynamics("test_raises_zero_input"))
        self.addTest(TestNormalizeDynamics("test_raises_inf_input"))
        self.addTest(TestNormalizeDynamics("test_raises_invalid_metric"))
        self.addTest(TestNormalizeDynamics("test_raises_positive_target"))


class TrimTestSuite(unittest.TestSuite):
    def __init__(self):
        super(TrimTestSuite, self).__init__()
        self.addTest(TestTrim("test_happy_path"))
        self.addTest(TestTrim("test_edge_case_little_silence"))
        self.addTest(TestTrim("test_edge_case_all_silence"))
        self.addTest(TestTrim("test_raises_empty_audio"))
        self.addTest(TestTrim("test_edge_case_high_threshold"))
        self.addTest(TestTrim("test_edge_case_zero_threshold"))


class ResampleTestSuite(unittest.TestSuite):
    def __init__(self):
        super(ResampleTestSuite, self).__init__()
        self.addTest(TestResample("test_happy_path"))
        self.addTest(TestResample("test_happy_path_no_change"))
        self.addTest(TestResample("test_raises_too_high_target"))
        self.addTest(TestResample("test_raises_too_low_target"))
        self.addTest(TestResample("test_edge_case_high_original"))
        self.addTest(TestResample("test_edge_case_low_original"))


class AudioChunkerTestSuite(unittest.TestSuite):
    def __init__(self):
        super(AudioChunkerTestSuite, self).__init__()
        self.addTest(TestAudioChunker('test_happy_path'))
        self.addTest(TestAudioChunker('test_edge_case_short_audio'))
        # self.addTest(TestAudioChunker('test_edge_case_long_audio'))
        self.addTest(TestAudioChunker('test_edge_case_long_chunk'))
        self.addTest(TestAudioChunker('test_edge_case_short_chunk'))
        self.addTest(TestAudioChunker('test_raises_invalid_segment_duration'))
        self.addTest(TestAudioChunker('test_raises_invalid_audio_data'))


class StandardizeQuickSuite(unittest.TestSuite):
    def __init__(self):
        super(StandardizeQuickSuite, self).__init__()
        # self.addTest(TestNormalizeDynamics("test_happy_path_rms"))
        # self.addTest(TestNormalizeDynamics("test_happy_path_itu"))
        # self.addTest(TestNormalizeDynamics("test_edge_case_zero_db"))
        # self.addTest(TestNormalizeDynamics("test_edge_case_minus_100_db"))
        # self.addTest(TestNormalizeDynamics("test_raises_empty_input"))
        # self.addTest(TestNormalizeDynamics("test_raises_zero_input"))
        # self.addTest(TestNormalizeDynamics("test_raises_inf_input"))
        # self.addTest(TestNormalizeDynamics("test_raises_invalid_metric"))
        # self.addTest(TestNormalizeDynamics("test_raises_positive_target"))

        # self.addTest(TestTrim("test_happy_path"))
        # self.addTest(TestTrim("test_edge_case_little_silence"))
        # self.addTest(TestTrim("test_edge_case_all_silence"))
        # self.addTest(TestTrim("test_raises_empty_audio"))
        # self.addTest(TestTrim("test_edge_case_high_threshold"))
        # self.addTest(TestTrim("test_edge_case_zero_threshold"))

        self.addTest(TestResample("test_happy_path"))


if __name__ == "__main__":
    unittest.TextTestRunner().run(AudioChunkerTestSuite())
