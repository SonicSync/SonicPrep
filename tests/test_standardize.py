import unittest
import numpy as np
import pyloudnorm as pyln
from sonicprep.standardize import DynamicsNormalizer
from datafactory.datafactory import generate_random_audio


class TestNormalizeDynamics(unittest.TestCase):
    def setUp(self):
        self.sr = 44100
        self.happy_audio_data = generate_random_audio(
            5, self.sr)

    def test_happy_path_mean(self):
        target = -10

        normalizer = DynamicsNormalizer(
            self.sr, target, metric="rms")

        result = normalizer.normalize(self.happy_audio_data)
        result = self._calculate_lvl(result, "rms")
        self.assertAlmostEqual(result, target)

    def test_happy_path_itu(self):
        target = -10

        normalizer = DynamicsNormalizer(
            self.sr, target, metric="itu")

        result = normalizer.normalize(self.happy_audio_data)
        result = self._calculate_lvl(result, "itu")
        self.assertAlmostEqual(result, target)

    def test_edge_case_zero_db(self):
        target = 0

        normalizer = DynamicsNormalizer(
            self.sr, target, metric="rms")

        result = normalizer.normalize(self.happy_audio_data)
        result = self._calculate_lvl(result, "rms")
        self.assertAlmostEqual(result, target)

    def test_edge_case_minus_100_db(self):
        target = -100

        normalizer = DynamicsNormalizer(
            self.sr, target, metric="rms")

        result = normalizer.normalize(self.happy_audio_data)
        result = self._calculate_lvl(result, "rms")
        self.assertAlmostEqual(result, target)

    def test_edge_case_large_input(self):
        target = -10
        audio_data = generate_random_audio(3600, self.sr)

        normalizer = DynamicsNormalizer(
            self.sr, target, metric="rms")

        result = normalizer.normalize(audio_data)
        result = self._calculate_lvl(result, "rms")
        self.assertAlmostEqual(result, target)

    def test_raises_empty_input(self):
        target = -10
        audio_data = np.ndarray(0, dtype=np.float32)
        normalizer = DynamicsNormalizer(
            self.sr, target, metric="rms")
        with self.assertRaises(ValueError):
            normalizer.normalize(audio_data)

    def test_raises_zero_input(self):
        target = -10
        audio_data = np.zeros(0, dtype=np.float32)
        normalizer = DynamicsNormalizer(
            self.sr, target, metric="rms")
        with self.assertRaises(ValueError):
            normalizer.normalize(audio_data)

    def test_raises_inf_input(self):
        target = -10
        audio_data = np.array([np.inf])
        normalizer = DynamicsNormalizer(
            self.sr, target, metric="rms")
        with self.assertRaises(ValueError):
            normalizer.normalize(audio_data)

    def test_raises_invalid_metric(self):
        metric = "peak"
        with self.assertRaises(ValueError):
            DynamicsNormalizer(
                self.sr, -10, metric=metric)

    def test_raises_positive_target(self):
        target = 10
        with self.assertRaises(ValueError):
            DynamicsNormalizer(
                self.sr, target, metric="rms")

    def _calculate_lvl(self, audio_data, metric):
        meter = pyln.Meter(self.sr)
        if metric == "rms":
            return 10 * np.log10(np.mean(audio_data**2))
        if metric == "itu":

            return meter.integrated_loudness(audio_data)


class StandardizeQuickSuite(unittest.TestSuite):
    def __init__(self):
        super(StandardizeQuickSuite, self).__init__()
        self.addTest(TestNormalizeDynamics("test_happy_path_mean"))
        self.addTest(TestNormalizeDynamics("test_happy_path_itu"))
        self.addTest(TestNormalizeDynamics("test_edge_case_zero_db"))
        self.addTest(TestNormalizeDynamics("test_edge_case_minus_100_db"))
        self.addTest(TestNormalizeDynamics("test_raises_empty_input"))
        self.addTest(TestNormalizeDynamics("test_raises_zero_input"))
        self.addTest(TestNormalizeDynamics("test_raises_inf_input"))
        self.addTest(TestNormalizeDynamics("test_raises_invalid_metric"))
        self.addTest(TestNormalizeDynamics("test_raises_positive_target"))


if __name__ == "__main__":
    unittest.TextTestRunner().run(StandardizeQuickSuite())
