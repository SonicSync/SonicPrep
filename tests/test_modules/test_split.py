# import cupy as cp
import torch
import unittest
from unittest.mock import patch


class TestSplit(unittest.TestCase):
    def setUp(self):
        pass

    def test_remodel(self):
        for test, channels in self.tests_remodel:
            with self.subTest(test=test, channels=channels):
                if isinstance(test, (np.ndarray, torch.Tensor)) \
                        and isinstance(channels, int) and channels > 0:
                    expected_waveform = test.shape[0] // channels
                    result = _remodel(test, channels)
                    self.assertEqual(result.shape[0], channels)
                    self.assertEqual(result.shape[1], expected_waveform)
                    self.assertIsInstance(result, torch.Tensor)
                elif channels < 0:
                    with self.assertRaises(ValueError):
                        _remodel(test, channels)
                else:
                    with self.assertRaises(TypeError):
                        _remodel(test, channels)

    def test_split_sources(self):
        pass


if __name__ == "__main__":
    unittest.main()
