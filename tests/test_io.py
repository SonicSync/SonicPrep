import os
from stat import S_IREAD, S_IRGRP, S_IROTH
import unittest
import tempfile
import numpy as np
from sonicprep.io import load_audio_file as load
from sonicprep.utils import calculate_array_duration
from sonicprep.exceptions import AudioTypeError
from datafactory.datafactory import generate_random_audio, save_audio


class TestLoadAudio(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.sr = 44100

    def tearDown(self):
        self.tempdir.cleanup()

    def test_happy_path(self):
        """
        Test the happy path for loading audio files.

        - Test loading mp3
        - Test loading wav
        - Test loading flac
        """
        test_cases = [
            ('test.mp3', 'mp3'),
            ('test.wav', 'wav'),
            ('test.flac', 'flac')
        ]
        duration = 180
        audio = generate_random_audio(duration, self.sr)

        for file_name, file_extension in test_cases:
            file_path = os.path.join(self.tempdir.name, file_name)
            save_audio(file_path, audio, self.sr, file_extension)

            result, result_sr = load(file_path, self.sr)
            result_duration = calculate_array_duration(result, self.sr)

            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result_duration, duration)
            self.assertEqual(result_sr, self.sr)

    def test_edge_cases(self):
        """
        Test edge cases for the audio processing function.

        - Test All silence audio
        - Test Large duration audio
        - Test Short duration audio
        - Test Incorrect sample rate audio
        """

        # Test case 1: All silence
        file_name = 'silence.wav'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 10
        sr = 44100
        silence_audio = np.zeros(int(sr * duration))
        save_audio(file_path, silence_audio, sr, 'wav')

        result, result_sr = load(file_path, sr)
        result_duration = calculate_array_duration(result, sr)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result_duration, duration)
        self.assertEqual(result_sr, sr)
        self.assertTrue(np.all(result == 0))

        # Test case 2: Large duration
        file_name = 'large_duration.wav'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 3600  # 1 hour
        audio = generate_random_audio(duration, sr)
        save_audio(file_path, audio, sr, 'wav')

        result, result_sr = load(file_path, sr)
        result_duration = calculate_array_duration(result, sr)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result_duration, duration)
        self.assertEqual(result_sr, sr)

        # Test case 3: Short duration
        file_name = 'short_duration.wav'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 0.01
        audio = generate_random_audio(duration, sr)
        save_audio(file_path, audio, sr, 'wav')

        result, result_sr = load(file_path, sr)
        result_duration = calculate_array_duration(result, sr)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result_duration, duration)
        self.assertEqual(result_sr, sr)

        # Test case 4: Incorrect sample rate
        file_name = 'incorrect_sr.wav'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 10
        load_sr = sr * 2
        audio = generate_random_audio(duration, sr)
        save_audio(file_path, audio, sr, 'wav')

        result, result_sr = load(file_path, load_sr)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result_sr, load_sr)
        self.assertEqual(calculate_array_duration(result, load_sr),
                         duration)

    def test_raises(self):
        """
        Test cases:
        - Non-audio file:
            - Create a non-audio file
            - Assert that loading the file raises an AudioTypeError
        - Incorrect audio type:
            - Create an audio file with an incorrect audio type
            - Assert that loading the file raises an AudioTypeError
        """
        # Test Non-audio file
        file_name = 'test.txt'
        file_path = os.path.join(self.tempdir.name, file_name)
        with open(file_path, 'w') as f:
            f.write("This is not an audio file.")
        self.assertRaises(AudioTypeError, load, file_path, self.sr)

        # Incorrect audio type
        file_name = 'test.aac'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 5
        audio = generate_random_audio(duration, self.sr)
        save_audio(file_path, audio, self.sr, 'aac')
        self.assertRaises(AudioTypeError, load, file_path, self.sr)


if __name__ == '__main__':
    unittest.main()
