import os
import random
import unittest
import tempfile
import numpy as np
from sonicprep.io import load_audio_file as load
from sonicprep.io import audio_batch_generator, find_all_files
from sonicprep.utils import calculate_array_duration
from sonicprep.exceptions import AudioTypeError, NoFilesError
from datafactory.datafactory import generate_random_audio, save_audio, save_many_audio


class TestLoadAudio(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.sr = 44100

    def tearDown(self):
        self.tempdir.cleanup()

    def test_happy_path(self):
        test_cases = [
            ('test.mp3', 'mp3'),
            ('test.wav', 'wav'),
            ('test.flac', 'flac')
        ]
        duration = 10
        audio = generate_random_audio(duration, self.sr)

        for file_name, file_extension in test_cases:
            with self.subTest('Loading {}'.format(file_name)):
                file_path = os.path.join(self.tempdir.name, file_name)
                save_audio(file_path, audio, self.sr, file_extension)

                result, result_sr = load(file_path, self.sr)
                result_duration = calculate_array_duration(result, self.sr)

                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result_duration, duration)
                self.assertEqual(result_sr, self.sr)

    def test_edge_cases_all_silence(self):
        # Test case 1: All silence
        file_name = 'silence.wav'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 10
        self.sr = 44100
        silence_audio = np.zeros(int(self.sr * duration))
        save_audio(file_path, silence_audio, self.sr, 'wav')

        result, result_sr = load(file_path, self.sr)
        result_duration = calculate_array_duration(result, self.sr)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result_duration, duration)
        self.assertEqual(result_sr, self.sr)
        self.assertTrue(np.all(result == 0))

    def test_edge_case_long(self):
        # Test case 2: Large duration
        file_name = 'large_duration.wav'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 3600  # 1 hour
        audio = generate_random_audio(duration, self.sr)
        save_audio(file_path, audio, self.sr, 'wav')

        result, result_sr = load(file_path, self.sr)
        result_duration = calculate_array_duration(result, self.sr)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result_duration, duration)
        self.assertEqual(result_sr, self.sr)

    def test_edge_case_short(self):
        # Test case 3: Short duration
        file_name = 'short_duration.wav'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 0.01
        audio = generate_random_audio(duration, self.sr)
        save_audio(file_path, audio, self.sr, 'wav')

        result, result_sr = load(file_path, self.sr)
        result_duration = calculate_array_duration(result, self.sr)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result_duration, duration)
        self.assertEqual(result_sr, self.sr)

    def test_edge_case_incorrect_sr(self):
        # Test case 4: Incorrect sample rate
        file_name = 'incorrect_sr.wav'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 10
        load_sr = self.sr * 2
        audio = generate_random_audio(duration, self.sr)
        save_audio(file_path, audio, self.sr, 'wav')

        result, result_sr = load(file_path, load_sr)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result_sr, load_sr)
        self.assertEqual(calculate_array_duration(result, load_sr),
                         duration)

    def test_raises_non_audio(self):
        # Test Non-audio file
        file_name = 'test.txt'
        file_path = os.path.join(self.tempdir.name, file_name)
        with open(file_path, 'w') as f:
            f.write("This is not an audio file.")
        self.assertRaises(AudioTypeError, load, file_path, self.sr)

    def test_raises_invalid_type(self):
        # Incorrect audio type
        file_name = 'test.aiff'
        file_path = os.path.join(self.tempdir.name, file_name)
        duration = 5
        audio = generate_random_audio(duration, self.sr)
        save_audio(file_path, audio, self.sr, 'aiff')
        self.assertRaises(AudioTypeError, load, file_path, self.sr)


class TestBatchLoadAudio(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.sr = 44100

    def tearDown(self):
        self.tempdir.cleanup()

    def test_happy_path_mp3(self):
        amount = 3
        duration = 10
        audio_data = generate_random_audio(duration, self.sr)
        self._save_audio(audio_data, amount, 'mp3')

        all_audio_files = find_all_files(self.tempdir.name)
        batch_gen = audio_batch_generator(all_audio_files, 10)
        batch = next(batch_gen)

        result_amount = len(batch)
        self.assertEqual(result_amount, amount)
        for result, result_sr in batch:
            result_duration = calculate_array_duration(result, self.sr)
            self.assertEqual(result_sr, self.sr)
            self.assertEqual(result_duration, duration)
            self.assertIsInstance(result, np.ndarray)

    def test_happy_path_wav(self):
        amount = 3
        duration = 10
        audio_data = generate_random_audio(duration, self.sr)
        self._save_audio(audio_data, amount, 'wav')

        all_audio_files = find_all_files(self.tempdir.name)
        batch_gen = audio_batch_generator(all_audio_files, 10)
        batch = next(batch_gen)

        result_amount = len(batch)
        self.assertEqual(result_amount, amount)
        for result, result_sr in batch:
            result_duration = calculate_array_duration(result, self.sr)
            self.assertEqual(result_sr, self.sr)
            self.assertEqual(result_duration, duration)
            self.assertIsInstance(result, np.ndarray)

    def test_happy_path_flac(self):
        duration = 10
        amount = 3
        audio_data = generate_random_audio(duration, self.sr)
        self._save_audio(audio_data, amount, 'flac')

        all_audio_files = find_all_files(self.tempdir.name)
        batch_gen = audio_batch_generator(all_audio_files, 10)
        batch = next(batch_gen)

        result_amount = len(batch)
        self.assertEqual(result_amount, amount)
        for result, result_sr in batch:
            result_duration = calculate_array_duration(result, self.sr)
            self.assertEqual(result_sr, self.sr)
            self.assertEqual(result_duration, duration)
            self.assertIsInstance(result, np.ndarray)

    def test_happy_path_mix(self):
        duration = 10
        amount = 3
        audio_data = generate_random_audio(duration, self.sr)
        self._save_audio(audio_data, amount)

        all_audio_files = find_all_files(self.tempdir.name)
        batch_gen = audio_batch_generator(all_audio_files, 10)
        batch = next(batch_gen)

        result_amount = len(batch)
        self.assertEqual(result_amount, amount)
        for result, result_sr in batch:
            result_duration = calculate_array_duration(result, self.sr)
            self.assertEqual(result_sr, self.sr)
            self.assertEqual(result_duration, duration)
            self.assertIsInstance(result, np.ndarray)

    def test_edge_case_all_silence(self):
        duration = 10
        amount = 3
        silence_audio = np.zeros(int(self.sr * duration))
        self._save_audio(silence_audio, amount)

        all_audio_files = find_all_files(self.tempdir.name)
        batch_gen = audio_batch_generator(all_audio_files, 10)
        batch = next(batch_gen)

        for result, result_sr in batch:
            result_duration = calculate_array_duration(result, self.sr)
            self.assertEqual(result_sr, self.sr)
            self.assertEqual(result_duration, duration)
            self.assertIsInstance(result, np.ndarray)

    def test_edge_case_long_duration(self):
        duration = 3600
        amount = 2
        audio_data = generate_random_audio(duration, self.sr)
        self._save_audio(audio_data, amount)

        all_audio_files = find_all_files(self.tempdir.name)
        batch_gen = audio_batch_generator(all_audio_files, 10)
        batch = next(batch_gen)

        for result, result_sr in batch:
            result_duration = calculate_array_duration(result, self.sr)
            self.assertEqual(result_sr, self.sr)
            self.assertEqual(result_duration, duration)
            self.assertIsInstance(result, np.ndarray)

    def test_edge_case_short_duration(self):
        duration = 0.01
        amount = 3
        audio_data = generate_random_audio(duration, self.sr)
        self._save_audio(audio_data, amount)

        all_audio_files = find_all_files(self.tempdir.name)
        batch_gen = audio_batch_generator(all_audio_files, 10)
        batch = next(batch_gen)

        for result, result_sr in batch:
            result_duration = calculate_array_duration(result, self.sr)
            self.assertEqual(result_sr, self.sr)
            self.assertEqual(result_duration, duration)
            self.assertIsInstance(result, np.ndarray)

    def test_edge_case_lots_of_files(self):
        duration = 10
        amount = 100
        audio_data = generate_random_audio(duration, self.sr)
        self._save_audio(audio_data, amount)

        all_audio_files = find_all_files(self.tempdir.name)
        num_batches = 0
        for batch in audio_batch_generator(all_audio_files, 10):
            num_batches += 1
            for result, result_sr in batch:
                result_duration = calculate_array_duration(result, self.sr)
                self.assertEqual(result_sr, self.sr)
                self.assertEqual(result_duration, duration)
                self.assertIsInstance(result, np.ndarray)
        self.assertEqual(num_batches, 10)

    def test_edge_case_large_batch_size(self):
        duration = 10
        amount = 100
        audio_data = generate_random_audio(duration, self.sr)
        self._save_audio(audio_data, amount)

        all_audio_files = find_all_files(self.tempdir.name)
        num_batches = 0
        for batch in audio_batch_generator(all_audio_files, 100):
            num_batches += 1
            for result, result_sr in batch:
                result_duration = calculate_array_duration(result, self.sr)
                self.assertEqual(result_sr, self.sr)
                self.assertEqual(result_duration, duration)
                self.assertIsInstance(result, np.ndarray)
        self.assertEqual(num_batches, 1)

    def test_edge_case_one_batch_size(self):
        duration = 10
        amount = 10
        audio_data = generate_random_audio(duration, self.sr)
        self._save_audio(audio_data, amount)

        all_audio_files = find_all_files(self.tempdir.name)
        num_batches = 0
        for batch in audio_batch_generator(all_audio_files, 1):
            num_batches += 1
            for result, result_sr in batch:
                result_duration = calculate_array_duration(result, self.sr)
                self.assertEqual(result_sr, self.sr)
                self.assertEqual(result_duration, duration)
                self.assertIsInstance(result, np.ndarray)
        self.assertEqual(num_batches, 10)

    def test_raises_no_files_error(self):
        with self.assertRaises(NoFilesError):
            find_all_files(self.tempdir.name)

    def _save_audio(self, audio_data, amount, type=None):
        if type:
            audio = [(audio_data, type, os.path.join(self.tempdir.name,
                                                     f'test{i}.{type}')) for i in range(amount)]
            save_many_audio(audio, self.sr)
        else:
            file_types = ['mp3', 'wav', 'flac']
            audio = []
            for i in range(amount):
                file_type = random.choice(file_types)
                file_name = f'test{i}.{file_type}'
                path = os.path.join(self.tempdir.name, file_name)
                audio.append((audio_data, file_type, path))
            save_many_audio(audio, self.sr)


class IOTestSuite(unittest.TestSuite):
    def __init__(self):
        super(IOTestSuite, self).__init__()
        self.addTest(TestLoadAudio('test_happy_path'))
        self.addTest(TestLoadAudio('test_raises_invalid_type'))
        self.addTest(TestBatchLoadAudio('test_happy_path_mp3'))
        self.addTest(TestBatchLoadAudio('test_happy_path_wav'))
        self.addTest(TestBatchLoadAudio('test_happy_path_flac'))
        self.addTest(TestBatchLoadAudio('test_happy_path_mix'))
        self.addTest(TestBatchLoadAudio('test_edge_case_all_silence'))
        self.addTest(TestBatchLoadAudio('test_edge_case_short_duration'))
        self.addTest(TestBatchLoadAudio('test_raises_no_files_error'))


if __name__ == '__main__':
    unittest.TextTestRunner().run(IOTestSuite())
    # unittest.main()
