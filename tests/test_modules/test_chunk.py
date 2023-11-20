import unittest
import torch
import numpy as np
# from sonicprep.chunk import Chunker
# from tests.mock_data import *


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


if __name__ == "__main__":
    unittest.main()
