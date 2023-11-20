from typing import List, Union
import numpy as np
from torch import Tensor


class Chunker:
    def __init__(self, **kwargs):
        """
        Initializes the class with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments that can be used to set the sample rate
                and segment duration.

        Returns:
            None

        Raises:
            TypeError: If the sample rate or segment duration is not an integer.
        """
        # Set the sample rate to the value provided in kwargs, or default to 44100
        self.sample_rate = kwargs.get('sample_rate', 44100)

        # Ensure that the sample rate is an integer
        if not isinstance(self.sample_rate, int):
            raise TypeError('sample_rate must be an int')

        # Set the segment duration to the value provided in kwargs, or default to 30
        self.segment_duration = kwargs.get('segment_duration', 30)

        # Ensure that the segment duration is an integer
        if not isinstance(self.segment_duration, int):
            raise TypeError('segment_duration must be an int')

        # Calculate the number of samples per segment based on the
        # sample rate and segment duration
        self.samples_per_segment = int(
            self.sample_rate * self.segment_duration)

    def chunk(self, audio_data: Union[np.ndarray, Tensor]) -> List[np.ndarray]:
        """
        Chunk the given audio data into segments.

        Args:
            audio_data (Union[np.ndarray, Tensor]): The audio data to be chunked. 
            It can be either a numpy array or a Tensor.

        Returns:
            List[np.ndarray]: A list of chunks, where each chunk is a numpy array.

        Raises:
            TypeError: If the audio_data is neither a numpy array nor a Tensor.
        """
        # Ensure that the audio data is either a numpy array
        # or a Tensor
        if not isinstance(audio_data, np.ndarray) and not isinstance(audio_data, Tensor):
            raise TypeError('audio_data must be a numpy array or Tensor.')

        # Calculate the number of segments and the remaining
        # samples in the audio data
        num_segments, remaining_samples = self._calculate_segments(audio_data)

        # Generate a list of chunks by iterating over the number of
        # segments and getting each chunk
        chunks = [self._get_chunk(i, audio_data) for i in range(num_segments)]

        # If there are remaining samples, create a last chunk and
        # add it to the list of chunks
        if remaining_samples > 0:
            last_chunk = self._get_last_chunk(
                audio_data, num_segments, remaining_samples)
            chunks.append(last_chunk)

        # Return the list of chunks
        return chunks

    def _calculate_segments(self, audio_data):
        # Calculate the number of segments by dividing the length of
        # the audio data by the number of samples per segment
        num_segments = len(audio_data) // self.samples_per_segment

        # Calculate the remaining samples by taking the modulo of the
        # length of the audio data by the number of samples per segment
        remaining_samples = len(audio_data) % self.samples_per_segment

        # Return the number of segments and the remaining samples
        return num_segments, remaining_samples

    def _get_chunk(self, i: int, audio_data: Union[np.ndarray, Tensor]) -> np.ndarray:
        # Calculate the start index of the chunk by multiplying the
        # segment index by the number of samples per segment
        start_idx = i * self.samples_per_segment

        # Calculate the end index of the chunk by adding the start
        # index and the number of samples per segment
        end_idx = (i + 1) * self.samples_per_segment

        # Return the chunk from the audio data based on the start and
        # end indices
        return audio_data[start_idx:end_idx]

    def _get_last_chunk(
        self,
        audio_data: Union[np.ndarray, Tensor],
        num_segments: int,
        remaining_samples: int
    ) -> np.ndarray:
        # Calculate the start index of the last chunk by multiplying
        # the number of segments by the number of samples per segment
        start_idx = num_segments * self.samples_per_segment

        # Calculate the end index of the last chunk by
        # adding the start index and the remaining samples
        end_idx = start_idx + remaining_samples

        # Return the last chunk from the audio data based
        # on the start and end indices
        return audio_data[start_idx:end_idx]
