import os
import random
import numpy as np
from faker import Faker
import pandas as pd
import pydub
import soundfile as sf

fake = Faker()


def generate_random_audio(duration_sec, sample_rate, amplitude_db=None, frequency=440.0, silence_before=0.0, silence_after=0.0, entirely_silent=False):
    """
    Generate random audio data.

    Parameters:
    - duration_sec: Duration of the audio in seconds.
    - sample_rate: Number of samples per second (sampling rate).
    - amplitude_db: Target peak amplitude in decibels. If None, peak amplitude is not adjusted.
    - frequency: Frequency of the audio waveform.
    - silence_before: Duration of silence to add at the beginning (seconds).
    - silence_after: Duration of silence to add at the end (seconds).
    - entirely_silent: If True, generate entirely silent audio (ignores other parameters).

    Returns:
    - audio_data: NumPy array representing the audio waveform.
    - time_axis: Time axis corresponding to the audio data.
    """
    if entirely_silent:
        # Generate entirely silent audio
        audio_data = np.zeros(int(sample_rate * duration_sec))
    else:
        total_duration = duration_sec + silence_before + silence_after

        # Time axis for the entire duration
        time_axis = np.linspace(0, total_duration, int(
            sample_rate * total_duration), endpoint=False)

        # Create the audio waveform (sine wave)
        audio_data = np.sin(2 * np.pi * frequency *
                            (time_axis - silence_before))

        # Add random noise to the audio data
        audio_data += np.random.normal(0, 0.5, audio_data.shape)

        # Apply silence at the beginning
        audio_data[:int(silence_before * sample_rate)] = 0.0

        # Apply silence at the end
        audio_data[-int(silence_after * sample_rate):] = 0.0

        # Adjust peak amplitude if specified
        if amplitude_db is not None:
            target_amplitude = 10 ** (amplitude_db / 20.0)
            current_amplitude = np.max(np.abs(audio_data))
            audio_data *= target_amplitude / current_amplitude

    return audio_data


def get_data():
    choices = ['a', 'b', 'c']
    type = random.choice(choices)
    if type == 'a':
        return np.random.randn(1000)
    elif type == 'b':
        return random.randint(0, 1000)
    elif type == 'c':
        return fake.sentence()


def generate_dataframe(r, c, data=True):
    """
    Generate a dataframe with random data.

    Parameters:
    - r (int): The number of rows in the dataframe.
    - c (int): The number of columns in the dataframe.

    Returns:
    - df (DataFrame): The generated dataframe.

    """
    data = []
    for _ in range(r):
        dict = {}
        if data:
            for i in range(c):
                dict[i] = str(get_data())
        data.append(dict)
    df = pd.DataFrame(data)
    return df


def save_audio(path, data, sr, format):
    if format == 'wav':
        sf.write(path, data, sr, subtype='PCM_24')
    elif format == 'mp3':
        data = pydub.AudioSegment(
            data.tobytes(), frame_rate=sr, sample_width=4, channels=2)
        data.export(out_f=path, format="mp3")
    elif format == 'flac':
        sf.write(path, data, sr, format=format,
                 subtype='PCM_24')


def save_many_audio(data, sr):
    for audio, format, path in data:
        save_audio(path, audio, sr, format)


def read_dataframe(path):
    _, ext = os.path.splitext(path)
    if ext == '.csv':
        df = pd.read_csv(path)
    if ext == '.xlsx':
        df = pd.read_excel(path)
    if ext == '.json':
        df = pd.read_json(path)
    return df
