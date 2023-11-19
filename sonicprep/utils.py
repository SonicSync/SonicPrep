
import numpy as np


def calculate_array_duration(audio, sr):
    num_samples = len(audio)
    duration_sec = num_samples / sr
    return int(duration_sec)


def calculate_array_duration_diff(array1, array2, sr):
    array1_dur = calculate_array_duration(array1, sr)
    array2_dur = calculate_array_duration(array2, sr)
    if array1_dur > array2_dur:
        return array1_dur - array2_dur
    if array1_dur < array2_dur:
        return array2_dur - array1_dur
    return 0


def calculate_array_len_diff(array1, array2):
    return len(array1) - len(array2)


def calculate_sample_rate(audio_data, duration):
    num_samples = len(audio_data)
    sample_rate = num_samples / duration
    return int(sample_rate)


def analyze_frequency(audio_array, sample_rate, high_freq, low_freq):
    # Perform FFT
    fft_result = np.fft.fft(audio_array)
    # Use only positive frequencies
    fft_result = np.abs(fft_result[:len(fft_result)//2])

    # Calculate corresponding frequencies
    frequencies = np.fft.fftfreq(len(fft_result), d=1/sample_rate)
    positive_frequencies = frequencies[:len(frequencies)//2]

    # Check if certain frequencies are present (e.g., beyond 1000 Hz)
    above = positive_frequencies[positive_frequencies >
                                 high_freq]
    below = positive_frequencies[positive_frequencies <
                                 low_freq]

    return above, below


def check_between_freqs(audio_array, sample_rate, high_freq, low_freq):
    above, below = analyze_frequency(
        audio_array, sample_rate, high_freq, low_freq)
    above = bool(len(above) > 0)
    below = bool(len(below) > 0)
    return any([above, below])
