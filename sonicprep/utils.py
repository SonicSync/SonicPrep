import numpy as np
from madmom.features import beats
from madmom.audio.signal import Signal


class Calculator:

    @staticmethod
    def mean_power(audio_data):
        return np.mean(audio_data**2)

    @staticmethod
    def peak_power(audio_data):
        return np.max(audio_data**2)

    @staticmethod
    def mean_dB(audio_data):
        return 10 * np.log10(Calculator.mean_power(audio_data))

    @staticmethod
    def peak_dB(audio_data):
        return 10 * np.log10(Calculator.peak_power(audio_data))

    @staticmethod
    def rms(audio_data):
        return np.sqrt(Calculator.mean_power(audio_data))

    @staticmethod
    def duration(audio_data, sr):
        return int(len(audio_data) / sr)

    @staticmethod
    def sample_rate(audio_data, duration):
        return int(len(audio_data) / duration)

    @staticmethod
    def duration_diff(audio_data1, audio_data2, sr):
        duration1 = Calculator.duration(audio_data1, sr)
        duration2 = Calculator.duration(audio_data2, sr)
        if duration1 > duration2:
            return duration1 - duration2
        if duration1 < duration2:
            return duration2 - duration1
        return 0

    @staticmethod
    def size_diff(audio_data1, audio_data2):
        return len(audio_data1) - len(audio_data2)

    @staticmethod
    def bpm(audio_data, sr):
        autocorr = np.correlate(audio_data, audio_data, mode='full')

        # Keep only positive lags (ignoring negative lags)
        autocorr = autocorr[len(autocorr)//2:]

        # Find the index of the maximum value in the autocorrelation
        peak_index = np.argmax(autocorr)

        # Convert the lag to time in seconds
        lag_in_seconds = peak_index / sr

        # Convert lag to BPM
        tempo = 60 / lag_in_seconds

        return tempo
