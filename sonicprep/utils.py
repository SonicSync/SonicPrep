import numpy as np


def calculate_array_duration(audio, sr):
    num_samples = len(audio)
    duration_sec = num_samples / sr
    return duration_sec