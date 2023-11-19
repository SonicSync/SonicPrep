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
