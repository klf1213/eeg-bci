from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def bandpass_filter(x, sampling_rate, low_frequency=4.0, high_frequency=40.0, order=4):
    nyquist = sampling_rate / 2
    sos = butter(order, [low_frequency / nyquist, high_frequency / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def resample_eeg(x, original_rate=250, target_rate=128):
    return resample_poly(x, up=target_rate, down=original_rate, axis=-1).astype(np.float32)


def preprocess_trials(x, original_rate, target_rate, low_frequency, high_frequency):
    return resample_eeg(bandpass_filter(x, original_rate, low_frequency, high_frequency), original_rate, target_rate)


def standardize_splits(train_x, *other_splits):
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(train_x.std(axis=(0, 2), keepdims=True), 1e-6)
    standardized = [((values - mean) / std).astype(np.float32) for values in (train_x, *other_splits)]
    return (*standardized, mean, std)

