import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def interpolate_nan(data):
    """Interpolate NaN values in a 1D array."""
    nans = np.isnan(data)
    if np.any(nans):
        not_nan = ~nans
        data[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nan), data[not_nan])
    return data

from scipy.ndimage import uniform_filter1d
def smooth_with_window(data, win_size):
    """Apply moving average smoothing with a fixed output size."""
    win_size = int(win_size)
    if win_size < 1:
        return data  # return original if window is too small
    return uniform_filter1d(data, size=win_size, mode='nearest')

def bandpass_filter(fs, lowcut, highcut, order, input_signal):
    """Apply Butterworth bandpass filter to retain key torsional frequencies."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, input_signal)

def clean_signal(signal_raw, fps, drift_removal=True, drift_window=10, lowcut=0.1, highcut=2.0, order=4):
    """
    Full torsion signal cleaning pipeline:
    - NaN interpolation
    - drift removal (moving average)
    - bandpass filtering
    """
    signal_interp = interpolate_nan(np.asarray(signal_raw, dtype=float).copy())

    if drift_removal:
        drift = smooth_with_window(signal_interp, win_size=int(fps * drift_window))
        signal_detrended = signal_interp - drift
    else:
        signal_detrended = signal_interp

    # bandpass filter ALWAYS applied
    nyq = 0.5 * fps
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    cleaned = signal.filtfilt(b, a, signal_detrended)

    return cleaned
