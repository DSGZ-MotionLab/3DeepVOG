import numpy as np
import pandas as pd
# import plotly.graph_objects as go
from scipy.ndimage import uniform_filter1d

def interpolate_nan(data):
    nans = np.isnan(data)
    if np.any(nans):
        not_nan = ~nans
        data[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nan), data[not_nan])
    return data

def clean_signal(signal_raw, fps, drift_removal=True, drift_window=10, lowcut=0.1, highcut=2.0, order=4):
    from scipy import signal
    sig = interpolate_nan(signal_raw.copy())
    if drift_removal:
        drift = uniform_filter1d(sig, size=int(fps * drift_window), mode='nearest')
        sig -= drift
    nyq = 0.5 * fps
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, sig)