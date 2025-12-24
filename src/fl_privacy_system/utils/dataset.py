import numpy as np
import torch
from torch.utils.data import TensorDataset


def make_supervised(series, window=1):
    """
    Convert a time series into supervised learning pairs.
    Vectorized, safe, and PyTorch-compatible.
    """
    values = series.values.astype(np.float32)

    if len(values) <= window:
        raise ValueError("Time series too short for given window size")

    # sliding window (view)
    x = np.lib.stride_tricks.sliding_window_view(values, window)[:-1]

    # make writable copy (required for PyTorch safety)
    x = x.copy()

    y = values[window:].copy()

    x = torch.from_numpy(x)
    y = torch.from_numpy(y).unsqueeze(1)

    return TensorDataset(x, y)
