"""
Data preprocessing and normalization module for EquiTS.
Enforces strict train-only fitting to guarantee zero target leakage.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


class TimeSeriesStandardScaler:
    """
    StandardScaler fitted strictly on training slices.
    Applies (x - mean) / (std + eps) independently per channel.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "TimeSeriesStandardScaler":
        """
        Fit mean and std strictly on training array [T_train, D].
        """
        if data.ndim == 1:
            data = data[:, None]
        self.mean = np.nanmean(data, axis=0, keepdims=True)
        self.std = np.nanstd(data, axis=0, keepdims=True)
        # Avoid division by zero for constant series
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet!")
        orig_dim = data.ndim
        if orig_dim == 1:
            data = data[:, None]
        norm = (data - self.mean) / self.std
        return norm if orig_dim > 1 else norm.squeeze(-1)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet!")
        orig_dim = data.ndim
        if orig_dim == 1:
            data = data[:, None]
        denorm = data * self.std + self.mean
        return denorm if orig_dim > 1 else denorm.squeeze(-1)

