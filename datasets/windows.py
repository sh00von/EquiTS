"""
Sliding window generator and PyTorch Dataset for multivariate time series.
Enforces non-overlapping or sliding window extraction with strict horizon separation.
"""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Constructs (X, Y) window pairs from a continuous array [T, D].
    X: context window of length L: [x_{t-L+1}, ..., x_t]
    Y: target future of length H: [x_{t+1}, ..., x_{t+H}]
    
    Zero future leakage: The earliest index in Y is t+1, strictly after the last index of X (t).
    """
    def __init__(self, data: np.ndarray, lookback: int = 96, horizon: int = 96, stride: int = 1):
        super().__init__()
        assert data.ndim == 2, f"Expected 2D array [T, D], got shape {data.shape}"
        assert len(data) >= lookback + horizon, (
            f"Series length {len(data)} is shorter than lookback {lookback} + horizon {horizon}"
        )
        self.data = data.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        
        # Total number of valid sliding windows
        self.num_windows = (len(self.data) - lookback - horizon) // stride + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_x = idx * self.stride
        end_x = start_x + self.lookback
        start_y = end_x
        end_y = start_y + self.horizon
        
        x = self.data[start_x:end_x] # [L, D]
        y = self.data[start_y:end_y] # [H, D]
        
        return torch.from_numpy(x), torch.from_numpy(y)

