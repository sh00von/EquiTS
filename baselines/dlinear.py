"""
DLinear: Decomposition Linear Model for Long-term Time Series Forecasting.
Reference: Zeng et al., AAAI 2023 (Are Transformers Effective for Time Series?)
"""

from typing import Tuple
import torch
import torch.nn as nn


class SeriesDecomp(nn.Module):
    """Series decomposition block (moving average)."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, D] -> [B, D, L]
        x_perm = x.permute(0, 2, 1)
        pad = nn.ReplicationPad1d((self.padding, self.padding))
        trend = nn.functional.avg_pool1d(pad(x_perm), kernel_size=self.kernel_size, stride=1)
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class Model(nn.Module):
    """
    Standard DLinear architecture with channel-independent linear projections.
    """
    def __init__(self, lookback: int = 96, horizon: int = 96, num_channels: int = 7, kernel_size: int = 25):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_channels = num_channels
        
        self.decompsition = SeriesDecomp(kernel_size=kernel_size)
        self.linear_seasonal = nn.Linear(lookback, horizon)
        self.linear_trend = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        seasonal, trend = self.decompsition(x)
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend_out = self.linear_trend(trend.permute(0, 2, 1))
        out = (seasonal_out + trend_out).permute(0, 2, 1) # [B, H, D]
        return out

