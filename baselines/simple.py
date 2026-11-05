"""
Reference baselines reported in the benchmark tables.
Persistence: repeats the last observed value (no parameters).
Linear: one channel-shared linear map from the lookback window to the horizon.
"""

import torch
import torch.nn as nn


class Persistence(nn.Module):
    def __init__(self, lookback: int = 96, horizon: int = 96, num_channels: int = 7):
        super().__init__()
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, H, D]
        return x[:, -1:, :].expand(-1, self.horizon, -1).contiguous()


class Linear(nn.Module):
    def __init__(self, lookback: int = 96, horizon: int = 96, num_channels: int = 7):
        super().__init__()
        self.proj = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)
