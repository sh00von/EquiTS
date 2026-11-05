"""
TCN: Temporal Convolutional Network Baseline for Long-Term Forecasting.
Reference: Bai et al., 2018.
"""

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.dropout(self.act(out))
        return out + self.residual(x)


class Model(nn.Module):
    def __init__(self, lookback: int = 96, horizon: int = 96, num_channels: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_channels = num_channels
        
        self.net = nn.Sequential(
            TemporalBlock(num_channels, hidden_dim, kernel_size=3, dilation=1),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, horizon * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, D, L]
        h = self.net(x.permute(0, 2, 1))
        z = self.pool(h).squeeze(-1)
        return self.head(z).view(-1, self.horizon, self.num_channels)

