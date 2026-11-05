"""
LSTM Forecaster: Standard Multi-layer Recurrent Baseline for Time Series.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, lookback: int = 96, horizon: int = 96, num_channels: int = 7, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_channels = num_channels
        
        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, horizon * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        _, (hn, _) = self.lstm(x) # hn[-1]: [B, hidden_dim]
        out = self.head(hn[-1])
        return out.view(-1, self.horizon, self.num_channels)

