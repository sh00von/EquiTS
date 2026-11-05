"""
RLinear-Gauss: RevIN + linear mean + linear log-variance.

Ablation baseline for EquiTS. It shares EquiTS's instance normalization (RevIN)
and its data-space variance mapping (predicted variance scales with the input
window's variance, plus the same variance floor), but has no convolutional
trunk. If EquiTS cannot beat this model, its uncertainty gains come from RevIN
scaling rather than from the architecture.
Point-forecast counterpart: RLinear (Li et al., 2023).
"""

import math
import torch
import torch.nn as nn

from equits.revin import RevIN


class RLinearGauss(nn.Module):
    probabilistic = True

    def __init__(self, lookback: int = 96, horizon: int = 96, num_channels: int = 7,
                 var_floor: float = 1e-2, min_logvar: float = -8.0, max_logvar: float = 6.0):
        super().__init__()
        self.revin = RevIN(num_features=num_channels, affine=True)
        self.mu_proj = nn.Linear(lookback, horizon)
        self.lv_proj = nn.Linear(lookback, horizon)
        self.log_var_floor = math.log(var_floor)
        self.min_logvar, self.max_logvar = min_logvar, max_logvar

    def forward(self, x: torch.Tensor):
        # x: [B, L, D] -> mu, logvar: [B, H, D] in data space
        z = self.revin(x, mode="norm").permute(0, 2, 1)          # [B, D, L]
        mu = self.mu_proj(z).permute(0, 2, 1)
        logvar = torch.clamp(self.lv_proj(z), self.min_logvar, self.max_logvar).permute(0, 2, 1)
        mu = self.revin(mu, mode="denorm")
        logvar = self.revin.denorm_logvar(logvar)
        logvar = torch.logaddexp(logvar, torch.full_like(logvar, self.log_var_floor))
        return mu, logvar
