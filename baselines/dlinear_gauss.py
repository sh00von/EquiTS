"""
DLinear-Gauss: DLinear with a Gaussian output head.

The probabilistic counterpart of DLinear used as a same-category baseline for
EquiTS: identical decomposition and linear mean path, plus a second
decomposition-linear path for the log-variance. Trained with the same
(beta-)Gaussian NLL and the same data-space variance floor as EquiTS.
"""

import math
import torch
import torch.nn as nn

from baselines.dlinear import SeriesDecomp


class DLinearGauss(nn.Module):
    probabilistic = True

    def __init__(self, lookback: int = 96, horizon: int = 96, num_channels: int = 7,
                 kernel_size: int = 25, var_floor: float = 1e-2,
                 min_logvar: float = -8.0, max_logvar: float = 6.0):
        super().__init__()
        self.decomposition = SeriesDecomp(kernel_size=kernel_size)
        self.mu_seasonal = nn.Linear(lookback, horizon)
        self.mu_trend = nn.Linear(lookback, horizon)
        self.lv_seasonal = nn.Linear(lookback, horizon)
        self.lv_trend = nn.Linear(lookback, horizon)
        self.log_var_floor = math.log(var_floor)
        self.min_logvar, self.max_logvar = min_logvar, max_logvar

    def forward(self, x: torch.Tensor):
        # x: [B, L, D] -> mu, logvar: [B, H, D]
        seasonal, trend = self.decomposition(x)
        s, t = seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)
        mu = (self.mu_seasonal(s) + self.mu_trend(t)).permute(0, 2, 1)
        logvar = (self.lv_seasonal(s) + self.lv_trend(t)).permute(0, 2, 1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        logvar = torch.logaddexp(logvar, torch.full_like(logvar, self.log_var_floor))
        return mu, logvar
