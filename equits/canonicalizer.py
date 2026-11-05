"""
Canonicalizers for affine-equivariant forecasting.

Setting. A window x in R^{L x D} is acted on by the positive affine group
G = {(a, b) : a > 0, b in R^D}, (a, b) . x = a x + b (per channel). A canonicalizer
returns a location m(x) in R^D and a scale s(x) > 0 such that, ideally,

    m(a x + b) = a m(x) + b,      s(a x + b) = a s(x).                  (EQ)

A forecaster  y = m + s * g((x - m) / s)  is then G-equivariant for any backbone g.

Impossibility on flat windows. Constant windows x = c 1 are fixed by the subgroup
{(a, (1 - a) c)}, so (EQ) forces s(c 1) = a s(c 1) for all a > 0, i.e. s(c 1) = 0;
by continuity s -> 0 near flat windows and (x - m) / s is unbounded there. Hence no
continuous canonicalizer with s > 0 can satisfy (EQ) on all of R^{L x D}. RevIN's
"+ 1e-5" inside the square root is an uncontrolled instance of this problem
(the constant is in raw data units).

Relaxation. We use  s_eps(x) = sqrt(s(x)^2 + (eps * sigma0)^2),  where sigma0 is a
fixed per-channel reference scale (1 for standardized data). Then
  * location equivariance is exact:      m(x + b) = m(x) + b,  s_eps(x + b) = s_eps(x)
  * scale equivariance has bounded error: |s_eps(a x) - a s_eps(x)| <= |a - 1| eps sigma0
    (for a fixed, exactly equivariant raw scale s), and the relative error vanishes
    as s(x) / (eps sigma0) -> infinity;
  * s_eps >= eps sigma0, so normalized inputs and predictive variances stay bounded.

Modes
  "revin"   : mean / sqrt(var + 1e-5)                       (RevIN / TSLib baseline)
  "relaxed" : mean / s_eps with the plain standard deviation
  "learned" : m = sum_t w_t x_t,  s^2 = sum_t v_t (x_t - m)^2, relaxed; the weights
              w = softmax(f(u)), v = softmax(h(u)) depend only on the normalized shape
              u = (x - mean) / s_eps(std), which is invariant under location shifts
              (exactly) and under scaling (up to the relaxation). f and h are
              channel-shared MLPs whose last layer is zero-initialized, so training
              starts exactly at the "relaxed" canonicalizer.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("revin", "relaxed", "learned")


def relaxed_scale(var: torch.Tensor, eps: float, sigma0: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(var + (eps * sigma0) ** 2)


class _WeightNet(nn.Module):
    """Maps a normalized window [N, L] to temporal logits [N, L]; zero-initialized output."""
    def __init__(self, seq_len: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(seq_len, hidden), nn.GELU(), nn.Linear(hidden, seq_len))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class Canonicalizer(nn.Module):
    def __init__(self, seq_len: int, num_channels: int, mode: str = "learned",
                 eps: float = 0.1, hidden: int = 64):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode}")
        self.mode = mode
        self.eps = eps
        # Reference scale per channel (1.0 for data standardized with the training scaler).
        self.register_buffer("sigma0", torch.ones(1, 1, num_channels))
        if mode == "learned":
            self.loc_net = _WeightNet(seq_len, hidden)
            self.scale_net = _WeightNet(seq_len, hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, L, D] -> m, s: [B, 1, D] (s > 0)."""
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        if self.mode == "revin":
            return mean, torch.sqrt(var + 1e-5)
        s_ref = relaxed_scale(var, self.eps, self.sigma0)
        if self.mode == "relaxed":
            return mean, s_ref

        B, L, D = x.shape
        u = ((x - mean) / s_ref).permute(0, 2, 1).reshape(B * D, L)       # invariant shape
        w = F.softmax(self.loc_net(u), dim=-1).view(B, D, L).permute(0, 2, 1)
        v = F.softmax(self.scale_net(u), dim=-1).view(B, D, L).permute(0, 2, 1)
        m = (w * x).sum(dim=1, keepdim=True)
        s2 = (v * (x - m) ** 2).sum(dim=1, keepdim=True)
        return m, relaxed_scale(s2, self.eps, self.sigma0)
