"""
EquiTS: a plug-in distributional normalization layer for any point forecaster.

    layer = EquiTS(seq_len=96, pred_len=96, num_channels=7)
    mu, logvar = layer(x, lambda z: backbone(z))           # z: normalized input

Mean path (trained with MSE, exactly "backbone + normalization"):
    (m, s) = canonicalizer(x),  z = (x - m) / s,  mu = m + s * backbone(z)

Variance path (trained with Gaussian NLL; sees only detached inputs, so it can
never change the mean forecast):
    u_x = z,  u_y = (mu - m) / s                (normalized window and forecast)
    sigma_h^2, tau^2 = scale_head(u_x, u_y)
    Var[Y_h] = s^2 * (sigma_h^2 + tau^2)

sigma_h^2 is per-step noise; tau^2 (optional "shift term") is a per-window level-shift
variance shared across the horizon, expressing uncertainty about the future level
relative to the input window. Both are predicted in normalized units, so the
predictive distribution transforms equivariantly with (m, s): under x -> a x + b,
mu -> a mu + b and Var -> a^2 Var (up to the canonicalizer's relaxation).
Note: with a diagonal Gaussian, sigma_h^2 and tau^2 enter only through their sum;
the split is a structural prior, not separately identifiable.
"""

from typing import Callable, Tuple

import torch
import torch.nn as nn

from equits.canonicalizer import Canonicalizer


class ScaleHead(nn.Module):
    """Channel-shared MLP: [normalized window (L) ; normalized mean forecast (H)] -> log variances."""
    def __init__(self, seq_len: int, pred_len: int, hidden: int = 128, shift_term: bool = True,
                 min_logvar: float = -8.0, max_logvar: float = 6.0):
        super().__init__()
        self.pred_len = pred_len
        self.shift_term = shift_term
        out = pred_len + (1 if shift_term else 0)
        self.net = nn.Sequential(
            nn.Linear(seq_len + pred_len, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out),
        )
        self.min_logvar, self.max_logvar = min_logvar, max_logvar

    def forward(self, u_x: torch.Tensor, u_y: torch.Tensor) -> torch.Tensor:
        """u_x: [N, L], u_y: [N, H] -> normalized variance [N, H]."""
        o = self.net(torch.cat([u_x, u_y], dim=-1))
        o = torch.clamp(o, self.min_logvar, self.max_logvar)
        var = torch.exp(o[:, : self.pred_len])
        if self.shift_term:
            var = var + torch.exp(o[:, self.pred_len:])        # broadcast tau^2 over the horizon
        return var


class EquiTS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_channels: int,
                 normalizer: str = "learned", shift_term: bool = True, eps: float = 0.1,
                 canon_hidden: int = 64, head_hidden: int = 128):
        super().__init__()
        self.seq_len, self.pred_len, self.num_channels = seq_len, pred_len, num_channels
        self.canonicalizer = Canonicalizer(seq_len, num_channels, mode=normalizer, eps=eps, hidden=canon_hidden)
        self.scale_head = ScaleHead(seq_len, pred_len, hidden=head_hidden, shift_term=shift_term)

    def mean_parameters(self):
        return self.canonicalizer.parameters()

    def variance_parameters(self):
        return self.scale_head.parameters()

    def forward(self, x: torch.Tensor, backbone: Callable[[torch.Tensor], torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D] raw (or globally standardized) input.
        backbone: callable mapping the normalized input [B, L, D] to a normalized forecast [B, H, D].
        Returns mu, logvar: [B, H, D].
        """
        B, L, D = x.shape
        m, s = self.canonicalizer(x)
        z = (x - m) / s
        y_n = backbone(z)[:, -self.pred_len:, :]
        mu = m + s * y_n

        # Variance path on detached quantities
        md, sd, zd, yd = m.detach(), s.detach(), z.detach(), y_n.detach()
        u_x = zd.permute(0, 2, 1).reshape(B * D, L)
        u_y = yd.permute(0, 2, 1).reshape(B * D, self.pred_len)
        var_n = self.scale_head(u_x, u_y).view(B, D, self.pred_len).permute(0, 2, 1)
        logvar = torch.log(var_n) + 2.0 * torch.log(sd)
        return mu, logvar
