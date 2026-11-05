"""
Dual Probabilistic Prediction Head for Diagonal Gaussian Future Distributions.
"""

from typing import Tuple
import torch
import torch.nn as nn


class DualProbabilisticHead(nn.Module):
    """
    Direct linear multi-horizon probabilistic projection.
    Maps representation z [B * D, d_model] to:
      mu: Forecast mean [B * D, H]
      logvar: Forecast log-variance [B * D, H]
    """
    def __init__(
        self,
        d_model: int = 64,
        horizon: int = 96,
        min_logvar: float = -8.0,
        max_logvar: float = 6.0
    ):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon)
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B * D, d_model]
        Returns:
          mu: [B * D, H]
          logvar: [B * D, H] (clamped for numerical stability)
        """
        mu = self.mu_head(z)
        logvar = self.logvar_head(z)
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        return mu, logvar
