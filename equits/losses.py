"""
Loss functions and evaluation metrics for EquiTS.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss for diagonal Gaussian forecasts.
    NLL = 0.5 * mean( logvar + (Y - mu)^2 * exp(-logvar) ) + constant
    """
    def __init__(self, include_constant: bool = False):
        super().__init__()
        self.include_constant = include_constant
        self.c = 0.5 * math.log(2.0 * math.pi) if include_constant else 0.0

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        mu: [..., H, D]
        logvar: [..., H, D]
        target: [..., H, D]
        mask: Optional boolean or float tensor where 1 indicates valid observation.
        """
        precision = torch.exp(-logvar)
        sq_err = (target - mu) ** 2
        elementwise_nll = 0.5 * (logvar + sq_err * precision) + self.c
        
        if mask is not None:
            elementwise_nll = elementwise_nll * mask
            return torch.sum(elementwise_nll) / (torch.sum(mask) + 1e-8)
        
        return torch.mean(elementwise_nll)


class EPSMetricAlignmentLoss(nn.Module):
    """
    Trains the Early Predictive Signature (EPS) keys so that cosine distance in key
    space tracks predictive equivalence, turning Stage 1 screening into a useful filter.

    Pairs whose predictive distributions are within `equiv_threshold` (same units as
    the memory merge threshold) are pulled together; other pairs are pushed beyond
    `key_margin` in cosine distance. Positive and negative terms are averaged
    separately because equivalent pairs are typically rare.
    """
    def __init__(self, equiv_threshold: float = 0.25, key_margin: float = 0.3):
        super().__init__()
        self.equiv_threshold = equiv_threshold
        self.key_margin = key_margin

    def forward(self, keys: torch.Tensor, predictive_dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        keys: [N, d] (unit-norm)
        predictive_dist_matrix: [N, N] pairwise predictive distances (no gradient needed)
        """
        N = keys.shape[0]
        key_dist = 1.0 - keys @ keys.t()                      # cosine distance in [0, 2]
        off_diag = ~torch.eye(N, dtype=torch.bool, device=keys.device)

        equiv = (predictive_dist_matrix <= self.equiv_threshold) & off_diag
        distinct = (~equiv) & off_diag

        pos = (key_dist ** 2)[equiv]
        neg = (F.relu(self.key_margin - key_dist) ** 2)[distinct]
        pos_loss = pos.mean() if pos.numel() > 0 else key_dist.new_zeros(())
        neg_loss = neg.mean() if neg.numel() > 0 else key_dist.new_zeros(())
        return pos_loss + neg_loss


def gaussian_interval_coverage(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, z: float = 1.6449) -> float:
    """Empirical coverage of the central interval mu +/- z*sigma (z=1.6449 -> nominal 90%)."""
    sigma = torch.exp(0.5 * logvar)
    inside = (y >= mu - z * sigma) & (y <= mu + z * sigma)
    return inside.float().mean().item()


def compute_forecast_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """
    Evaluates MAE, MSE, and RMSE.
    """
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = math.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}
