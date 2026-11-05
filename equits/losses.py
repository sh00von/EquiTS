"""
Loss functions and evaluation metrics for PEC-Net.
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
    Encourages cosine similarity of EPS keys (k_i . k_j) to correlate inversely
    with their predictive discrepancy, turning Stage 1 screening into a valid metric filter.
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, keys: torch.Tensor, predictive_dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        keys: [B_eff, d] (unit-norm)
        predictive_dist_matrix: [B_eff, B_eff] pairwise predictive distances
        """
        # Cosine distance in key space = 1 - (k_i . k_j)
        key_cos_sim = torch.mm(keys, keys.t()) # [B_eff, B_eff] in [-1, 1]
        key_dist = 1.0 - key_cos_sim
        
        # Binary target: 1 if within equivalence threshold, 0 otherwise
        equiv_pairs = (predictive_dist_matrix <= self.margin).float()
        
        # Margin contrastive loss: pull equivalent pairs together, push distinct pairs apart
        pos_loss = equiv_pairs * (key_dist ** 2)
        neg_loss = (1.0 - equiv_pairs) * (F.relu(self.margin - key_dist) ** 2)
        
        return torch.mean(pos_loss + neg_loss)


def compute_forecast_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """
    Evaluates MAE, MSE, and RMSE.
    """
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = math.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}
