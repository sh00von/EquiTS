"""
Evaluation metrics: point, probabilistic (NLL, CRPS, interval score) and split conformal intervals.

All tensors are [N, H, D] in the standardized data space. Interval metrics use a
central (1 - alpha) interval [l, u]; default alpha = 0.1 (nominal 90%).

Metrics
  coverage       : fraction of targets inside [l, u]
  width          : mean u - l
  interval score : Gneiting & Raftery (2007), proper scoring rule for intervals,
                   IS = (u - l) + (2/alpha)(l - y)1[y < l] + (2/alpha)(y - u)1[y > u]
                   (lower is better; it rewards narrow intervals and penalizes misses)
  CRPS           : closed form for Gaussian forecasts (lower is better)

Conformal intervals (split conformal, calibrated on the validation split,
separately for every horizon step h and channel d):
  absolute  : score |y - mu|           -> mu +/- q            (fixed width per (h, d))
  normalized: score |y - mu| / sigma   -> mu +/- q * sigma    (adaptive width)
With n calibration windows the quantile is the ceil((n + 1)(1 - alpha))-th
smallest score, which gives finite-sample marginal coverage >= 1 - alpha under
exchangeability. Consecutive windows overlap, so exchangeability holds only
approximately for time series; coverage is reported on the test split as a check.
"""

import math
from typing import Dict, Tuple

import torch

SQRT_PI = math.sqrt(math.pi)


def normal_z(alpha: float) -> float:
    """Two-sided standard normal quantile z_{1 - alpha/2}."""
    return float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha / 2.0)))


def gaussian_crps(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> float:
    """Mean closed-form CRPS of N(mu, exp(logvar)) at y."""
    sigma = torch.exp(0.5 * logvar)
    z = (y - mu) / sigma
    std = torch.distributions.Normal(0.0, 1.0)
    pdf = torch.exp(std.log_prob(z))
    cdf = std.cdf(z)
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / SQRT_PI)
    return crps.mean().item()


def interval_metrics(lower: torch.Tensor, upper: torch.Tensor, y: torch.Tensor, alpha: float = 0.1) -> Dict[str, float]:
    below = (lower - y).clamp(min=0.0)
    above = (y - upper).clamp(min=0.0)
    score = (upper - lower) + (2.0 / alpha) * (below + above)
    inside = (y >= lower) & (y <= upper)
    return {
        "coverage": inside.float().mean().item(),
        "width": (upper - lower).mean().item(),
        "interval_score": score.mean().item(),
    }


def conformal_quantile(scores: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    scores: [n, H, D] calibration non-conformity scores.
    Returns [H, D]: the ceil((n+1)(1-alpha))-th smallest score per (h, d)
    (+inf if that rank exceeds n).
    """
    n = scores.shape[0]
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return torch.full(scores.shape[1:], float("inf"))
    return torch.sort(scores, dim=0).values[k - 1]


def conformal_interval(
    mu_cal: torch.Tensor, y_cal: torch.Tensor, mu_test: torch.Tensor,
    sigma_cal: torch.Tensor = None, sigma_test: torch.Tensor = None, alpha: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Absolute (sigma=None) or normalized split conformal interval for the test set."""
    if sigma_cal is None:
        q = conformal_quantile((y_cal - mu_cal).abs(), alpha)
        half = q.unsqueeze(0).expand_as(mu_test)
    else:
        q = conformal_quantile((y_cal - mu_cal).abs() / sigma_cal, alpha)
        half = q.unsqueeze(0) * sigma_test
    return mu_test - half, mu_test + half


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, include_constant: bool = True) -> torch.Tensor:
    """Mean Gaussian negative log-likelihood (differentiable; returns a scalar tensor)."""
    nll = 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))
    if include_constant:
        nll = nll + 0.5 * math.log(2.0 * math.pi)
    return nll.mean()


def point_metrics(pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    err = pred - y
    return {"MSE": (err ** 2).mean().item(), "MAE": err.abs().mean().item()}
