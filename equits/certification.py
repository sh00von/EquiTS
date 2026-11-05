"""
Statistical certification of the EquiTS computational bypass.

False merge (paper Sec. 3.3). For a query i that is served by prototype j instead
of the full network,
    E_i = 1[ l(q_j, Y_i) - l(q_i, Y_i) > tau ]
where q_i is the full-network forecast, q_j the bypass forecast and l the MAE over
the horizon in the standardized data space.

Controlled risk. Both procedures below control the per-query false-bypass risk
    L_i(lambda) = 1[ d*_i <= lambda ] * E_i            (bypass AND false merge)
where d*_i is the Stage 2 verification score of the prototype selected for
query i and lambda is the memory's bypass_threshold. Because the selected
prototype does not depend on lambda, L_i is monotone
non-decreasing in lambda. The conditional rate among merges,
FMR = sum(L) / sum(bypass), is reported empirically but is not the controlled
quantity.

1. split_crc_threshold: offline Conformal Risk Control (Angelopoulos et al., 2024)
   on a held-out calibration split. Under exchangeability, E[L_test] <= alpha.

2. OnlineAdaptiveCRC: online threshold update for the test stream, where
   exchangeability fails (serial correlation, drift). Update rule:
       lambda_{t+1} = clip(lambda_t - eta * (Lhat_t - alpha), lambda_min, lambda_max)
   applied once per query. Lhat_t is an importance-weighted estimate of L_t: only a
   random fraction p of bypassed queries is audited (full network also run), and
   audited false merges are weighted by 1/p, so E[Lhat_t] = E[L_t]. Ground truth
   for a window ending at time t arrives H steps later, so feedback is delayed.
   With lambda_min = 0 (no bypass possible), the lower clip only binds when no
   bypass happens, and the standard argument gives, deterministically,
       (1/T) sum_t (Lhat_t - alpha) <= (lambda_max - lambda_min + eta * Lmax) / (eta * T),
   with Lmax = 1/p.
"""

from typing import List
import numpy as np
import torch


def false_merge_indicator(
    bypass_mu: torch.Tensor,
    full_mu: torch.Tensor,
    target: torch.Tensor,
    tolerance: float
) -> torch.Tensor:
    """
    Per-row false-merge indicator. Tensors: [N, H] in the standardized data space.
    Returns float tensor [N] of 0/1.
    """
    err_bypass = torch.mean(torch.abs(bypass_mu - target), dim=-1)
    err_full = torch.mean(torch.abs(full_mu - target), dim=-1)
    return ((err_bypass - err_full) > tolerance).float()


def split_crc_threshold(d_star: torch.Tensor, false_merge: torch.Tensor, alpha: float) -> float:
    """
    Conformal Risk Control for a monotone 0/1 loss L_i(lambda) = 1[d*_i <= lambda] * E_i.
    Returns the largest lambda with (sum_i L_i(lambda) + 1) / (n + 1) <= alpha.
    Returns 0.0 (bypass disabled) if no such lambda exists.

    d_star: [n] (+inf where no verified prototype); false_merge: [n] 0/1.
    """
    n = d_star.numel()
    finite = torch.isfinite(d_star)
    d = d_star[finite].double().cpu()
    e = false_merge[finite].double().cpu()
    if d.numel() == 0:
        return 0.0
    order = torch.argsort(d)
    d, e = d[order], e[order]
    risk = (torch.cumsum(e, 0) + 1.0) / (n + 1.0)       # bound when lambda = d[k]
    ok = (risk <= alpha).nonzero(as_tuple=True)[0]
    if ok.numel() == 0:
        return 0.0
    k = int(ok.max().item())
    # risk is non-decreasing in k, so every lambda in [d[k], d[k+1]) has the same risk.
    # Stay just below the next distance so the next calibration point is excluded.
    if k + 1 < d.numel():
        return float(d[k] + 0.5 * (d[k + 1] - d[k]))
    return float(d[k])


class OnlineAdaptiveCRC:
    """
    Online adaptive risk control of the bypass threshold (see module docstring).
    """
    def __init__(
        self,
        target_risk_alpha: float = 0.05,
        eta: float = 0.001,
        initial_threshold: float = 0.1,
        min_threshold: float = 0.0,
        max_threshold: float = 2.0
    ):
        self.alpha = target_risk_alpha
        self.eta = eta
        self.current_threshold = float(np.clip(initial_threshold, min_threshold, max_threshold))
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.num_decisions = 0
        self.sum_loss_estimate = 0.0
        self.threshold_history: List[float] = [self.current_threshold]

    def get_threshold(self) -> float:
        return self.current_threshold

    def update(self, num_decisions: int, weighted_false_merges: float) -> float:
        """
        Applies the per-query update for a block of resolved decisions.
        num_decisions: number of queries whose ground truth just resolved.
        weighted_false_merges: sum over those queries of Lhat_t
                               (= audited false merges / audit_prob).
        """
        if num_decisions <= 0:
            return self.current_threshold
        grad = weighted_false_merges - self.alpha * num_decisions
        self.current_threshold = float(np.clip(
            self.current_threshold - self.eta * grad,
            self.min_threshold,
            self.max_threshold
        ))
        self.num_decisions += num_decisions
        self.sum_loss_estimate += weighted_false_merges
        self.threshold_history.append(self.current_threshold)
        return self.current_threshold

    def estimated_risk(self) -> float:
        """Importance-weighted estimate of the per-query false-bypass rate."""
        if self.num_decisions == 0:
            return 0.0
        return self.sum_loss_estimate / self.num_decisions
