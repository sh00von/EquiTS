"""
Statistical Certification and Conformal Risk Control (CRC) for PEC-Net.
Provides non-asymptotic and online-adaptive guarantees on the False-Merge Rate (FMR)
under temporal serial correlation and non-stationarity.
"""

from typing import Tuple, List, Optional
import numpy as np
import torch


class OnlineAdaptiveCRC:
    """
    Online Adaptive Conformal Risk Control (O-CRC) for streaming time-series quotienting.
    Adjusts the merge threshold lambda_t via no-regret online gradient descent:
        lambda_{t+1} = clip(lambda_t - eta * (Loss_t - alpha), lambda_min, lambda_max)
    
    Guarantees:
        limsup_{T -> infty} (1/T) sum_{t=1}^T Loss_t <= alpha + O(1 / sqrt(T))
    regardless of serial correlation, seasonal shifts, or non-stationarity.
    """
    def __init__(
        self,
        target_risk_alpha: float = 0.05,
        eta: float = 0.005,
        initial_threshold: float = 0.25,
        min_threshold: float = 0.02,
        max_threshold: float = 1.00
    ):
        self.alpha = target_risk_alpha
        self.eta = eta
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        self.loss_history: List[float] = []
        self.threshold_history: List[float] = [initial_threshold]

    def get_threshold(self) -> float:
        return self.current_threshold

    def update(self, realized_loss: float) -> float:
        """
        realized_loss: 1.0 if merge was a false-merge (error exceeded tolerance tau), 0.0 otherwise.
        """
        self.loss_history.append(float(realized_loss))
        
        # Online gradient descent update
        grad = realized_loss - self.alpha
        self.current_threshold = float(np.clip(
            self.current_threshold - self.eta * grad,
            self.min_threshold,
            self.max_threshold
        ))
        self.threshold_history.append(self.current_threshold)
        return self.current_threshold

    def empirical_risk(self) -> float:
        if not self.loss_history:
            return 0.0
        return float(np.mean(self.loss_history))


class FalseMergeMonitor:
    """
    Monitors empirical false-merge events:
    A merge of context i into prototype j is certified valid if the realized evaluation loss
    under prototype q_j does not inflate beyond tolerance tau compared to the model's unmerged prediction q_i:
        Loss(q_j, Y) - Loss(q_i, Y) <= tau
    """
    def __init__(self, tolerance_tau: float = 0.10):
        self.tau = tolerance_tau
        self.total_merges = 0
        self.false_merges = 0

    def evaluate_merge(
        self,
        proto_mu: torch.Tensor,
        proto_logvar: torch.Tensor,
        unmerged_mu: torch.Tensor,
        unmerged_logvar: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> float:
        """
        Computes 0/1 false-merge indicator.
        """
        err_proto = torch.mean(torch.abs(proto_mu - ground_truth)).item()
        err_unmerged = torch.mean(torch.abs(unmerged_mu - ground_truth)).item()
        
        loss_inflation = err_proto - err_unmerged
        is_false_merge = 1.0 if loss_inflation > self.tau else 0.0
        
        self.total_merges += 1
        if is_false_merge > 0.5:
            self.false_merges += 1
            
        return is_false_merge

    def false_merge_rate(self) -> float:
        if self.total_merges == 0:
            return 0.0
        return self.false_merges / self.total_merges
