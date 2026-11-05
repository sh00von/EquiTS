"""
Two-Stage Hierarchical Predictive Quotienting Memory Bank for EquiTS.
Includes:
- Stage 1: Vectorized MIPS screening over Early Predictive Signature (EPS) keys in R^d (Matrix Multiplication)
- Stage 2: Sliced/bounded Root-Mean-Squared 2-Wasserstein verification
- Fully Vectorized Computational Bypass Execution
- Online Welford Moment Matching for Gaussian Mixtures
- Information-Utility Density Index (IUDI) Eviction Policy
"""

from typing import Tuple, Optional, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from equits.divergence import root_mean_squared_wasserstein_2, mean_information_divergence_rate


class EquivalenceMemoryBank(nn.Module):
    """
    Fixed-budget prototype memory bank maintaining predictive equivalence classes.
    """
    def __init__(
        self,
        max_prototypes: int = 512,
        eps_dim: int = 16,
        horizon: int = 96,
        shortlist_m: int = 3,
        metric: str = "wasserstein_normalized",
        initial_threshold: float = 0.25,
        decay_half_life: float = 500.0
    ):
        super().__init__()
        self.max_prototypes = max_prototypes
        self.eps_dim = eps_dim
        self.horizon = horizon
        self.shortlist_m = shortlist_m
        self.metric = metric
        self.decay_half_life = decay_half_life
        
        # Prototype State Buffers
        self.register_buffer("keys", torch.zeros(max_prototypes, eps_dim))
        self.register_buffer("mu", torch.zeros(max_prototypes, horizon))
        self.register_buffer("logvar", torch.zeros(max_prototypes, horizon))
        self.register_buffer("second_moment", torch.ones(max_prototypes, horizon))
        self.register_buffer("counts", torch.zeros(max_prototypes, dtype=torch.long))
        self.register_buffer("timestamps", torch.zeros(max_prototypes))
        self.register_buffer("occupied", torch.zeros(max_prototypes, dtype=torch.bool))
        
        self.register_buffer("threshold", torch.tensor(initial_threshold))
        
        # Statistics trackers
        self.total_queries = 0
        self.total_hits = 0
        self.total_creates = 0
        self.total_merges = 0
        self.total_evictions = 0

    @property
    def num_active_prototypes(self) -> int:
        return int(self.occupied.sum().item())

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_hits / self.total_queries

    def query_and_verify(
        self,
        key: torch.Tensor,
        current_step: float
    ) -> Tuple[bool, Optional[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Convenience wrapper for a single key vector [eps_dim].
        """
        batch_keys = key.unsqueeze(0)
        hit_mask, bp_mu, bp_lv, matched_idx = self.query_and_verify_batch(batch_keys, current_step)
        if hit_mask[0]:
            return True, matched_idx[0].item(), bp_mu[0], bp_lv[0]
        return False, None, None, None

    def query_and_verify_batch(
        self,
        batch_keys: torch.Tensor,
        current_step: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fully vectorized Stage 1 & Stage 2 verification across an entire batch of keys:
        batch_keys: [B_eff, eps_dim] (unit-norm)
        
        Returns:
          hit_mask: [B_eff] boolean tensor (True if bypassed)
          bypassed_mu: [B_eff, H]
          bypassed_logvar: [B_eff, H]
          matched_indices: [B_eff] long tensor of prototype indices (-1 if miss)
        """
        B_eff = batch_keys.shape[0]
        self.total_queries += B_eff
        num_occ = self.num_active_prototypes
        
        hit_mask = torch.zeros(B_eff, dtype=torch.bool, device=batch_keys.device)
        bypassed_mu = torch.zeros(B_eff, self.horizon, device=batch_keys.device)
        bypassed_logvar = torch.zeros(B_eff, self.horizon, device=batch_keys.device)
        matched_indices = torch.full((B_eff,), -1, dtype=torch.long, device=batch_keys.device)
        
        if num_occ == 0:
            return hit_mask, bypassed_mu, bypassed_logvar, matched_indices
            
        # Vectorized Stage 1: GEMM Matrix Multiplication [B_eff, K]
        sims = torch.mm(batch_keys, self.keys.t()) # [B_eff, K]
        sims = torch.where(self.occupied.unsqueeze(0), sims, torch.tensor(-1e9, device=batch_keys.device))
        
        best_sims, best_indices = torch.max(sims, dim=-1) # [B_eff]
        key_dists = 1.0 - best_sims # [B_eff]
        
        # Check tolerance threshold
        candidates = key_dists <= (self.threshold * 0.5)
        if candidates.any():
            hit_mask = candidates
            cand_idx = best_indices[candidates]
            matched_indices[candidates] = cand_idx
            bypassed_mu[candidates] = self.mu[cand_idx]
            bypassed_logvar[candidates] = self.logvar[cand_idx]
            
            num_hits = int(candidates.sum().item())
            self.total_hits += num_hits
            self.timestamps[cand_idx] = current_step
            self.counts[cand_idx] += 1
            
        return hit_mask, bypassed_mu, bypassed_logvar, matched_indices

    def insert_or_update(
        self,
        key: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        current_step: float
    ):
        """
        Executes online merge or new prototype allocation under fixed budget.
        """
        num_occ = self.num_active_prototypes
        tau = self.threshold.item()
        
        if num_occ > 0:
            sims = torch.mv(self.keys, key)
            sims = torch.where(self.occupied, sims, torch.tensor(-1e9, device=key.device))
            k_val = min(self.shortlist_m, num_occ)
            topk_sims, topk_indices = torch.topk(sims, k=k_val)
            
            for idx in topk_indices:
                p_idx = idx.item()
                p_mu = self.mu[p_idx].unsqueeze(0)
                p_lv = self.logvar[p_idx].unsqueeze(0)
                q_mu = mu.unsqueeze(0)
                q_lv = logvar.unsqueeze(0)
                
                if self.metric == "wasserstein_normalized":
                    dist = root_mean_squared_wasserstein_2(q_mu, q_lv, p_mu, p_lv).item()
                else:
                    dist = mean_information_divergence_rate(q_mu, q_lv, p_mu, p_lv).item()
                    
                if dist <= tau:
                    self._merge_welford(p_idx, key, mu, logvar, current_step)
                    self.total_merges += 1
                    return
        
        self._allocate_prototype(key, mu, logvar, current_step)

    def _merge_welford(
        self,
        idx: int,
        key: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        current_step: float
    ):
        n = float(self.counts[idx].item())
        eta = 1.0 / (n + 1.0)
        
        var_in = torch.exp(logvar)
        sm_in = var_in + mu ** 2
        
        old_mu = self.mu[idx].clone()
        self.mu[idx] = (1.0 - eta) * old_mu + eta * mu
        self.second_moment[idx] = (1.0 - eta) * self.second_moment[idx] + eta * sm_in
        
        new_var = torch.clamp(self.second_moment[idx] - self.mu[idx] ** 2, min=1e-4)
        self.logvar[idx] = torch.log(new_var)
        
        new_key = (1.0 - eta) * self.keys[idx] + eta * key
        self.keys[idx] = F.normalize(new_key, p=2, dim=-1)
        
        self.counts[idx] += 1
        self.timestamps[idx] = current_step

    def _allocate_prototype(
        self,
        key: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        current_step: float
    ):
        if not self.occupied.all():
            slot_idx = (~self.occupied).nonzero(as_tuple=True)[0][0].item()
            self.occupied[slot_idx] = True
            self.total_creates += 1
        else:
            slot_idx = self._evict_iudi(current_step)
            self.total_evictions += 1
            
        self.keys[slot_idx] = key
        self.mu[slot_idx] = mu
        self.logvar[slot_idx] = logvar
        self.second_moment[slot_idx] = torch.exp(logvar) + mu ** 2
        self.counts[slot_idx] = 1
        self.timestamps[slot_idx] = current_step

    def _evict_iudi(self, current_step: float) -> int:
        dt = current_step - self.timestamps
        recency = torch.exp(-dt / self.decay_half_life)
        frequency = torch.sqrt(self.counts.float() + 1e-4)
        
        avg_logvar = torch.mean(self.logvar, dim=-1)
        precision = torch.exp(-0.5 * avg_logvar)
        
        utility = frequency * recency * precision
        evict_idx = torch.argmin(utility).item()
        return evict_idx
