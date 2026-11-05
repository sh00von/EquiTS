"""
Two-Stage Hierarchical Predictive Quotienting Memory Bank for EquiTS.

Lookup (inference, no deep trunk required):
  Stage 1 - Retrieval: cosine similarity (one GEMM) between the query's Early
            Predictive Signature (EPS) key and all prototype keys; keep the
            top-m candidates (optionally only those within a fixed key gate).
  Stage 2 - Verification score: RMS distance between the query's linear-highway
            forecast and each candidate's stored highway centroid. Both are
            point forecasts in RevIN-normalized units, so this is the normalized
            2-Wasserstein distance between the two Dirac measures. The candidate
            with the smallest score s* is selected.
  Decision - The query bypasses the deep trunk iff s* <= bypass_threshold.

  The selected prototype does not depend on bypass_threshold, so the bypass
  decision (and the false-merge loss) is monotone in bypass_threshold. This is
  what makes conformal risk control of bypass_threshold valid.
  (If the model has no linear highway, the key distance is used as the score.)

Insertion (prepopulation / optional online updates):
  Uses the full predictive distribution of the deep residual: a sample merges
  into a top-m candidate if their normalized W2 (or KL rate) distance is
  <= merge_threshold; otherwise a new prototype is allocated (IUDI eviction
  when the budget is full). Merges use Welford moment matching.
"""

from typing import Tuple, Optional
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
        merge_threshold: float = 0.25,
        bypass_threshold: float = 0.1,
        key_threshold: float = 2.0,
        decay_half_life: float = 500.0
    ):
        super().__init__()
        if metric not in ("wasserstein_normalized", "kl_mean_rate"):
            raise ValueError(f"Unknown metric: {metric}")
        self.max_prototypes = max_prototypes
        self.eps_dim = eps_dim
        self.horizon = horizon
        self.shortlist_m = shortlist_m
        self.metric = metric
        self.decay_half_life = decay_half_life

        # Prototype state
        self.register_buffer("keys", torch.zeros(max_prototypes, eps_dim))
        self.register_buffer("mu", torch.zeros(max_prototypes, horizon))          # deep residual mean
        self.register_buffer("logvar", torch.zeros(max_prototypes, horizon))      # deep predictive log-variance
        self.register_buffer("second_moment", torch.ones(max_prototypes, horizon))
        self.register_buffer("lin_mu", torch.zeros(max_prototypes, horizon))      # linear-highway centroid (Stage 2)
        self.register_buffer("counts", torch.zeros(max_prototypes, dtype=torch.long))  # merged samples
        self.register_buffer("hits", torch.zeros(max_prototypes, dtype=torch.long))    # times reused at lookup
        self.register_buffer("timestamps", torch.zeros(max_prototypes))
        self.register_buffer("occupied", torch.zeros(max_prototypes, dtype=torch.bool))
        self.register_buffer("clock", torch.zeros((), dtype=torch.float64))

        # Thresholds (buffers so they are saved with the model and adjustable by CRC)
        self.register_buffer("merge_threshold", torch.tensor(float(merge_threshold)))
        self.register_buffer("bypass_threshold", torch.tensor(float(bypass_threshold)))  # CRC-controlled
        self.register_buffer("key_threshold", torch.tensor(float(key_threshold)))        # fixed Stage 1 gate (2.0 = off)

        self.reset_stats()

    # ------------------------------------------------------------------ utils
    def reset_stats(self):
        self.total_queries = 0
        self.total_hits = 0
        self.total_creates = 0
        self.total_merges = 0
        self.total_evictions = 0

    def reset(self):
        """Empty the bank (thresholds are kept)."""
        self.occupied.zero_()
        self.counts.zero_()
        self.hits.zero_()
        self.timestamps.zero_()
        self.clock.zero_()
        self.reset_stats()

    @property
    def num_active_prototypes(self) -> int:
        return int(self.occupied.sum().item())

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_hits / self.total_queries

    def _distance(self, q_mu, q_lv, p_mu, p_lv) -> torch.Tensor:
        if self.metric == "wasserstein_normalized":
            return root_mean_squared_wasserstein_2(q_mu, q_lv, p_mu, p_lv)
        return mean_information_divergence_rate(q_mu, q_lv, p_mu, p_lv)

    # ----------------------------------------------------------------- lookup
    def nearest_verified(
        self,
        batch_keys: torch.Tensor,
        lin_forecast: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1 retrieval + Stage 2 scoring, without applying bypass_threshold.
        batch_keys: [N, eps_dim] (unit-norm); lin_forecast: [N, H] or None.
        Returns:
          score: [N] verification score of the selected prototype (+inf if none)
          idx:   [N] its slot index (-1 if none)
        """
        N = batch_keys.shape[0]
        device = batch_keys.device
        score = torch.full((N,), float("inf"), device=device)
        idx = torch.full((N,), -1, dtype=torch.long, device=device)
        num_occ = self.num_active_prototypes
        if num_occ == 0:
            return score, idx

        sims = batch_keys @ self.keys.t()                                   # [N, K]
        sims = sims.masked_fill(~self.occupied.unsqueeze(0), float("-inf"))
        m = min(self.shortlist_m, num_occ)
        cand_sims, cand_idx = torch.topk(sims, k=m, dim=-1)                 # [N, m]
        key_dist = 1.0 - cand_sims

        if lin_forecast is not None:
            diff = lin_forecast.unsqueeze(1) - self.lin_mu[cand_idx]         # [N, m, H]
            cand_score = torch.sqrt(torch.mean(diff * diff, dim=-1) + 1e-12)
        else:
            cand_score = key_dist.clone()
        cand_score = cand_score.masked_fill(key_dist > self.key_threshold, float("inf"))

        best, pos = torch.min(cand_score, dim=-1)
        found = torch.isfinite(best)
        score[found] = best[found]
        idx[found] = cand_idx[found, pos[found]]
        return score, idx

    def query_and_verify_batch(
        self,
        batch_keys: torch.Tensor,
        lin_forecast: Optional[torch.Tensor] = None,
        record: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          hit_mask: [N] bool, bypassed_mu: [N, H], bypassed_logvar: [N, H],
          matched_indices: [N] long (-1 if miss)
        """
        N = batch_keys.shape[0]
        score, idx = self.nearest_verified(batch_keys, lin_forecast)
        hit_mask = score <= self.bypass_threshold
        matched = torch.where(hit_mask, idx, torch.full_like(idx, -1))

        bypassed_mu = torch.zeros(N, self.horizon, device=batch_keys.device)
        bypassed_logvar = torch.zeros(N, self.horizon, device=batch_keys.device)
        if hit_mask.any():
            sel = matched[hit_mask]
            bypassed_mu[hit_mask] = self.mu[sel]
            bypassed_logvar[hit_mask] = self.logvar[sel]
            if record:
                self.clock += 1
                self.hits.index_add_(0, sel, torch.ones_like(sel))
                self.timestamps[sel] = self.clock.float()

        if record:
            self.total_queries += N
            self.total_hits += int(hit_mask.sum().item())
        return hit_mask, bypassed_mu, bypassed_logvar, matched

    def query_and_verify(self, key: torch.Tensor, lin_forecast: Optional[torch.Tensor] = None):
        """Single-key convenience wrapper. key: [eps_dim]."""
        lf = lin_forecast.unsqueeze(0) if lin_forecast is not None else None
        hit, mu, lv, idx = self.query_and_verify_batch(key.unsqueeze(0), lf)
        if hit[0]:
            return True, int(idx[0].item()), mu[0], lv[0]
        return False, None, None, None

    # -------------------------------------------------------------- insertion
    @torch.no_grad()
    def insert_or_update_batch(
        self,
        keys: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        lin_mu: Optional[torch.Tensor] = None
    ):
        """
        keys: [N, eps_dim]; mu, logvar: [N, H] deep residual distribution; lin_mu: [N, H].
        Merge candidates are screened against the bank state at the start of the
        batch (vectorized); merges/allocations are then applied sequentially.
        """
        N = keys.shape[0]
        if N == 0:
            return
        if lin_mu is None:
            lin_mu = torch.zeros_like(mu)
        self.clock += 1
        now = float(self.clock.item())

        target = torch.full((N,), -1, dtype=torch.long, device=keys.device)
        num_occ = self.num_active_prototypes
        if num_occ > 0:
            sims = keys @ self.keys.t()
            sims = sims.masked_fill(~self.occupied.unsqueeze(0), float("-inf"))
            m = min(self.shortlist_m, num_occ)
            _, cand_idx = torch.topk(sims, k=m, dim=-1)                       # [N, m]
            dist = self._distance(mu.unsqueeze(1), logvar.unsqueeze(1),
                                  self.mu[cand_idx], self.logvar[cand_idx])    # [N, m]
            dist = dist.masked_fill(dist > self.merge_threshold, float("inf"))
            best, pos = torch.min(dist, dim=-1)
            ok = torch.isfinite(best)
            target[ok] = cand_idx[ok, pos[ok]]

        overwritten = set()
        target_list = target.tolist()
        for i in range(N):
            t = target_list[i]
            if t >= 0 and t not in overwritten:
                self._merge_welford(t, keys[i], mu[i], logvar[i], lin_mu[i], now)
                self.total_merges += 1
            else:
                slot = self._allocate_prototype(keys[i], mu[i], logvar[i], lin_mu[i], now)
                overwritten.add(slot)

    def insert_or_update(self, key, mu, logvar, lin_mu=None):
        """Single-sample convenience wrapper."""
        self.insert_or_update_batch(
            key.unsqueeze(0), mu.unsqueeze(0), logvar.unsqueeze(0),
            lin_mu.unsqueeze(0) if lin_mu is not None else None
        )

    def _merge_welford(self, idx, key, mu, logvar, lin_mu, now):
        n = float(self.counts[idx].item())
        eta = 1.0 / (n + 1.0)

        sm_in = torch.exp(logvar) + mu ** 2
        self.mu[idx] = (1.0 - eta) * self.mu[idx] + eta * mu
        self.second_moment[idx] = (1.0 - eta) * self.second_moment[idx] + eta * sm_in
        new_var = torch.clamp(self.second_moment[idx] - self.mu[idx] ** 2, min=1e-4)
        self.logvar[idx] = torch.log(new_var)
        self.lin_mu[idx] = (1.0 - eta) * self.lin_mu[idx] + eta * lin_mu
        self.keys[idx] = F.normalize((1.0 - eta) * self.keys[idx] + eta * key, p=2, dim=-1)

        self.counts[idx] += 1
        self.timestamps[idx] = now

    def _allocate_prototype(self, key, mu, logvar, lin_mu, now) -> int:
        if not self.occupied.all():
            slot = int((~self.occupied).nonzero(as_tuple=True)[0][0].item())
            self.occupied[slot] = True
            self.total_creates += 1
        else:
            slot = self._evict_iudi(now)
            self.total_evictions += 1

        self.keys[slot] = key
        self.mu[slot] = mu
        self.logvar[slot] = logvar
        self.second_moment[slot] = torch.exp(logvar) + mu ** 2
        self.lin_mu[slot] = lin_mu
        self.counts[slot] = 1
        self.hits[slot] = 0
        self.timestamps[slot] = now
        return slot

    def _evict_iudi(self, now: float) -> int:
        """Information-Utility Density Index: frequency x recency x precision."""
        dt = now - self.timestamps
        recency = torch.exp(-dt / self.decay_half_life)
        frequency = torch.sqrt((self.counts + self.hits).float() + 1e-4)
        precision = torch.exp(-0.5 * torch.mean(self.logvar, dim=-1))
        utility = frequency * recency * precision
        return int(torch.argmin(utility).item())
