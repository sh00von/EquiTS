"""
EquiTS: Equivalence-Quotiented Time-Series Network.
Unified architecture integrating:
- Reversible Instance Normalization (RevIN)
- Channel-Independent Large-Kernel Depthwise 1D Convolutional Backbone
- Early Predictive Signature (EPS) Projections
- Residual Linear Highway
- Direct Multi-Horizon Probabilistic Head
- Two-Stage Hierarchical Memory Bank with Computational Bypass
"""

import math
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

from equits.revin import RevIN
from equits.encoder import CIModernEncoder
from equits.predictor import DualProbabilisticHead
from equits.memory import EquivalenceMemoryBank


class EquiTS(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        ds_cfg = config["dataset"]
        bb_cfg = config["backbone"]
        pred_cfg = config["prediction_head"]
        mem_cfg = config.get("memory", {})

        self.lookback = ds_cfg["lookback"]
        self.horizon = ds_cfg["horizon"]
        self.num_channels = ds_cfg["num_channels"]

        # 1. Reversible Instance Normalization
        self.use_revin = bb_cfg.get("use_revin", True)
        if self.use_revin:
            self.revin = RevIN(num_features=self.num_channels, affine=bb_cfg.get("revin_affine", True))

        # 2. Channel-Independent Modern Depthwise Encoder
        self.d_model = bb_cfg.get("d_model", 64)
        self.eps_dim = mem_cfg.get("eps_dim", 16)
        self.encoder = CIModernEncoder(
            lookback=self.lookback,
            d_model=self.d_model,
            kernel_size=bb_cfg.get("kernel_size", 31),
            eps_dim=self.eps_dim,
            dropout=bb_cfg.get("dropout", 0.1),
            eps_detach_backbone=mem_cfg.get("eps_detach_backbone", True)
        )

        # 2b. Residual Linear Highway (direct input-to-horizon projection)
        self.use_linear_highway = bb_cfg.get("use_linear_highway", True)
        if self.use_linear_highway:
            self.linear_highway = nn.Linear(self.lookback, self.horizon)
            nn.init.xavier_uniform_(self.linear_highway.weight, gain=1.0)
            nn.init.zeros_(self.linear_highway.bias)

        # Data-space variance floor (standardized units). Without it, an input window
        # with a flat channel has RevIN stdev ~ 0, so the predicted variance collapses
        # and the NLL / its gradients explode.
        self.log_var_floor = math.log(pred_cfg.get("data_var_floor", 1e-2))

        # 3. Probabilistic Prediction Head
        self.pred_head = DualProbabilisticHead(
            d_model=self.d_model,
            horizon=self.horizon,
            min_logvar=pred_cfg.get("min_logvar", -8.0),
            max_logvar=pred_cfg.get("max_logvar", 6.0)
        )

        # 4. Two-Stage Predictive Quotienting Memory Bank (always built so that
        #    checkpoints are interchangeable; `memory_enabled` toggles its use).
        self.memory_enabled = mem_cfg.get("enabled", False)
        self.memory = EquivalenceMemoryBank(
            max_prototypes=mem_cfg.get("max_prototypes", 512),
            eps_dim=self.eps_dim,
            horizon=self.horizon,
            shortlist_m=mem_cfg.get("shortlist_m", 3),
            metric=mem_cfg.get("metric", "wasserstein_normalized"),
            merge_threshold=mem_cfg.get("merge_threshold", 0.25),
            bypass_threshold=mem_cfg.get("initial_bypass_threshold", 0.1),
            key_threshold=mem_cfg.get("key_gate", 2.0),
            decay_half_life=mem_cfg.get("decay_half_life", 500.0)
        )

    # ------------------------------------------------------------------ helpers
    def _encode_early(self, x: torch.Tensor):
        """x: [B, L, D] -> x_ci [B*D, 1, L], h1, eps keys [B*D, d], linear forecast [B*D, H]."""
        B, L, D = x.shape
        x_norm = self.revin(x, mode="norm") if self.use_revin else x
        x_ci = x_norm.permute(0, 2, 1).contiguous().view(B * D, 1, L)
        h1, eps_keys = self.encoder.forward_early(x_ci)
        if self.use_linear_highway:
            mu_linear = self.linear_highway(x_ci.squeeze(1))
        else:
            mu_linear = torch.zeros(B * D, self.horizon, device=x.device, dtype=x.dtype)
        return h1, eps_keys, mu_linear

    def _to_output(self, rows_mu: torch.Tensor, rows_logvar: torch.Tensor, B: int, D: int):
        """[B*D, H] rows in normalized space -> data-space [B, H, D] mean and log-variance."""
        mu = rows_mu.view(B, D, self.horizon).permute(0, 2, 1).contiguous()
        logvar = rows_logvar.view(B, D, self.horizon).permute(0, 2, 1).contiguous()
        if self.use_revin:
            mu = self.revin(mu, mode="denorm")
            logvar = self.revin.denorm_logvar(logvar)
        # log(var + floor), smooth in var
        logvar = torch.logaddexp(logvar, torch.full_like(logvar, self.log_var_floor))
        return mu, logvar

    def _stage2_input(self, mu_linear: torch.Tensor):
        return mu_linear if self.use_linear_highway else None

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = False,
        audit_prob: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        x: [B, L, D]
        update_memory: insert non-bypassed samples into the memory bank (online mode).
        audit_prob: fraction of bypassed samples for which the full network is also run,
                    so the risk controller can check whether the bypass was a false merge.
                    The served forecast is always the bypass forecast.
        Returns:
          mu:       [B, H, D] forecast mean (data space)
          logvar:   [B, H, D] forecast log-variance (data space)
          eps_keys: [B*D, eps_dim]
          aux:      dict with bypass statistics; in training mode also the deep
                    residual distribution rows (res_mu, res_logvar) for the EPS
                    alignment loss; with audits, `audit_mask` and `mu_full`.
        """
        B, L, D = x.shape
        N = B * D
        h1, eps_keys, mu_linear = self._encode_early(x)

        res_mu = torch.empty(N, self.horizon, device=x.device)
        res_logvar = torch.empty(N, self.horizon, device=x.device)
        hit_mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        audit_mask = torch.zeros(N, dtype=torch.bool, device=x.device)

        use_memory = self.memory_enabled and not self.training
        if use_memory:
            hit_mask, bp_mu, bp_lv, _ = self.memory.query_and_verify_batch(
                eps_keys, self._stage2_input(mu_linear)
            )
            if hit_mask.any():
                res_mu[hit_mask] = bp_mu[hit_mask]
                res_logvar[hit_mask] = bp_lv[hit_mask]
                if audit_prob > 0.0:
                    audit_mask = hit_mask & (torch.rand(N, device=x.device) < audit_prob)

        # Deep trunk only for misses (+ audited hits)
        compute_mask = (~hit_mask) | audit_mask
        full_res_mu = full_res_lv = None
        if compute_mask.any():
            z_deep = self.encoder.forward_deep(h1[compute_mask])
            sub_mu, sub_logvar = self.pred_head(z_deep)
            miss_in_sub = (~hit_mask)[compute_mask]
            miss_mask = ~hit_mask
            res_mu[miss_mask] = sub_mu[miss_in_sub]
            res_logvar[miss_mask] = sub_logvar[miss_in_sub]

            if audit_mask.any():
                full_res_mu = res_mu.clone()
                full_res_lv = res_logvar.clone()
                full_res_mu[audit_mask] = sub_mu[~miss_in_sub]
                full_res_lv[audit_mask] = sub_logvar[~miss_in_sub]

            if use_memory and update_memory and miss_mask.any():
                self.memory.insert_or_update_batch(
                    eps_keys[miss_mask].detach(),
                    res_mu[miss_mask].detach(),
                    res_logvar[miss_mask].detach(),
                    mu_linear[miss_mask].detach()
                )

        mu, logvar = self._to_output(mu_linear + res_mu, res_logvar, B, D)

        bypassed = int(hit_mask.sum().item()) if use_memory else 0
        aux: Dict[str, Any] = {
            "bypassed_count": bypassed,
            "audited_count": int(audit_mask.sum().item()),
            "total_tokens": N,
            "bypass_ratio": bypassed / float(N),
            "active_prototypes": self.memory.num_active_prototypes if use_memory else 0,
        }
        if self.training:
            aux["res_mu"] = res_mu
            aux["res_logvar"] = res_logvar
        if full_res_mu is not None:
            mu_full, _ = self._to_output(mu_linear + full_res_mu, full_res_lv, B, D)
            aux["audit_mask"] = audit_mask
            aux["mu_full"] = mu_full
        return mu, logvar, eps_keys, aux

    @torch.no_grad()
    def forward_highway_only(self, x: torch.Tensor) -> torch.Tensor:
        """Ablation: forecast from the linear highway alone (deep residual = 0). Returns [B, H, D]."""
        B, L, D = x.shape
        _, _, mu_linear = self._encode_early(x)
        zeros = torch.zeros_like(mu_linear)
        mu, _ = self._to_output(mu_linear, zeros, B, D)
        return mu

    # -------------------------------------------------------------- memory ops
    @torch.no_grad()
    def populate_memory(self, x: torch.Tensor):
        """Runs the full network on x and inserts every row into the memory bank."""
        B, L, D = x.shape
        h1, eps_keys, mu_linear = self._encode_early(x)
        z = self.encoder.forward_deep(h1)
        res_mu, res_lv = self.pred_head(z)
        self.memory.insert_or_update_batch(eps_keys, res_mu, res_lv, mu_linear)

    @torch.no_grad()
    def calibration_forward(self, x: torch.Tensor):
        """
        For conformal calibration. Returns, per row [B*D]:
          score:     Stage 2 verification score of the selected prototype (+inf if none)
          mu_full:   [B*D, H] full-network forecast (data space)
          mu_bypass: [B*D, H] forecast that would be served on bypass (data space;
                     equals mu_full where no prototype was found)
        """
        B, L, D = x.shape
        h1, eps_keys, mu_linear = self._encode_early(x)
        z = self.encoder.forward_deep(h1)
        res_mu, res_lv = self.pred_head(z)
        score, idx = self.memory.nearest_verified(eps_keys, self._stage2_input(mu_linear))

        bp_mu = res_mu.clone()
        found = idx >= 0
        bp_mu[found] = self.memory.mu[idx[found]]

        mu_full, _ = self._to_output(mu_linear + res_mu, res_lv, B, D)
        mu_bp, _ = self._to_output(mu_linear + bp_mu, res_lv, B, D)
        to_rows = lambda t: t.permute(0, 2, 1).reshape(B * D, self.horizon)
        return score, to_rows(mu_full), to_rows(mu_bp)
