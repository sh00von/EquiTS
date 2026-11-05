"""
EquiTS: Equivalence-Quotiented Time-Series Network.
Unified architecture integrating:
- Reversible Instance Normalization (RevIN)
- Channel-Independent Large-Kernel Depthwise 1D Convolutional Backbone
- Early Predictive Signature (EPS) Projections
- Direct Multi-Horizon Probabilistic Head
- Fully Vectorized Two-Stage Hierarchical Memory Bank with Computational Bypass
"""

from typing import Tuple, Dict, Any, Optional
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
            self.revin = RevIN(
                num_features=self.num_channels,
                affine=bb_cfg.get("revin_affine", True)
            )
            
        # 2. Channel-Independent Modern Depthwise Encoder
        self.d_model = bb_cfg.get("d_model", 64)
        self.eps_dim = mem_cfg.get("eps_dim", 16)
        self.encoder = CIModernEncoder(
            lookback=self.lookback,
            d_model=self.d_model,
            kernel_size=bb_cfg.get("kernel_size", 31),
            eps_dim=self.eps_dim,
            dropout=bb_cfg.get("dropout", 0.1)
        )
        
        # 3. Probabilistic Prediction Head
        self.pred_head = DualProbabilisticHead(
            d_model=self.d_model,
            horizon=self.horizon,
            min_logvar=pred_cfg.get("min_logvar", -8.0),
            max_logvar=pred_cfg.get("max_logvar", 6.0)
        )
        
        # 4. Two-Stage Predictive Quotienting Memory Bank
        self.memory_enabled = mem_cfg.get("enabled", False)
        if self.memory_enabled:
            self.memory = EquivalenceMemoryBank(
                max_prototypes=mem_cfg.get("max_prototypes", 512),
                eps_dim=self.eps_dim,
                horizon=self.horizon,
                shortlist_m=mem_cfg.get("shortlist_m", 3),
                metric=mem_cfg.get("metric", "wasserstein_normalized"),
                initial_threshold=mem_cfg.get("initial_threshold", 0.25),
                decay_half_life=mem_cfg.get("decay_half_life", 500.0)
            )
        else:
            self.memory = None

    def forward(
        self,
        x: torch.Tensor,
        current_step: float = 0.0,
        update_memory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        x: [B, L, D]
        Returns:
          mu_denorm: [B, H, D]
          logvar: [B, H, D]
          eps_keys: [B * D, eps_dim]
          aux_info: dictionary containing hit_rate, bypass_count, etc.
        """
        B, L, D = x.shape
        B_eff = B * D
        
        # 1. RevIN Normalization
        if self.use_revin:
            x_norm = self.revin(x, mode="norm")
        else:
            x_norm = x
            
        # Reshape to Channel-Independent representation: [B * D, 1, L]
        x_ci = x_norm.permute(0, 2, 1).contiguous().view(B_eff, 1, L)
        
        # 2. Early Feature Stage & EPS Key Generation
        h1, eps_keys = self.encoder.forward_early(x_ci)
        
        out_mu = torch.empty(B_eff, self.horizon, device=x.device)
        out_logvar = torch.empty(B_eff, self.horizon, device=x.device)
        hit_mask = torch.zeros(B_eff, dtype=torch.bool, device=x.device)
        
        # 3. Fast Vectorized Memory Screening & Bypass
        if self.memory_enabled and not self.training:
            hit_mask, bp_mu, bp_lv, matched_idx = self.memory.query_and_verify_batch(eps_keys, current_step)
            if hit_mask.any():
                out_mu[hit_mask] = bp_mu[hit_mask]
                out_logvar[hit_mask] = bp_lv[hit_mask]
                
        # 4. Compute deep trunk & head only for non-bypassed samples (misses)
        miss_mask = ~hit_mask
        if miss_mask.any():
            miss_h1 = h1[miss_mask]
            z_deep = self.encoder.forward_deep(miss_h1)
            sub_mu, sub_logvar = self.pred_head(z_deep)
            
            out_mu[miss_mask] = sub_mu
            out_logvar[miss_mask] = sub_logvar
            
            # Online memory update for misses during streaming
            if self.memory_enabled and update_memory:
                miss_indices = miss_mask.nonzero(as_tuple=True)[0]
                for i, orig_idx in enumerate(miss_indices):
                    self.memory.insert_or_update(
                        key=eps_keys[orig_idx],
                        mu=sub_mu[i].detach(),
                        logvar=sub_logvar[i].detach(),
                        current_step=current_step
                    )
                    
        # Reshape forecasts back to [B, H, D]
        mu_reshaped = out_mu.view(B, D, self.horizon).permute(0, 2, 1).contiguous()
        logvar_reshaped = out_logvar.view(B, D, self.horizon).permute(0, 2, 1).contiguous()
        
        # 5. RevIN Denormalization
        if self.use_revin:
            mu_denorm = self.revin(mu_reshaped, mode="denorm")
        else:
            mu_denorm = mu_reshaped
            
        bypassed_count = int(hit_mask.sum().item())
        aux_info = {
            "bypassed_count": bypassed_count,
            "total_tokens": B_eff,
            "bypass_ratio": bypassed_count / float(B_eff),
            "active_prototypes": self.memory.num_active_prototypes if self.memory_enabled else 0
        }
        
        return mu_denorm, logvar_reshaped, eps_keys, aux_info
