"""
Channel-Independent Large-Kernel Depthwise 1D Convolutional Encoder with Early Predictive Signature (EPS) Tap.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LargeKernelDepthwiseBlock(nn.Module):
    """
    Channel-Independent Depthwise 1D Conv block.
    Covers large temporal context without dilation gaps.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv_dw = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.norm = nn.BatchNorm1d(in_channels)
        self.act = nn.GELU()
        self.conv_pw = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_eff, C, L]
        res = self.residual(x)
        out = self.conv_dw(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv_pw(out)
        out = self.dropout(out)
        return out + res


class CIModernEncoder(nn.Module):
    """
    Channel-Independent Temporal Feature Encoder.
    Processes each variable channel independently through shared weights:
    Input: [B, L, D] -> Reshaped to [B * D, 1, L].
    Outputs:
      - Early Predictive Signature (EPS) key k_i: [B * D, eps_dim] (unit-norm)
      - Deep Context Representation z: [B * D, d_model]
    """
    def __init__(
        self,
        lookback: int = 96,
        d_model: int = 64,
        kernel_size: int = 31,
        eps_dim: int = 16,
        dropout: float = 0.1,
        eps_detach_backbone: bool = True
    ):
        super().__init__()
        self.lookback = lookback
        self.d_model = d_model
        self.eps_dim = eps_dim
        # If True, the EPS alignment loss only trains the EPS projection, so the
        # forecasting backbone (and the no-memory ablation) is unaffected by it.
        self.eps_detach_backbone = eps_detach_backbone
        
        # Layer 1: Shallow Temporal Feature Extractor
        self.layer1 = LargeKernelDepthwiseBlock(in_channels=1, out_channels=d_model, kernel_size=kernel_size, dropout=dropout)
        
        # Early Predictive Signature (EPS) projection: taps Layer 1 representation
        # Projects pooled Layer 1 activation to compact hyperspherical key R^d
        self.eps_pool = nn.AdaptiveAvgPool1d(1)
        self.eps_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, eps_dim)
        )
        
        # Deep Trunk: Layer 2
        self.layer2 = LargeKernelDepthwiseBlock(in_channels=d_model, out_channels=d_model, kernel_size=15, dropout=dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.out_norm = nn.LayerNorm(d_model)

    def forward_early(self, x_ci: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes only the shallow early stage.
        x_ci: [B * D, 1, L]
        Returns:
          h1: Layer 1 activations [B * D, d_model, L]
          k: Early Predictive Signature key [B * D, eps_dim] (unit-normalized)
        """
        h1 = self.layer1(x_ci)
        pooled = self.eps_pool(h1).squeeze(-1) # [B * D, d_model]
        if self.eps_detach_backbone:
            pooled = pooled.detach()
        k = F.normalize(self.eps_proj(pooled), p=2, dim=-1) # Unit-norm key in S^{d-1}
        return h1, k

    def forward_deep(self, h1: torch.Tensor) -> torch.Tensor:
        """
        Executes deep trunk from h1 activations.
        h1: [B * D, d_model, L]
        Returns:
          z: Deep context representation [B * D, d_model]
        """
        h2 = self.layer2(h1)
        z = self.global_pool(h2).squeeze(-1) # [B * D, d_model]
        z = self.out_norm(z)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full encoder execution:
        x: [B, L, D]
        Returns:
          z: [B * D, d_model]
          k: [B * D, eps_dim]
        """
        B, L, D = x.shape
        # Reshape to Channel-Independent representation [B * D, 1, L]
        x_ci = x.permute(0, 2, 1).contiguous().view(B * D, 1, L)
        
        h1, k = self.forward_early(x_ci)
        z = self.forward_deep(h1)
        return z, k
