"""
PatchLinear: Channel-Independent Patch-based Linear Forecaster.
Captures sub-series patch inductive biases inspired by PatchTST (Nie et al., ICLR 2023).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, lookback: int = 96, horizon: int = 96, num_channels: int = 7, patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.num_channels = num_channels
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate number of patches
        self.num_patches = (lookback - patch_len) // stride + 1
        
        # Linear projection from flattened patch tokens to target horizon H
        self.patch_proj = nn.Linear(self.num_patches * patch_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B * D, L]
        B, L, D = x.shape
        x_ci = x.permute(0, 2, 1).contiguous().view(B * D, L)
        
        # Unfold into patches: [B * D, num_patches, patch_len]
        patches = x_ci.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        flat = patches.contiguous().view(B * D, -1)
        
        out = self.patch_proj(flat) # [B * D, H]
        out = out.view(B, D, self.horizon).permute(0, 2, 1).contiguous() # [B, H, D]
        return out

