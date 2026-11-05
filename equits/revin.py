"""
Reversible Instance Normalization (RevIN) for Non-Stationary Time Series.
Normalizes mean and variance per instance and inverts the transformation at the output.
Reference: Kim et al., ICLR 2022.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        num_features: number of variables D
        affine: learnable scale and shift
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
            
        self.mean: torch.Tensor = None
        self.stdev: torch.Tensor = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        x: [B, L, D] in 'norm' mode, or [B, H, D] in 'denorm' mode
        mode: 'norm' or 'denorm'
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")
        return x

    def _get_statistics(self, x: torch.Tensor):
        # Compute mean and standard deviation along the temporal dimension L (dim=1)
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
