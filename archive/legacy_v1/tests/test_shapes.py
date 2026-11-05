"""
Unit tests for EquiTS model tensor shapes and RevIN statistics.
"""

import pytest
import yaml
import torch
from equits.model import EquiTS
from equits.revin import RevIN


@pytest.fixture
def base_config():
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_equits_forward_shapes(base_config):
    B = 16
    L = base_config["dataset"]["lookback"]
    H = base_config["dataset"]["horizon"]
    D = base_config["dataset"]["num_channels"]
    eps_dim = base_config["memory"]["eps_dim"]
    
    model = EquiTS(base_config)
    x = torch.randn(B, L, D)
    
    model.eval()
    mu, logvar, eps_keys, aux = model(x)
    
    assert mu.shape == (B, H, D), f"Expected mu shape {(B, H, D)}, got {mu.shape}"
    assert logvar.shape == (B, H, D), f"Expected logvar shape {(B, H, D)}, got {logvar.shape}"
    assert eps_keys.shape == (B * D, eps_dim), f"Expected eps_keys shape {(B * D, eps_dim)}, got {eps_keys.shape}"
    
    # Check that eps_keys are on unit sphere S^{d-1}
    norms = torch.norm(eps_keys, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "EPS keys are not unit-normalized!"


def test_revin_identity():
    B, L, D = 8, 96, 7
    revin = RevIN(num_features=D, affine=True)
    x = torch.randn(B, L, D) * 5.0 + 12.0
    
    norm_x = revin(x, mode="norm")
    # Means across L should be ~0, stds ~1
    assert torch.allclose(norm_x.mean(dim=1), torch.zeros(B, 1, D), atol=1e-5)
    
    recon_x = revin(norm_x, mode="denorm")
    assert torch.allclose(recon_x, x, atol=1e-5), "RevIN denorm failed to reconstruct original values!"

