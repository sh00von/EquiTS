"""
Tests for the EquiTS layer: shapes, decoupling of mean and variance paths, and
equivariance of the predictive distribution.
"""

import pytest
import torch
import torch.nn as nn

from equits import EquiTS, gaussian_nll


class LinearBackbone(nn.Module):
    def __init__(self, L, H):
        super().__init__()
        self.proj = nn.Linear(L, H)

    def forward(self, z):
        return self.proj(z.permute(0, 2, 1)).permute(0, 2, 1)


def _setup(normalizer="learned", shift_term=True, seed=0):
    torch.manual_seed(seed)
    L, H, D = 48, 24, 3
    return EquiTS(L, H, D, normalizer=normalizer, shift_term=shift_term), LinearBackbone(L, H)


@pytest.mark.parametrize("normalizer", ["revin", "relaxed", "learned"])
@pytest.mark.parametrize("shift_term", [True, False])
def test_shapes_and_finiteness(normalizer, shift_term):
    layer, bb = _setup(normalizer, shift_term)
    mu, lv = layer(torch.randn(5, 48, 3), bb)
    assert mu.shape == lv.shape == (5, 24, 3)
    assert torch.isfinite(mu).all() and torch.isfinite(lv).all()


def test_nll_does_not_touch_mean_path():
    layer, bb = _setup("learned")
    for p in layer.canonicalizer.parameters():
        torch.nn.init.normal_(p, std=0.3)
    x, y = torch.randn(5, 48, 3), torch.randn(5, 24, 3)
    mu, lv = layer(x, bb)
    gaussian_nll(mu.detach(), lv, y).backward()
    for p in list(bb.parameters()) + list(layer.canonicalizer.parameters()):
        assert p.grad is None or torch.all(p.grad == 0)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.scale_head.parameters())


def test_mean_does_not_depend_on_scale_head():
    layer, bb = _setup("learned")
    x = torch.randn(5, 48, 3)
    mu1, _ = layer(x, bb)
    for p in layer.scale_head.parameters():
        torch.nn.init.normal_(p, std=1.0)
    mu2, _ = layer(x, bb)
    assert torch.equal(mu1, mu2)


def test_predictive_distribution_is_equivariant():
    layer, bb = _setup("learned")
    layer.eval()
    x = torch.randn(4, 48, 3) * 1000.0            # large scale: relaxation negligible
    a, b = 2.5, torch.tensor([3.0, -7.0, 100.0]).view(1, 1, 3)
    mu1, lv1 = layer(x, bb)
    mu2, lv2 = layer(a * x + b, bb)
    assert torch.allclose(mu2, a * mu1 + b, rtol=1e-3, atol=1e-1)
    assert torch.allclose(lv2, lv1 + 2 * torch.log(torch.tensor(a)), atol=1e-3)


def test_flat_input_gives_bounded_variance():
    layer, bb = _setup("relaxed")
    mu, lv = layer(torch.full((2, 48, 3), 5.0), bb)
    assert torch.isfinite(lv).all()
    assert (lv >= 2 * torch.log(torch.tensor(layer.canonicalizer.eps)) - 8.0 - 1e-4).all()
