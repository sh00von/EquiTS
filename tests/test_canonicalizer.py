"""
Property tests for the canonicalizers: these check the statements made in
equits/canonicalizer.py (exact shift equivariance, bounded scale error, flat windows).
"""

import pytest
import torch

from equits import Canonicalizer


def _canon(mode, L=48, D=3, eps=0.1, seed=0):
    torch.manual_seed(seed)
    return Canonicalizer(L, D, mode=mode, eps=eps).eval()


@pytest.mark.parametrize("mode", ["relaxed", "learned"])
def test_shift_equivariance_is_exact(mode):
    c = _canon(mode)
    if mode == "learned":   # make the weights non-trivial
        for p in c.parameters():
            torch.nn.init.normal_(p, std=0.3)
    x = torch.randn(5, 48, 3)
    b = torch.randn(1, 1, 3) * 10
    m1, s1 = c(x)
    m2, s2 = c(x + b)
    assert torch.allclose(m2, m1 + b, atol=1e-4)
    assert torch.allclose(s2, s1, atol=1e-5)


@pytest.mark.parametrize("a", [0.01, 0.3, 2.0, 50.0])
def test_relaxed_scale_error_bound(a):
    c = _canon("relaxed")
    x = torch.randn(64, 48, 3) * torch.rand(64, 1, 1) * 3
    _, s1 = c(x)
    _, s2 = c(a * x)
    err = (s2 - a * s1).abs()
    bound = abs(a - 1) * c.eps * c.sigma0 + 1e-6
    assert (err <= bound).all()


def test_scale_equivariance_holds_asymptotically_for_learned():
    c = _canon("learned")
    for p in c.parameters():
        torch.nn.init.normal_(p, std=0.3)
    x = torch.randn(8, 48, 3) * 1000.0          # s >> eps * sigma0
    m1, s1 = c(x)
    m2, s2 = c(3.0 * x)
    assert torch.allclose(m2, 3.0 * m1, rtol=1e-3, atol=1e-2)
    assert torch.allclose(s2, 3.0 * s1, rtol=1e-3)


@pytest.mark.parametrize("mode", ["relaxed", "learned"])
def test_flat_windows_are_safe(mode):
    c = _canon(mode)
    x = torch.full((4, 48, 3), 7.0)
    x[1] += 1e-7 * torch.randn(48, 3)            # nearly flat
    m, s = c(x)
    assert (s >= c.eps * c.sigma0 - 1e-7).all()
    z = (x - m) / s
    assert torch.isfinite(z).all() and z.abs().max() < 1.0


def test_revin_collapses_on_flat_windows():
    """Documents the failure mode the relaxation fixes."""
    c = _canon("revin")
    x = torch.full((2, 48, 3), 7.0)
    _, s = c(x)
    assert (s < 1e-2).all()


def test_learned_starts_exactly_at_relaxed():
    x = torch.randn(4, 48, 3)
    m_r, s_r = _canon("relaxed")(x)
    m_l, s_l = _canon("learned")(x)
    assert torch.allclose(m_r, m_l, atol=1e-6)
    assert torch.allclose(s_r, s_l, atol=1e-6)
