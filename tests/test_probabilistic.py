"""
Tests for probabilistic metrics, conformal intervals and the DLinear-Gauss baseline.
"""

import torch
from equits.probabilistic import (
    gaussian_crps, interval_metrics, conformal_quantile, conformal_interval, normal_z
)
from baselines import DLinearGauss


def test_gaussian_crps_matches_monte_carlo():
    torch.manual_seed(0)
    mu, sigma, y = torch.tensor([[[0.3]]]), torch.tensor([[[1.7]]]), torch.tensor([[[1.2]]])
    closed = gaussian_crps(mu, 2 * torch.log(sigma), y)
    x = mu + sigma * torch.randn(400000, 1, 1)
    x2 = mu + sigma * torch.randn(400000, 1, 1)
    mc = ((x - y).abs().mean() - 0.5 * (x - x2).abs().mean()).item()
    assert abs(closed - mc) < 0.01


def test_interval_metrics_penalizes_misses():
    y = torch.tensor([0.0, 2.0])
    m = interval_metrics(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), y, alpha=0.1)
    assert m["coverage"] == 0.5
    assert abs(m["width"] - 2.0) < 1e-6
    assert abs(m["interval_score"] - (2.0 + 0.5 * 20.0 * 1.0)) < 1e-5


def test_conformal_coverage_iid():
    torch.manual_seed(0)
    n, H, D = 2000, 4, 2
    y_cal, y_test = torch.randn(n, H, D) * 2, torch.randn(n, H, D) * 2
    zeros = torch.zeros(n, H, D)
    lo, hi = conformal_interval(zeros, y_cal, zeros, alpha=0.1)
    cov = interval_metrics(lo, hi, y_test, 0.1)["coverage"]
    assert 0.88 < cov < 0.92


def test_normalized_conformal_is_adaptive():
    torch.manual_seed(0)
    n = 4000
    sigma = torch.where(torch.rand(n, 1, 1) < 0.5, 0.5, 3.0)
    y_cal = sigma * torch.randn(n, 1, 1)
    y_test = sigma * torch.randn(n, 1, 1)
    zeros = torch.zeros(n, 1, 1)
    lo_a, hi_a = conformal_interval(zeros, y_cal, zeros, alpha=0.1)
    lo_n, hi_n = conformal_interval(zeros, y_cal, zeros, sigma_cal=sigma, sigma_test=sigma, alpha=0.1)
    m_abs = interval_metrics(lo_a, hi_a, y_test, 0.1)
    m_norm = interval_metrics(lo_n, hi_n, y_test, 0.1)
    assert 0.87 < m_norm["coverage"] < 0.93
    assert m_norm["interval_score"] < m_abs["interval_score"]


def test_conformal_quantile_infinite_when_too_few_points():
    assert torch.isinf(conformal_quantile(torch.rand(5, 1, 1), alpha=0.1)).all()


def test_normal_z():
    assert abs(normal_z(0.1) - 1.6449) < 1e-3


def test_dlinear_gauss_shapes():
    m = DLinearGauss(lookback=96, horizon=24, num_channels=3)
    mu, lv = m(torch.randn(5, 96, 3))
    assert mu.shape == lv.shape == (5, 24, 3)
    assert torch.isfinite(lv).all()
