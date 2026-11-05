"""
Unit tests for exact Gaussian KL divergence and Wasserstein distance metrics.
"""

import pytest
import torch
from equits.divergence import (
    exact_diagonal_gaussian_kl,
    symmetric_gaussian_kl,
    mean_information_divergence_rate,
    root_mean_squared_wasserstein_2
)


def test_kl_identity_and_nonnegativity():
    H = 96
    mu = torch.randn(10, H)
    logvar = torch.randn(10, H)
    
    # KL(P || P) must be 0
    kl_self = exact_diagonal_gaussian_kl(mu, logvar, mu, logvar)
    assert torch.allclose(kl_self, torch.zeros_like(kl_self), atol=1e-6), "KL(P||P) is not zero"
    
    # Random P and Q
    mu_q = torch.randn(10, H)
    logvar_q = torch.randn(10, H)
    kl_pq = exact_diagonal_gaussian_kl(mu, logvar, mu_q, logvar_q)
    assert (kl_pq >= -1e-6).all(), "KL divergence produced negative values"


def test_symmetry():
    H = 96
    p_mu, p_lv = torch.randn(5, H), torch.randn(5, H)
    q_mu, q_lv = torch.randn(5, H), torch.randn(5, H)
    
    sym_kl_1 = symmetric_gaussian_kl(p_mu, p_lv, q_mu, q_lv)
    sym_kl_2 = symmetric_gaussian_kl(q_mu, q_lv, p_mu, p_lv)
    assert torch.allclose(sym_kl_1, sym_kl_2), "Symmetric KL is not symmetric"
    
    w2_1 = root_mean_squared_wasserstein_2(p_mu, p_lv, q_mu, q_lv)
    w2_2 = root_mean_squared_wasserstein_2(q_mu, q_lv, p_mu, p_lv)
    assert torch.allclose(w2_1, w2_2), "Wasserstein distance is not symmetric"


def test_hand_calculated_1d_case():
    # P = N(0, 1), Q = N(1, 4)
    # var_p = 1, var_q = 4, logvar_p = 0, logvar_q = ln(4) = 1.386294
    # KL(P || Q) = 0.5 * ( (1 + 1) / 4 - 1 - (0 - ln(4)) ) = 0.5 * ( 0.5 - 1 + 1.386294 ) = 0.5 * 0.886294 = 0.443147
    mu_p = torch.tensor([[0.0]])
    lv_p = torch.tensor([[0.0]])
    mu_q = torch.tensor([[1.0]])
    lv_q = torch.tensor([[torch.log(torch.tensor(4.0))]])
    
    kl = exact_diagonal_gaussian_kl(mu_p, lv_p, mu_q, lv_q)
    assert torch.isclose(kl, torch.tensor([0.443147]), atol=1e-4)


def test_numerical_stability_extreme_values():
    # Clamped bounds: logvar in [-8, 6]
    mu_p = torch.randn(4, 96) * 10.0
    lv_p = torch.full((4, 96), -8.0) # very small variance
    
    mu_q = torch.randn(4, 96) * 10.0
    lv_q = torch.full((4, 96), 6.0)  # large variance
    
    w2 = root_mean_squared_wasserstein_2(mu_p, lv_p, mu_q, lv_q)
    assert not torch.isnan(w2).any() and not torch.isinf(w2).any(), "Wasserstein produced NaN or Inf"
    
    kl_rate = mean_information_divergence_rate(mu_p, lv_p, mu_q, lv_q)
    assert not torch.isnan(kl_rate).any() and not torch.isinf(kl_rate).any(), "KL rate produced NaN or Inf"
