"""
Tests for conformal calibration, online risk control and the RevIN variance mapping.
"""

import torch
from equits.certification import split_crc_threshold, OnlineAdaptiveCRC, false_merge_indicator
from equits.revin import RevIN


def test_split_crc_respects_risk_bound():
    torch.manual_seed(0)
    n = 2000
    d = torch.rand(n)
    e = (torch.rand(n) < d).float()          # false merges more likely for distant keys
    alpha = 0.05
    lam = split_crc_threshold(d, e, alpha)
    risk = ((d <= lam).float() * e).sum().item()
    assert (risk + 1) / (n + 1) <= alpha + 1e-9
    assert lam > 0.0


def test_split_crc_disables_bypass_when_impossible():
    d = torch.zeros(10)
    e = torch.ones(10)
    assert split_crc_threshold(d, e, 0.05) == 0.0


def test_online_crc_moves_in_right_direction():
    crc = OnlineAdaptiveCRC(target_risk_alpha=0.1, eta=0.01, initial_threshold=0.5)
    up = crc.update(num_decisions=10, weighted_false_merges=0.0)
    assert up > 0.5
    down = crc.update(num_decisions=10, weighted_false_merges=10.0)
    assert down < up


def test_false_merge_indicator():
    y = torch.zeros(2, 4)
    full = torch.zeros(2, 4)
    bypass = torch.tensor([[0.05] * 4, [1.0] * 4])
    assert false_merge_indicator(bypass, full, y, 0.1).tolist() == [0.0, 1.0]


def test_revin_logvar_maps_variance_to_data_scale():
    torch.manual_seed(0)
    B, L, H, D = 4, 64, 16, 3
    revin = RevIN(num_features=D, affine=True)
    with torch.no_grad():
        revin.affine_weight.fill_(2.0)
    x = torch.randn(B, L, D) * 7.0 + 3.0
    revin(x, mode="norm")
    lv = revin.denorm_logvar(torch.zeros(B, H, D))
    expected = (revin.stdev / (2.0 + revin.eps ** 2)) ** 2
    assert torch.allclose(torch.exp(lv), expected.expand(B, H, D), rtol=1e-4)
