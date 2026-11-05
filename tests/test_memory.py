"""
Unit tests for EquiTS Two-Stage Memory Bank, Welford updates, and IUDI eviction.
"""

import pytest
import torch
import torch.nn.functional as F
from equits.memory import EquivalenceMemoryBank


def test_memory_insert_and_bypass_hit():
    K = 10
    eps_dim = 16
    H = 96
    memory = EquivalenceMemoryBank(
        max_prototypes=K,
        eps_dim=eps_dim,
        horizon=H,
        shortlist_m=3,
        initial_threshold=0.30
    )
    
    assert memory.num_active_prototypes == 0
    
    # Create key and forecast
    key = F.normalize(torch.randn(eps_dim), p=2, dim=-1)
    mu = torch.randn(H)
    logvar = torch.full((H,), -1.0)
    
    # 1. First query should be a miss
    hit, idx, c_mu, c_lv = memory.query_and_verify(key, current_step=1.0)
    assert not hit, "First query should miss on empty memory"
    
    # 2. Insert prototype
    memory.insert_or_update(key, mu, logvar, current_step=1.0)
    assert memory.num_active_prototypes == 1
    
    # 3. Query with identical key should hit and bypass
    hit2, idx2, c_mu2, c_lv2 = memory.query_and_verify(key, current_step=2.0)
    assert hit2, "Query with identical key should hit"
    assert idx2 == 0
    assert torch.allclose(c_mu2, mu, atol=1e-5)


def test_memory_budget_and_iudi_eviction():
    K = 4 # Small budget for testing eviction
    eps_dim = 16
    H = 96
    memory = EquivalenceMemoryBank(
        max_prototypes=K,
        eps_dim=eps_dim,
        horizon=H,
        shortlist_m=2,
        initial_threshold=0.01 # strict threshold to force new prototype allocations
    )
    
    # Insert K distinct prototypes
    for i in range(K):
        key = F.normalize(torch.randn(eps_dim), p=2, dim=-1)
        mu = torch.randn(H) * 10.0
        logvar = torch.zeros(H)
        memory.insert_or_update(key, mu, logvar, current_step=float(i))
        
    assert memory.num_active_prototypes == K
    
    # Insert K+1th prototype, should trigger IUDI eviction without expanding past K
    key_new = F.normalize(torch.randn(eps_dim), p=2, dim=-1)
    mu_new = torch.randn(H) * 10.0
    logvar_new = torch.zeros(H)
    memory.insert_or_update(key_new, mu_new, logvar_new, current_step=10.0)
    
    assert memory.num_active_prototypes == K, "Memory expanded beyond max_prototypes budget!"
    assert memory.total_evictions >= 1, "Eviction was not recorded"

