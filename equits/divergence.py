"""
Exact Analytical Divergence and Distance Metrics for Diagonal Gaussian Predictive Distributions.
Includes:
- Exact Analytical Gaussian KL Divergence
- Symmetrized KL Divergence: max(KL(P||Q), KL(Q||P))
- Mean Information Divergence Rate: KL / H
- Dimension-Normalized Root-Mean-Squared 2-Wasserstein Distance: W2_bar
"""

import torch


def exact_diagonal_gaussian_kl(
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Computes exact analytical KL divergence KL(P || Q) for diagonal Gaussians:
    P = Normal(mu_p, diag(exp(logvar_p)))
    Q = Normal(mu_q, diag(exp(logvar_q)))
    
    KL(P || Q) = 0.5 * sum( exp(logvar_p - logvar_q) + (mu_p - mu_q)^2 / exp(logvar_q) - 1 - (logvar_p - logvar_q) )
    """
    var_p = torch.exp(logvar_p)
    var_q = torch.exp(logvar_q)
    
    kl_elementwise = 0.5 * (
        (var_p + (mu_p - mu_q) ** 2) / (var_q + 1e-9)
        - 1.0
        - (logvar_p - logvar_q)
    )
    # Sum over the target horizon/dimension
    return torch.sum(kl_elementwise, dim=dim)


def symmetric_gaussian_kl(
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Candidate symmetric distance proposed in Section 3 of the brief:
    d(P, Q) = max( KL(P || Q), KL(Q || P) )
    """
    kl_pq = exact_diagonal_gaussian_kl(mu_p, logvar_p, mu_q, logvar_q, dim=dim)
    kl_qp = exact_diagonal_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p, dim=dim)
    return torch.maximum(kl_pq, kl_qp)


def mean_information_divergence_rate(
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Dimension-normalized KL divergence: KL / H.
    Prevents threshold explosion as horizon H increases.
    """
    kl = symmetric_gaussian_kl(mu_p, logvar_p, mu_q, logvar_q, dim=dim)
    horizon = mu_p.shape[dim]
    return kl / float(horizon)


def root_mean_squared_wasserstein_2(
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Root-Mean-Squared 2-Wasserstein Distance:
    W2^2(P, Q) = ||mu_p - mu_q||_2^2 + ||sigma_p - sigma_q||_2^2
    W2_bar = sqrt( W2^2 / H )
    
    Dimension-invariant, scale-interpretable in normalized std units, and immune to zero-division explosions.
    """
    sigma_p = torch.exp(0.5 * logvar_p)
    sigma_q = torch.exp(0.5 * logvar_q)
    
    sq_diff = (mu_p - mu_q) ** 2 + (sigma_p - sigma_q) ** 2
    mean_sq = torch.mean(sq_diff, dim=dim)
    return torch.sqrt(torch.clamp(mean_sq, min=0.0) + 1e-9)


def compute_prototype_distances(
    query_mu: torch.Tensor,
    query_logvar: torch.Tensor,
    proto_mu: torch.Tensor,
    proto_logvar: torch.Tensor,
    metric: str = "wasserstein_normalized"
) -> torch.Tensor:
    """
    Computes pairwise distance between query [B_eff, H] and prototypes [K, H].
    Returns distance matrix [B_eff, K].
    """
    # query: [B_eff, 1, H], proto: [1, K, H]
    q_mu = query_mu.unsqueeze(1)
    q_lv = query_logvar.unsqueeze(1)
    p_mu = proto_mu.unsqueeze(0)
    p_lv = proto_logvar.unsqueeze(0)
    
    if metric == "wasserstein_normalized":
        return root_mean_squared_wasserstein_2(q_mu, q_lv, p_mu, p_lv, dim=-1)
    elif metric == "kl_mean_rate":
        return mean_information_divergence_rate(q_mu, q_lv, p_mu, p_lv, dim=-1)
    elif metric == "kl_symmetric_sum":
        return symmetric_gaussian_kl(q_mu, q_lv, p_mu, p_lv, dim=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
