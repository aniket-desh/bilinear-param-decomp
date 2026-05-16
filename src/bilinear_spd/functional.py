"""Probe-conditioned interaction matrix M_r and friends.

Given a bilinear MLP with weights A, B, C and a probe r in output space,
the scalar r^T y(x) equals x^T M_r x exactly, where

    M_r = 0.5 * sum_i (Cr)_i * ( a_i b_i^T + b_i a_i^T )

with a_i = A[:, i], b_i = B[:, i].

Day 2 will add eigendecomposition + projector grouping; today we only need
the matrix itself plus the equivalence check.
"""

from __future__ import annotations

import torch


def construct_M_r(
    A: torch.Tensor,  # [d, m]
    B: torch.Tensor,  # [d, m]
    C: torch.Tensor,  # [m, p]
    r: torch.Tensor,  # [p]
) -> torch.Tensor:  # [d, d]
    alpha = C @ r  # [m]
    raw = torch.einsum("m,dm,em->de", alpha, A, B)
    return 0.5 * (raw + raw.T)


def functional_scalar(y: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """r^T y for a batch of logits.

    y : [batch, p]   r : [p]   -> [batch]
    """
    return y @ r


def quadratic_scalar(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """x^T M x for a batch of inputs.

    x : [batch, d]   M : [d, d]   -> [batch]
    """
    return torch.einsum("bd,de,be->b", x, M, x)


def centered_class_probe(k: int, p: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """r_k = e_k - (1/p) * 1."""
    r = -torch.ones(p, device=device) / p
    r[k] += 1.0
    return r


def pairwise_probe(k: int, k_prime: int, p: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """r_{k, k'} = e_k - e_{k'}."""
    r = torch.zeros(p, device=device)
    r[k] = 1.0
    r[k_prime] = -1.0
    return r
