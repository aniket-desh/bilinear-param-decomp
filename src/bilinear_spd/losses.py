"""Loss terms for the gated rank-one decomposition.

Four terms (spec § 6.2):

  L_behavior      KL( softmax(y_target) || softmax(y_hat) )
  L_param_A/B     relative Frobenius reconstruction of A or B from its atoms
  L_sparse        L_p on gates (push individual gates toward 0)
  L_freq          per-component frequency penalty (push usage toward 0)

`kl_logits` and `frequency_penalty` are cherry-picked from Goodfire's
nano_param_decomp/run.py § E (with attribution). The freq penalty drops
the `world_size` multiplier since we're single-process.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --- cherry-picked from Goodfire nano_param_decomp/run.py § E ---


def kl_logits(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL( softmax(target) || softmax(pred) ).

    `target` is detached: this is the reconstruction objective, not a
    distillation-style soft-target gradient on the target.
    """
    log_q = F.log_softmax(pred, dim=-1)
    p = F.softmax(target.detach(), dim=-1)
    kl_per_row = F.kl_div(log_q, p, reduction="none").sum(dim=-1)
    return kl_per_row.mean()


def frequency_penalty(
    g: torch.Tensor,
    beta: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-component mean + beta * mean * log2(1 + sum).

    Discourages a component from being weakly active everywhere (the additive
    `mean` term) and *also* heavily active on many examples (the multiplicative
    `log2(1+sum)` term). Sum dominates for components used by many examples;
    mean dominates for components used very lightly.

    Equivalent in form to nano's `importance_minimality_loss` with
    `world_size = 1` and `p = 1` (we apply L_p separately via `sparsity_lp`).
    """
    batch_dims = tuple(range(g.ndim - 1))
    sum_c = g.sum(dim=batch_dims)             # [C]
    n = max(1, int(g.shape[:-1].numel()))
    mean_c = sum_c / n
    return (mean_c + beta * mean_c * torch.log2(1.0 + sum_c + eps)).sum()


# --- our own ---


def relative_param_recon(target: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """||target − sum_c U[c,:] V[c,:]^T||_F² / ||target||_F².

    U: [C, d_in], V: [C, d_hidden], target: [d_in, d_hidden].
    Bounded in roughly [0, 1] at random init; small means atoms reconstruct target well.
    """
    reconstructed = torch.einsum("cd,ch->dh", U, V)
    delta = target - reconstructed
    return delta.pow(2).sum() / (target.pow(2).sum() + 1e-12)


def sparsity_lp(g: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """E_x [ sum_c |g_c(x)|^p ]: gate-norm penalty.

    For p = 1 this is the standard L1 sparsity. For p < 1 it more aggressively
    pushes individual gates toward 0 (concave penalty), at the cost of training
    stability.
    """
    return g.abs().pow(p).sum(dim=-1).mean()


def gate_l0(g: torch.Tensor, threshold: float = 1e-3) -> float:
    """Mean #components with gate > threshold across the batch."""
    return float((g > threshold).float().sum(dim=-1).mean().item())
