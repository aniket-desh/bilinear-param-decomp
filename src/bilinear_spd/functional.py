"""Probe-conditioned interaction matrix M_r and friends.

Given a bilinear MLP with weights A, B, C and a probe r in output space,
the scalar r^T y(x) equals x^T M_r x exactly, where

    M_r = 0.5 * sum_i (Cr)_i * ( a_i b_i^T + b_i a_i^T )

with a_i = A[:, i], b_i = B[:, i].

Symmetric M_r => real eigenvalues. Degenerate eigenvalues are merged into
projectors P_lambda = sum_{j in group} q_j q_j^T so that downstream alignment
metrics compare to invariants rather than to arbitrary rotations within a
degenerate eigenspace.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def build_probes(
    probe_cfg: dict,
    p: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Materialize the probes declared in a config block.

    Recognized keys:
      type: "centered_class" or "pairwise"
      classes: list of class indices for centered_class probes
      include_pairwise: list of [k, k'] pairs (always added regardless of `type`)
    """
    out: dict[str, torch.Tensor] = {}
    if probe_cfg.get("type") == "centered_class":
        for k in probe_cfg.get("classes", []):
            out[f"centered_{k}"] = centered_class_probe(k, p, device=device)
    for pair in probe_cfg.get("include_pairwise", []):
        k, k_prime = int(pair[0]), int(pair[1])
        out[f"pairwise_{k}_{k_prime}"] = pairwise_probe(k, k_prime, p, device=device)
    return out


def eigendecompose_M(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigendecomposition sorted by |lambda| descending.

    Returns
    -------
    evals : [d]      real eigenvalues, sorted so |evals[0]| is largest
    evecs : [d, d]   columns are corresponding eigenvectors
    """
    evals, evecs = torch.linalg.eigh(M)
    idx = torch.argsort(evals.abs(), descending=True)
    return evals[idx], evecs[:, idx]


@dataclass
class EigenGroup:
    indices: list[int]              # positions in the |lambda|-sorted eigvec matrix
    evals: torch.Tensor             # [rank] signed eigenvalues
    projector: torch.Tensor         # [d, d]  P = Q Q^T over this group
    rank: int
    mean_abs_eval: float
    mean_eval: float


def group_eigenspaces(
    evals: torch.Tensor,
    evecs: torch.Tensor,
    rel_tol: float = 1e-3,
    min_abs_eval: float = 1e-6,
) -> list[EigenGroup]:
    """Merge eigenvalues into degenerate eigenspaces.

    Two eigenvalues belong to the same group iff `|lambda_i - lambda_j| <= rel_tol * max_k |lambda_k|`.
    The grouping is *signed*: +lambda and -lambda are different eigenspaces of M_r,
    even if their magnitudes coincide.

    Eigenvalues with `|lambda| < min_abs_eval` are dropped — they contribute nothing
    above the floating-point floor.

    Returned groups are sorted by mean |lambda| descending.
    """
    d = evals.shape[0]
    max_abs = float(evals.abs().max().item()) if d > 0 else 0.0
    tol = rel_tol * max_abs

    keep = (evals.abs() >= min_abs_eval).nonzero().flatten().tolist()
    if not keep:
        return []

    # walk through kept indices ordered by signed value, group consecutive within tol
    keep_sorted_by_value = sorted(keep, key=lambda i: evals[i].item())
    groups_idx: list[list[int]] = [[keep_sorted_by_value[0]]]
    for k in range(1, len(keep_sorted_by_value)):
        i_prev = groups_idx[-1][-1]
        i_curr = keep_sorted_by_value[k]
        if abs(evals[i_curr].item() - evals[i_prev].item()) <= tol:
            groups_idx[-1].append(i_curr)
        else:
            groups_idx.append([i_curr])

    out: list[EigenGroup] = []
    for indices in groups_idx:
        Q = evecs[:, indices]
        P = Q @ Q.T
        ev = evals[indices]
        out.append(
            EigenGroup(
                indices=indices,
                evals=ev,
                projector=P,
                rank=len(indices),
                mean_abs_eval=float(ev.abs().mean().item()),
                mean_eval=float(ev.mean().item()),
            )
        )

    out.sort(key=lambda g: -g.mean_abs_eval)
    return out
