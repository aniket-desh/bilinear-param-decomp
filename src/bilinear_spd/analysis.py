"""Atom and cluster perturbations $\\Delta M_r$ vs. probe-conditioned eigenspaces.

For each atom $c \\in \\{1, \\dots, C_A\\}$ in the decomposition's A-side, the
rank-one contribution is $\\Delta A_c = u^A_c (v^A_c)^\\top$ with $\\Delta B_c = 0$;
symmetrically for B-side atoms. The induced change in the probe-conditioned
interaction matrix is

    $\\Delta M_r(\\{c\\}) = M_r(A, B, C) - M_r(A - \\Delta A_c, B - \\Delta B_c, C).$

For a cluster $S$ that bundles multiple atoms (possibly across both A and B),
we sum the rank-ones first and recompute $M_r$ on the perturbed weights — this
keeps the bilinear cross-term right (spec § 5.6).

A useful sanity identity (spec § 5.6 / theory § 7.2) is the perturbation
expansion: for any $\\Delta A_S, \\Delta B_S$,

    $\\Delta M_r(S) = M_r(\\Delta A_S, B, C) + M_r(A, \\Delta B_S, C) - M_r(\\Delta A_S, \\Delta B_S, C).$

We test this identity in `tests/test_perturbation_expansion.py`.

Outputs are stored as `[n_atoms_or_clusters, d_in, d_in]` tensors of $\\Delta M_r$,
keyed by probe name when there are multiple probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .decomposition import BilinearComponentMLP
from .functional import construct_M_r


@dataclass
class AtomDeltas:
    """ΔM_r for every atom on a single probe.

    Convention: rows 0..C_A−1 are A-side atoms, rows C_A..C_A+C_B−1 are B-side.
    """

    delta_M: torch.Tensor      # [C_A + C_B, d_in, d_in]
    norms: torch.Tensor        # [C_A + C_B]   ||ΔM||_F
    a_or_b: torch.Tensor       # [C_A + C_B]   0 for A-atom, 1 for B-atom


@torch.no_grad()
def atom_perturbations(decomp: BilinearComponentMLP, r: torch.Tensor) -> AtomDeltas:
    """Compute $\\Delta M_r(\\{c\\})$ for every atom $c$ in A and B.

    Returns a struct holding the stacked [C_A + C_B, d, d] tensor plus
    Frobenius norms. The full M_r is computed once; per-atom deltas are
    built by subtracting one rank-one update at a time.

    Cost: $O((C_A + C_B) \\cdot d^2 \\cdot m_\\text{eff})$ with $m_\\text{eff} = 1$
    per atom — a small matmul per atom plus a symmetrize. We do this in pure
    PyTorch (no batched einsum across atoms) because $C \\sim O(2 d_\\text{hidden})$
    is small here and clarity beats vectorization.
    """
    A = decomp.A_target
    B = decomp.B_target
    C = decomp.C_target

    M_full = construct_M_r(A, B, C, r)
    d = M_full.shape[0]
    C_A = decomp.C_A
    C_B = decomp.C_B

    out = torch.zeros(C_A + C_B, d, d, device=A.device, dtype=A.dtype)
    norms = torch.zeros(C_A + C_B, device=A.device, dtype=A.dtype)
    a_or_b = torch.zeros(C_A + C_B, dtype=torch.int8)

    U_A = decomp.U_A         # [C_A, d_in]
    V_A = decomp.V_A         # [C_A, d_hidden]
    U_B = decomp.U_B
    V_B = decomp.V_B

    for c in range(C_A):
        dA = torch.outer(U_A[c], V_A[c])           # [d_in, d_hidden]
        M_minus = construct_M_r(A - dA, B, C, r)
        delta = M_full - M_minus
        out[c] = delta
        norms[c] = torch.linalg.matrix_norm(delta, "fro")
        a_or_b[c] = 0

    for c in range(C_B):
        dB = torch.outer(U_B[c], V_B[c])
        M_minus = construct_M_r(A, B - dB, C, r)
        delta = M_full - M_minus
        out[C_A + c] = delta
        norms[C_A + c] = torch.linalg.matrix_norm(delta, "fro")
        a_or_b[C_A + c] = 1

    return AtomDeltas(delta_M=out, norms=norms, a_or_b=a_or_b)


@torch.no_grad()
def cluster_perturbations(
    decomp: BilinearComponentMLP,
    r: torch.Tensor,
    clusters: list[list[int]],
) -> torch.Tensor:
    """Compute $\\Delta M_r(S)$ for each cluster $S$.

    `clusters` is a list of clusters; each cluster is a list of *flat* atom
    indices in the [0..C_A + C_B) convention used by `atom_perturbations`.
    A-atom and B-atom contributions are summed *into the weights* before
    constructing $M_r$ so the bilinear cross-term ($\\Delta A_S \\Delta B_S$) is
    handled correctly.

    Returns `[n_clusters, d, d]` (Frobenius norms can be computed by the
    caller if needed).
    """
    A = decomp.A_target
    B = decomp.B_target
    C = decomp.C_target

    M_full = construct_M_r(A, B, C, r)
    d = M_full.shape[0]
    C_A = decomp.C_A

    U_A = decomp.U_A
    V_A = decomp.V_A
    U_B = decomp.U_B
    V_B = decomp.V_B

    out = torch.zeros(len(clusters), d, d, device=A.device, dtype=A.dtype)
    for ci, S in enumerate(clusters):
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)
        for atom_idx in S:
            if atom_idx < C_A:
                c = atom_idx
                dA = dA + torch.outer(U_A[c], V_A[c])
            else:
                c = atom_idx - C_A
                dB = dB + torch.outer(U_B[c], V_B[c])
        M_minus = construct_M_r(A - dA, B - dB, C, r)
        out[ci] = M_full - M_minus

    return out


@torch.no_grad()
def collect_gates(decomp: BilinearComponentMLP, x: torch.Tensor) -> torch.Tensor:
    """Concatenated deterministic gates $[g_A \\,|\\, g_B] \\in \\mathbb{R}^{N \\times (C_A + C_B)}$.

    Determined (not stochastic) so clusters are stable across calls. Matches the
    indexing convention used by `atom_perturbations` and `cluster_perturbations`:
    the first $C_A$ columns are A-side atoms, the next $C_B$ are B-side.
    """
    _, aux = decomp(x, mask_mode="deterministic", return_aux=True)
    return torch.cat([aux.g_A, aux.g_B], dim=-1)
