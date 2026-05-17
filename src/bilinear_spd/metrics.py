"""Alignment metrics between matrices and between matrices/projectors.

`fro_cos` and `signed_fro_cos` measure how much two matrices share direction
in Frobenius (Hilbert–Schmidt) inner-product space — applied here to
$\\Delta M_r$ from an atom/cluster vs. an eigenspace projector $P_\\lambda$.

`projector_energy` measures how much of $\\Delta M_r$'s energy lies inside
the eigenspace: $\\|P \\Delta M_r P\\|_F^2 / \\|\\Delta M_r\\|_F^2 \\in [0, 1]$.

`mean_max_alignment` summarises an `[atoms × eigenspaces]` matrix by, for
each eigenspace, taking the best-aligned atom (column max) and averaging:
"on average, how well-explained is each eigenspace by *some* atom?" This is
the column-wise version — atom-vs-eigenspace asks the dual question
(row-wise max means "on average, how aligned is each atom to *some*
eigenspace?"). Spec § 5.7 uses the eigenspace-coverage convention.
"""

from __future__ import annotations

import torch


def fro_cos(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Unsigned cosine of A and B in the Frobenius inner product, in [0, 1]."""
    num = torch.abs(torch.sum(A * B))
    den = torch.linalg.norm(A) * torch.linalg.norm(B) + eps
    return num / den


def signed_fro_cos(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Signed cosine of A and B in the Frobenius inner product, in [-1, 1]."""
    num = torch.sum(A * B)
    den = torch.linalg.norm(A) * torch.linalg.norm(B) + eps
    return num / den


def projector_energy(
    delta_M: torch.Tensor,
    P: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Fraction of $\\|\\Delta M\\|_F^2$ that lies inside the eigenspace projector P.

    `P` should be an orthogonal projector ($P^2 = P$, $P^\\top = P$); the
    quantity $\\|P \\Delta M P\\|_F^2 / \\|\\Delta M\\|_F^2 \\in [0, 1]$ — 0
    iff $\\Delta M$ has no component in the eigenspace, 1 iff it lives
    entirely in it.
    """
    projected = P @ delta_M @ P
    num = torch.linalg.matrix_norm(projected, "fro").pow(2)
    den = torch.linalg.matrix_norm(delta_M, "fro").pow(2) + eps
    return num / den


def mean_max_alignment(alignment_matrix: torch.Tensor, axis: str = "eigenspaces") -> torch.Tensor:
    """Summary statistic for an [n_atoms_or_clusters × n_eigenspaces] alignment matrix.

    axis="eigenspaces" (default): for each eigenspace (column), take the best
    atom/cluster; average across eigenspaces. This is the "eigenspace coverage"
    number — high means every eigenspace has *some* atom/cluster that explains it.

    axis="atoms": for each atom/cluster (row), take its best eigenspace; average
    across atoms. High means most atoms are *individually* well-aligned to some
    eigenspace.

    Spec § 5.7's `mean_max_alignment` is the "atoms" axis on alignment_matrix.T
    (rows are atoms there); we expose both with an explicit kw to avoid axis
    confusion in callers.
    """
    if axis == "eigenspaces":
        return alignment_matrix.max(dim=0).values.mean()
    if axis == "atoms":
        return alignment_matrix.max(dim=1).values.mean()
    raise ValueError(f"axis must be 'eigenspaces' or 'atoms', got {axis!r}")
