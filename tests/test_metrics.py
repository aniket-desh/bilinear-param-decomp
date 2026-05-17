"""Tests for the alignment metrics in `bilinear_spd.metrics`."""

import torch

from bilinear_spd.metrics import (
    fro_cos,
    mean_max_alignment,
    projector_energy,
    signed_fro_cos,
)
from bilinear_spd.utils import set_seed


def test_fro_cos_self_is_one():
    set_seed(0)
    A = torch.randn(5, 5)
    assert torch.allclose(fro_cos(A, A), torch.tensor(1.0), atol=1e-6)


def test_signed_fro_cos_sign_flip():
    set_seed(0)
    A = torch.randn(4, 4)
    assert torch.allclose(signed_fro_cos(A, A), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(signed_fro_cos(A, -A), torch.tensor(-1.0), atol=1e-6)
    assert torch.allclose(fro_cos(A, -A), torch.tensor(1.0), atol=1e-6)


def test_projector_energy_in_eigenspace_is_one():
    """A matrix built from columns of a projector lives entirely inside it."""
    set_seed(0)
    # build a rank-2 orthogonal projector
    Q, _ = torch.linalg.qr(torch.randn(6, 2))
    P = Q @ Q.T
    # any matrix of the form P @ X @ P lives in the eigenspace
    X = torch.randn(6, 6)
    M_in = P @ X @ P
    e = projector_energy(M_in, P)
    assert torch.allclose(e, torch.tensor(1.0), atol=1e-5)


def test_projector_energy_orthogonal_is_zero():
    set_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(6, 6))
    P1 = Q[:, :2] @ Q[:, :2].T
    P2 = Q[:, 2:4] @ Q[:, 2:4].T
    X = torch.randn(6, 6)
    M_in_P1 = P1 @ X @ P1
    e = projector_energy(M_in_P1, P2)
    assert e.item() < 1e-5


def test_mean_max_alignment_axis_kwargs():
    # 3 atoms × 2 eigenspaces. Eigenspace 0 best aligned to atom 0 (=0.9),
    # eigenspace 1 best aligned to atom 2 (=0.7).
    A = torch.tensor([[0.9, 0.1], [0.1, 0.2], [0.3, 0.7]])
    eig_axis = mean_max_alignment(A, axis="eigenspaces")
    atom_axis = mean_max_alignment(A, axis="atoms")
    assert torch.allclose(eig_axis, torch.tensor((0.9 + 0.7) / 2), atol=1e-6)
    assert torch.allclose(atom_axis, torch.tensor((0.9 + 0.2 + 0.7) / 3), atol=1e-6)
