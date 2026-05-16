"""Per-group projector invariants: P^2 = P, P^T = P, tr(P) = rank(P).

Run twice: once on a random symmetric M_r (decoupled from training), once on
a freshly trained modadd_p7 checkpoint (real eigenstructure with degeneracies).
"""

import torch

from bilinear_spd.data import make_modular_addition_dataset
from bilinear_spd.functional import (
    centered_class_probe,
    construct_M_r,
    eigendecompose_M,
    group_eigenspaces,
)
from bilinear_spd.models import BilinearMLP
from bilinear_spd.train import train_bilinear
from bilinear_spd.utils import set_seed


def _check_projector_invariants(groups, d: int) -> None:
    for g in groups:
        P = g.projector
        assert P.shape == (d, d)
        assert torch.allclose(P @ P, P, atol=1e-5), "P^2 != P"
        assert torch.allclose(P, P.T, atol=1e-6), "P^T != P"
        # tr(P) should equal rank(P) for an orthogonal projector
        assert abs(P.trace().item() - g.rank) < 1e-4, (
            f"tr(P)={P.trace().item():.4f} but rank={g.rank}"
        )


def test_projectors_on_random_symmetric():
    set_seed(0)
    d = 26
    A = torch.randn(d, d)
    M = 0.5 * (A + A.T)
    evals, evecs = eigendecompose_M(M)
    groups = group_eigenspaces(evals, evecs, rel_tol=1e-3, min_abs_eval=1e-6)
    assert len(groups) > 0
    _check_projector_invariants(groups, d)
    # sanity: sum of group ranks should equal the count of evals above the floor
    above_floor = (evals.abs() >= 1e-6).sum().item()
    assert sum(g.rank for g in groups) == above_floor


def test_projectors_on_trained_modadd_p7():
    set_seed(0)
    p = 7
    data = make_modular_addition_dataset(p=p)
    model = BilinearMLP(d_in=2 * p, d_hidden=24, d_out=p, init_scale=0.5)
    train_bilinear(
        model,
        data["x"],
        data["y"],
        cfg={
            "lr": 0.003,
            "steps": 600,
            "log_every": 1000,
            "target_accuracy": 0.99,
            "target_acc_patience": 200,
        },
        verbose=False,
    )
    r = centered_class_probe(0, p)
    M = construct_M_r(model.A, model.B, model.C, r)
    evals, evecs = eigendecompose_M(M)
    groups = group_eigenspaces(evals, evecs, rel_tol=1e-3, min_abs_eval=1e-6)
    assert len(groups) > 0
    _check_projector_invariants(groups, d=2 * p)


def test_degenerate_grouping_with_repeated_eigenvalue():
    """Synthetic: project onto a known 3-dim subspace with eigenvalue 1, rest 0."""
    d = 10
    set_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    # eigenvalues: 1, 1, 1, 0.5, -0.5, 0, ..., 0
    diag = torch.zeros(d)
    diag[0:3] = 1.0
    diag[3] = 0.5
    diag[4] = -0.5
    M = Q @ torch.diag(diag) @ Q.T
    M = 0.5 * (M + M.T)
    evals, evecs = eigendecompose_M(M)
    groups = group_eigenspaces(evals, evecs, rel_tol=1e-3, min_abs_eval=1e-6)
    # should be 3 groups: {+1 ×3}, {+0.5}, {-0.5}
    assert len(groups) == 3
    ranks = sorted(g.rank for g in groups)
    assert ranks == [1, 1, 3]
    _check_projector_invariants(groups, d=d)
