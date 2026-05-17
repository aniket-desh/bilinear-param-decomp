"""Tests for `bilinear_spd.ablations`.

The key invariants:
1. Ablating *no* atoms returns the original target logits.
2. Ablating *all* atoms (A + B) yields a network whose effective weights
   are $(\\Delta A, \\Delta B)$. With our trained decompositions $\\Delta$
   is tiny, so the resulting logits should NOT match the target — but on a
   fresh untrained decomposition $\\Delta = A - \\hat A$ is whatever the
   atoms didn't pick up.
3. The KL of an empty-ablation run is 0.
4. random_size_matched_sets respects `exclude` and the requested size.
"""

import torch

from bilinear_spd.ablations import (
    ablate_weights,
    bilinear_forward,
    evaluate_ablation,
    random_size_matched_sets,
)
from bilinear_spd.decomposition import BilinearComponentMLP
from bilinear_spd.models import BilinearMLP
from bilinear_spd.utils import set_seed


def _make_decomp():
    set_seed(0)
    target = BilinearMLP(d_in=10, d_hidden=8, d_out=5, use_bias=False)
    decomp = BilinearComponentMLP(target, C_A=16, C_B=16, gate_hidden=12)
    return target, decomp


def test_ablate_empty_set_recovers_target_weights():
    _, decomp = _make_decomp()
    A_minus, B_minus = ablate_weights(decomp, [])
    assert torch.allclose(A_minus, decomp.A_target, atol=1e-6)
    assert torch.allclose(B_minus, decomp.B_target, atol=1e-6)


def test_evaluate_ablation_kl_is_zero_for_unperturbed_weights():
    target, decomp = _make_decomp()
    x = torch.randn(8, target.d_in)
    y_target = target(x)
    metrics = evaluate_ablation(decomp, x, y_target,
                                A_minus=decomp.A_target, B_minus=decomp.B_target, r=None)
    assert metrics["kl"] < 1e-6
    assert abs(metrics["d_acc"]) < 1e-6
    assert abs(metrics["d_margin"]) < 1e-6


def test_ablate_single_atom_changes_weights_by_outer_product():
    _, decomp = _make_decomp()
    A_minus, B_minus = ablate_weights(decomp, [3])  # A-atom 3
    expected_A = decomp.A_target - torch.outer(decomp.U_A[3], decomp.V_A[3])
    assert torch.allclose(A_minus, expected_A, atol=1e-6)
    assert torch.allclose(B_minus, decomp.B_target, atol=1e-6)
    # B-side ablation
    A_minus2, B_minus2 = ablate_weights(decomp, [decomp.C_A + 5])  # B-atom 5
    assert torch.allclose(A_minus2, decomp.A_target, atol=1e-6)
    expected_B = decomp.B_target - torch.outer(decomp.U_B[5], decomp.V_B[5])
    assert torch.allclose(B_minus2, expected_B, atol=1e-6)


def test_bilinear_forward_matches_model_forward():
    target, decomp = _make_decomp()
    x = torch.randn(8, target.d_in)
    y_model = target(x)
    y_func = bilinear_forward(x, decomp.A_target, decomp.B_target, decomp.C_target,
                              b_out=decomp.b_out)
    assert torch.allclose(y_model, y_func, atol=1e-6)


def test_random_size_matched_sets_respects_exclude_and_size():
    sets = random_size_matched_sets(n_atoms=20, size=3, n_samples=10, exclude=[0, 1, 2, 3])
    assert len(sets) == 10
    for s in sets:
        assert len(s) == 3
        assert all(a not in {0, 1, 2, 3} for a in s)
        assert all(0 <= a < 20 for a in s)


def test_random_size_matched_sets_raises_when_infeasible():
    import pytest
    with pytest.raises(ValueError):
        random_size_matched_sets(n_atoms=5, size=10, n_samples=1)
