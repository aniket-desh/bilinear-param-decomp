"""The ΔM_r perturbation expansion (theory § 7.2 / spec § 5.6 / § 8.4).

For any choice of $\\Delta A_S, \\Delta B_S$:

    $M_r(A, B, C) - M_r(A - \\Delta A_S, B - \\Delta B_S, C)$
        $= M_r(\\Delta A_S, B, C) + M_r(A, \\Delta B_S, C) - M_r(\\Delta A_S, \\Delta B_S, C).$

The identity is purely algebraic — bilinearity of $M_r$ in $(A, B)$ — so it
should hold for any input matrices, not just trained ones. We test on random
matrices, on an atom-shaped perturbation, and on a multi-atom cluster (A + B
mixed) — the cross-term presence is the whole reason for the expansion.
"""

import torch

from bilinear_spd.analysis import atom_perturbations, cluster_perturbations
from bilinear_spd.decomposition import BilinearComponentMLP
from bilinear_spd.functional import centered_class_probe, construct_M_r
from bilinear_spd.models import BilinearMLP
from bilinear_spd.utils import set_seed


def _make_target(d_in: int = 10, d_hidden: int = 8, d_out: int = 5) -> BilinearMLP:
    set_seed(0)
    return BilinearMLP(d_in=d_in, d_hidden=d_hidden, d_out=d_out, use_bias=False)


def _expansion(A, B, dA, dB, C, r):
    return (
        construct_M_r(dA, B, C, r)
        + construct_M_r(A, dB, C, r)
        - construct_M_r(dA, dB, C, r)
    )


def test_expansion_random_perturbations():
    set_seed(0)
    d_in, d_hidden, d_out = 10, 8, 5
    A = torch.randn(d_in, d_hidden)
    B = torch.randn(d_in, d_hidden)
    C = torch.randn(d_hidden, d_out)
    r = torch.randn(d_out)
    dA = torch.randn(d_in, d_hidden) * 0.1
    dB = torch.randn(d_in, d_hidden) * 0.1

    direct = construct_M_r(A, B, C, r) - construct_M_r(A - dA, B - dB, C, r)
    expansion = _expansion(A, B, dA, dB, C, r)

    assert torch.allclose(direct, expansion, atol=1e-5)


def test_atom_perturbations_match_expansion():
    """For a single A-atom, dB = 0, so the cross-term vanishes."""
    target = _make_target()
    decomp = BilinearComponentMLP(target, C_A=8, C_B=8, gate_hidden=12)
    r = centered_class_probe(0, target.d_out)
    deltas = atom_perturbations(decomp, r)
    # atom 3 is an A-atom; build its expected ΔM_r by the expansion
    c = 3
    dA = torch.outer(decomp.U_A[c], decomp.V_A[c])
    dB = torch.zeros_like(decomp.B_target)
    expected = _expansion(decomp.A_target, decomp.B_target, dA, dB, decomp.C_target, r)
    assert torch.allclose(deltas.delta_M[c], expected, atol=1e-5)


def test_cluster_perturbation_includes_cross_term():
    """A cluster containing both an A-atom and a B-atom needs the cross-term.

    If we *erroneously* summed ΔM_r's of the two atoms in isolation, we'd
    miss $M_r(dA_S, dB_S, C, r)$. The correct cluster API recomputes M_r on
    the perturbed weights, so it must agree with the explicit expansion.
    """
    target = _make_target()
    decomp = BilinearComponentMLP(target, C_A=8, C_B=8, gate_hidden=12)
    r = centered_class_probe(0, target.d_out)

    # Cluster: A-atom 2 + B-atom 1 (flat index = C_A + 1 = 9)
    cluster = [2, 8 + 1]
    cluster_delta = cluster_perturbations(decomp, r, [cluster])[0]

    dA = torch.outer(decomp.U_A[2], decomp.V_A[2])
    dB = torch.outer(decomp.U_B[1], decomp.V_B[1])
    expansion = _expansion(decomp.A_target, decomp.B_target, dA, dB, decomp.C_target, r)
    assert torch.allclose(cluster_delta, expansion, atol=1e-5)

    # And the "naive sum of atom deltas" must NOT equal the cluster delta
    # (the cross-term is real). Compare via Frobenius distance.
    deltas = atom_perturbations(decomp, r)
    naive = deltas.delta_M[2] + deltas.delta_M[8 + 1]
    err_naive = torch.linalg.matrix_norm(cluster_delta - naive, "fro").item()
    cross_norm = torch.linalg.matrix_norm(
        construct_M_r(dA, dB, decomp.C_target, r), "fro"
    ).item()
    assert err_naive > 0.5 * cross_norm  # cross-term is the missing piece
