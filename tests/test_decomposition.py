"""Forward + ablation sanity checks for BilinearComponentMLP.

Three identities a correct implementation must satisfy:

  1. forward_target(x) reproduces the frozen BilinearMLP's logits exactly.
  2. forward_with_masks(x, m_A=1, m_B=1, include_delta_*=True) reproduces them too
     — because Â + Δ = A by construction.
  3. forward_with_masks(x, m_A=0, m_B=0, include_delta_*=False) gives zeros
     (no bias was used in our configs).
"""

import torch

from bilinear_spd.decomposition import BilinearComponentMLP
from bilinear_spd.models import BilinearMLP
from bilinear_spd.utils import set_seed


def _make_target(d_in: int = 10, d_hidden: int = 8, d_out: int = 5) -> BilinearMLP:
    set_seed(0)
    m = BilinearMLP(d_in=d_in, d_hidden=d_hidden, d_out=d_out, use_bias=False)
    return m


def test_forward_target_matches_bilinear_mlp():
    target = _make_target()
    decomp = BilinearComponentMLP(target, C_A=16, C_B=16, gate_hidden=12)
    x = torch.randn(4, target.d_in)
    expected = target(x)
    got = decomp.forward_target(x)
    assert torch.allclose(got, expected, atol=1e-6)


def test_full_atoms_plus_delta_equals_target():
    target = _make_target()
    decomp = BilinearComponentMLP(target, C_A=16, C_B=16, gate_hidden=12)
    x = torch.randn(4, target.d_in)
    expected = target(x)
    # all gates = 1, delta included → A_g = sum_c U_A V_A^T + (A - sum U V^T) = A
    got = decomp.forward_with_masks(
        x,
        m_A=torch.ones(decomp.C_A),
        m_B=torch.ones(decomp.C_B),
        include_delta_A=True,
        include_delta_B=True,
    )
    assert torch.allclose(got, expected, atol=1e-5)


def test_zero_atoms_zero_delta_gives_zero_logits():
    target = _make_target()
    decomp = BilinearComponentMLP(target, C_A=16, C_B=16, gate_hidden=12)
    x = torch.randn(4, target.d_in)
    got = decomp.forward_with_masks(
        x,
        m_A=torch.zeros(decomp.C_A),
        m_B=torch.zeros(decomp.C_B),
        include_delta_A=False,
        include_delta_B=False,
    )
    # A_g = 0, B_g = 0  =>  (xA)*(xB) = 0  =>  y = 0
    assert torch.allclose(got, torch.zeros_like(got), atol=1e-6)


def test_gates_clamp_to_unit_interval():
    target = _make_target()
    decomp = BilinearComponentMLP(target, C_A=16, C_B=16, gate_hidden=12)
    x = torch.randn(4, target.d_in)
    _, aux = decomp(x, mask_mode="deterministic", return_aux=True)
    assert (aux.g_A >= 0).all() and (aux.g_A <= 1).all()
    assert (aux.g_B >= 0).all() and (aux.g_B <= 1).all()


def test_stochastic_masks_in_unit_interval_and_above_gates():
    """mask = g + (1-g)*U ∈ [g, 1]. Should hold sample-by-sample."""
    target = _make_target()
    decomp = BilinearComponentMLP(target, C_A=16, C_B=16, gate_hidden=12)
    x = torch.randn(8, target.d_in)
    _, aux = decomp(x, mask_mode="stochastic", return_aux=True)
    assert (aux.m_A >= aux.g_A - 1e-6).all()
    assert (aux.m_A <= 1.0 + 1e-6).all()
    assert (aux.m_B >= aux.g_B - 1e-6).all()
    assert (aux.m_B <= 1.0 + 1e-6).all()
