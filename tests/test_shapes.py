import torch

from bilinear_spd.data import make_modular_addition_dataset, make_xor_dataset
from bilinear_spd.functional import construct_M_r
from bilinear_spd.models import BilinearMLP


def test_modular_addition_dataset_shapes():
    p = 11
    d = make_modular_addition_dataset(p=p)
    assert d["x"].shape == (p * p, 2 * p)
    assert d["a"].shape == (p * p,)
    assert d["b"].shape == (p * p,)
    assert d["y"].shape == (p * p,)
    # spot-check the algebra
    assert ((d["a"] + d["b"]) % p == d["y"]).all()


def test_xor_dataset_shapes():
    d = make_xor_dataset()
    assert d["x"].shape == (4, 4)
    assert (d["y"] == (d["a"] ^ d["b"])).all()


def test_bilinear_forward_shape():
    p, m = 13, 32
    model = BilinearMLP(d_in=2 * p, d_hidden=m, d_out=p)
    x = torch.randn(7, 2 * p)
    y = model(x)
    assert y.shape == (7, p)


def test_construct_M_r_shape_and_symmetry():
    p, m, d = 13, 32, 2 * 13
    model = BilinearMLP(d_in=d, d_hidden=m, d_out=p)
    r = torch.randn(p)
    M = construct_M_r(model.A, model.B, model.C, r)
    assert M.shape == (d, d)
    assert torch.allclose(M, M.T, atol=1e-6)
