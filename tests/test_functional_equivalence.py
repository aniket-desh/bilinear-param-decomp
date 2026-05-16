"""r^T y(x) == x^T M_r x must hold to floating-point precision.

This is the central correctness check for the whole project. It must pass
at random init AND after training — if it ever fails, the M_r formula or
shape conventions are wrong.
"""

import torch

from bilinear_spd.data import make_modular_addition_dataset
from bilinear_spd.functional import (
    centered_class_probe,
    construct_M_r,
    functional_scalar,
    pairwise_probe,
    quadratic_scalar,
)
from bilinear_spd.models import BilinearMLP
from bilinear_spd.train import train_bilinear
from bilinear_spd.utils import set_seed


def _equivalence_error(model: BilinearMLP, x: torch.Tensor, r: torch.Tensor) -> float:
    with torch.no_grad():
        scalar_y = functional_scalar(model(x), r)
        M = construct_M_r(model.A, model.B, model.C, r)
        scalar_x = quadratic_scalar(x, M)
    return (scalar_y - scalar_x).abs().max().item()


def test_equivalence_random_init():
    set_seed(0)
    p, m = 13, 32
    model = BilinearMLP(d_in=2 * p, d_hidden=m, d_out=p)
    x = torch.randn(64, 2 * p)
    # try a centered class probe, a pairwise probe, and a random probe
    for r in (
        centered_class_probe(3, p),
        pairwise_probe(0, 7, p),
        torch.randn(p),
    ):
        err = _equivalence_error(model, x, r)
        assert err < 1e-4, f"equivalence failed for probe with err={err:.2e}"


def test_equivalence_post_training_modadd_p7():
    """Train a tiny model briefly and re-check equivalence — must still hold."""
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
            "weight_decay": 0.0,
            "steps": 200,  # not aiming for convergence; just exercising the path
            "log_every": 1000,
            "target_accuracy": 1.01,  # disable early stop
            "target_acc_patience": 10_000,
        },
        verbose=False,
    )
    for k in (0, 1, 2):
        r = centered_class_probe(k, p)
        err = _equivalence_error(model, data["x"], r)
        assert err < 1e-4, f"post-train equivalence failed for probe e_{k}-1/p: {err:.2e}"
