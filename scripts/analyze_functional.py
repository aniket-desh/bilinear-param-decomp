"""Build M_r for the configured probes, eigendecompose, group, save + plot.

    python -m scripts.analyze_functional configs/modadd_p13.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bilinear_spd.data import input_dim, make_dataset, output_dim
from bilinear_spd.functional import (
    build_probes,
    construct_M_r,
    eigendecompose_M,
    functional_scalar,
    group_eigenspaces,
    quadratic_scalar,
)
from bilinear_spd.models import BilinearMLP
from bilinear_spd.plotting import plot_spectrum, plot_top_eigenvectors
from bilinear_spd.utils import (
    load_checkpoint,
    load_config,
    run_dir,
    select_device,
    set_seed,
)


def _slot_split(task_cfg: dict) -> int | None:
    """For concat-onehot tasks, the boundary between a-slot and b-slot in x."""
    if task_cfg["name"] == "modular_addition":
        return task_cfg["p"]
    if task_cfg["name"] == "xor":
        return 2
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    out = run_dir(cfg)

    ckpt = load_checkpoint(out / "checkpoints" / "bilinear_mlp.pt", map_location=device)
    model = BilinearMLP(
        d_in=input_dim(cfg["task"]),
        d_hidden=cfg["model"]["d_hidden"],
        d_out=output_dim(cfg["task"]),
        use_bias=cfg["model"]["use_bias"],
        init_scale=cfg["model"]["init_scale"],
    ).to(device)
    model.load_state_dict(ckpt.state_dict)
    model.eval()

    data = make_dataset(cfg["task"], device=device)
    p = output_dim(cfg["task"])
    probes = build_probes(cfg["probes"], p=p, device=device)
    if not probes:
        raise SystemExit("no probes built — check the probes block in your config")
    print(f"probes: {list(probes)}")

    mr_slices: dict[str, torch.Tensor] = {}
    eigenspaces: dict[str, dict] = {}
    summary_per_probe: dict[str, dict] = {}

    rel_tol = float(cfg["eigenspaces"]["rel_tol"])
    min_abs_eval = float(cfg["eigenspaces"]["min_abs_eval"])

    for name, r in probes.items():
        with torch.no_grad():
            M = construct_M_r(model.A, model.B, model.C, r)
            # equivalence sanity check on this probe
            y_scalar = functional_scalar(model(data["x"]), r)
            x_scalar = quadratic_scalar(data["x"], M)
            eq_err = (y_scalar - x_scalar).abs().max().item()

            evals, evecs = eigendecompose_M(M)
            groups = group_eigenspaces(
                evals, evecs, rel_tol=rel_tol, min_abs_eval=min_abs_eval
            )

        mr_slices[name] = M.detach().cpu()
        eigenspaces[name] = {
            "evals": evals.detach().cpu(),
            "evecs": evecs.detach().cpu(),
            "groups": [
                {
                    "indices": g.indices,
                    "evals": g.evals.detach().cpu(),
                    "projector": g.projector.detach().cpu(),
                    "rank": g.rank,
                    "mean_abs_eval": g.mean_abs_eval,
                    "mean_eval": g.mean_eval,
                }
                for g in groups
            ],
        }

        slot_split = _slot_split(cfg["task"])
        plot_spectrum(
            evals,
            groups,
            out_path=out / "figures" / f"spectrum_{name}.png",
            title=f"{cfg['run_name']}  probe={name}",
            min_abs_eval=min_abs_eval,
        )
        plot_top_eigenvectors(
            evals,
            evecs,
            out_path=out / "figures" / f"eigenvectors_{name}.png",
            k=min(8, evecs.shape[1]),
            slot_split=slot_split,
            title=f"{cfg['run_name']}  probe={name}",
        )

        # human-readable line
        n_groups = len(groups)
        n_degenerate = sum(1 for g in groups if g.rank >= 2)
        top_lambda = float(evals[0].abs().item()) if len(evals) > 0 else 0.0
        print(
            f"  {name:14s} ‖r⊤y − x⊤Mᵣx‖∞={eq_err:.2e}  "
            f"groups={n_groups:3d}  degenerate={n_degenerate:2d}  "
            f"top|λ|={top_lambda:.3f}"
        )

        summary_per_probe[name] = {
            "equivalence_max_err": eq_err,
            "n_groups": n_groups,
            "n_degenerate_groups": n_degenerate,
            "top_abs_eval": top_lambda,
            "group_ranks": [g.rank for g in groups],
            "group_mean_abs_evals": [g.mean_abs_eval for g in groups],
        }

    torch.save(mr_slices, out / "artifacts" / "mr_slices.pt")
    torch.save(eigenspaces, out / "artifacts" / "eigenspaces.pt")

    with open(out / "reports" / "functional_summary.json", "w") as f:
        json.dump(summary_per_probe, f, indent=2)
    print(f"wrote {out / 'reports' / 'functional_summary.json'}")


if __name__ == "__main__":
    main()
