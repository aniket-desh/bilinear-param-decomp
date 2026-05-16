"""Train the bilinear MLP and persist artifacts.

    python -m scripts.train_bilinear configs/modadd_p13.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from bilinear_spd.data import input_dim, make_dataset, output_dim
from bilinear_spd.functional import (
    build_probes,
    centered_class_probe,
    construct_M_r,
    functional_scalar,
    quadratic_scalar,
)
from bilinear_spd.models import BilinearMLP
from bilinear_spd.plotting import plot_grokking_curves, plot_training_curves
from bilinear_spd.train import train_bilinear
from bilinear_spd.utils import (
    load_config,
    run_dir,
    save_checkpoint,
    save_config,
    select_device,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    print(f"device: {device}")

    out = run_dir(cfg)
    save_config(cfg, out / "config.yaml")

    data = make_dataset(cfg["task"], device=device)
    torch.save(data, out / "artifacts" / "dataset.pt")

    model = BilinearMLP(
        d_in=input_dim(cfg["task"]),
        d_hidden=cfg["model"]["d_hidden"],
        d_out=output_dim(cfg["task"]),
        use_bias=cfg["model"]["use_bias"],
        init_scale=cfg["model"]["init_scale"],
    ).to(device)
    print(f"model: d_in={model.d_in} d_hidden={model.d_hidden} d_out={model.d_out}  "
          f"params={sum(p.numel() for p in model.parameters())}")

    # optional grok-metric tracking
    train_cfg = cfg["train_model"]
    probe_for_grok = None
    grok_eval_every = train_cfg.get("grok_metrics_every")
    grok_rel_tols = tuple(train_cfg.get("grok_rel_tols", (0.01, 0.05)))
    if grok_eval_every is not None:
        all_probes = build_probes(cfg["probes"], p=output_dim(cfg["task"]), device=device)
        grok_probe_name = train_cfg.get("grok_probe", next(iter(all_probes)))
        if grok_probe_name not in all_probes:
            raise SystemExit(
                f"grok_probe '{grok_probe_name}' not in built probes {list(all_probes)}"
            )
        probe_for_grok = all_probes[grok_probe_name]
        print(f"grok tracking enabled: probe={grok_probe_name} every {grok_eval_every} steps "
              f"τ={list(grok_rel_tols)}")

    metrics = train_bilinear(
        model,
        data["x"],
        data["y"],
        train_cfg,
        probe_for_grok=probe_for_grok,
        grok_eval_every=grok_eval_every,
        grok_rel_tols=grok_rel_tols,
        verbose=True,
    )

    save_checkpoint(
        out / "checkpoints" / "bilinear_mlp.pt",
        model,
        cfg,
        asdict(metrics),
    )

    with torch.no_grad():
        logits = model(data["x"]).cpu()
    torch.save(logits, out / "artifacts" / "logits.pt")

    # Quick equivalence sanity check on the saved model. Probe = e_0.
    p = output_dim(cfg["task"])
    r = centered_class_probe(0, p, device=device)
    with torch.no_grad():
        y_scalar = functional_scalar(model(data["x"]), r)
        M = construct_M_r(model.A, model.B, model.C, r)
        x_scalar = quadratic_scalar(data["x"], M)
    max_err = (y_scalar - x_scalar).abs().max().item()
    print(f"equivalence check on probe e_0 - 1/p: max |r^T y - x^T M_r x| = {max_err:.2e}")
    assert max_err < 1e-4, f"equivalence test failed: {max_err}"

    fig_path = out / "figures" / "training_curves.png"
    plot_training_curves(metrics.history, fig_path, title=cfg["run_name"])

    if probe_for_grok is not None:
        plot_grokking_curves(
            metrics.history,
            out / "figures" / "grokking_curves.png",
            title=cfg["run_name"],
        )

    summary = {
        "final_step": metrics.final_step,
        "final_loss": metrics.final_loss,
        "final_accuracy": metrics.final_accuracy,
        "final_logit_margin": metrics.final_logit_margin,
        "early_stopped": metrics.early_stopped,
        "equivalence_max_err_probe_0": max_err,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    with open(out / "reports" / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
