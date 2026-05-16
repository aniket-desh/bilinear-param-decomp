"""Train the gated rank-one decomposition against a frozen bilinear MLP.

    python -m scripts.train_decomp configs/modadd_p13_grok.yaml

Loads `runs/<run_name>/checkpoints/bilinear_mlp.pt`, freezes it, trains
U_A, V_A, U_B, V_B, gate_A, gate_B against the four-term loss, and writes
the decomposition + history + figures to the same run dir.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from bilinear_spd.data import input_dim, make_dataset, output_dim
from bilinear_spd.decomposition import BilinearComponentMLP
from bilinear_spd.models import BilinearMLP
from bilinear_spd.plotting import plot_decomposition_curves
from bilinear_spd.train import train_decomposition
from bilinear_spd.utils import (
    load_checkpoint,
    load_config,
    run_dir,
    save_checkpoint,
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
    out = run_dir(cfg)

    # Load the frozen bilinear target.
    ckpt = load_checkpoint(out / "checkpoints" / "bilinear_mlp.pt", map_location=device)
    target = BilinearMLP(
        d_in=input_dim(cfg["task"]),
        d_hidden=cfg["model"]["d_hidden"],
        d_out=output_dim(cfg["task"]),
        use_bias=cfg["model"]["use_bias"],
        init_scale=cfg["model"]["init_scale"],
    ).to(device)
    target.load_state_dict(ckpt.state_dict)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    decomp_cfg = cfg["decomposition"]
    train_cfg = cfg["train_decomp"]

    decomp = BilinearComponentMLP(
        target=target,
        C_A=int(decomp_cfg["C_A"]),
        C_B=int(decomp_cfg["C_B"]),
        gate_hidden=int(decomp_cfg["gate_hidden"]),
    ).to(device)
    n_params = sum(p.numel() for p in decomp.parameters() if p.requires_grad)
    print(f"decomposition: C_A={decomp.C_A} C_B={decomp.C_B} gate_hidden={decomp_cfg['gate_hidden']}  "
          f"trainable params={n_params:,}")

    # Precompute target logits on the dataset (full table).
    data = make_dataset(cfg["task"], device=device)
    with torch.no_grad():
        y_target_logits = target(data["x"])

    metrics = train_decomposition(
        decomp,
        data["x"],
        y_target_logits,
        cfg=dict(train_cfg, use_stochastic_masks=decomp_cfg.get("use_stochastic_masks", True)),
        mask_samples_per_step=int(decomp_cfg.get("mask_samples_per_batch", 1)),
        verbose=True,
    )

    # Save artifacts.
    save_checkpoint(
        out / "checkpoints" / "decomposition.pt",
        decomp,
        cfg,
        asdict(metrics),
    )

    # Gates over the full dataset, deterministic — useful for clustering on Day 4.
    decomp.eval()
    with torch.no_grad():
        _, aux = decomp(data["x"], mask_mode="deterministic", return_aux=True)
    torch.save(
        {
            "g_A": aux.g_A.cpu(),
            "g_B": aux.g_B.cpu(),
            "g_A_upper": aux.g_A_upper.cpu(),
            "g_B_upper": aux.g_B_upper.cpu(),
        },
        out / "artifacts" / "gates.pt",
    )

    fig_path = out / "figures" / "decomposition_curves.png"
    plot_decomposition_curves(metrics.history, fig_path, title=cfg["run_name"])

    summary = {
        "final_step": metrics.final_step,
        "final_loss": metrics.final_loss,
        "final_kl": metrics.final_kl,
        "final_param_recon_A": metrics.final_param_recon_A,
        "final_param_recon_B": metrics.final_param_recon_B,
        "final_l0_A": metrics.final_l0_A,
        "final_l0_B": metrics.final_l0_B,
        "C_A": decomp.C_A,
        "C_B": decomp.C_B,
        "trainable_params": n_params,
    }
    with open(out / "reports" / "decomposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out / 'reports' / 'decomposition_summary.json'}")


if __name__ == "__main__":
    main()
