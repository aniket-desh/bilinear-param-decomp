"""Day 5 supplement: per-atom gate heatmap + ablation mask over the (a, b) grid.

For the top-K atoms by $\\|\\Delta M_r\\|_F$ on a chosen probe:

  1. plot the deterministic gate value $g_c(a, b)$ as a $p \\times p$
     heatmap — the shard hypothesis predicts a clean horizontal or
     vertical stripe;
  2. plot the per-input ablation-error mask: for each $(a, b)$, did
     the model's argmax flip after surgically removing this single atom?
     Red cell = flip, white cell = no flip.

Two side-by-side grids per atom, K atoms per page.

    python -m scripts.run_grid_visuals configs/modadd_p47_grok.yaml --probe centered_0 --top-k 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bilinear_spd.ablations import ablate_weights, bilinear_forward
from bilinear_spd.analysis import atom_perturbations
from bilinear_spd.data import input_dim, make_dataset, output_dim
from bilinear_spd.decomposition import BilinearComponentMLP
from bilinear_spd.functional import build_probes
from bilinear_spd.models import BilinearMLP
from bilinear_spd.plotting import apply_style
from bilinear_spd.utils import (
    load_checkpoint,
    load_config,
    run_dir,
    select_device,
    set_seed,
)


def _load_decomposition(cfg: dict, out: Path, device: torch.device) -> BilinearComponentMLP:
    target = BilinearMLP(
        d_in=input_dim(cfg["task"]),
        d_hidden=cfg["model"]["d_hidden"],
        d_out=output_dim(cfg["task"]),
        use_bias=cfg["model"]["use_bias"],
        init_scale=cfg["model"]["init_scale"],
    ).to(device)
    bi_ckpt = load_checkpoint(out / "checkpoints" / "bilinear_mlp.pt", map_location=device)
    target.load_state_dict(bi_ckpt.state_dict); target.eval()

    decomp_cfg = cfg["decomposition"]
    decomp = BilinearComponentMLP(
        target=target, C_A=int(decomp_cfg["C_A"]),
        C_B=int(decomp_cfg["C_B"]), gate_hidden=int(decomp_cfg["gate_hidden"]),
    ).to(device)
    d_ckpt = load_checkpoint(out / "checkpoints" / "decomposition.pt", map_location=device)
    decomp.load_state_dict(d_ckpt.state_dict); decomp.eval()
    return decomp


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--probe", default="centered_0")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg["task"]["name"] != "modular_addition":
        raise SystemExit("grid visuals are only meaningful for modular_addition tasks")
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    out = run_dir(cfg)
    p = int(cfg["task"]["p"])

    decomp = _load_decomposition(cfg, out, device)
    data = make_dataset(cfg["task"], device=device)
    probes = build_probes(cfg["probes"], p=output_dim(cfg["task"]), device=device)
    r = probes[args.probe]

    # Deterministic gates over the full table.
    _, aux = decomp(data["x"], mask_mode="deterministic", return_aux=True)
    g_all = torch.cat([aux.g_A, aux.g_B], dim=-1).cpu()   # [p*p, C_A + C_B]
    C_A = decomp.C_A

    # ΔM_r norms — rank atoms.
    deltas = atom_perturbations(decomp, r)
    norms = deltas.norms.cpu()
    top_atoms = torch.argsort(norms, descending=True)[: args.top_k].tolist()

    # Target argmaxes.
    y_target = decomp.forward_target(data["x"])
    target_argmax = y_target.argmax(dim=-1).cpu()
    a_idx = data["a"].cpu()
    b_idx = data["b"].cpu()

    # Reshape helper: [p*p] in (a, b) row-major (a outer, b inner from data.py).
    def to_grid(vec):
        return vec.reshape(p, p).numpy()

    apply_style()
    from bilinear_spd.plotting import COLORS
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    # Sequential cream → deep blue for gates; binary white→shard-red for flips.
    gate_cmap = LinearSegmentedColormap.from_list(
        "gate_cream_blue",
        ["#FCFAF6", "#A5C3DD", COLORS["atom"], COLORS["accent_dark"]],
    )
    flip_cmap = ListedColormap(["#FFFFFF", COLORS["shard"]])

    # Decide grid layout — prefer 4 cols × ceil rows.
    cols = 4 if args.top_k >= 8 else 3
    rows = int(np.ceil(args.top_k / cols))

    # --- gate heatmaps ---
    fig, axes = plt.subplots(rows, cols,
                             figsize=(2.55 * cols + 0.7, 2.55 * rows),
                             squeeze=False, sharex=True, sharey=True)
    im = None
    for k, atom_idx in enumerate(top_atoms):
        ax = axes[k // cols][k % cols]
        grid = to_grid(g_all[:, atom_idx])
        im = ax.imshow(grid, cmap=gate_cmap, vmin=0.0, vmax=1.0,
                       aspect="equal", origin="lower",
                       interpolation="nearest")
        side = "A" if atom_idx < C_A else "B"
        ax.set_title(f"atom {atom_idx} · {side}-side · rank {k + 1}", fontsize=9.5)
        ax.set_xticks([0, p - 1]); ax.set_yticks([0, p - 1])
        ax.set_xticklabels([0, p - 1] if k // cols == rows - 1 else [])
        ax.set_yticklabels([0, p - 1] if k % cols == 0 else [])
        if k // cols == rows - 1:
            ax.set_xlabel("$b$", fontsize=10)
        if k % cols == 0:
            ax.set_ylabel("$a$", fontsize=10)
        ax.grid(False)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_color(COLORS["grid"])
    for k in range(args.top_k, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.93, 0.18, 0.018, 0.64])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("deterministic gate $g_c(a, b)$", fontsize=9.5)
    cb.outline.set_visible(False)
    fig.savefig(out / "figures" / f"gate_grid_top_norm_{args.probe}.png",
                bbox_inches="tight")
    plt.close(fig)

    # --- ablation-error masks (binary white / red) ---
    fig, axes = plt.subplots(rows, cols,
                             figsize=(2.55 * cols + 0.4, 2.55 * rows),
                             squeeze=False, sharex=True, sharey=True)
    for k, atom_idx in enumerate(top_atoms):
        ax = axes[k // cols][k % cols]
        A_m, B_m = ablate_weights(decomp, [int(atom_idx)])
        y_abl = bilinear_forward(data["x"], A_m, B_m, decomp.C_target, b_out=decomp.b_out)
        abl_argmax = y_abl.argmax(dim=-1).cpu()
        flips = (abl_argmax != target_argmax).float()
        n_flipped = int(flips.sum().item())
        grid = to_grid(flips)
        ax.imshow(grid, cmap=flip_cmap, vmin=0.0, vmax=1.0,
                  aspect="equal", origin="lower", interpolation="nearest")
        side = "A" if atom_idx < C_A else "B"
        ax.set_title(f"atom {atom_idx} · {side} · {n_flipped}/{p * p} flipped", fontsize=9.5)
        ax.set_xticks([0, p - 1]); ax.set_yticks([0, p - 1])
        ax.set_xticklabels([0, p - 1] if k // cols == rows - 1 else [])
        ax.set_yticklabels([0, p - 1] if k % cols == 0 else [])
        if k // cols == rows - 1:
            ax.set_xlabel("$b$", fontsize=10)
        if k % cols == 0:
            ax.set_ylabel("$a$", fontsize=10)
        ax.grid(False)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_color(COLORS["grid"])
    for k in range(args.top_k, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.savefig(out / "figures" / f"ablation_mask_top_norm_{args.probe}.png",
                bbox_inches="tight")
    plt.close(fig)

    print(
        f"wrote {out / 'figures' / f'gate_grid_top_norm_{args.probe}.png'} "
        f"and {out / 'figures' / f'ablation_mask_top_norm_{args.probe}.png'}"
    )


if __name__ == "__main__":
    main()
