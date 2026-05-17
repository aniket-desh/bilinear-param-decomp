"""Day 5 supplement: ablate atoms one-by-one ranked by ΔM_r norm vs. alignment.

The Day 5 main ablation (`run_ablation.py`) showed that the alignment-
selected atoms are NOT load-bearing. This diagnostic asks the dual: what
*are* the load-bearing atoms? Sort atoms by (a) Frobenius norm of ΔM_r
and (b) alignment to the best-aligned eigenspace, walk from top to
bottom, ablate one at a time, and record KL. Produces a clean visual
that shows "load-bearing ∝ norm, not ∝ alignment."

    python -m scripts.run_rank_ablation configs/modadd_p47_grok.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bilinear_spd.ablations import ablate_weights, evaluate_ablation
from bilinear_spd.analysis import atom_perturbations
from bilinear_spd.data import input_dim, make_dataset, output_dim
from bilinear_spd.decomposition import BilinearComponentMLP
from bilinear_spd.functional import build_probes
from bilinear_spd.metrics import fro_cos
from bilinear_spd.models import BilinearMLP
from bilinear_spd.plotting import COLORS, apply_style
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
        C_B=int(decomp_cfg["C_B"]),
        gate_hidden=int(decomp_cfg["gate_hidden"]),
    ).to(device)
    d_ckpt = load_checkpoint(out / "checkpoints" / "decomposition.pt", map_location=device)
    decomp.load_state_dict(d_ckpt.state_dict); decomp.eval()
    return decomp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--probe", default="centered_0")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    out = run_dir(cfg)

    decomp = _load_decomposition(cfg, out, device)
    data = make_dataset(cfg["task"], device=device)
    p = output_dim(cfg["task"])
    probes = build_probes(cfg["probes"], p=p, device=device)
    r = probes[args.probe]

    with torch.no_grad():
        y_target = decomp.forward_target(data["x"])

    deltas = atom_perturbations(decomp, r)
    norms = deltas.norms.cpu()

    # Load eigenspaces to compute per-atom alignment to the top-eigenspace.
    eigenspaces_blob = torch.load(
        out / "artifacts" / "eigenspaces.pt", weights_only=False, map_location=device
    )
    groups = eigenspaces_blob[args.probe]["groups"]
    top_proj = groups[0]["projector"].to(device)
    align_top = torch.tensor(
        [fro_cos(deltas.delta_M[i], top_proj).item() for i in range(deltas.delta_M.shape[0])]
    )

    rank_norm = torch.argsort(norms, descending=True).tolist()
    rank_align = torch.argsort(align_top, descending=True).tolist()

    def _walk(rank_order):
        rows = []
        for k, idx in enumerate(rank_order[: args.top_n]):
            A_m, B_m = ablate_weights(decomp, [int(idx)])
            m = evaluate_ablation(decomp, data["x"], y_target, A_m, B_m, r=r,
                                  targets_for_acc=data.get("y"))
            rows.append({
                "rank": k,
                "atom": int(idx),
                "norm": float(norms[int(idx)].item()),
                "alignment_top_eigenspace": float(align_top[int(idx)].item()),
                "kl": m["kl"],
                "d_acc": m["d_acc"],
            })
        return rows

    by_norm = _walk(rank_norm)
    by_align = _walk(rank_align)

    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    x = np.arange(args.top_n)
    axes[0].plot(x, [r["kl"] for r in by_norm], color=COLORS["orange"],
                 marker="o", label="rank by ‖ΔM_r‖_F")
    axes[0].plot(x, [r["kl"] for r in by_align], color=COLORS["blue"],
                 marker="x", label="rank by |fro_cos| to top eigenspace")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("single-atom ablation KL")
    axes[0].set_title(f"{cfg['run_name']}  probe={args.probe}  one-at-a-time ablation")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, [r["d_acc"] for r in by_norm], color=COLORS["orange"],
                 marker="o", label="rank by ‖ΔM_r‖_F")
    axes[1].plot(x, [r["d_acc"] for r in by_align], color=COLORS["blue"],
                 marker="x", label="rank by alignment")
    axes[1].set_ylabel(r"$\Delta$ accuracy")
    axes[1].set_xlabel("rank (top-K)")
    axes[1].axhline(0, color=COLORS["gray"], linewidth=0.6, linestyle="--")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figures" / f"rank_ablation_{args.probe}.png")
    plt.close(fig)

    with open(out / "reports" / f"rank_ablation_{args.probe}.json", "w") as f:
        json.dump({"by_norm": by_norm, "by_alignment": by_align}, f, indent=2)
    print(f"wrote {out / 'figures' / f'rank_ablation_{args.probe}.png'}")


if __name__ == "__main__":
    main()
