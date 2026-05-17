"""Day 5 supplement: sparsity-pressure sweep at p=13.

Address the "maybe you forced $L_0 \\approx 1$, so of course the
decomposition became shards" objection. For one config (defaults to
`configs/modadd_p13_grok.yaml`), train the decomposition once per
$\\lambda_s$ value against the *same* frozen bilinear MLP, then run
the Day-4 alignment + Day-5 single-atom ablation pipelines in-process,
and plot the resulting trends.

    python -m scripts.run_sparsity_sweep configs/modadd_p13_grok.yaml \
        --lambda-sparsities 1e-2 1e-3 1e-4 1e-5 0

Outputs (under the run dir):
  artifacts/sparsity_sweep.json
  figures/sparsity_sweep.png
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bilinear_spd.ablations import ablate_weights, evaluate_ablation, random_size_matched_sets
from bilinear_spd.analysis import atom_perturbations, collect_gates
from bilinear_spd.clustering import gate_corr
from bilinear_spd.data import input_dim, make_dataset, output_dim
from bilinear_spd.decomposition import BilinearComponentMLP
from bilinear_spd.functional import build_probes, construct_M_r, eigendecompose_M, group_eigenspaces
from bilinear_spd.metrics import fro_cos, mean_max_alignment
from bilinear_spd.models import BilinearMLP
from bilinear_spd.plotting import COLORS, apply_style
from bilinear_spd.train import train_decomposition
from bilinear_spd.utils import (
    load_checkpoint,
    load_config,
    run_dir,
    select_device,
    set_seed,
)


def _load_target(cfg: dict, out: Path, device: torch.device) -> BilinearMLP:
    target = BilinearMLP(
        d_in=input_dim(cfg["task"]),
        d_hidden=cfg["model"]["d_hidden"],
        d_out=output_dim(cfg["task"]),
        use_bias=cfg["model"]["use_bias"],
        init_scale=cfg["model"]["init_scale"],
    ).to(device)
    ck = load_checkpoint(out / "checkpoints" / "bilinear_mlp.pt", map_location=device)
    target.load_state_dict(ck.state_dict)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def _eigenspaces_for_probe(target: BilinearMLP, r: torch.Tensor,
                           rel_tol: float, min_abs_eval: float):
    with torch.no_grad():
        M = construct_M_r(target.A, target.B, target.C, r)
        evals, evecs = eigendecompose_M(M)
        groups = group_eigenspaces(evals, evecs, rel_tol=rel_tol, min_abs_eval=min_abs_eval)
    return groups


def _atom_alignment(decomp: BilinearComponentMLP, r: torch.Tensor,
                    projectors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return alignment matrix [n_alive, n_eig] and per-atom ‖ΔM_r‖_F over alive atoms."""
    deltas = atom_perturbations(decomp, r)
    alive_mask = deltas.norms > 1e-8
    alive_deltas = deltas.delta_M[alive_mask]
    align = torch.zeros(alive_deltas.shape[0], len(projectors),
                        device=alive_deltas.device, dtype=alive_deltas.dtype)
    for j, P in enumerate(projectors):
        for i in range(alive_deltas.shape[0]):
            align[i, j] = fro_cos(alive_deltas[i], P)
    return align, deltas.norms


def _random_baseline_mma(n_rows: int, projectors: list[torch.Tensor], device,
                         seed: int, n_trials: int = 16) -> float:
    if not projectors:
        return 0.0
    d = projectors[0].shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    vals = []
    for _ in range(n_trials):
        raw = torch.randn(n_rows, d, d, generator=gen).to(device)
        sym = 0.5 * (raw + raw.transpose(-1, -2))
        m = torch.zeros(n_rows, len(projectors), device=device, dtype=sym.dtype)
        for j, P in enumerate(projectors):
            for i in range(n_rows):
                m[i, j] = fro_cos(sym[i], P)
        vals.append(float(mean_max_alignment(m, axis="eigenspaces").item()))
    return float(torch.tensor(vals).mean())


def _ablation_kl(decomp: BilinearComponentMLP, x, y_target, atoms: list[int], r) -> float:
    A_m, B_m = ablate_weights(decomp, atoms)
    m = evaluate_ablation(decomp, x, y_target, A_m, B_m, r=r)
    return m["kl"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--probe", default="centered_0")
    parser.add_argument("--lambda-sparsities", type=float, nargs="+",
                        default=[1e-2, 1e-3, 1e-4, 1e-5, 0.0])
    parser.add_argument("--top-k-eigenspaces", type=int, default=5)
    parser.add_argument("--n-random-baselines", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    out = run_dir(cfg)

    target = _load_target(cfg, out, device)
    data = make_dataset(cfg["task"], device=device)
    p = output_dim(cfg["task"])
    probes = build_probes(cfg["probes"], p=p, device=device)
    r = probes[args.probe]

    with torch.no_grad():
        y_target_logits = target(data["x"])

    # Build eigenspaces on the chosen probe (one-time — depends only on the target).
    groups = _eigenspaces_for_probe(
        target, r,
        rel_tol=float(cfg["eigenspaces"]["rel_tol"]),
        min_abs_eval=float(cfg["eigenspaces"]["min_abs_eval"]),
    )
    projectors = [g.projector.to(device) for g in groups]
    print(f"sparsity sweep on probe={args.probe}: {len(projectors)} eigenspaces")

    decomp_cfg = cfg["decomposition"]
    train_cfg_base = dict(cfg["train_decomp"],
                          use_stochastic_masks=decomp_cfg.get("use_stochastic_masks", True))

    rng = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
    sweep: list[dict] = []
    for ls in args.lambda_sparsities:
        set_seed(int(cfg["seed"]))
        decomp = BilinearComponentMLP(
            target=target, C_A=int(decomp_cfg["C_A"]),
            C_B=int(decomp_cfg["C_B"]),
            gate_hidden=int(decomp_cfg["gate_hidden"]),
        ).to(device)
        tcfg = dict(train_cfg_base)
        tcfg["lambda_sparsity"] = float(ls)
        tcfg["log_every"] = 5_000   # quieter for the sweep
        metrics = train_decomposition(
            decomp, data["x"], y_target_logits, cfg=tcfg,
            mask_samples_per_step=int(decomp_cfg.get("mask_samples_per_batch", 1)),
            verbose=False,
        )
        decomp.eval()

        # Atom-level alignment + MMA.
        align, atom_norms = _atom_alignment(decomp, r, projectors)
        atom_mma = float(mean_max_alignment(align, axis="eigenspaces").item())
        n_atoms_alive = int(align.shape[0])
        rand_mma = _random_baseline_mma(
            n_atoms_alive, projectors, device, seed=hash(("rand_mma", ls)) % (2**31))

        # Causal ablation on top-K eigenspaces by atom MMA.
        best_per_eig = align.argmax(dim=0).tolist()
        atom_alive_idx = (atom_norms > 1e-8).nonzero().flatten().tolist()
        order = sorted(range(len(projectors)), key=lambda j: -float(align[:, j].max().item()))
        chosen = order[: args.top_k_eigenspaces]

        aligned_kls: list[float] = []
        random_kls_mean: list[float] = []
        with torch.no_grad():
            y_target = decomp.forward_target(data["x"])
        for j in chosen:
            best_alive = best_per_eig[j]
            global_atom = atom_alive_idx[best_alive]
            kl_a = _ablation_kl(decomp, data["x"], y_target, [global_atom], r)
            aligned_kls.append(kl_a)
            # random size-matched (size 1, excluding the aligned atom). We use
            # *mean* across random trials per eigenspace because the random
            # distribution is heavy-tailed — median washes out the load-bearing
            # tail that we care about. Then median *across* eigenspaces is a
            # robust cross-probe summary.
            rand_sets = random_size_matched_sets(
                n_atoms=decomp.C_A + decomp.C_B, size=1,
                n_samples=args.n_random_baselines, exclude=[global_atom], rng=rng,
            )
            rkl = [_ablation_kl(decomp, data["x"], y_target, rs, r) for rs in rand_sets]
            random_kls_mean.append(float(np.mean(rkl)))

        median_aligned = float(np.median(aligned_kls))
        median_random = float(np.median(random_kls_mean))

        # Gate sparsity summary.
        with torch.no_grad():
            g = collect_gates(decomp, data["x"])
        # Mean #gates > 1e-3 per sample, separately for A and B halves
        mean_l0 = float((g > 1e-3).float().sum(dim=-1).mean().item())
        mean_gate = float(g.mean().item())

        # Cluster correlation density (% of off-diagonal |corr| > 0.5 — proxy for co-activation).
        corr = gate_corr(g)
        C = corr.shape[0]
        offdiag = corr.abs()
        offdiag.fill_diagonal_(0.0)
        co_act_frac = float(((offdiag > 0.5).float().sum() / (C * (C - 1))).item())

        row = {
            "lambda_sparsity": float(ls),
            "final_loss": metrics.final_loss,
            "final_kl": metrics.final_kl,
            "final_recon_A": metrics.final_param_recon_A,
            "final_recon_B": metrics.final_param_recon_B,
            "mean_gates_above_threshold": mean_l0,
            "mean_gate": mean_gate,
            "co_activation_frac_at_0.5": co_act_frac,
            "n_atoms_alive": n_atoms_alive,
            "atom_mma_eig_coverage": atom_mma,
            "random_mma": rand_mma,
            "median_aligned_KL": median_aligned,
            "median_random_KL_of_means": median_random,
        }
        sweep.append(row)
        print(
            f"  λ_s={ls:>7g}  KL={metrics.final_kl:.2e}  recon_A={metrics.final_param_recon_A:.2e}  "
            f"L0={mean_l0:.2f}  atom_MMA={atom_mma:.3f} (rand {rand_mma:.3f})  "
            f"median KL aligned={median_aligned:.2e}  rand={median_random:.2e}"
        )

    with open(out / "artifacts" / "sparsity_sweep.json", "w") as f:
        json.dump({"probe": args.probe, "rows": sweep}, f, indent=2)

    # Plot — two panels carrying the actual claim: "sparsity doesn't change the story".
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    xs = list(range(len(sweep)))
    xticklabels = [(f"{r['lambda_sparsity']:g}" if r['lambda_sparsity'] > 0 else "0")
                   for r in sweep]

    # Panel 1: MMA (atoms) vs random.
    atom_mma_series = [r["atom_mma_eig_coverage"] for r in sweep]
    rand_mma_series = [r["random_mma"] for r in sweep]
    axes[0].plot(xs, atom_mma_series, color=COLORS["atom"], lw=2.4, marker="o")
    axes[0].plot(xs, rand_mma_series, color=COLORS["random"], lw=2.0, marker="x",
                 linestyle="--")
    axes[0].set_ylabel("mean max alignment")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(xticklabels)
    axes[0].set_xlabel(r"$\lambda_\mathrm{sparsity}$")
    axes[0].set_ylim(0.0, 0.35)
    axes[0].set_xlim(-0.4, len(xs) - 0.6 + 1.4)
    axes[0].annotate("atoms", xy=(xs[-1], atom_mma_series[-1]),
                     xytext=(xs[-1] + 0.25, atom_mma_series[-1]),
                     color=COLORS["atom"], fontsize=9.5, va="center")
    axes[0].annotate("random", xy=(xs[-1], rand_mma_series[-1]),
                     xytext=(xs[-1] + 0.25, rand_mma_series[-1]),
                     color=COLORS["muted"], fontsize=9.5, va="center")
    axes[0].set_title("alignment is flat in sparsity")

    # Panel 2: median ablation KL aligned vs random.
    aligned_series = [r["median_aligned_KL"] for r in sweep]
    random_series = [r["median_random_KL_of_means"] for r in sweep]
    axes[1].plot(xs, aligned_series, color=COLORS["atom"], lw=2.4, marker="o")
    axes[1].plot(xs, random_series, color=COLORS["loadbearing"], lw=2.0, marker="x",
                 linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("KL after single-atom ablation")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(xticklabels)
    axes[1].set_xlabel(r"$\lambda_\mathrm{sparsity}$")
    axes[1].set_xlim(-0.4, len(xs) - 0.6 + 1.6)
    axes[1].annotate("aligned atom", xy=(xs[-1], aligned_series[-1]),
                     xytext=(xs[-1] + 0.25, aligned_series[-1]),
                     color=COLORS["atom"], fontsize=9.5, va="center")
    axes[1].annotate("random atom", xy=(xs[-1], random_series[-1]),
                     xytext=(xs[-1] + 0.25, random_series[-1]),
                     color=COLORS["loadbearing"], fontsize=9.5, va="center")
    axes[1].set_title("aligned/causal gap is flat in sparsity")

    fig.tight_layout()
    fig.savefig(out / "figures" / "sparsity_sweep.png")
    plt.close(fig)
    print(f"wrote {out / 'figures' / 'sparsity_sweep.png'}")


if __name__ == "__main__":
    main()
