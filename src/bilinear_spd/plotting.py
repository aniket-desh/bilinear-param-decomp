"""Matplotlib style and figure helpers.

Style: clean white background, no top/right spines, light grid, small palette.
All figures save at 300 dpi by default.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "gray": "#9D9DA1",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_training_curves(
    history: list[dict],
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """Three stacked panels: loss, accuracy, logit margin vs step."""
    apply_style()
    steps = [h["step"] for h in history]
    loss = [h["loss"] for h in history]
    acc = [h["accuracy"] for h in history]
    margin = [h["logit_margin"] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
    axes[0].plot(steps, loss, color=COLORS["blue"])
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_yscale("log")

    axes[1].plot(steps, acc, color=COLORS["green"])
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(1.0, color=COLORS["gray"], linewidth=0.8, linestyle="--")

    axes[2].plot(steps, margin, color=COLORS["orange"])
    axes[2].set_ylabel("logit margin")
    axes[2].set_xlabel("step")
    axes[2].axhline(0.0, color=COLORS["gray"], linewidth=0.8, linestyle="--")

    if title is not None:
        fig.suptitle(title, y=1.0)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --- Day 2.5: grokking visualization ---


def plot_grokking_curves(
    history: list[dict],
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """4-panel grokking diagnostic.

    Top: loss (log) + train accuracy on a twin axis.
    2nd: parameter L2 norm — should rise during memorization, fall during grokking.
    3rd: degenerate-group count at multiple τ — flat-zero during memorization,
         steps up during grokking.
    4th: top |λ| of the tracked M_r — typically grows then plateaus.
    """
    apply_style()
    grok_rows = [h for h in history if "param_l2_norm" in h]
    if not grok_rows:
        raise ValueError("history has no grok metrics; pass probe_for_grok to train_bilinear")

    steps = [h["step"] for h in grok_rows]
    loss = [h["loss"] for h in grok_rows]
    acc = [h["accuracy"] for h in grok_rows]
    pn = [h["param_l2_norm"] for h in grok_rows]
    top_lam = [h["top_abs_eval"] for h in grok_rows]
    deg_keys = sorted(k for k in grok_rows[0] if k.startswith("n_degenerate_"))

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 9.5), sharex=True)

    axes[0].plot(steps, loss, color=COLORS["blue"], label="loss")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("cross-entropy")
    twin = axes[0].twinx()
    twin.plot(steps, acc, color=COLORS["green"], label="accuracy")
    twin.set_ylim(-0.05, 1.05)
    twin.set_ylabel("accuracy", color=COLORS["green"])
    twin.tick_params(axis="y", colors=COLORS["green"])
    twin.spines["top"].set_visible(False)
    twin.grid(False)

    axes[1].plot(steps, pn, color=COLORS["purple"])
    axes[1].set_ylabel(r"$\|\theta\|^2$")

    deg_palette = [COLORS["orange"], COLORS["red"], COLORS["blue"]]
    for k, dk in enumerate(deg_keys):
        vals = [h[dk] for h in grok_rows]
        tau = dk.split("_")[-1]
        axes[2].plot(steps, vals, color=deg_palette[k % len(deg_palette)],
                     label=f"τ = {tau}")
    axes[2].set_ylabel("# degenerate groups")
    axes[2].legend(loc="best", frameon=False, fontsize=9)

    axes[3].plot(steps, top_lam, color=COLORS["green"])
    axes[3].set_ylabel(r"top $|\lambda|$")
    axes[3].set_xlabel("step")

    if title is not None:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --- Day 3: decomposition training ---


def plot_decomposition_curves(
    history: list[dict],
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """4-panel decomposition diagnostic.

    Top: total loss + KL behavior loss (twin axis).
    2nd: relative parameter reconstruction error for A and B.
    3rd: gate L0 (count of gates > 1e-3) for A and B, plus C_A / C_B reference lines.
    4th: mean gate value over the batch (the "usage frequency").
    """
    apply_style()
    if not history:
        raise ValueError("history is empty")
    steps = [h["step"] for h in history]
    loss = [h["loss"] for h in history]
    kl = [h["kl"] for h in history]
    recon_A = [h["recon_A"] for h in history]
    recon_B = [h["recon_B"] for h in history]
    l0_A = [h["l0_A"] for h in history]
    l0_B = [h["l0_B"] for h in history]
    mg_A = [h["mean_gate_A"] for h in history]
    mg_B = [h["mean_gate_B"] for h in history]

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 9.5), sharex=True)

    axes[0].plot(steps, loss, color=COLORS["blue"], label="total loss")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("total loss")
    twin = axes[0].twinx()
    twin.plot(steps, kl, color=COLORS["orange"], label="KL")
    twin.set_yscale("log")
    twin.set_ylabel("KL behavior", color=COLORS["orange"])
    twin.tick_params(axis="y", colors=COLORS["orange"])
    twin.spines["top"].set_visible(False)
    twin.grid(False)

    axes[1].plot(steps, recon_A, color=COLORS["blue"], label="A")
    axes[1].plot(steps, recon_B, color=COLORS["green"], label="B")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\|\Delta\|^2 / \|W\|^2$")
    axes[1].legend(loc="best", frameon=False, fontsize=9)

    axes[2].plot(steps, l0_A, color=COLORS["blue"], label="L0 (A)")
    axes[2].plot(steps, l0_B, color=COLORS["green"], label="L0 (B)")
    axes[2].set_ylabel(r"# gates > $10^{-3}$")
    axes[2].legend(loc="best", frameon=False, fontsize=9)

    axes[3].plot(steps, mg_A, color=COLORS["blue"], label="mean g_A")
    axes[3].plot(steps, mg_B, color=COLORS["green"], label="mean g_B")
    axes[3].set_ylabel("mean gate")
    axes[3].set_xlabel("step")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc="best", frameon=False, fontsize=9)

    if title is not None:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --- Day 4: alignment heatmaps ---


def plot_alignment_heatmaps(
    align_atoms: torch.Tensor,
    align_clusters: torch.Tensor,
    out_path: str | Path,
    title: str | None = None,
    summary_text: str | None = None,
) -> None:
    """Side-by-side heatmaps: atoms × eigenspaces and clusters × eigenspaces.

    Rows (atoms / clusters) are sorted by max alignment descending so the
    "obvious" diagonal — well-aligned atoms next to their best eigenspace —
    shows up at the top. Columns (eigenspaces) are in mean-|λ| descending
    order, matching the spectrum plot's convention.
    """
    apply_style()
    A = align_atoms.detach().cpu().numpy()
    Cc = align_clusters.detach().cpu().numpy()

    # sort rows by max alignment desc
    a_idx = np.argsort(-A.max(axis=1))
    c_idx = np.argsort(-Cc.max(axis=1))
    A_sorted = A[a_idx]
    C_sorted = Cc[c_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    vmax = float(max(A_sorted.max(), C_sorted.max(), 1.0))
    for ax, M, label in zip(axes, [A_sorted, C_sorted], ["atoms", "clusters"]):
        im = ax.imshow(M, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xlabel("eigenspace (|λ| desc)")
        ax.set_ylabel(f"{label} (max-align desc)")
        ax.set_title(f"{label} × eigenspaces")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="|cos|")

    if summary_text is not None:
        fig.text(0.5, -0.02, summary_text, ha="center", va="top", fontsize=10)

    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_alignment_bars(
    atom_mma_by_probe: dict,
    cluster_mma_by_probe: dict,
    out_path: str | Path,
    title: str | None = None,
    baseline_by_probe: dict | None = None,
) -> None:
    """Per-probe paired bars: atom MMA vs cluster MMA vs random-baseline MMA.

    The headline H1/H2/H3 comparison: do atoms/clusters meaningfully beat a
    random size-matched "decomposition" at covering eigenspaces? Each
    baseline entry should be a dict with `eig_coverage_mean` (+ optional
    `eig_coverage_std` for the errorbar).
    """
    apply_style()
    probes = list(atom_mma_by_probe.keys())
    x = np.arange(len(probes))
    atom_vals = [atom_mma_by_probe[p] for p in probes]
    cluster_vals = [cluster_mma_by_probe[p] for p in probes]

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(probes), 4.2))
    if baseline_by_probe is not None:
        rand_mean = [baseline_by_probe[p]["eig_coverage_mean"] for p in probes]
        rand_std = [baseline_by_probe[p].get("eig_coverage_std", 0.0) for p in probes]
        width = 0.26
        ax.bar(x - width, atom_vals, width, color=COLORS["blue"], label="atoms")
        ax.bar(x, cluster_vals, width, color=COLORS["orange"], label="clusters")
        ax.bar(x + width, rand_mean, width, yerr=rand_std, color=COLORS["gray"],
               capsize=3, label="random size-matched (mean ± std)")
    else:
        width = 0.36
        ax.bar(x - width / 2, atom_vals, width, color=COLORS["blue"], label="atoms")
        ax.bar(x + width / 2, cluster_vals, width, color=COLORS["orange"], label="clusters")
    ax.set_xticks(x)
    ax.set_xticklabels(probes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean max alignment (eigenspace coverage)")
    ax.set_ylim(0.0, max(1.0, max(atom_vals + cluster_vals) * 1.1))
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --- Day 5: ablation comparison ---


def plot_ablation_bars(
    rows: list[dict],
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """Bar chart of $\\Delta$KL for {aligned cluster, random size-matched, low-align}.

    `rows` is a list of dicts with keys:
      eigenspace (str), kl_aligned, kl_random_mean, kl_random_std, kl_low_align.
    Sorted by aligned-cluster KL descending.
    """
    apply_style()
    rows = sorted(rows, key=lambda r: -r["kl_aligned"])
    labels = [r["eigenspace"] for r in rows]
    aligned = [r["kl_aligned"] for r in rows]
    rand_mean = [r["kl_random_mean"] for r in rows]
    rand_std = [r["kl_random_std"] for r in rows]
    low = [r["kl_low_align"] for r in rows]

    x = np.arange(len(rows))
    width = 0.28
    fig, ax = plt.subplots(figsize=(0.7 * len(rows) + 2.5, 4.5))
    ax.bar(x - width, aligned, width, color=COLORS["orange"], label="aligned cluster")
    ax.bar(x, rand_mean, width, yerr=rand_std, color=COLORS["gray"],
           capsize=3, label="random size-matched (mean ± std)")
    ax.bar(x + width, low, width, color=COLORS["blue"], label="low-alignment cluster")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$\Delta$ KL (ablated vs full)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --- Day 2: functional analysis ---


def _group_palette(n: int) -> list[str]:
    """Cycle a small palette across groups so degeneracies are visually distinct."""
    base = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["red"],
        COLORS["purple"],
    ]
    return [base[i % len(base)] for i in range(n)]


def plot_spectrum(
    evals: torch.Tensor,
    groups,
    out_path: str | Path,
    title: str | None = None,
    min_abs_eval: float = 1e-6,
) -> None:
    """Stem plot of eigenvalues ordered by |lambda| desc, colored by group.

    Annotates degenerate groups (rank >= 2) with their mean lambda and rank.
    """
    apply_style()
    evals_np = evals.detach().cpu().numpy()
    d = len(evals_np)
    # `evals` already comes |lambda|-sorted from eigendecompose_M
    idx_to_group: dict[int, int] = {}
    for gi, g in enumerate(groups):
        for i in g.indices:
            idx_to_group[i] = gi
    palette = _group_palette(len(groups))
    colors = [palette[idx_to_group[i]] if i in idx_to_group else COLORS["gray"]
              for i in range(d)]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(d)
    ax.bar(x, np.abs(evals_np), color=colors, width=0.7, linewidth=0)
    if min_abs_eval > 0:
        ax.axhline(min_abs_eval, color=COLORS["gray"], linewidth=0.6, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index (|λ| desc)")
    ax.set_ylabel("|λ|")
    if title is not None:
        ax.set_title(title)

    # summary box rather than per-group annotations (which collide when many
    # degenerate groups sit at similar |λ|).
    degenerate = [g for g in groups if g.rank >= 2]
    if degenerate:
        max_rank = max(g.rank for g in degenerate)
        total_vec = sum(g.rank for g in degenerate)
        box_text = (
            f"groups: {len(groups)}\n"
            f"degenerate (rank ≥ 2): {len(degenerate)}\n"
            f"max group rank: {max_rank}\n"
            f"eigvecs in degenerate groups: {total_vec}"
        )
    else:
        box_text = f"groups: {len(groups)}\nno degenerate groups"
    ax.text(
        0.99,
        0.97,
        box_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": COLORS["gray"], "alpha": 0.9, "boxstyle": "round,pad=0.4"},
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_eigenvectors(
    evals: torch.Tensor,
    evecs: torch.Tensor,
    out_path: str | Path,
    k: int = 8,
    slot_split: int | None = None,
    title: str | None = None,
) -> None:
    """Heatmap of the top-k eigenvectors as columns.

    Eigenvectors are sorted by |lambda| desc (matching `eigendecompose_M`).
    If `slot_split` is given, draws a horizontal dashed line at that row to
    separate the a-slot and b-slot of a concat-onehot input.
    """
    apply_style()
    k = min(k, evecs.shape[1])
    Q = evecs[:, :k].detach().cpu().numpy()
    lam = evals[:k].detach().cpu().numpy()

    vmax = float(np.max(np.abs(Q)))
    fig, ax = plt.subplots(figsize=(0.55 * k + 2.0, 0.16 * Q.shape[0] + 1.5))
    im = ax.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"λ={l:+.2f}" for l in lam], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("eigenvector (|λ| desc)")
    ax.set_ylabel("input coordinate")
    if slot_split is not None:
        ax.axhline(slot_split - 0.5, color="black", linewidth=0.7, linestyle="--")
        # tick labels split by slot
        d = Q.shape[0]
        yticks = list(range(0, slot_split, max(1, slot_split // 4))) + list(
            range(slot_split, d, max(1, (d - slot_split) // 4))
        )
        ax.set_yticks(yticks)
    if title is not None:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="eigenvector value")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
