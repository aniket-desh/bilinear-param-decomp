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

    # annotate the leading degenerate groups
    annotated = 0
    for gi, g in enumerate(groups):
        if g.rank >= 2 and annotated < 4:
            label_x = max(g.indices)
            label_y = max(np.abs(evals_np[i]) for i in g.indices)
            ax.annotate(
                f"rank {g.rank}, μ|λ|={g.mean_abs_eval:.2f}",
                xy=(label_x, label_y),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=8,
                color=palette[gi],
            )
            annotated += 1

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
