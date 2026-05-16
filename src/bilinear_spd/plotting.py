"""Matplotlib style and figure helpers.

Style follows docs/02 §6: clean white background, no top/right spines, light grid,
small Goodfire-ish palette. All figures save at 300 dpi by default.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

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
