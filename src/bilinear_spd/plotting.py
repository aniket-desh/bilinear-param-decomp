"""Matplotlib style and figure helpers.

Publication-style plotting for the post: clean white background, restrained
sequential / muted-categorical palettes, claim-driven titles by default,
direct line labels at the right edge instead of legends where possible.

Semantic color convention used across the post — pick by *meaning*, not by
default cycle:

- `aligned`  / `atom`        blue  — alignment-selected (geometric) objects
- `cluster`                   amber — cluster aggregations
- `loadbearing` / `random`    orange-red / gray — causal load-bearing / chance baseline
- `shard`                     deep red — ablation flips on the (a, b) grid
- `muted` / `ink` / `grid`    neutral text/structure colors

Use `COLORS["atom"]` / `COLORS["loadbearing"]` etc. in new figures so the
narrative stays color-consistent across the post.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Semantic palette. The keys are intent-named so the figure code reads as
# "color the aligned-atom line blue", not "color this thing #4C78A8".
COLORS = {
    "ink": "#1F2933",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "atom": "#2563A6",          # blue — alignment / geometric selection
    "aligned": "#2563A6",       # alias
    "cluster": "#D97706",       # amber
    "random": "#A3A3A3",        # gray — chance / size-matched baseline
    "loadbearing": "#C2410C",   # orange-red — norm-selected / causal
    "low": "#64748B",           # muted blue-gray — low-alignment control
    "shard": "#B91C1C",         # red — ablation flips
    "accent": "#2563A6",        # generic accent (blue)
    "accent_dark": "#0F2A4A",   # darker accent (for largest degenerate group)
    "noise": "#D1D5DB",         # very light gray — noise floor / singletons
    # legacy aliases so older figure code keeps working
    "blue": "#2563A6",
    "orange": "#D97706",
    "green": "#0F766E",
    "red": "#B91C1C",
    "purple": "#7C3AED",
    "gray": "#A3A3A3",
}


def apply_style() -> None:
    """Publication-style rcParams. Idempotent; safe to call from every plot fn."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


# --- generic helpers ---


def despine(ax) -> None:
    """Hide top + right spines (rcParams sets this globally, but stays here for in-figure overrides)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def direct_label_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    color: str,
    *,
    dx: float = 0.5,
    fontsize: float = 9.5,
    va: str = "center",
) -> None:
    """Annotate a line at its right edge instead of via a legend."""
    ax.annotate(
        label,
        xy=(float(x[-1]), float(y[-1])),
        xytext=(float(x[-1]) + dx, float(y[-1])),
        ha="left", va=va,
        color=color, fontsize=fontsize,
        annotation_clip=False,
    )


# --- Day 1: training curves ---


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
    axes[0].plot(steps, loss, color=COLORS["atom"])
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_yscale("log")

    axes[1].plot(steps, acc, color=COLORS["green"])
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(1.0, color=COLORS["muted"], linewidth=0.8, linestyle="--")

    axes[2].plot(steps, margin, color=COLORS["cluster"])
    axes[2].set_ylabel("logit margin")
    axes[2].set_xlabel("step")
    axes[2].axhline(0.0, color=COLORS["muted"], linewidth=0.8, linestyle="--")

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
    """4-panel grokking diagnostic."""
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

    axes[0].plot(steps, loss, color=COLORS["atom"], label="loss")
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

    deg_palette = [COLORS["cluster"], COLORS["loadbearing"], COLORS["atom"]]
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
    """4-panel decomposition diagnostic."""
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

    axes[0].plot(steps, loss, color=COLORS["atom"], label="total loss")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("total loss")
    twin = axes[0].twinx()
    twin.plot(steps, kl, color=COLORS["cluster"], label="KL")
    twin.set_yscale("log")
    twin.set_ylabel("KL behavior", color=COLORS["cluster"])
    twin.tick_params(axis="y", colors=COLORS["cluster"])
    twin.spines["top"].set_visible(False)
    twin.grid(False)

    axes[1].plot(steps, recon_A, color=COLORS["atom"], label="A")
    axes[1].plot(steps, recon_B, color=COLORS["green"], label="B")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\|\Delta\|^2 / \|W\|^2$")
    axes[1].legend(loc="best", frameon=False, fontsize=9)

    axes[2].plot(steps, l0_A, color=COLORS["atom"], label="L0 (A)")
    axes[2].plot(steps, l0_B, color=COLORS["green"], label="L0 (B)")
    axes[2].set_ylabel(r"# gates > $10^{-3}$")
    axes[2].legend(loc="best", frameon=False, fontsize=9)

    axes[3].plot(steps, mg_A, color=COLORS["atom"], label="mean g_A")
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


# --- Day 4: alignment ---


def plot_alignment_bars(
    atom_mma_by_probe: dict,
    cluster_mma_by_probe: dict,
    out_path: str | Path,
    title: str | None = None,
    baseline_by_probe: dict | None = None,
    y_max: float = 0.32,
) -> None:
    """Per-probe paired bars: atom MMA vs cluster MMA vs random-baseline MMA.

    y-axis is capped at `y_max` (default 0.32) — the H1/H2/H3 story is about
    the *separation from random*, not closeness to 1, so we don't visually
    compress real signal into the bottom 20% of the canvas. Direct labels
    inside the first probe's bars replace a legend.
    """
    apply_style()
    probes = list(atom_mma_by_probe.keys())
    x = np.arange(len(probes))
    atom_vals = [atom_mma_by_probe[p] for p in probes]
    cluster_vals = [cluster_mma_by_probe[p] for p in probes]

    pretty_probes = [_pretty_probe(p) for p in probes]

    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * len(probes), 3.6))
    if baseline_by_probe is not None:
        rand_mean = [baseline_by_probe[p]["eig_coverage_mean"] for p in probes]
        rand_std = [baseline_by_probe[p].get("eig_coverage_std", 0.0) for p in probes]
        width = 0.26
        b_a = ax.bar(x - width, atom_vals, width, color=COLORS["atom"], label="atoms")
        b_c = ax.bar(x, cluster_vals, width, color=COLORS["cluster"], label="clusters")
        b_r = ax.bar(x + width, rand_mean, width, yerr=rand_std, color=COLORS["random"],
                     capsize=3, label="random")
        # Compact legend top-right of the canvas — y-axis has plenty of room.
        ax.legend([b_a, b_c, b_r], ["atoms", "clusters", "random"],
                  loc="upper right", frameon=False, fontsize=9)
    else:
        width = 0.36
        ax.bar(x - width / 2, atom_vals, width, color=COLORS["atom"], label="atoms")
        ax.bar(x + width / 2, cluster_vals, width, color=COLORS["cluster"], label="clusters")
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_probes, rotation=0, fontsize=9)
    ax.set_ylabel("best Frobenius cosine (mean over eigenspaces)")
    ax.set_ylim(0.0, y_max)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _pretty_probe(name: str) -> str:
    if name.startswith("centered_"):
        return f"centered {name.split('_')[1]}"
    if name.startswith("pairwise_"):
        parts = name.split("_")
        return f"{parts[1]} vs {parts[2]}"
    return name


def plot_alignment_coverage(
    atom_best: torch.Tensor,
    cluster_best: torch.Tensor,
    random_per_eigenspace: dict,
    out_path: str | Path,
    title: str | None = None,
    y_max: float = 0.35,
) -> None:
    """Sorted best-match alignment per eigenspace (atoms / clusters / random ±2σ).

    Eigenspaces are sorted by atom best alignment descending so the figure
    reads left-to-right as "how well are the easiest-to-recover eigenspaces
    actually recovered?" Random baseline is a 2σ band, atoms and clusters are
    line plots. If both lines sit inside or barely above the random band, the
    alignment signal is small. If atoms (blue) climbs but clusters (amber)
    doesn't, H2 is false. If neither approaches y=1, H1 is false.

    `random_per_eigenspace` must contain `mean` and `std` arrays of length
    n_eigenspaces (same indexing as atom_best/cluster_best).
    """
    apply_style()
    a = atom_best.detach().cpu().numpy()
    c = cluster_best.detach().cpu().numpy()
    rm = np.asarray(random_per_eigenspace["mean"], dtype=float)
    rs = np.asarray(random_per_eigenspace["std"], dtype=float)

    order = np.argsort(-a)
    a, c, rm, rs = a[order], c[order], rm[order], rs[order]
    x = np.arange(len(a))

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.fill_between(x, rm - 2 * rs, rm + 2 * rs,
                    color=COLORS["random"], alpha=0.22, linewidth=0)
    ax.plot(x, rm, color=COLORS["random"], lw=1.0, linestyle="--")
    # cluster underneath (thick amber), atom on top (thinner blue) — if the two
    # curves are visually identical (typical in this regime) the amber halo
    # still peeks out at the edges.
    ax.plot(x, c, color=COLORS["cluster"], lw=4.0, alpha=0.55)
    ax.plot(x, a, color=COLORS["atom"], lw=2.0)

    # Direct labels at right edge — stack vertically so atom/cluster labels
    # don't collide even when the two curves end at the same value.
    a_label_y = a[-1]
    c_label_y = a_label_y - 0.020
    r_label_y = max(rm[-1] + 0.012, a_label_y + 0.020)
    ax.annotate("atoms", xy=(x[-1], a[-1]),
                xytext=(x[-1] + 0.6, a_label_y), color=COLORS["atom"],
                fontsize=9.5, va="center", annotation_clip=False)
    ax.annotate("clusters", xy=(x[-1], c[-1]),
                xytext=(x[-1] + 0.6, c_label_y), color=COLORS["cluster"],
                fontsize=9.5, va="center", annotation_clip=False)
    ax.annotate("random ±2σ", xy=(x[-1], rm[-1]),
                xytext=(x[-1] + 0.6, r_label_y), color=COLORS["muted"],
                fontsize=9.5, va="center", annotation_clip=False)

    ax.set_xlim(-0.5, x[-1] + 9)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Functional eigenspace, sorted by best atom alignment")
    ax.set_ylabel("Best Frobenius cosine")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_alignment_vs_ablation(
    atom_best_alignment: torch.Tensor,
    atom_ablation_kl: torch.Tensor,
    atom_norms: torch.Tensor,
    out_path: str | Path,
    title: str | None = None,
    kl_floor: float = 1e-8,
) -> None:
    """Scatter: per-atom max eigenspace alignment vs single-atom ablation KL.

    Color = $\\log_{10} \\|\\Delta M_r\\|_F$. Load-bearing atoms (high KL when
    ablated) cluster at the top of the y-axis; aligned atoms (high
    Frobenius cosine to some eigenspace) cluster on the right. The post's
    main visual claim is that these two clusters barely overlap.
    """
    apply_style()
    x = atom_best_alignment.detach().cpu().numpy()
    y = atom_ablation_kl.detach().cpu().numpy().clip(min=kl_floor)
    c = np.log10(atom_norms.detach().cpu().numpy() + 1e-12)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    sc = ax.scatter(x, y, c=c, s=24, alpha=0.85, cmap="viridis", linewidths=0)
    ax.set_yscale("log")
    ax.set_xlabel("best eigenspace alignment")
    ax.set_ylabel("single-atom ablation KL")
    if title is not None:
        ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                      label=r"$\log_{10}\,\|\Delta M_r\|_F$")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_alignment_heatmaps(
    align_atoms: torch.Tensor,
    align_clusters: torch.Tensor,
    out_path: str | Path,
    title: str | None = None,
    summary_text: str | None = None,
    vmax: float = 0.30,
    max_rows: int = 40,
    max_cols: int = 30,
) -> None:
    """Side-by-side heatmaps capped at `vmax` and trimmed to the top region.

    Truncated to `max_rows × max_cols` (atoms × eigenspaces, both sorted
    so the "interesting" corner is at the top-left). Rendered as a
    diagnostic / appendix figure — the main-text alignment story uses
    `plot_alignment_coverage` and `plot_alignment_vs_ablation`.
    """
    apply_style()
    A = align_atoms.detach().cpu().numpy()
    Cc = align_clusters.detach().cpu().numpy()

    a_idx = np.argsort(-A.max(axis=1))[:max_rows]
    c_idx = np.argsort(-Cc.max(axis=1))[:max_rows]
    eig_idx_atoms = np.argsort(-A[a_idx].max(axis=0))[:max_cols]
    eig_idx_clusters = np.argsort(-Cc[c_idx].max(axis=0))[:max_cols]
    A_sorted = A[np.ix_(a_idx, eig_idx_atoms)]
    C_sorted = Cc[np.ix_(c_idx, eig_idx_clusters)]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
    for ax, M, label in zip(axes, [A_sorted, C_sorted], ["atoms", "clusters"]):
        im = ax.imshow(M, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xlabel("Functional eigenspace (top-30 by max alignment)")
        ax.set_ylabel(f"Rank-one {label} (top-40 by max alignment)")
        ax.set_title(f"{label} × eigenspaces")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                          label=f"Frobenius cosine (cap {vmax:g})")
        cb.outline.set_visible(False)

    if summary_text is not None:
        fig.text(0.5, -0.02, summary_text, ha="center", va="top", fontsize=10)
    if title is not None:
        fig.suptitle(title, y=1.02)
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

    Sorted by aligned-cluster KL descending. Same semantic palette as the
    rest of the post: blue = aligned/geometric, gray = random baseline,
    muted blue-gray = low-alignment control.
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
    fig, ax = plt.subplots(figsize=(0.7 * len(rows) + 2.5, 4.0))
    ax.bar(x - width, aligned, width, color=COLORS["aligned"], label="aligned")
    ax.bar(x, rand_mean, width, yerr=rand_std, color=COLORS["random"],
           capsize=3, label="random size-matched (mean ± std)")
    ax.bar(x + width, low, width, color=COLORS["low"], label="low-alignment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("KL after single-atom ablation")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --- Day 2: functional analysis ---


def _spectrum_colors(groups, d: int) -> list[str]:
    """Color rule: singletons -> noise gray; degenerate -> accent.

    Among the degenerate groups, the largest one *above the median group's
    mean |λ|* gets the dark accent — this avoids highlighting the M_r
    kernel (which can be the largest-rank "degenerate" group but is
    near-noise and uninteresting structurally).
    """
    if not groups:
        return [COLORS["noise"]] * d
    degenerate = [g for g in groups if g.rank >= 2]
    if degenerate:
        sorted_means = sorted(g.mean_abs_eval for g in groups)
        median_mean = sorted_means[len(sorted_means) // 2]
        above_median = [g for g in degenerate if g.mean_abs_eval >= median_mean]
        if above_median:
            biggest = max(above_median, key=lambda g: g.rank)
        else:
            biggest = None
    else:
        biggest = None
    idx_to_color: dict[int, str] = {}
    for g in groups:
        if g.rank == 1:
            color = COLORS["noise"]
        elif biggest is not None and g is biggest:
            color = COLORS["accent_dark"]
        else:
            color = COLORS["accent"]
        for i in g.indices:
            idx_to_color[i] = color
    return [idx_to_color.get(i, COLORS["noise"]) for i in range(d)]


def _group_palette(n: int) -> list[str]:
    """Backwards-compatible cycler (kept for any external caller)."""
    base = [COLORS["atom"], COLORS["cluster"], COLORS["green"], COLORS["red"], COLORS["purple"]]
    return [base[i % len(base)] for i in range(n)]


def plot_spectrum(
    evals: torch.Tensor,
    groups,
    out_path: str | Path,
    title: str | None = None,
    min_abs_eval: float = 1e-6,
) -> None:
    """Bar plot of |λ| ordered desc, colored by *type* of eigenspace (not by group id).

    Colors carry semantic load: noise gray = singleton, accent = degenerate
    group (rank ≥ 2), dark accent = the largest degenerate group. A subtle
    vertical bracket on the spectral cliff (between leading modes and the
    noise floor) makes the structure of the spectrum legible at a glance.
    """
    apply_style()
    evals_np = evals.detach().cpu().numpy()
    d = len(evals_np)
    colors = _spectrum_colors(groups, d)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(d)
    ax.bar(x, np.abs(evals_np), color=colors, width=0.8, linewidth=0)
    if min_abs_eval > 0:
        ax.axhline(min_abs_eval, color=COLORS["muted"], linewidth=0.6, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"Eigenvalue index ($|\lambda|$ descending)")
    ax.set_ylabel(r"$|\lambda|$")
    if title is not None:
        ax.set_title(title)

    # Annotate the spectral cliff. We look for the biggest log-jump in |λ|
    # within the kept (above-floor) eigenvalues.
    kept = np.where(np.abs(evals_np) >= max(min_abs_eval, 1e-12))[0]
    if len(kept) >= 5:
        log_ev = np.log10(np.abs(evals_np[kept]) + 1e-30)
        jumps = log_ev[:-1] - log_ev[1:]
        cliff_local = int(np.argmax(jumps[max(2, len(jumps) // 4):])) + max(2, len(jumps) // 4)
        cliff = int(kept[cliff_local + 1])
        if cliff > 5 and cliff < d - 2:
            ax.axvline(cliff - 0.5, color=COLORS["muted"], linewidth=0.7, linestyle="--")
            ax.text(cliff - 0.5, min_abs_eval * 8, " noise floor",
                    color=COLORS["muted"], fontsize=8.5, va="bottom", ha="left")

    # Compact summary box (top-right).
    degenerate = [g for g in groups if g.rank >= 2]
    if degenerate:
        box_text = (
            f"{len(groups)} eigenspaces, "
            f"{len(degenerate)} degenerate (max rank {max(g.rank for g in degenerate)})"
        )
    else:
        box_text = f"{len(groups)} eigenspaces, no degenerate groups"
    ax.text(0.99, 0.97, box_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": COLORS["grid"],
                  "alpha": 0.9, "boxstyle": "round,pad=0.4"})

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

    Cleaner publication style:
    - x-axis ticks are "mode 1..k" (eigenvalues live in the caption);
    - if `slot_split` is given, y-axis is annotated with "a-slot" / "b-slot"
      brackets rather than a generic "input coordinate" label;
    - symmetric RdBu_r scaling kept (sign-meaningful).
    """
    apply_style()
    k = min(k, evecs.shape[1])
    Q = evecs[:, :k].detach().cpu().numpy()

    vmax = float(np.max(np.abs(Q)))
    d = Q.shape[0]
    fig, ax = plt.subplots(figsize=(0.5 * k + 2.6, 0.135 * d + 1.5))
    im = ax.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"mode {i + 1}" for i in range(k)], rotation=0, fontsize=9)
    ax.set_xlabel(r"Eigenvector (top $k$ by $|\lambda|$)")
    if slot_split is not None:
        ax.axhline(slot_split - 0.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks([slot_split // 2, slot_split + (d - slot_split) // 2])
        ax.set_yticklabels(["a-slot", "b-slot"], rotation=90, va="center", fontsize=9.5)
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Input coordinate")
    if title is not None:
        ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="eigenvector value")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
