"""Threshold-graph clustering on the atom co-activation correlation matrix.

Given the per-sample gate matrix $g \\in \\mathbb{R}^{N \\times (C_A + C_B)}$,
we compute the Pearson correlation between gate columns and form an undirected
graph whose edges are gate-pair correlations exceeding a threshold $\\tau$.
Connected components of this graph become clusters (spec § 5.8).

The threshold is a tunable knob: low $\\tau$ merges everything into one giant
cluster, high $\\tau$ leaves every atom as a singleton. Spec § 5.8 says: sweep
$\\tau$, report sensitivity, pick something defensible (we use the config's
`selected_threshold`).
"""

from __future__ import annotations

import torch


def gate_corr(g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation matrix between atom gates over the dataset.

    g: [N, C]. Output: [C, C] with diagonal 1.

    Standardised over the N axis. A column with zero variance (a dead atom)
    produces NaN correlations — we replace those with 0 (treating dead atoms
    as uncorrelated with everything).
    """
    g_centered = g - g.mean(dim=0, keepdim=True)
    std = g_centered.std(dim=0, unbiased=False, keepdim=True)
    g_normed = g_centered / (std + eps)
    n = g.shape[0]
    corr = (g_normed.T @ g_normed) / n
    # any column with std≈0 → entire row/col is ~0 already from normalization;
    # the diagonal might be 0 there too. Force diagonal = 1 for non-dead atoms
    # by clamping any near-zero std column's diagonal cleanly:
    dead = (std.squeeze(0) < eps)
    if dead.any():
        # zero out their rows/cols (already near-zero) and clamp diag separately.
        corr[dead, :] = 0.0
        corr[:, dead] = 0.0
    # set diagonal to 1 for live atoms, 0 for dead — dead atoms shouldn't merge.
    diag = torch.where(dead, torch.zeros_like(std.squeeze(0)), torch.ones_like(std.squeeze(0)))
    corr = corr - torch.diag(torch.diagonal(corr)) + torch.diag(diag)
    return corr


def threshold_components(corr: torch.Tensor, threshold: float) -> list[list[int]]:
    """Connected components of the graph $\\{(i, j) : |\\mathrm{corr}_{ij}| > \\tau\\}$.

    Uses signed correlations' absolute value: negatively-correlated gates are
    treated as still part of the same mechanism (they fire on disjoint inputs
    but anti-correlate). Diagonal is ignored. Returns a list of clusters, each
    a list of atom indices (sorted ascending). Singletons are included.
    """
    C = corr.shape[0]
    adj = (corr.abs() > threshold).clone()
    adj.fill_diagonal_(False)
    adj = adj.cpu().numpy()

    visited = [False] * C
    clusters: list[list[int]] = []
    for start in range(C):
        if visited[start]:
            continue
        # BFS
        stack = [start]
        comp: list[int] = []
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            comp.append(u)
            neighbours = adj[u].nonzero()[0].tolist()
            for v in neighbours:
                if not visited[v]:
                    stack.append(v)
        clusters.append(sorted(comp))
    return clusters


def cluster_size_summary(clusters: list[list[int]]) -> dict[str, float | int]:
    """Quick stats over a clustering — n_clusters, sizes, max/median/mean."""
    sizes = sorted((len(c) for c in clusters), reverse=True)
    if not sizes:
        return {"n_clusters": 0, "max_size": 0, "median_size": 0.0, "mean_size": 0.0,
                "n_singletons": 0, "n_nontrivial": 0}
    n = len(sizes)
    return {
        "n_clusters": n,
        "max_size": sizes[0],
        "median_size": float(sizes[n // 2]),
        "mean_size": sum(sizes) / n,
        "n_singletons": sum(1 for s in sizes if s == 1),
        "n_nontrivial": sum(1 for s in sizes if s >= 2),
    }
