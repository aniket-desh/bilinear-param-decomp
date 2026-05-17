"""Tests for `bilinear_spd.clustering`."""

import torch

from bilinear_spd.clustering import (
    cluster_size_summary,
    gate_corr,
    threshold_components,
)


def test_gate_corr_self_diagonal_is_one_for_live_atoms():
    g = torch.tensor([
        [0.1, 0.9, 0.5],
        [0.2, 0.8, 0.5],   # column 2 has zero variance → dead
        [0.3, 0.7, 0.5],
        [0.4, 0.6, 0.5],
    ])
    corr = gate_corr(g)
    assert torch.isclose(corr[0, 0], torch.tensor(1.0))
    assert torch.isclose(corr[1, 1], torch.tensor(1.0))
    # dead atom: diag = 0, off-diag = 0
    assert torch.isclose(corr[2, 2], torch.tensor(0.0))
    assert torch.isclose(corr[2, 0], torch.tensor(0.0))


def test_threshold_components_basic_topology():
    # 4 atoms: {0,1} perfectly correlated, {2,3} anticorrelated, {0,1}⊥{2,3}.
    # At threshold 0.5: 0-1 merge, 2-3 merge (|corr|=1 each), result is 2 clusters.
    g = torch.stack([
        torch.tensor([1.0, 1.0, 1.0, -1.0]),
        torch.tensor([2.0, 2.0, -1.0, 1.0]),
        torch.tensor([3.0, 3.0, 1.0, -1.0]),
        torch.tensor([4.0, 4.0, -1.0, 1.0]),
    ])
    corr = gate_corr(g)
    components = threshold_components(corr, threshold=0.5)
    assert {tuple(sorted(c)) for c in components} == {(0, 1), (2, 3)}

    # At threshold = exactly 1.0, the strict `> threshold` test cuts every edge,
    # so all atoms are singletons regardless of perfect correlations.
    singletons = threshold_components(corr, threshold=1.0)
    assert {tuple(sorted(c)) for c in singletons} == {(0,), (1,), (2,), (3,)}


def test_cluster_size_summary_handles_singletons():
    clusters = [[0, 1, 2], [3], [4, 5]]
    s = cluster_size_summary(clusters)
    assert s["n_clusters"] == 3
    assert s["max_size"] == 3
    assert s["n_singletons"] == 1
    assert s["n_nontrivial"] == 2
