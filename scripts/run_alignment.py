"""Day 4: atom & cluster perturbation → eigenspace alignment.

    python -m scripts.run_alignment configs/modadd_p47_grok.yaml

Loads the trained decomposition + eigenspaces and, for every probe:

  - computes ΔM_r per atom and per cluster (per spec § 5.6),
  - builds atom×eigenspace and cluster×eigenspace alignment matrices using
    `fro_cos` against the eigenspace projector (spec § 5.7),
  - records mean-max-alignment (eigenspace-coverage axis) for atoms vs
    clusters — the H1/H2/H3 headline number,
  - dumps CSVs and heatmaps per probe, plus a cross-probe summary bar chart.

Clustering uses the threshold from `cfg['clustering']['selected_threshold']`
on the gate-correlation graph; threshold sensitivity is logged but not
swept in the main figure (spec § 5.8: "do not overfit threshold in the
main post").
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from bilinear_spd.analysis import (
    atom_perturbations,
    cluster_perturbations,
    collect_gates,
)
from bilinear_spd.clustering import (
    cluster_size_summary,
    gate_corr,
    threshold_components,
)
from bilinear_spd.data import input_dim, make_dataset, output_dim
from bilinear_spd.decomposition import BilinearComponentMLP
from bilinear_spd.functional import build_probes
from bilinear_spd.metrics import fro_cos, mean_max_alignment, projector_energy
from bilinear_spd.models import BilinearMLP
from bilinear_spd.ablations import ablate_weights, evaluate_ablation
from bilinear_spd.plotting import (
    plot_alignment_bars,
    plot_alignment_coverage,
    plot_alignment_heatmaps,
    plot_alignment_vs_ablation,
)
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
    target.load_state_dict(bi_ckpt.state_dict)
    target.eval()

    decomp_cfg = cfg["decomposition"]
    decomp = BilinearComponentMLP(
        target=target,
        C_A=int(decomp_cfg["C_A"]),
        C_B=int(decomp_cfg["C_B"]),
        gate_hidden=int(decomp_cfg["gate_hidden"]),
    ).to(device)
    d_ckpt = load_checkpoint(out / "checkpoints" / "decomposition.pt", map_location=device)
    decomp.load_state_dict(d_ckpt.state_dict)
    decomp.eval()
    return decomp


def _alignment_matrix(deltas: torch.Tensor, projectors: list[torch.Tensor]) -> torch.Tensor:
    """[n_rows × n_eigenspaces] |Frobenius cosine| of each ΔM_r vs each projector."""
    n_rows = deltas.shape[0]
    n_eig = len(projectors)
    M = torch.zeros(n_rows, n_eig, device=deltas.device, dtype=deltas.dtype)
    for j, P in enumerate(projectors):
        for i in range(n_rows):
            M[i, j] = fro_cos(deltas[i], P)
    return M


def _random_baseline_alignment(
    n_rows: int,
    projectors: list[torch.Tensor],
    n_trials: int = 64,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, float | list[float]]:
    """Mean-max-alignment for a random "decomposition" with the same shape.

    Returns both the scalar MMA summaries (eig_coverage_mean ± std,
    atom_axis_mean ± std) and the *per-eigenspace* "best random atom"
    distribution (mean + std over trials), so we can render the coverage
    plot's gray ±2σ band.
    """
    if not projectors:
        return {"eig_coverage_mean": 0.0, "eig_coverage_std": 0.0,
                "atom_axis_mean": 0.0, "atom_axis_std": 0.0,
                "per_eigenspace_mean": [], "per_eigenspace_std": []}
    d = projectors[0].shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eig_vals = []
    atom_vals = []
    per_eig_best_trials = []   # [n_trials, n_eig]
    for _ in range(n_trials):
        deltas_raw = torch.randn(n_rows, d, d, generator=gen).to(device)
        deltas_sym = 0.5 * (deltas_raw + deltas_raw.transpose(-1, -2))
        align = _alignment_matrix(deltas_sym, projectors)
        eig_vals.append(float(mean_max_alignment(align, axis="eigenspaces").item()))
        atom_vals.append(float(mean_max_alignment(align, axis="atoms").item()))
        per_eig_best_trials.append(align.max(dim=0).values.cpu())
    eig_t = torch.tensor(eig_vals)
    atom_t = torch.tensor(atom_vals)
    per_eig_t = torch.stack(per_eig_best_trials)   # [n_trials, n_eig]
    return {
        "eig_coverage_mean": float(eig_t.mean()),
        "eig_coverage_std": float(eig_t.std(unbiased=False)),
        "atom_axis_mean": float(atom_t.mean()),
        "atom_axis_std": float(atom_t.std(unbiased=False)),
        "per_eigenspace_mean": per_eig_t.mean(dim=0).tolist(),
        "per_eigenspace_std": per_eig_t.std(dim=0, unbiased=False).tolist(),
    }


def _energy_matrix(deltas: torch.Tensor, projectors: list[torch.Tensor]) -> torch.Tensor:
    n_rows = deltas.shape[0]
    n_eig = len(projectors)
    E = torch.zeros(n_rows, n_eig, device=deltas.device, dtype=deltas.dtype)
    for j, P in enumerate(projectors):
        for i in range(n_rows):
            E[i, j] = projector_energy(deltas[i], P)
    return E


def _write_csv(path: Path, matrix: torch.Tensor, row_labels: list[str], col_labels: list[str]) -> None:
    rows = matrix.detach().cpu().numpy().tolist()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row"] + col_labels)
        for label, row in zip(row_labels, rows):
            w.writerow([label] + [f"{v:.6f}" for v in row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    out = run_dir(cfg)

    decomp = _load_decomposition(cfg, out, device)
    data = make_dataset(cfg["task"], device=device)
    p = output_dim(cfg["task"])
    probes = build_probes(cfg["probes"], p=p, device=device)

    # Gates + clusters are probe-independent — compute once.
    gates = collect_gates(decomp, data["x"])
    corr = gate_corr(gates)

    threshold = float(cfg["clustering"]["selected_threshold"])
    thresholds_to_sweep = list(cfg["clustering"]["corr_thresholds"])

    sweep_summary = []
    for tau in thresholds_to_sweep:
        cl = threshold_components(corr, tau)
        s = cluster_size_summary(cl)
        s["threshold"] = tau
        sweep_summary.append(s)

    clusters = threshold_components(corr, threshold)
    print(f"selected threshold τ = {threshold}: "
          f"{len(clusters)} clusters "
          f"({sum(1 for c in clusters if len(c) == 1)} singletons, "
          f"{sum(1 for c in clusters if len(c) >= 2)} nontrivial, "
          f"max size {max(len(c) for c in clusters) if clusters else 0})")

    atom_mma_per_probe: dict[str, float] = {}
    cluster_mma_per_probe: dict[str, float] = {}
    atom_axis_per_probe: dict[str, float] = {}
    cluster_axis_per_probe: dict[str, float] = {}
    per_probe_records: dict[str, dict] = {}

    # Save alignment matrices into artifacts/, CSVs into artifacts/, heatmaps into figures/.
    artifacts = out / "artifacts"
    figures = out / "figures"

    raw_alignments: dict[str, dict[str, torch.Tensor]] = {}

    for probe_name, r in probes.items():
        # Load this probe's eigenspaces.
        # mr_slices.pt + eigenspaces.pt were written by analyze_functional.
        eigenspaces_blob = torch.load(
            artifacts / "eigenspaces.pt", weights_only=False, map_location=device
        )
        if probe_name not in eigenspaces_blob:
            print(f"[skip] {probe_name}: no eigenspaces saved")
            continue
        groups = eigenspaces_blob[probe_name]["groups"]
        projectors = [g["projector"].to(device) for g in groups]
        eig_labels = [f"g{i:02d}(λ={g['mean_eval']:+.2f},r={g['rank']})"
                      for i, g in enumerate(groups)]

        # Perturbations on this probe.
        atom_deltas = atom_perturbations(decomp, r)
        cluster_deltas = cluster_perturbations(decomp, r, clusters)

        # Filter out atoms / clusters with essentially-zero ΔM_r — they're
        # numerically dead and would give meaningless cosines.
        eps = 1e-8
        atom_alive = atom_deltas.norms > eps
        cluster_norms = torch.linalg.matrix_norm(cluster_deltas, "fro", dim=(-2, -1))
        cluster_alive = cluster_norms > eps

        atoms_alive_deltas = atom_deltas.delta_M[atom_alive]
        clusters_alive_deltas = cluster_deltas[cluster_alive]

        atom_align = _alignment_matrix(atoms_alive_deltas, projectors)
        cluster_align = _alignment_matrix(clusters_alive_deltas, projectors)
        atom_energy = _energy_matrix(atoms_alive_deltas, projectors)
        cluster_energy = _energy_matrix(clusters_alive_deltas, projectors)

        atom_mma = float(mean_max_alignment(atom_align, axis="eigenspaces").item())
        cluster_mma = float(mean_max_alignment(cluster_align, axis="eigenspaces").item())
        atom_axis = float(mean_max_alignment(atom_align, axis="atoms").item())
        cluster_axis = float(mean_max_alignment(cluster_align, axis="atoms").item())
        # Match size — same #rows as atoms (n_rows = C_A + C_B effective live count).
        baseline = _random_baseline_alignment(
            n_rows=int(atoms_alive_deltas.shape[0]),
            projectors=projectors,
            n_trials=32,
            device=device,
            seed=int(cfg["seed"]) + sum(ord(c) for c in probe_name),
        )
        atom_mma_per_probe[probe_name] = atom_mma
        cluster_mma_per_probe[probe_name] = cluster_mma
        atom_axis_per_probe[probe_name] = atom_axis
        cluster_axis_per_probe[probe_name] = cluster_axis

        # Per-probe label lists (alive atom/cluster indices), then CSVs.
        atom_idx = atom_alive.nonzero().flatten().tolist()
        cluster_idx = cluster_alive.nonzero().flatten().tolist()
        atom_labels = [
            f"a{i}_{'A' if atom_deltas.a_or_b[i].item() == 0 else 'B'}"
            for i in atom_idx
        ]
        cluster_labels = [f"c{i}(|S|={len(clusters[i])})" for i in cluster_idx]

        _write_csv(artifacts / f"alignment_atoms_{probe_name}.csv",
                   atom_align, atom_labels, eig_labels)
        _write_csv(artifacts / f"alignment_clusters_{probe_name}.csv",
                   cluster_align, cluster_labels, eig_labels)
        _write_csv(artifacts / f"energy_atoms_{probe_name}.csv",
                   atom_energy, atom_labels, eig_labels)
        _write_csv(artifacts / f"energy_clusters_{probe_name}.csv",
                   cluster_energy, cluster_labels, eig_labels)

        raw_alignments[probe_name] = {
            "atom_align": atom_align.cpu(),
            "cluster_align": cluster_align.cpu(),
            "atom_energy": atom_energy.cpu(),
            "cluster_energy": cluster_energy.cpu(),
            "atom_idx": atom_idx,
            "cluster_idx": cluster_idx,
            "atom_norms": atom_deltas.norms.cpu(),
            "cluster_norms": cluster_norms.cpu(),
        }

        summary_text = (
            f"mean max alignment (atoms→eigenspaces): {atom_axis:.3f}     "
            f"eigenspace coverage by best atom: {atom_mma:.3f}\n"
            f"mean max alignment (clusters→eigenspaces): {cluster_axis:.3f}     "
            f"eigenspace coverage by best cluster: {cluster_mma:.3f}"
        )
        # Diagnostic heatmap (capped vmax, top-region only). Coverage + scatter
        # below are the main-text visuals.
        plot_alignment_heatmaps(
            atom_align, cluster_align,
            out_path=figures / f"alignment_{probe_name}.png",
            title=f"{cfg['run_name']}  probe={probe_name}  "
                  f"(diagnostic, {atoms_alive_deltas.shape[0]} live atoms, "
                  f"{clusters_alive_deltas.shape[0]} live clusters, "
                  f"{len(projectors)} eigenspaces)",
            summary_text=summary_text,
        )

        # --- Main-text visual 1: sorted best-match coverage curve ---
        atom_best = atom_align.max(dim=0).values
        cluster_best = cluster_align.max(dim=0).values
        plot_alignment_coverage(
            atom_best, cluster_best,
            random_per_eigenspace={
                "mean": baseline["per_eigenspace_mean"],
                "std": baseline["per_eigenspace_std"],
            },
            out_path=figures / f"alignment_coverage_{probe_name}.png",
            title=None,
        )

        # --- Main-text visual 2: alignment-vs-ablation scatter ---
        # Compute single-atom ablation KL for *every alive atom* on this probe.
        # Cost: ~O(C_A + C_B) plain bilinear forwards on the full table.
        with torch.no_grad():
            y_target = decomp.forward_target(data["x"])
        per_atom_kl = torch.zeros(int(atoms_alive_deltas.shape[0]),
                                  device=device, dtype=atom_deltas.norms.dtype)
        per_atom_max_align = atom_align.max(dim=1).values
        atom_norms_alive = atom_deltas.norms[atom_alive]
        for k, global_atom in enumerate(atom_idx):
            A_m, B_m = ablate_weights(decomp, [int(global_atom)])
            metrics = evaluate_ablation(
                decomp, data["x"], y_target, A_m, B_m, r=r,
                targets_for_acc=data.get("y"),
            )
            per_atom_kl[k] = metrics["kl"]
        plot_alignment_vs_ablation(
            per_atom_max_align, per_atom_kl, atom_norms_alive,
            out_path=figures / f"alignment_vs_ablation_{probe_name}.png",
            title=None,
        )
        raw_alignments[probe_name]["per_atom_kl"] = per_atom_kl.cpu()
        raw_alignments[probe_name]["per_atom_max_alignment"] = per_atom_max_align.cpu()

        per_probe_records[probe_name] = {
            "n_eigenspaces": len(projectors),
            "n_atoms_alive": int(atoms_alive_deltas.shape[0]),
            "n_clusters_alive": int(clusters_alive_deltas.shape[0]),
            "atom_mean_max_alignment_eigenspace_coverage": atom_mma,
            "cluster_mean_max_alignment_eigenspace_coverage": cluster_mma,
            "atom_mean_max_alignment_atom_axis": atom_axis,
            "cluster_mean_max_alignment_atom_axis": cluster_axis,
            "atom_max_per_eigenspace": atom_align.max(dim=0).values.cpu().tolist(),
            "cluster_max_per_eigenspace": cluster_align.max(dim=0).values.cpu().tolist(),
            "random_baseline": baseline,
        }
        print(
            f"  {probe_name:14s}  atoms_mma_eig={atom_mma:.3f}  "
            f"clusters_mma_eig={cluster_mma:.3f}  "
            f"random_mma_eig={baseline['eig_coverage_mean']:.3f}±{baseline['eig_coverage_std']:.3f}  "
            f"(atom-axis: a={atom_axis:.3f}/c={cluster_axis:.3f}/rand={baseline['atom_axis_mean']:.3f})"
        )

    # cross-probe bars (eigenspace-coverage MMA — the "do mechanisms get covered?" number).
    # No title — caption in the post carries the claim.
    baseline_by_probe = {
        name: per_probe_records[name]["random_baseline"] for name in atom_mma_per_probe
    }
    plot_alignment_bars(
        atom_mma_per_probe,
        cluster_mma_per_probe,
        out_path=figures / "alignment_mma_by_probe.png",
        title=None,
        baseline_by_probe=baseline_by_probe,
    )

    # Save tensors for downstream Day 5 use.
    torch.save(
        {
            "gate_corr": corr.cpu(),
            "clusters": clusters,
            "threshold": threshold,
            "alignments": raw_alignments,
        },
        artifacts / "alignment.pt",
    )

    cl_sizes = cluster_size_summary(clusters)
    summary = {
        "clustering": {
            "selected_threshold": threshold,
            **cl_sizes,
            "threshold_sweep": sweep_summary,
        },
        "per_probe": per_probe_records,
        "atom_mean_max_alignment_eigenspace_coverage": atom_mma_per_probe,
        "cluster_mean_max_alignment_eigenspace_coverage": cluster_mma_per_probe,
        "atom_mean_max_alignment_atom_axis": atom_axis_per_probe,
        "cluster_mean_max_alignment_atom_axis": cluster_axis_per_probe,
    }
    with open(out / "reports" / "alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out / 'reports' / 'alignment_summary.json'}")


if __name__ == "__main__":
    main()
