"""Day 5: causal ablations — aligned cluster vs. random vs. low-alignment.

    python -m scripts.run_ablation configs/modadd_p47_grok.yaml

For each probe, picks the top-K eigenspaces by atom-MMA from Day 4 and, per
eigenspace, ablates three sets of atoms (all of the same size) and measures
behaviour change vs. the unperturbed target:

  - **aligned cluster** — the cluster containing the best-aligned atom for
    this eigenspace (usually a singleton in our regime),
  - **random size-matched** — `n_random_baselines` random atom sets of the
    same size (default 100), all excluding the aligned cluster,
  - **low-alignment** — atoms with bottom-quartile alignment to this
    eigenspace whose cumulative $\\|\\Delta M_r\\|_F$ is roughly matched to
    the aligned set.

For each ablation we record KL behaviour error, accuracy change, mean
top-2 logit margin change, and the probe-scalar change $\\Delta\\,r^\\top y(x)$.

Outputs:
  reports/ablation_summary.json
  artifacts/ablation_results.csv
  figures/ablation_kl_<probe>.png      — bars: aligned vs random±std vs low-align
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from bilinear_spd.ablations import (
    ablate_weights,
    evaluate_ablation,
    low_alignment_atom_set,
    random_size_matched_sets,
)
from bilinear_spd.analysis import atom_perturbations
from bilinear_spd.data import input_dim, make_dataset, output_dim
from bilinear_spd.decomposition import BilinearComponentMLP
from bilinear_spd.functional import build_probes
from bilinear_spd.models import BilinearMLP
from bilinear_spd.plotting import plot_ablation_bars
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


def _atom_to_cluster(clusters: list[list[int]]) -> dict[int, int]:
    out = {}
    for ci, c in enumerate(clusters):
        for atom in c:
            out[atom] = ci
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--top-k-eigenspaces", type=int, default=5,
                        help="how many eigenspaces per probe to ablate (descending atom MMA)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    out = run_dir(cfg)

    decomp = _load_decomposition(cfg, out, device)
    data = make_dataset(cfg["task"], device=device)
    p = output_dim(cfg["task"])
    probes = build_probes(cfg["probes"], p=p, device=device)

    # Behaviour targets
    with torch.no_grad():
        y_target = decomp.forward_target(data["x"])
    targets_for_acc = data["y"] if "y" in data else None

    # Day 4 alignment artifact: gives us clusters + per-probe alignment matrices.
    align_blob = torch.load(out / "artifacts" / "alignment.pt",
                            weights_only=False, map_location=device)
    clusters: list[list[int]] = align_blob["clusters"]
    atom_to_cluster = _atom_to_cluster(clusters)
    alignments_per_probe: dict = align_blob["alignments"]

    n_random = int(cfg["ablations"]["n_random_baselines"])
    rng = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

    summary_records: list[dict] = []
    figure_rows_per_probe: dict[str, list[dict]] = {}

    for probe_name, r in probes.items():
        if probe_name not in alignments_per_probe:
            continue
        rec = alignments_per_probe[probe_name]
        atom_align = rec["atom_align"].to(device)          # [n_atoms_alive, n_eigenspaces]
        atom_alive_idx: list[int] = rec["atom_idx"]
        atom_norms_all = rec["atom_norms"].to(device)      # [C_A + C_B]
        # Per-eigenspace: the alive-atom with the highest |fro_cos|.
        n_alive, n_eig = atom_align.shape
        best_atom_per_eig = atom_align.argmax(dim=0).tolist()
        atom_max_align = atom_align.max(dim=0).values.tolist()

        # Refresh ΔM_r norms on the full atom universe for this probe (alignment
        # tensor uses *alive* atoms only; we ablate on the full index set).
        deltas_all = atom_perturbations(decomp, r)
        delta_norms = deltas_all.norms  # [C_A + C_B]
        total_atoms = decomp.C_A + decomp.C_B

        # Rank eigenspaces by atom-MMA-best (descending) and take the top-K.
        order = sorted(range(n_eig), key=lambda j: -atom_max_align[j])
        chosen_eig = order[: args.top_k_eigenspaces]

        figure_rows: list[dict] = []
        for j in chosen_eig:
            best_alive_atom = best_atom_per_eig[j]
            # Map alive index back to global atom index.
            global_atom = atom_alive_idx[best_alive_atom]
            cluster_id = atom_to_cluster.get(global_atom, -1)
            aligned_set = sorted(clusters[cluster_id]) if cluster_id != -1 else [global_atom]
            cluster_size = len(aligned_set)

            # 1) aligned-cluster ablation
            A_minus, B_minus = ablate_weights(decomp, aligned_set)
            m_aligned = evaluate_ablation(decomp, data["x"], y_target,
                                          A_minus, B_minus, r=r,
                                          targets_for_acc=targets_for_acc)

            # 2) random size-matched baseline (exclude aligned cluster atoms)
            rand_sets = random_size_matched_sets(
                n_atoms=total_atoms,
                size=cluster_size,
                n_samples=n_random,
                exclude=aligned_set,
                rng=rng,
            )
            rand_metrics: list[dict] = []
            for rs in rand_sets:
                A_r, B_r = ablate_weights(decomp, rs)
                rand_metrics.append(
                    evaluate_ablation(decomp, data["x"], y_target,
                                      A_r, B_r, r=r,
                                      targets_for_acc=targets_for_acc)
                )

            def _stats(key: str) -> tuple[float, float]:
                vals = torch.tensor([m[key] for m in rand_metrics])
                return float(vals.mean()), float(vals.std(unbiased=False))

            rand_kl_mean, rand_kl_std = _stats("kl")
            rand_dacc_mean, rand_dacc_std = _stats("d_acc")
            rand_dprobe_mean, rand_dprobe_std = _stats("d_probe_scalar")

            # 3) low-alignment baseline
            low_set = low_alignment_atom_set(
                align_to_eigenspace=atom_align[:, j].cpu(),
                delta_norms=delta_norms[atom_alive_idx].cpu(),
                target_size=cluster_size,
                aligned_set={atom_alive_idx.index(a) for a in aligned_set if a in atom_alive_idx},
            )
            # low_set is in *alive* indices — map back to global
            low_set_global = sorted(atom_alive_idx[i] for i in low_set)
            A_l, B_l = ablate_weights(decomp, low_set_global)
            m_low = evaluate_ablation(decomp, data["x"], y_target,
                                      A_l, B_l, r=r,
                                      targets_for_acc=targets_for_acc)

            z_kl = (m_aligned["kl"] - rand_kl_mean) / (rand_kl_std + 1e-12)
            z_dprobe = abs(m_aligned["d_probe_scalar"] - rand_dprobe_mean) / (rand_dprobe_std + 1e-12)

            record = {
                "probe": probe_name,
                "eigenspace_idx": j,
                "atom_max_alignment": atom_max_align[j],
                "aligned_cluster_size": cluster_size,
                "kl_aligned": m_aligned["kl"],
                "kl_random_mean": rand_kl_mean,
                "kl_random_std": rand_kl_std,
                "kl_low_align": m_low["kl"],
                "d_acc_aligned": m_aligned["d_acc"],
                "d_acc_random_mean": rand_dacc_mean,
                "d_acc_random_std": rand_dacc_std,
                "d_acc_low_align": m_low["d_acc"],
                "d_probe_aligned": m_aligned["d_probe_scalar"],
                "d_probe_random_mean": rand_dprobe_mean,
                "d_probe_random_std": rand_dprobe_std,
                "d_probe_low_align": m_low["d_probe_scalar"],
                "z_kl_vs_random": z_kl,
                "z_dprobe_vs_random": z_dprobe,
            }
            summary_records.append(record)
            figure_rows.append({
                "eigenspace": f"g{j:02d}",
                "kl_aligned": m_aligned["kl"],
                "kl_random_mean": rand_kl_mean,
                "kl_random_std": rand_kl_std,
                "kl_low_align": m_low["kl"],
            })
            print(
                f"  {probe_name:14s} g{j:02d}  align={atom_max_align[j]:.3f}  size={cluster_size}  "
                f"KL aligned={m_aligned['kl']:.3e}  rand={rand_kl_mean:.3e}±{rand_kl_std:.1e}  "
                f"low={m_low['kl']:.3e}  z_kl={z_kl:+.2f}"
            )

        figure_rows_per_probe[probe_name] = figure_rows
        plot_ablation_bars(
            figure_rows,
            out_path=out / "figures" / f"ablation_kl_{probe_name}.png",
            title=f"{cfg['run_name']}  probe={probe_name}  "
                  f"(top {len(chosen_eig)} eigenspaces by atom MMA)",
        )

    # CSV + JSON
    csv_path = out / "artifacts" / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        if summary_records:
            w = csv.DictWriter(f, fieldnames=list(summary_records[0]))
            w.writeheader()
            w.writerows(summary_records)

    with open(out / "reports" / "ablation_summary.json", "w") as f:
        json.dump(
            {
                "top_k_eigenspaces": args.top_k_eigenspaces,
                "n_random_baselines": n_random,
                "records": summary_records,
            },
            f, indent=2,
        )
    print(f"wrote {out / 'reports' / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
