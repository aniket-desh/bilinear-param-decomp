# Bilinear SPD Benchmark

**Do rank-one parameter decompositions recover bilinear functional mechanisms?**

This repo trains a tiny bilinear MLP to grok modular addition, computes the
exact probe-conditioned interaction matrix $M_r$ and its eigenspaces, learns
an SPD-style rank-one gated parameter decomposition of the bilinear weights
$(A, B)$, and asks whether the learned atoms — or clusters of co-active
atoms — align with the eigenspaces of $M_r$. The short answer turns out
to be **no, the decomposition converges to a different structure entirely**
(row/column shards of the $p \times p$ lookup table), which is the subject
of the post below.

The full writeup lives in **[`post.md`](post.md)** (LessWrong-style, ~3.5k
words, inline figures, methods appendix). Read that first; this README is
just a code-pointer.

## Quickstart

```bash
git clone https://github.com/aniket-desh/bilinear-param-decomp.git
cd bilinear-param-decomp
uv sync                          # or:  pip install -e ".[dev]"
pytest                           # 31 passes
```

## Reproducing the post

```bash
for c in modadd_p13_grok modadd_p23_grok modadd_p47_grok; do
    python -m scripts.train_bilinear       configs/${c}.yaml
    python -m scripts.analyze_functional   configs/${c}.yaml
    python -m scripts.train_decomp         configs/${c}.yaml
    python -m scripts.run_alignment        configs/${c}.yaml
    python -m scripts.run_ablation         configs/${c}.yaml
    python -m scripts.run_rank_ablation    configs/${c}.yaml --probe centered_0
    python -m scripts.run_grid_visuals     configs/${c}.yaml --probe centered_0 --top-k 12
done
# Robustness check on the headline: shard ontology vs sparsity pressure
python -m scripts.run_sparsity_sweep configs/modadd_p13_grok.yaml
```

p47 needs ~10 min for bilinear training plus ~10 min for the decomposition
on an A6000; the smaller configs finish in 1–2 min each. Outputs land in
`runs/<run_name>/{checkpoints, artifacts, figures, reports}/` — only the
`figures/` and `reports/` directories are committed; `checkpoints/` and
`artifacts/` are gitignored.

## Layout

```
src/bilinear_spd/   library code
  ├ models.py         BilinearMLP
  ├ functional.py     M_r, eigendecomposition, eigenspace grouping
  ├ decomposition.py  gated rank-one decomposition (BilinearComponentMLP)
  ├ losses.py         4-term loss (cherry-picked KL + freq penalty from Goodfire nano)
  ├ sigmoids.py       lower/upper-leaky sigmoids (cherry-picked from Goodfire nano)
  ├ analysis.py       atom/cluster ΔM_r perturbations
  ├ metrics.py        Frobenius cosine, projector energy, mean-max-alignment
  ├ clustering.py     gate-correlation threshold-CC
  ├ ablations.py      surgical weight ablation + size-matched random baseline
  └ plotting.py       publication-style figure helpers

scripts/            CLI entry points (see Reproducing the post above)
configs/            three grok configs: p ∈ {13, 23, 47}
tests/              31 pytests (shapes, M_r equivalence, projector invariants,
                    decomposition forward/ablation, perturbation expansion,
                    alignment metrics, clustering, ablation invariants)
runs/               training/analysis outputs (figures + reports committed,
                    checkpoints + artifacts gitignored)
post.md             the writeup
```

## Cherry-picked code

Four small pieces of `src/bilinear_spd/{sigmoids,losses}.py` were copied
from [Goodfire's `nano_param_decomp/run.py`](https://github.com/goodfire-ai/param-decomp/tree/spd-paper)
with attribution comments at the call sites: `lower_leaky` / `upper_leaky`,
`kl_logits`, `frequency_penalty`, and the stochastic-mask sampler
$m = g + (1 - g) U(0, 1)$. Everything else (bilinear MLP, $M_r$,
eigenspace grouping, gated decomposition over raw $(A, B)$ parameters,
alignment metrics, clustering, ablation, plotting, tests) is from scratch.
