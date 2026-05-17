# Bilinear-SPD Benchmark — Progress Summary

Living document. Updated as days roll in.

## Project one-line

Train a tiny bilinear MLP on modular addition, build the exact probe-conditioned
interaction matrix $M_r$, learn an SPD-style rank-one gated decomposition of the
bilinear weights $(A, B)$, and ask whether atoms (or clusters of co-active atoms)
align with the eigenspaces of $M_r$.

Theory writeup and codebase spec are kept locally and not committed.

## Reuse decisions (made before Day 1)

The Goodfire `param-decomp` repo was inspected and not vendored. The four
small pieces we will cherry-pick from it on later days (sigmoids with leaky
backward, importance-minimality loss, stochastic-mask sampler, cosine LR) are
~100 lines total and will be pasted with attribution comments. Everything else
— the bilinear MLP, $M_r$, eigenspace grouping, gated decomposition over
raw $A,B$ parameters, clustering by gate correlations, alignment metrics,
ablations, plotting, tests — is written from scratch because Goodfire's
implementation is tightly coupled to LM/transformer/DDP/WandB/SLURM
infrastructure we don't need for this benchmark.

## Day 1 — repo skeleton + bilinear MLP + $M_r$ equivalence

### What landed

- **Package**: `src/bilinear_spd/` with `utils.py`, `data.py`, `models.py`,
  `functional.py`, `train.py`, `plotting.py`.
- **Configs**: `configs/{xor, modadd_p7, modadd_p13}.yaml`. Device defaults
  to `cpu`; auto-detect available for cuda/mps.
- **CLI**: `python -m scripts.train_bilinear configs/<name>.yaml` writes
  `runs/<run_name>/{config.yaml, checkpoints/, artifacts/, figures/, reports/}`.
- **Tests**: 6 pytests covering dataset shapes, model forward shape, $M_r$
  shape + symmetry, and the central correctness check
  $r^\top y(x) \;=\; x^\top M_r x$ at both random init and post-training.

### Functional-equivalence sanity check

The most important sanity check in the project. For all three configs, after
training, $\max_n |r^\top y(x_n) - x_n^\top M_r x_n|$ stays at floating-point
noise.

| run | params | steps to 99% acc | wall (CPU) | $\|r^\top y - x^\top M_r x\|_\infty$ |
|---|---:|---:|---:|---:|
| `xor` | 80 | 9 | <1 s | 4.8e-07 |
| `modadd_p7` | 840 | 105 | ~1 s | 9.5e-07 |
| `modadd_p13` | 2 080 | 111 | 3.6 s | 1.7e-06 |

All numbers come from `runs/<name>/reports/run_summary.json` after a fresh CPU
run on an M2 Air. Probe used for the check is the centered class probe
$r = e_0 - \tfrac{1}{p}\mathbf{1}$.

### Test suite

```text
tests/test_functional_equivalence.py ..  (random-init + post-train probes)
tests/test_shapes.py                ....  (dataset, forward, M_r shape+symmetry)
6 passed in 20.58s
```

The 20 s wall is dominated by torch import; raw test work is <1 s.

### Training curves

#### XOR (`runs/xor_seed0/`)

![xor training curves](runs/xor_seed0/figures/training_curves.png)

#### Modular addition, $p = 7$ (`runs/modadd_p7_seed0/`)

![modadd_p7 training curves](runs/modadd_p7_seed0/figures/training_curves.png)

#### Modular addition, $p = 13$ (`runs/modadd_p13_seed0/`)

![modadd_p13 training curves](runs/modadd_p13_seed0/figures/training_curves.png)

All three runs reach 100% accuracy well before the configured step budget and
early-stop on the spec's 99%-for-N-steps criterion. Logit margins are healthy
(>4 by the early-stop point), so the models are confidently fitting the table
rather than sitting at a decision boundary.

### Hardware note

Whole project is comfortably CPU on a 16 GB M2 Air. The largest run
(`modadd_p13`) is 2 080 parameters and finishes in under 4 seconds wall.

## Day 2 — eigendecomposition + projector grouping + spectrum figures

### What landed

- `functional.py`: `eigendecompose_M` (eigh, sorted by $|\lambda|$ desc) and
  `group_eigenspaces` (signed-proximity grouping with $\tau \cdot \max_k|\lambda_k|$
  tolerance, $\min |\lambda|$ floor, returns `EigenGroup` dataclass with
  `indices`, `evals`, `projector`, `rank`, `mean_abs_eval`).
- `build_probes(cfg, p)`: materializes centered-class and pairwise probes from
  the YAML config block.
- `plotting.py`: `plot_spectrum` (bar plot of $|\lambda|$ colored by group,
  degeneracy annotations) and `plot_top_eigenvectors` (heatmap split at the
  a-slot / b-slot boundary).
- `scripts/analyze_functional.py`: load checkpoint → build probes → $M_r$ per
  probe → equivalence sanity check → eigendecompose → group → save
  `artifacts/{mr_slices.pt, eigenspaces.pt}` → save spectrum +
  eigenvector figures per probe → write `reports/functional_summary.json`.
- `tests/test_eigenspace_projectors.py`: $P^2 = P$, $P^\top = P$,
  $\mathrm{tr}\,P = \mathrm{rank}\,P$ on a random symmetric matrix, on a
  freshly trained $p=7$ checkpoint, AND on a hand-constructed matrix with a
  known rank-3 degenerate eigenspace.

Full test suite: **9/9 pass**.

### Spectra

| run | probe | $\max\,\|r^\top y - x^\top M_r x\|$ | groups @ τ=1e-3 | top $\|λ\|$ |
|---|---|---:|---:|---:|
| xor | pairwise_0_1 | 9.5e-07 | 4 (0 degenerate) | 7.13 |
| modadd_p7 | centered_0 | 9.5e-07 | 14 (0 degenerate) | 10.09 |
| modadd_p7 | pairwise_0_1 | 1.9e-06 | 14 (0 degenerate) | 17.84 |
| modadd_p13 | centered_0 | 1.7e-06 | 26 (0 degenerate) | 16.89 |
| modadd_p13 | pairwise_0_1 | 2.9e-06 | 26 (0 degenerate) | 29.41 |

Equivalence holds at floating-point noise across all probes.

#### modadd_p13 — centered class probe $r = e_0 - \tfrac{1}{p}\mathbf{1}$

![spectrum centered_0 p13](runs/modadd_p13_seed0/figures/spectrum_centered_0.png)
![eigvecs centered_0 p13](runs/modadd_p13_seed0/figures/eigenvectors_centered_0.png)

### The interesting finding: the model is in the memorization regime

The current trained model produces **zero degenerate eigenspaces at $\tau = 10^{-3}$** across all probes. Theory (`docs/01` §12) predicts that a model learning the
*algorithmic* solution to modular addition should exhibit degenerate eigenspaces
from translation equivariance — same-sign pairs of eigenvalues corresponding to
the cosine/sine modes of each Fourier frequency. We see none of that.

Sweeping the tolerance on `modadd_p13 / centered_0`:

| $\tau$ | groups | degenerate |
|---:|---:|---:|
| 0.001 | 26 | 0 |
| 0.01 | 26 | 0 |
| 0.02 | 25 | 1 |
| 0.05 | 19 | 6 |
| 0.10 | 8 | 4 |

You have to loosen $\tau$ to ~5% before degeneracies appear, and the eigenvalue
pairs the model *does* produce are close but *opposite-sign* (e.g. $-16.89$
vs $+16.81$, $-15.44$ vs $+15.08$) — not the same-sign sine/cosine pairs of a
Fourier algorithm. Combined with the eigenvector heatmap (no periodic
structure across the a-slot/b-slot), this is consistent with the model
memorizing the table rather than learning translation-equivariant features.

`docs/01` §12.1 anticipates exactly this:

> The bilinear MLP may memorize the table in a non-Fourier way, especially for
> small $p$ and overparameterized hidden dimension.

Our setup is squarely in that regime: $m = 32 > p = 13$ hidden units, no weight
decay, large initialization ($\sigma = 0.5$ vs the spec's suggested $0.02$),
trained for only ~600 steps before early stop. Day 1's "100% accuracy in 110
steps" was a *symptom* of the problem, not a success — it's the model finding a
local lookup-table minimum well before any algorithmic structure has time to
emerge.

This is not a bug in any of the code we wrote. Equivalence holds, projectors
are correct, plots render. The trained model is simply uninteresting from a
mechanistic standpoint.

## Day 2.5 — training-regime fix + p sweep

### What landed

- `train_bilinear` now optionally tracks "grok metrics" every $N$ steps:
  parameter $L^2$ norm, top $|\lambda|$ of $M_r$ for a tracked probe, and
  the number of degenerate eigenspace groups at one or more tolerances.
- `plot_grokking_curves`: 4-panel diagnostic (loss + acc, $\|\theta\|^2$,
  degenerate-group count at $\tau \in \{0.01, 0.05\}$, top $|\lambda|$).
- `plot_spectrum`: replaced the colliding per-group text annotations with a
  single corner summary box (groups / degenerate / max rank / eigvecs in
  degenerate).
- Three grokking configs: `modadd_p{13,23,47}_grok.yaml`. All use
  `init_scale = 0.02`, `weight_decay = 1.0`, `target_accuracy = 1.01` (no
  early stop), and 30k / 40k / 60k steps respectively.

### Sweep result at $\tau = 0.01$

For each run I compare the Day 1/2 baseline (no weight decay, early stop on
accuracy, large init) against the Day 2.5 grokked model on the same probe
$r = e_0 - \tfrac{1}{p}\mathbf{1}$:

| run | wd | init | steps | top $\|λ\|$ | # groups | # degenerate | max group rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `modadd_p13` (Day 1/2) | 0.0 | 0.5 | 612 (early stop) | 16.9 | 26 | **0** | 1 |
| `modadd_p13_grok` | 1.0 | 0.02 | 30 000 | 15.6 | 22 | **2** | 3 |
| `modadd_p23_grok` | 1.0 | 0.02 | 40 000 | 21.3 | 40 | **4** | 4 |
| `modadd_p47_grok` | 1.0 | 0.02 | 60 000 | 31.0 | 56 | **12** | **20** |

The qualitative shift between the memorization baseline and the grokked
models is the whole point. With weight decay turned on, the degenerate
eigenspace structure emerges and *grows with $p$* — at $p=47$ a single
eigenspace projector has rank 20, exactly what we want for downstream
projector-vs-atom alignment.

Wall time on M2 Air CPU: 18 s + 35 s + 3 min. The whole sweep is under
4 minutes.

### Grokking curves

#### p = 47 — the cleanest signal

![p47 grokking curves](runs/modadd_p47_grok_seed0/figures/grokking_curves.png)

Train accuracy hits 1.0 around step 200. Weight norm spikes early, sags,
and then slowly recovers. Degenerate-group count is already non-zero by
step 500 — the strong weight decay + small init combination skips the
memorization plateau entirely. From step ~2000 onward the model sits in a
quasi-Fourier regime with 5–18 degenerate groups oscillating (the
oscillation is due to the τ-threshold flipping nearby pairs in and out).

This is *not* the classical Power-et-al delayed-grokking curve where train
accuracy saturates long before test accuracy and the algorithmic phase
takes thousands of steps to emerge. With these hyperparameters the model
goes directly into the algorithmic basin.

### Functional spectra (post-grokking)

#### modadd_p47_grok — centered class probe

![p47 spectrum grok](runs/modadd_p47_grok_seed0/figures/spectrum_centered_0.png)
![p47 eigvecs grok](runs/modadd_p47_grok_seed0/figures/eigenvectors_centered_0.png)

The spectrum has ~65 leading eigenvalues at similar magnitude ($\sim 10^1$),
a sharp two-decade cliff at index 65, and a noise floor below index 73.
**46 of the 94 eigenvectors live inside rank-≥2 degenerate eigenspaces** —
the signature of translation-equivariant feature pairing.

The eigenvector heatmap is the visual confirmation: clear periodic
patterns repeat across the a-slot (rows 0–46) and b-slot (rows 47–93), with
column-wise sinusoidal structure. Compare to the Day 2 memorization
eigenvectors, which looked like random noise. The model is doing
something Fourier-flavored.

Caveat: "Fourier-flavored" is not "fully Fourier." Several columns are
muddy. A future check is to FFT each eigenvector along the a-slot and
report what fraction of energy concentrates at a single frequency — that
would be the rigorous test.

#### modadd_p13_grok (smaller, fewer degeneracies but same regime)

![p13 spectrum grok](runs/modadd_p13_grok_seed0/figures/spectrum_centered_0.png)

Only 2 degenerate groups (vs 12 at $p=47$). The model has chosen ~2 Fourier
frequencies to encode $p=13$ — enough capacity in $m=32$ for more, but
weight decay encourages parsimony. This matches the Nanda et al. finding
that grokking-mode transformers on modular addition typically use a small
"key frequency set."

### What this unblocks

Day 3 (the SPD-style rank-one parameter decomposition) finally has a
meaningful target. The grokked checkpoints have:

- nontrivial degenerate eigenspaces to compare atoms against (projector
  alignment is well-defined);
- multi-rank groups (notably the rank-20 group at $p=47$) where the
  alignment-vs-projector distinction the theory doc emphasizes actually
  matters;
- Fourier-flavored eigenvectors, so visual sanity checks during alignment
  are interpretable.

## Day 3 — gated rank-one decomposition trained on all three grokked targets

### What landed (code)

- `src/bilinear_spd/sigmoids.py`, `losses.py`, `decomposition.py` —
  `BilinearComponentMLP` wraps a frozen `BilinearMLP` with rank-one atoms
  $u^A_c (v^A_c)^\top$ for $A$ and likewise for $B$, plus a small
  per-atom gate MLP. The atom-only reconstruction is $\hat A = \sum_c u^A_c (v^A_c)^\top$
  with implicit residual $\Delta = A - \hat A$ kept *unmasked* — masking
  affects only the atoms, so the parameter-reconstruction loss
  $\|\Delta\|^2 / \|A\|^2$ stays interpretable.
- Cherry-picked from Goodfire `nano_param_decomp/run.py` (with attribution
  in the source): `lower_leaky` / `upper_leaky` (§ B), `kl_logits` and
  `frequency_penalty` (§ E), and the stochastic-mask sampler
  $m = g + (1 - g) \cdot U(0,1)$ (§ E).
- `train_decomposition` (4-term loss: KL behavior + relative param recon
  for $A$ and $B$ + $L_1$ gate sparsity + frequency penalty), `train_decomp`
  CLI, `plot_decomposition_curves` (loss / KL / recon / $L_0$ / mean gate).
- 5 unit tests in `tests/test_decomposition.py` covering forward identity,
  unmasked atom + Δ identity, full-zero ablation invariant, gate clamp,
  stochastic mask bounds.

### Results

Decomposition trained for **20 000 steps × 4 mask samples** per config on an
RTX A6000, full-table batches, AdamW lr = 1e-3, $\lambda_\text{behavior} = \lambda_\text{recon} = 1.0$,
$\lambda_\text{sparsity} = \lambda_\text{frequency} = 10^{-3}$, $C_A = C_B = 2 d_\text{hidden}$.

| run | $C_A = C_B$ | final loss | KL | recon $A$ | recon $B$ | $L_0(A)$ / $C_A$ | $L_0(B)$ / $C_B$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `modadd_p13_grok` | 64 | 7.4e-03 | 5.6e-04 | 1.4e-05 | 1.2e-05 | 1.24 / 64 | 1.19 / 64 |
| `modadd_p23_grok` | 96 | 6.7e-03 | 2.3e-04 | 1.3e-05 | 1.7e-05 | 1.11 / 96 | 1.09 / 96 |
| `modadd_p47_grok` | 192 | 6.5e-03 | 4.2e-04 | 1.6e-05 | 1.3e-05 | 1.04 / 192 | 1.10 / 192 |

All three pass the acceptance criteria from `PASS_OFF.md` § 6:

- relative parameter reconstruction error $\le 5\%$ (we hit $\sim 10^{-5}$,
  three orders of magnitude inside spec);
- behavior KL $\le 10^{-2}$ (we hit $\sim 10^{-4}$);
- gate $L_0$ substantially less than $C_A / C_B$ (we hit ≈ 1 active gate
  per sample regardless of config — the gates are *very* sparse).

Wall time on the A6000: ~1 min for `p13_grok`, ~1.5 min for `p23_grok`,
~10 min for `p47_grok`. Bilinear MLPs were re-trained from scratch on
GPU (Day 1 + 2.5 are gitignored at the `.pt` level) — that took
~40 s + ~70 s + ~10 min on the same card.

### What the L₀ ≈ 1 result already suggests

The gate $L_0$ across all three runs sits at ≈ 1 *active gate per input
sample*, despite $C \in \{64, 96, 192\}$ atoms available. That's an
order-of-magnitude sparser than the spec's expected "5–20 atoms per
eigenspace" target, and a heads-up for Day 4: the decomposition seems to
have organized itself into a near-orthogonal per-input lookup rather than
into shared mechanisms. The behavior is right ($KL \sim 10^{-4}$) and the
parameter reconstruction is right ($\|\Delta\|^2 \sim 10^{-5}$), so the
decomposition is mathematically valid — but Day 4 is the test of whether
those atoms correspond to the bilinear MLP's *functional* eigenspaces.

### Decomposition curve — p47

![p47 decomposition curves](runs/modadd_p47_grok_seed0/figures/decomposition_curves.png)

Loss + KL drop sharply in the first ~2k steps then converge; recon error
crashes to $10^{-5}$ inside ~500 steps and stays there; gate $L_0$
collapses from $C$ at step 0 (everything half-on by init) down to ≈ 1 in
the first 5k steps; mean gate value settles near 0.005 — almost every gate
is essentially off for almost every input.

### Test suite

```text
tests/test_shapes.py                    ....   [4]
tests/test_functional_equivalence.py    ..     [2]
tests/test_eigenspace_projectors.py     ...    [3]
tests/test_decomposition.py             .....  [5]
14 passed
```

## Up next — Day 4

- `analysis.py`: per-atom and per-cluster $\Delta M_r$.
- `metrics.py`: $|\text{Frobenius cosine}|$ and projector-energy alignment.
- `clustering.py`: gate-correlation threshold-CC clustering.
- `run_alignment.py`: produce atom×eigenspace and cluster×eigenspace
  heatmaps + the H1/H2/H3 mean-max-alignment number per probe.
