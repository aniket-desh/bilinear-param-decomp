# Pass-off: bilinear-spd-benchmark, Day 3 onward

You are picking up a partially-complete mini-research project. Your job is to
**finish Day 3 (decomposition training on the three grokked checkpoints)** and
then start Day 4 (atom/cluster alignment vs eigenspaces). You have GPU access;
the previous run was on a 16 GB M2 Air CPU and stalled out on Day 3 due to
mask-sample-averaging cost. **All Day 3 code is already written, unit-tested,
and committed.** You only need to run it.

This document is a complete memory dump. Read it once end-to-end, then read the
other docs listed below before touching anything.

---

## 0. TL;DR (~30 seconds)

The project tests whether SPD-style rank-one parameter decompositions of a
bilinear MLP recover its probe-conditioned functional eigenspaces. Days 1–2.5
built and verified: a small bilinear MLP, exact $M_r = \tfrac12\sum_i (Cr)_i (a_i b_i^\top + b_i a_i^\top)$, eigendecomposition with τ-grouped projectors,
a grokking-friendly training regime, a $p$ sweep. Day 3 wrote the
`BilinearComponentMLP` gated decomposition and the training loop. **All 14
existing tests pass.** Day 3 training on the smallest grokked checkpoint
hadn't finished on CPU after 14 minutes (4 mask samples × 20 k steps), so I
killed it. Your task: run it on GPU.

---

## 1. First commands

```bash
git clone https://github.com/aniket-desh/bilinear-param-decomp.git
cd bilinear-param-decomp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # expect 14 passed
```

Then change `device: cpu` → `device: auto` in the three grok configs
(`configs/modadd_p{13,23,47}_grok.yaml`), and run the sweep:

```bash
# Day 1/2.5 has already been done — checkpoints are committed under
# runs/modadd_p{13,23,47}_grok_seed0/checkpoints/bilinear_mlp.pt.
# You only need to run the decomp training.

for c in modadd_p13_grok modadd_p23_grok modadd_p47_grok; do
  python -m scripts.train_decomp configs/${c}.yaml
done
```

Then re-run analysis on the new decomp artifacts (this is Day 4 work, code
not yet written — see § 6).

---

## 2. Read these next, in order

1. **`summary.md`** — running progress log. Already covers Days 1, 2, 2.5 with
   embedded figures. Skim, then come back here.
2. **`docs/01_theory_breakdown_bilinear_spd_benchmark.md`** — the full theory.
   Read § 0–9 carefully, skim § 10–16. Key formulas you will use today:
   the $M_r$ construction (§ 4.2), eigenspace projectors (§ 5),
   atom/cluster perturbation expansion (§ 7), alignment metrics (§ 8),
   clustering rule (§ 9), hypotheses (§ 10).
3. **`docs/02_experimental_codebase_spec_bilinear_spd_benchmark.md`** — the
   codebase spec. Read § 5.5 (decomposition training), § 5.6 (analysis),
   § 5.7 (metrics), § 5.8 (clustering), § 5.9 (ablations), § 6 (plots).
   Sections 1–4 you can skim; 11–16 are about post-writing.
4. **`docs/03_lesswrong_citations_and_style_guide.md`** — only if/when you
   reach Day 5 (post draft).

The PDFs in `docs/` (apd, spd, bilinear, bilinear-lee) are reference papers,
not required reading for the immediate task. If you want context: bilinear-lee
is most relevant (Lee et al. 2024 bilinear-MLP paper), apd/spd describe the
Goodfire parameter-decomposition lineage we're benchmarking.

---

## 3. Project overview (~3 minutes)

### What we're testing

Bilinear MLPs admit an **exact** rewrite of any scalar probe of the output as
a quadratic form in the input:

$$
r^\top y(x) = x^\top M_r x \quad \text{with} \quad M_r = \tfrac{1}{2}\sum_i (Cr)_i\,(a_i b_i^\top + b_i a_i^\top).
$$

The eigenspaces of $M_r$ are a canonical, basis-independent functional
mechanism decomposition (up to degenerate-eigenspace projector ambiguity).
SPD-style methods decompose the *weights* into rank-one atoms with sparse
causal-importance gates. **The central question:** do atoms (or clusters of
co-active atoms) align with eigenspaces of $M_r$?

Three possible findings:
- **H1**: atoms ≈ eigenspaces (rank-one ontology validated)
- **H2**: clusters ≈ eigenspaces, atoms alone don't (rank-one too fine-grained)
- **H3**: nothing aligns (negative benchmark, still informative)

### Why modular addition

Translation equivariance produces degenerate eigenvalues, which forces us to
work with eigenspace projectors instead of arbitrary eigenvector choices —
exactly the identifiability subtlety the theory doc emphasizes. We use $p ∈ \{13, 23, 47\}$;
larger $p$ ⇒ more frequencies ⇒ cleaner Fourier-flavored eigenspaces, so $p=47$
is the headline run.

### Why "SPD-style," not VPD

We do **not** use Goodfire's `param-decomp` repo. We cherry-picked four
small things from `nano_param_decomp/run.py` (with attribution comments) and
wrote everything else from scratch: their code is tightly coupled to LM
data / DDP / WandB / SLURM. See `src/bilinear_spd/sigmoids.py` and
`src/bilinear_spd/losses.py` for what was copied and from where.

Honest naming: call the method "SPD-style rank-one gated parameter
decomposition," **not** SPD or VPD.

---

## 4. Codebase layout

```
src/bilinear_spd/
  __init__.py
  utils.py            # set_seed, select_device, load_config, save_checkpoint, load_checkpoint, run_dir
  data.py             # make_modular_addition_dataset(p), make_xor_dataset(), make_dataset(cfg), input_dim, output_dim
  models.py           # BilinearMLP(d_in, d_hidden, d_out, use_bias, init_scale)
                      # Shapes: A,B ∈ [d_in, m]; C ∈ [m, d_out]
                      # Forward: y = ((x @ A) * (x @ B)) @ C
  functional.py       # construct_M_r(A, B, C, r) -> [d, d]
                      # eigendecompose_M(M) -> evals, evecs   (sorted by |λ| desc)
                      # group_eigenspaces(evals, evecs, rel_tol, min_abs_eval) -> list[EigenGroup]
                      # build_probes(probe_cfg, p) -> dict[str, Tensor]
                      # centered_class_probe(k, p), pairwise_probe(k, k', p)
                      # functional_scalar(y, r), quadratic_scalar(x, M)
  sigmoids.py         # lower_leaky(x, alpha), upper_leaky(x, alpha)
                      # *** CHERRY-PICKED FROM GOODFIRE nano_param_decomp/run.py § B ***
  losses.py           # kl_logits, frequency_penalty   (cherry-picked from nano § E)
                      # relative_param_recon, sparsity_lp, gate_l0   (ours)
  decomposition.py    # BilinearComponentMLP(target, C_A, C_B, gate_hidden)
                      # GateNet(d_in, gate_hidden, C) — small MLP, init so initial gates ≈ 0.5
                      # forward(x, mask_mode="stochastic"|"deterministic", return_aux=False)
                      # forward_target(x)   — frozen passthrough
                      # forward_with_masks(x, m_A, m_B, include_delta_A, include_delta_B)  — for ablation
  train.py            # train_bilinear(model, x, y, cfg, probe_for_grok=..., ...)
                      # train_decomposition(model, x, y_target_logits, cfg, mask_samples_per_step=1)
  plotting.py         # apply_style, COLORS, _group_palette
                      # plot_training_curves  (bilinear)
                      # plot_grokking_curves  (Day 2.5)
                      # plot_spectrum         (eigenvalues colored by group + summary box)
                      # plot_top_eigenvectors (heatmap, a-slot/b-slot split)
                      # plot_decomposition_curves (Day 3: loss/KL/recon/L0/mean-gate)

scripts/
  __init__.py
  train_bilinear.py      # CLI: trains the bilinear MLP, saves checkpoint + dataset + logits + figures + JSON summary
  analyze_functional.py  # CLI: builds M_r per probe, eigendecomposes, groups, saves mr_slices.pt + eigenspaces.pt + spectrum/eigvec figures
  train_decomp.py        # CLI: trains the gated rank-one decomposition; loads bilinear_mlp.pt, freezes it, saves decomposition.pt + gates.pt + decomp curve figure
                         # *** THIS IS THE NEXT THING TO RUN ***

configs/
  xor.yaml                  # Day 1 toy (memorization-regime baseline kept for reference)
  modadd_p7.yaml            # Day 1 toy
  modadd_p13.yaml           # Day 1 — memorization-regime baseline
  modadd_p13_grok.yaml      # Day 2.5 — wd=1.0, init=0.02, no early stop, grok metric tracking
  modadd_p23_grok.yaml      # Day 2.5
  modadd_p47_grok.yaml      # Day 2.5 — headline run, p=47, 60k steps, hidden=96

tests/
  test_shapes.py                    # 4 tests — data + model + M_r shapes
  test_functional_equivalence.py    # 2 tests — r⊤y(x) == x⊤M_r x at random init AND post-training
  test_eigenspace_projectors.py     # 3 tests — P²=P, P⊤=P, tr P = rank P on random/trained/synthetic
  test_decomposition.py             # 5 tests — forward identity, ablation correctness, mask invariants

runs/                              # one directory per (run_name, seed); checkpoints/artifacts gitignored
                                   # figures/ and reports/ ARE tracked so summary.md renders on GitHub
```

---

## 5. State as handed off

### What is done and verified (committed to git)

**Day 1** — bilinear MLP, modular-addition + XOR datasets, exact $M_r$
construction, functional-equivalence test (`r⊤y(x) == x⊤M_r x` to ~1e-6 on
all configs at random init and post-training).

**Day 2** — `eigendecompose_M`, `group_eigenspaces` with signed-proximity
$τ \cdot \max|λ|$ tolerance, projector tests, spectrum + top-eigenvector
figures. Discovered the Day 1 trained models were in the memorization regime
(zero degenerate eigenspaces at τ=0.001) — exactly the failure the theory
doc § 12.1 predicted for `wd=0` + overparameterized hidden.

**Day 2.5** — grokking-regime configs (`modadd_p{13,23,47}_grok`) with
`wd=1.0`, `init_scale=0.02`, no accuracy-based early stop, 30/40/60 k steps.
Training extended to track param L2 norm + top |λ| + degenerate-group count
during training. Result table at τ=0.01 on the centered_0 probe:

| run | top \|λ\| | # groups | # degenerate | max rank |
|---|---:|---:|---:|---:|
| `modadd_p13_grok` | 15.6 | 22 | 2 | 3 |
| `modadd_p23_grok` | 21.3 | 40 | 4 | 4 |
| `modadd_p47_grok` | 31.0 | 56 | **12** | **20** |

p47 has a rank-20 degenerate eigenspace. Eigenvectors visibly periodic across
the a-slot/b-slot. Bilinear MLP checkpoints, $M_r$ artifacts, eigenspace
groups, spectrum + eigenvector figures all committed under
`runs/modadd_p{13,23,47}_grok_seed0/`.

**Day 3** — code is written and committed:
- `src/bilinear_spd/sigmoids.py`, `losses.py`, `decomposition.py`
- `train_decomposition` added to `src/bilinear_spd/train.py`
- `scripts/train_decomp.py` CLI
- `plot_decomposition_curves` added to `src/bilinear_spd/plotting.py`
- `tests/test_decomposition.py` (5 tests, all pass)

**Day 3 has NOT been run end-to-end.** No `decomposition.pt` exists yet under
any `runs/*/checkpoints/`. The CPU run on p13_grok stalled at 14 minutes and
was killed.

### What is NOT done

- Decomposition training on any of the grokked targets (your immediate task).
- Day 4: atom + cluster perturbations, alignment metrics, clustering,
  alignment heatmaps. Spec § 5.6/5.7/5.8.
- Day 5: causal ablations + random/low-alignment baselines + final figures
  + post draft. Spec § 5.9/6.

---

## 6. Your job, concretely

### Phase A — verify the handoff

```bash
# 1. fresh clone + install + tests
git clone https://github.com/aniket-desh/bilinear-param-decomp.git
cd bilinear-param-decomp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                              # expect 14 passed

# 2. flip device: cpu → device: auto in the three grok configs
#    (sed -i works on Linux; on macOS use sed -i '')
for c in configs/modadd_p{13,23,47}_grok.yaml; do
  sed -i 's/^device: cpu$/device: auto/' "$c"
done
```

### Phase B — finish Day 3 (decomposition training)

```bash
# Use the original, high-quality settings: 20k steps × 4 mask samples.
# These are the values already in the grok configs. On GPU they should be
# fast (estimate based on shapes: each ~1-5 min per run on a single A100,
# ~5-15 min on a T4).

for c in modadd_p13_grok modadd_p23_grok modadd_p47_grok; do
  python -m scripts.train_decomp configs/${c}.yaml 2>&1 | tee /tmp/decomp_${c}.log
done
```

**Acceptance criteria for Day 3:**
- Each run writes `runs/<name>/checkpoints/decomposition.pt`,
  `artifacts/gates.pt`, `figures/decomposition_curves.png`,
  `reports/decomposition_summary.json`.
- Final `recon_A` and `recon_B` (relative Frobenius error) should be
  ≤ 0.05 — ideally ≤ 0.01 — meaning the atoms reconstruct A and B almost
  perfectly. If recon stays high (> 0.1), the loss balance is off; bump
  `lambda_param_A` / `lambda_param_B` in the config.
- Final KL behavior loss should be ≤ 1e-2 ideally, ≤ 1e-1 acceptable.
- Final `l0_A` / `l0_B` (count of active gates) should be **substantially
  less than C_A / C_B**, ideally on the order of the number of bilinear
  eigenspace mechanisms — say 5–20 for p47.
- All 14 unit tests still pass.

If recon never drops or KL stays high: see § 8 "gotchas / failure modes."

### Phase C — start Day 4 (alignment + clustering)

The spec § 5.6/5.7/5.8 describes this in detail. You will need to write:
- `src/bilinear_spd/analysis.py`: `atom_perturbations()` and
  `cluster_perturbations()` — computing $\Delta M_r(S) = M_r(A,B,C) - M_r(A-\Delta A_S, B-\Delta B_S, C)$
  for each atom $c$ and each cluster $S$. Use the perturbation expansion
  identity from theory § 7.2 to write a test.
- `src/bilinear_spd/metrics.py`: `fro_cos`, `signed_fro_cos`,
  `projector_energy`, `mean_max_alignment` — § 5.7 has the exact formulas.
- `src/bilinear_spd/clustering.py`: gate correlation matrix + threshold
  connected components (BFS, ~30 lines). § 5.8.
- `scripts/run_alignment.py`: loads decomposition, computes
  atom/cluster perturbations, computes atom×eigenspace and
  cluster×eigenspace alignment matrices, saves CSVs + heatmaps.
- A new test file: perturbation expansion (theory § 7.2), projector energy
  invariant, alignment metric sanity.

Plot deliverables (`plotting.py` extensions):
- `plot_alignment_heatmap(alignment_matrix, ...)` — atoms-vs-eigenspaces and
  clusters-vs-eigenspaces side-by-side.
- Optional: `plot_coactivation_graph(corr, ...)` if time permits.

**Acceptance criteria for Day 4:**
- One alignment-heatmap figure per run × probe combination.
- A `reports/alignment_summary.json` per run containing
  `mean_max_alignment_atoms` and `mean_max_alignment_clusters`.
- Tests still pass.
- `summary.md` updated with a Day 4 section showing the p47 alignment
  heatmap and the atoms-vs-clusters mean-max-align comparison (this is the
  H1/H2/H3 result).

### Phase D — Day 5 (causal ablations + final figures + post)

Spec § 5.9 for ablation, § 6 for figure aesthetics, § 16 for the strong
publishable target. Don't start this until Day 4 is committed.

---

## 7. Key design decisions (so you don't have to rediscover them)

1. **No persistent PGD.** The spec lists it as optional; nano's
   `PersistentPGD` is ~300 lines of DDP-coupled code. v1 uses pure
   stochastic-mask sampling, which is what the user (Aniket) asked for at
   Day 0. Defer PPGD until alignment is unblocked.

2. **`Δ` is implicit, not a parameter.** `BilinearComponentMLP` computes
   `delta_A = self.A_target - reconstruct_A()` on each forward. The Δ is
   *always unmasked* in the gated forward; only the atoms get masked. This
   matches the spec's § 2.2 formulation and keeps parameter reconstruction
   well-defined (recon error is `||delta_A||² / ||A||²`).

3. **Stochastic masks from day one.** `mask = g + (1-g) * U(0,1)` for atoms,
   independently per (batch, atom). Cherry-picked from Goodfire nano § E.
   The user explicitly chose this over deterministic-first when offered the
   choice on Day 0.

4. **`init_scale=0.02`, `weight_decay=1.0`.** These come from the Day 2.5
   investigation: without them you get memorization, not grokking. The Day 1
   configs (without `_grok` suffix) are kept as the memorization baseline
   for comparison; the headline runs are the `_grok` variants.

5. **`figures/` and `reports/` are tracked; `checkpoints/` and `artifacts/`
   are gitignored.** This means `summary.md` images render on GitHub but
   the repo doesn't bloat with PT files. See `.gitignore`.

6. **`mask_samples_per_batch: 4` in configs.** Variance reduction —
   average four stochastic-mask forward passes per training step. Costs 4×
   wall time. With GPU you can keep this; on CPU it was the bottleneck. If
   you ever need to drop it to 1, that's safe — the gradient will be
   noisier but the optimum should be the same.

7. **Gate net init: last-layer bias = 0.5.** `GateNet.__init__` zeros the
   last layer's weights and sets bias to 0.5, so initial raw logits are
   0.5 → `lower_leaky(0.5) = 0.5` → every atom contributes half-strength at
   step 0. Without this, gates start at 0 → atoms are dead → no gradient.

8. **All probes use `centered_class_probe` or `pairwise_probe`** from
   `src/bilinear_spd/functional.py`. Don't analyze raw logits directly —
   spec § 4.1 explains why.

---

## 8. Gotchas / failure modes

### Decomp recon error doesn't drop below 0.1

The atom-recon loss is *relative* (`||Δ||² / ||target||²`), so 0.1 means 10%
unexplained variance. If it plateaus there:
- Check that `C_A` and `C_B` are at least equal to `d_in` of the target.
  Configs are at 2 × d_hidden which should be plenty.
- Try increasing `lambda_param_A` / `lambda_param_B` from 1.0 to 10.
- Verify that the **frozen bilinear target** is the **grokked one**, not the
  Day 1 memorization checkpoint. The grok run dirs are
  `runs/modadd_p<p>_grok_seed0/`.

### KL behavior loss stays high (> 0.1)

- The atoms aren't reconstructing A/B well (see above).
- Or `lambda_behavior` is too low relative to recon. Try bumping to 10.0.
- Or the gates are saturating — check `l0_A` / `l0_B` in the curve. If they
  are at 0 (all dead), the gate-net init is broken; check that
  `GateNet.layers[-1].bias` is ~0.5 at step 0.

### Some grok config's bilinear training didn't groke

Unlikely — the committed checkpoints already groked — but if you re-train:
the canonical signal is `n_degenerate_tau_0.01 > 0` and growing during
training. If it stays at 0, increase `weight_decay`, use a smaller
`init_scale`, or run longer.

### Tests fail after editing

The two most fragile invariants:
- `tests/test_functional_equivalence.py` checks `r⊤y(x) == x⊤M_r x`. If you
  break the shape convention of A, B, C (specifically the `m` dimension
  placement), this breaks immediately.
- `tests/test_decomposition.py::test_full_atoms_plus_delta_equals_target`
  requires that `Δ = target − Σ U V⊤` is computed correctly. Don't refactor
  `BilinearComponentMLP.reconstruct_A`/`reconstruct_B` without re-running
  this test.

### CPU vs GPU tensor placement

`select_device` in `utils.py` returns the correct device for the
`device:` config key. `auto` picks cuda > mps > cpu. The training scripts
move both the model and the dataset to that device. If you see "Expected
all tensors to be on the same device" errors, the `make_dataset` call
might be using a stale device — pass `device=device` explicitly.

---

## 9. Where to push your work

The repo lives at https://github.com/aniket-desh/bilinear-param-decomp .
You should have push access already (or be in a fork). Commit message style
established in the existing log (see `git log` for examples):

```
Day 3: train gated rank-one decomposition on three grokked targets

<a few sentences on what changed and what the headline numbers are>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

(If you're a different model, use your own model name. The "Co-Authored-By
Claude" convention is fine to keep across model versions.)

When you commit Day 3, update `summary.md` with:
- the Day 3 section header
- final decomp metrics table for the three runs
- the `decomposition_curves.png` figures (or one of them — pick p47)
- queue Day 4 in "Up next"

---

## 10. Aniket's preferences (observed)

- Wants honest reporting of negative results, not just success stories. The
  Day 2 memorization finding was committed prominently rather than hidden;
  follow that pattern if Day 3 turns up something unexpected.
- Wants the LessWrong post audience in mind throughout — figures should be
  clean, captions explanatory, claims hedged.
- Likes one summary doc (`summary.md`) that grows across days, not a
  scatter of per-day READMEs.
- Wants `git add` to be explicit (avoid `git add .`). Use named files / dirs.
- Wants `docs/` PDFs and theory docs committed (this is the current state
  after you read this file).
- Defers to the spec when a design decision is in scope (read § 5–9 of
  `docs/02`). Asks before deviating.
- Likes terse, calibrated communication. Don't pad. If you don't know,
  say so. If something's working, one sentence is fine.

Good luck.
