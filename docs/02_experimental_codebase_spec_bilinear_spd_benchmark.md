# Experimental Codebase Specification: Minimal Bilinear-SPD Benchmark

**Project name:** `bilinear-spd-benchmark`  
**Goal:** build a minimal, reproducible codebase that trains a toy bilinear MLP, extracts exact probe-conditioned functional eigenspaces, learns an SPD-style rank-one gated parameter decomposition, and compares parameter atoms/clusters to exact bilinear mechanisms.

## 0. Final deliverable

The repo should support one command:

```bash
python -m scripts.run_all configs/modadd_p13.yaml
```

and produce:

```text
runs/modadd_p13_seed0/
  config.yaml
  checkpoints/
    bilinear_mlp.pt
    decomposition.pt
  artifacts/
    dataset.pt
    logits.pt
    mr_slices.pt
    eigenspaces.pt
    gates.pt
    atom_perturbations.pt
    cluster_assignments.json
    alignment_atoms.csv
    alignment_clusters.csv
    ablation_results.csv
  figures/
    fig1_spectrum.png
    fig2_alignment_heatmaps.png
    fig3_ablation.png
    fig_main_three_panel.png
  reports/
    metrics.md
    run_summary.json
```

The LessWrong post should be possible to write from these artifacts.

## 1. Design constraints

### 1.1 Minimal but not sloppy

The codebase should be small enough to write quickly but serious enough that a Goodfire/MATS reader can inspect it without cringing.

Priorities:

1. correctness of \(M_r\);
2. reproducibility;
3. readable, typed modules;
4. stable figures;
5. explicit sanity checks;
6. honest failure modes.

Non-priorities:

1. scaling to large language models;
2. implementing all of VPD;
3. supporting every architecture;
4. fancy UI;
5. large hyperparameter sweeps.

### 1.2 Honest naming

If you implement a simplified decomposition, call it:

- `MinimalGatedRankOneDecomp`
- `spd_style`
- `rank_one_gated_decomp`

Do **not** call it `VPD` unless you actually use the Goodfire implementation.

## 2. External codebases and links

### 2.1 Goodfire parameter-decomposition repos

- Goodfire organization: <https://github.com/goodfire-ai>
- Current parameter decomposition repo: <https://github.com/goodfire-ai/param-decomp>
- SPD branch in the same repo: <https://github.com/goodfire-ai/param-decomp/tree/spd-paper>
- `nano_param_decomp`: <https://github.com/goodfire-ai/param-decomp/tree/main/nano_param_decomp>
- VPD paper/post: <https://www.goodfire.ai/research/interpreting-lm-parameters>
- SPD explainer: <https://www.goodfire.ai/research/stochastic-param-decomp>
- SPD arXiv: <https://arxiv.org/abs/2506.20790>
- APD arXiv: <https://arxiv.org/abs/2501.14926>

The Goodfire repo is especially useful because it now includes a `nano_param_decomp` directory described as a self-contained implementation of the method. Inspect this before deciding how much of your own minimal implementation to write.

### 2.2 Bilinear MLP references

- Bilinear MLPs enable weight-based mechanistic interpretability: <https://arxiv.org/abs/2410.08417>
- Technical note on bilinear layers: <https://arxiv.org/abs/2305.03452>
- Earlier bilinear decomposition preprint: <https://arxiv.org/abs/2406.03947>

### 2.3 Useful libraries

- PyTorch: <https://pytorch.org/>
- einops: <https://github.com/arogozhnikov/einops>
- jaxtyping, optional: <https://github.com/patrick-kidger/jaxtyping>
- rich, optional CLI prettiness: <https://github.com/Textualize/rich>
- matplotlib: <https://matplotlib.org/>
- seaborn: <https://seaborn.pydata.org/>
- plotly, optional: <https://plotly.com/python/>
- scikit-learn: <https://scikit-learn.org/stable/>

Use matplotlib/seaborn for static post figures; optionally use plotly for debugging.

## 3. Repository structure

```text
bilinear-spd-benchmark/
  README.md
  pyproject.toml
  configs/
    modadd_p7.yaml
    modadd_p13.yaml
    xor.yaml
  scripts/
    run_all.py
    train_bilinear.py
    train_decomp.py
    analyze_functional.py
    plot_figures.py
  src/
    bilinear_spd/
      __init__.py
      data.py
      models.py
      functional.py
      decomposition.py
      train.py
      analysis.py
      clustering.py
      ablations.py
      plotting.py
      metrics.py
      utils.py
  tests/
    test_functional_equivalence.py
    test_shapes.py
    test_eigenspace_projectors.py
    test_perturbation_expansion.py
  notebooks/
    00_sanity.ipynb
    01_figures.ipynb
  runs/
    .gitkeep
```

## 4. Config design

Example:

```yaml
project: bilinear_spd_benchmark
run_name: modadd_p13_seed0

seed: 0
device: cuda

task:
  name: modular_addition
  p: 13
  input_encoding: concat_onehot
  train_fraction: 1.0
  eval_fraction: 1.0

model:
  d_hidden: 32
  use_bias: false
  init_scale: 0.02

train_model:
  optimizer: adamw
  lr: 0.003
  weight_decay: 0.0
  steps: 10000
  batch_size: 169
  log_every: 100
  target_accuracy: 0.99

probes:
  type: centered_class
  classes: [0, 1, 2, 3]
  include_pairwise:
    - [0, 1]
    - [0, 2]

eigenspaces:
  rel_tol: 0.001
  min_abs_eval: 1.0e-6

decomposition:
  method: minimal_gated_rank_one
  decompose: [A, B]
  C_A: 64
  C_B: 64
  gate_hidden: 64
  gate_activation: relu
  gate_temperature: 1.0
  use_stochastic_masks: true
  mask_samples_per_batch: 4

train_decomp:
  optimizer: adamw
  lr: 0.001
  steps: 20000
  batch_size: 169
  lambda_behavior: 1.0
  lambda_param_A: 1.0
  lambda_param_B: 1.0
  lambda_sparsity: 0.001
  lambda_frequency: 0.001
  p_sparsity: 1.0
  log_every: 100

clustering:
  method: threshold_connected_components
  corr_thresholds: [0.5, 0.6, 0.7, 0.8, 0.9]
  selected_threshold: 0.7

ablations:
  n_random_baselines: 100
  low_alignment_quantile: 0.25

plotting:
  style: goodfire_light
  dpi: 300
  save_svg: true
```

## 5. Module details

## 5.1 `data.py`

### Function: `make_modular_addition_dataset`

Signature:

```python
def make_modular_addition_dataset(
    p: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    ...
```

Inputs:

- prime or small integer modulus \(p\);
- device.

Outputs:

```python
{
    "x": Tensor[n_examples, 2*p],
    "a": Tensor[n_examples],
    "b": Tensor[n_examples],
    "y": Tensor[n_examples],
}
```

Where:

\[
x=[e_a;e_b],
\qquad
y=(a+b)\bmod p.
\]

### Implementation notes

```python
def one_hot(i, p):
    return F.one_hot(i, num_classes=p).float()
```

For all pairs:

```python
pairs = [(a, b) for a in range(p) for b in range(p)]
```

No need to overcomplicate train/test split for the first version. Exact table learning is acceptable because the benchmark is about learned weight structure, not generalization.

### Optional tasks

- XOR:
  \[
  y=a\oplus b.
  \]
- Sparse parity:
  \[
  y=\sum_i x_i \mod 2.
  \]
- Modular subtraction:
  \[
  y=(a-b)\bmod p.
  \]

Keep these optional.

## 5.2 `models.py`

### Class: `BilinearMLP`

```python
class BilinearMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, use_bias: bool = False):
        super().__init__()
        self.A = nn.Parameter(torch.empty(d_in, d_hidden))
        self.B = nn.Parameter(torch.empty(d_in, d_hidden))
        self.C = nn.Parameter(torch.empty(d_hidden, d_out))
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = x @ self.A
        z2 = x @ self.B
        h = z1 * z2
        y = h @ self.C
        return y
```

Shape convention:

\[
A,B\in\mathbb R^{d_{\text{in}}\times m},
\qquad
C\in\mathbb R^{m\times d_{\text{out}}}.
\]

This convention matches the theory:

\[
h=(A^\top x)\odot(B^\top x),
\quad
y=C^\top h.
\]

In batch notation:

\[
X A,\quad X B,\quad (XA)\odot(XB),\quad ((XA)\odot(XB))C.
\]

### Class: `MinimalGatedRankOneDecomp`

This model wraps a frozen bilinear MLP and replaces \(A,B\) with gated rank-one decompositions.

Key parameters:

```python
self.U_A: [C_A, d_in]
self.V_A: [C_A, d_hidden]
self.U_B: [C_B, d_in]
self.V_B: [C_B, d_hidden]
```

Reconstruct:

```python
A_hat = torch.einsum("cd,ch->dh", U_A, V_A)
B_hat = torch.einsum("cd,ch->dh", U_B, V_B)
```

Gates:

```python
gate_net_A: x -> [C_A]
gate_net_B: x -> [C_B]
```

Gated reconstruction for each batch item:

```python
A_g[n,d,h] = sum_c g_A[n,c] U_A[c,d] V_A[c,h]
B_g[n,d,h] = sum_c g_B[n,c] U_B[c,d] V_B[c,h]
```

Use `einsum`:

```python
A_g = torch.einsum("nc,cd,ch->ndh", g_A, U_A, V_A)
B_g = torch.einsum("nc,cd,ch->ndh", g_B, U_B, V_B)
```

Then:

```python
zA = torch.einsum("nd,ndh->nh", x, A_g)
zB = torch.einsum("nd,ndh->nh", x, B_g)
y_hat = (zA * zB) @ C
```

### Gate network

Minimal version:

```python
self.gate_A = nn.Sequential(
    nn.Linear(d_in, gate_hidden),
    nn.ReLU(),
    nn.Linear(gate_hidden, C_A),
    nn.Sigmoid(),
)
```

Same for \(B\).

Optional hard-concrete / Gumbel gates can wait.

## 5.3 `functional.py`

This is the most important module.

### Function: `construct_M_r`

```python
def construct_M_r(
    A: torch.Tensor,  # [d, m]
    B: torch.Tensor,  # [d, m]
    C: torch.Tensor,  # [m, p]
    r: torch.Tensor,  # [p]
) -> torch.Tensor:    # [d, d]
    alpha = C @ r  # [m]
    raw = torch.einsum("m,dm,em->de", alpha, A, B)
    M = 0.5 * (raw + raw.T)
    return M
```

Check:

\[
\alpha_i=(Cr)_i.
\]

### Function: `functional_scalar`

```python
def functional_scalar(model, x, r):
    return model(x) @ r
```

### Function: `quadratic_scalar`

```python
def quadratic_scalar(x, M):
    return torch.einsum("bd,de,be->b", x, M, x)
```

### Test

```python
with torch.no_grad():
    y1 = functional_scalar(model, x, r)
    M = construct_M_r(model.A, model.B, model.C, r)
    y2 = quadratic_scalar(x, M)
    assert torch.max(torch.abs(y1 - y2)) < 1e-5
```

This test is mandatory.

## 5.4 `functional.py`: eigenspaces

### Function: `eigendecompose_M`

```python
def eigendecompose_M(M: torch.Tensor):
    evals, evecs = torch.linalg.eigh(M)
    idx = torch.argsort(torch.abs(evals), descending=True)
    return evals[idx], evecs[:, idx]
```

### Function: `group_eigenspaces`

Input:

- eigenvalues \(\lambda_j\);
- eigenvectors \(q_j\);
- relative tolerance.

Output:

```python
[
  {
    "indices": [0, 1],
    "evals": Tensor[2],
    "projector": Tensor[d, d],
    "rank": 2,
    "mean_abs_eval": float,
  },
  ...
]
```

Projector:

```python
Q_group = evecs[:, indices]
P = Q_group @ Q_group.T
```

Tests:

1. \(P^2\approx P\)
2. \(P^\top\approx P\)
3. \(\operatorname{tr}(P)\approx \operatorname{rank}(P)\)

## 5.5 `train.py`

### Training the bilinear MLP

Loss:

```python
loss = F.cross_entropy(logits, y)
```

Metrics:

- accuracy;
- loss;
- logit margin;
- parameter norms.

Stop early if:

\[
\text{accuracy}\geq 0.99
\]

for 500 consecutive steps.

### Training the decomposition

Freeze \(C\) and target \(A,B\). Train \(U,V\) and gates.

Loss pieces:

```python
logits_target = frozen_model(x).detach()
logits_decomp, gates = decomp_model(x)

loss_behavior = kl_divergence(logits_target, logits_decomp)
loss_param_A = mse(A_hat, A_target)
loss_param_B = mse(B_hat, B_target)
loss_sparse = gates.abs().mean()
loss_freq = frequency_penalty(gates)

loss = (
    lambda_behavior * loss_behavior
    + lambda_param_A * loss_param_A
    + lambda_param_B * loss_param_B
    + lambda_sparsity * loss_sparse
    + lambda_frequency * loss_freq
)
```

### KL function

```python
def kl_logits(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return F.kl_div(q, p.exp(), reduction="batchmean")
```

Be careful: PyTorch's `kl_div(input, target)` expects input log-probs and target probabilities.

### Stochastic masks

Optional but recommended.

Instead of using \(g\) directly:

\[
m_c\sim \mathrm{Bernoulli}(g_c)
\]

with straight-through estimator, or

\[
m_c = g_c\cdot \epsilon_c,\quad \epsilon_c\sim \mathrm{Uniform}(0,1)
\]

for a softer SPD-style stochastic mask.

Minimal first version: use deterministic \(g\). Add stochastic masks if time permits.

## 5.6 `analysis.py`

### Extract gates

```python
def collect_gates(decomp_model, x):
    with torch.no_grad():
        _, aux = decomp_model(x, return_aux=True)
    return {
        "g_A": aux["g_A"],  # [N, C_A]
        "g_B": aux["g_B"],  # [N, C_B]
        "g": torch.cat([g_A, g_B], dim=-1),
    }
```

### Atom perturbations

For an atom \(c\) in \(A\):

\[
\Delta A_c=u^A_c(v^A_c)^\top,
\qquad
\Delta B_c=0.
\]

For an atom \(c\) in \(B\):

\[
\Delta A_c=0,
\qquad
\Delta B_c=u^B_c(v^B_c)^\top.
\]

Then compute:

\[
\Delta M_r(\{c\})=M_r(A,B,C)-M_r(A-\Delta A_c,B-\Delta B_c,C).
\]

### Cluster perturbations

Sum atoms first:

```python
delta_A = sum_A_atoms(cluster)
delta_B = sum_B_atoms(cluster)
delta_M = M_full - construct_M_r(A - delta_A, B - delta_B, C, r)
```

Do not sum individual \(\Delta M\)'s blindly when clusters include both \(A\)- and \(B\)-atoms, because the bilinear cross term matters.

### Perturbation expansion test

Check:

\[
\Delta M_r(S)
=
M_r(\Delta A_S,B,C)
+
M_r(A,\Delta B_S,C)
-
M_r(\Delta A_S,\Delta B_S,C).
\]

Implement a test that compares direct ablation to the expansion.

## 5.7 `metrics.py`

### Frobenius alignment

```python
def fro_cos(A, B, eps=1e-12):
    return torch.abs(torch.sum(A * B)) / (torch.linalg.norm(A) * torch.linalg.norm(B) + eps)
```

### Signed alignment

```python
def signed_fro_cos(A, B, eps=1e-12):
    return torch.sum(A * B) / (torch.linalg.norm(A) * torch.linalg.norm(B) + eps)
```

### Energy capture

```python
def projector_energy(delta_M, P, eps=1e-12):
    projected = P @ delta_M @ P
    return torch.linalg.norm(projected, "fro")**2 / (torch.linalg.norm(delta_M, "fro")**2 + eps)
```

### Mean max alignment

```python
def mean_max_alignment(alignment_matrix):
    # rows: atoms/clusters, cols: eigenspaces
    return alignment_matrix.max(dim=0).values.mean()
```

## 5.8 `clustering.py`

### Correlation matrix

```python
def gate_corr(g):
    g_centered = g - g.mean(dim=0, keepdim=True)
    g_normed = g_centered / (g_centered.std(dim=0, keepdim=True) + 1e-8)
    return (g_normed.T @ g_normed) / (g.shape[0] - 1)
```

### Threshold connected components

```python
def threshold_components(corr, threshold):
    adjacency = corr > threshold
    adjacency.fill_diagonal_(False)
    # BFS/DFS connected components
```

### Cluster selection

For each threshold:

- number of clusters;
- median cluster size;
- mean max cluster alignment;
- ablation KL.

Select threshold that gives:

- nontrivial clusters;
- stable alignment;
- interpretable cluster sizes.

Do not overfit threshold in the main post. Show sensitivity in appendix or GitHub README.

## 5.9 `ablations.py`

### Function: `ablate_atoms`

For a cluster \(S\), construct:

\[
A_{-S}=A-\Delta A_S,\quad B_{-S}=B-\Delta B_S.
\]

Evaluate:

```python
logits_base = model(x)
logits_ablated = bilinear_forward_with_matrices(x, A_minus, B_minus, C)
```

Metrics:

- accuracy change;
- KL;
- cross entropy;
- mean logit margin;
- probe scalar change;
- functional projector change.

### Random baseline

```python
def random_size_matched_sets(n_atoms, size, n_samples, exclude=None):
    ...
```

For every aligned cluster:

- sample \(100\) random sets of same size;
- compute metric distribution;
- plot mean ± std or full violin.

### Low-alignment baseline

Choose clusters with:

- similar \(\|\Delta M_r(S)\|_F\);
- low alignment to the target eigenspace.

## 5.10 `plotting.py`

## 6. Figure aesthetic: Goodfire-inspired but not copycat

Goodfire papers/posts tend to use:

- clean white background;
- readable explanatory captions;
- diagrams that introduce the object before the result;
- simple color palettes;
- lots of whitespace;
- figure panels that answer one question each;
- accessible prose surrounding technical figures;
- visible tables with clear labels.

For your figures:

### General style

```python
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
```

Use colors like:

```python
COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "gray": "#9D9DA1",
}
```

### Figure 1: Functional spectrum

Panel A:

- x-axis: eigenspace index;
- y-axis: \(|\lambda|\) or summed \(|\lambda|\) in eigenspace;
- annotate degeneracies.

Panel B:

- heatmap or line plots of top eigenvectors/eigenspaces in the input basis;
- split first \(p\) coordinates and second \(p\) coordinates;
- mark \(a\)-slot and \(b\)-slot.

### Figure 2: Alignment heatmaps

Two subpanels:

1. atoms vs eigenspaces;
2. clusters vs eigenspaces.

Rows sorted by max alignment.

Add summary text:

```text
Mean max alignment:
atoms = 0.xx
clusters = 0.yy
```

### Figure 3: Causal ablation

For selected eigenspaces:

- aligned cluster;
- random size-matched baseline;
- low-alignment cluster.

Metrics:

- \(\Delta\) KL;
- \(\Delta\) accuracy;
- \(\Delta\) functional coefficient.

Use one primary metric in the main figure, put others in supplementary figures.

### Figure 4: Optional co-activation graph

Nodes = atoms. Edges = high correlation. Color nodes by most-aligned eigenspace.

Only include if it looks clean. Otherwise leave in README.

## 7. Scripts

### `scripts/train_bilinear.py`

```bash
python -m scripts.train_bilinear configs/modadd_p13.yaml
```

Steps:

1. load config;
2. seed everything;
3. create dataset;
4. train bilinear MLP;
5. save checkpoint;
6. write model metrics.

### `scripts/analyze_functional.py`

```bash
python -m scripts.analyze_functional runs/modadd_p13_seed0/config.yaml
```

Steps:

1. load bilinear MLP;
2. construct probes;
3. compute \(M_r\);
4. verify scalar equivalence;
5. eigendecompose \(M_r\);
6. group eigenspaces;
7. save artifacts and spectrum plot.

### `scripts/train_decomp.py`

```bash
python -m scripts.train_decomp runs/modadd_p13_seed0/config.yaml
```

Steps:

1. load frozen bilinear MLP;
2. instantiate decomposition;
3. train;
4. save gates and decomposition checkpoint;
5. write loss curves.

### `scripts/run_alignment.py`

```bash
python -m scripts.run_alignment runs/modadd_p13_seed0/config.yaml
```

Steps:

1. load decomposition;
2. compute atom perturbations;
3. compute atom/eigenspace alignment;
4. collect gates;
5. cluster gates;
6. compute cluster perturbations;
7. compute cluster/eigenspace alignment;
8. save CSVs and heatmaps.

### `scripts/run_ablation.py`

```bash
python -m scripts.run_ablation runs/modadd_p13_seed0/config.yaml
```

Steps:

1. choose best-aligned clusters;
2. run ablations;
3. run random baselines;
4. run low-alignment baselines;
5. save results;
6. plot ablation figure.

### `scripts/run_all.py`

Runs everything in order.

## 8. Tests

### 8.1 Shape tests

- `BilinearMLP.forward` returns `[batch, p]`.
- `construct_M_r` returns `[d, d]`.
- gated matrices return `[batch, d, hidden]`.
- gates return `[batch, C]`.

### 8.2 Functional equivalence test

For random \(A,B,C,x,r\):

\[
r^\top y(x)=x^\top M_r x.
\]

This should pass before any training.

### 8.3 Eigenspace projector test

For every projector \(P\):

\[
P^2=P,
\qquad
P^\top=P,
\qquad
\operatorname{tr}(P)=\operatorname{rank}(P).
\]

### 8.4 Perturbation expansion test

Compare:

\[
M_r(A,B,C)-M_r(A-\Delta A,B-\Delta B,C)
\]

with

\[
M_r(\Delta A,B,C)+M_r(A,\Delta B,C)-M_r(\Delta A,\Delta B,C).
\]

### 8.5 Save/load test

Train a tiny model for 3 steps, save, load, ensure logits match.

## 9. Metrics to log

### Bilinear model

- train loss;
- train accuracy;
- logit margin;
- parameter norms;
- spectrum summary after training.

### Functional analysis

- max scalar equivalence error;
- eigenvalues;
- eigenspace ranks;
- number of significant eigenspaces;
- degeneracy groups.

### Decomposition

- parameter reconstruction error:
  \[
  \|A-\hat A\|_F/\|A\|_F
  \]
  and same for \(B\);
- KL reconstruction;
- prediction agreement;
- average gate L0:
  \[
  \mathbb E_x \sum_c \mathbf 1[g_c(x)>\epsilon];
  \]
- mean gate activation;
- dead components.

### Alignment

- atom mean max alignment;
- cluster mean max alignment;
- best cluster per eigenspace;
- threshold sensitivity.

### Ablation

- \(\Delta\) accuracy;
- KL;
- \(\Delta\) logit margin;
- \(\Delta\) functional coefficient;
- random baseline z-score.

## 10. Good failure-mode handling

### 10.1 Bilinear model does not train

Try:

- \(p=7\);
- hidden dimension \(m=64\);
- lower learning rate;
- train on full table;
- use Adam instead of AdamW;
- initialize \(A,B,C\) with larger scale.

### 10.2 Spectrum is degenerate

This is not a failure. Use projectors.

### 10.3 Spectrum is messy / no Fourier-like structure

Still valid, but the post becomes less elegant. Options:

- reduce hidden dimension;
- increase regularization;
- try multiple seeds;
- use pairwise probe \(e_0-e_1\);
- switch to XOR/parity for a cleaner known mechanism.

### 10.4 Decomposition does not learn sparse gates

Options:

- increase \(\lambda_g\);
- initialize atoms from SVD of \(A,B\);
- freeze atoms and train gates first;
- train atoms for parameter reconstruction first, then add behavior loss;
- use deterministic gates before stochastic masks.

### 10.5 Atoms/clusters do not align

This is scientifically interesting. Investigate:

- SVD baseline;
- whether gates cluster by input class rather than functional eigenspace;
- whether probe \(r\) choice changes result;
- whether alignment improves for data-weighted metric.

### 10.6 Goodfire repo integration is hard

Do not get stuck. Use `nano_param_decomp` as a reference and proceed with your minimal implementation.

## 11. Development timeline

### Day 1

- repo skeleton;
- modular addition data;
- bilinear model training;
- save checkpoints;
- functional equivalence test.

### Day 2

- \(M_r\) constructor;
- eigenspaces/projectors;
- spectrum and eigenvector plots;
- first draft of theory section.

### Day 3

- minimal rank-one decomposition;
- gates;
- train decomposition;
- reconstruction metrics.

### Day 4

- atom perturbations;
- cluster gates;
- alignment heatmaps;
- ablation metrics.

### Day 5

- final figures;
- README;
- LessWrong post draft;
- push GitHub.

## 12. README structure

```markdown
# Bilinear SPD Benchmark

This repo tests whether rank-one parameter decompositions recover exact bilinear functional mechanisms.

## Quickstart

...

## Theory

...

## Experiments

...

## Figures

...

## Reproduction

...

## Limitations

...
```

## 13. Reproduction commands

```bash
git clone https://github.com/aniket/.../bilinear-spd-benchmark
cd bilinear-spd-benchmark
uv sync
uv run python -m scripts.run_all configs/modadd_p13.yaml
```

Or:

```bash
conda create -n bilinear-spd python=3.11
conda activate bilinear-spd
pip install -e .
python -m scripts.run_all configs/modadd_p13.yaml
```

## 14. Suggested GitHub issues/tasks

Open these as issues in your repo:

1. Implement modular addition dataset.
2. Implement bilinear MLP.
3. Prove and test \(r^\top y=x^\top M_rx\).
4. Implement eigenspace projector grouping.
5. Implement minimal gated rank-one decomposition.
6. Implement perturbation expansion.
7. Implement atom/eigenspace alignment.
8. Implement gate co-activation clustering.
9. Implement ablation baselines.
10. Generate Goodfire-style figures.
11. Write LessWrong post.

## 15. Minimum publishable result

If time gets tight, the minimum viable post is:

1. train bilinear model;
2. compute \(M_r\);
3. implement SVD-initialized rank-one parameter atoms;
4. cluster gates or atom effects;
5. show alignment heatmap;
6. discuss limitations honestly.

The ablation panel is desirable but not strictly necessary if the alignment result is crisp.

## 16. Strong publishable result

The strong version has:

1. exact \(M_r\) derivation;
2. projector handling for degeneracy;
3. atom vs cluster comparison;
4. causal ablation baseline;
5. GitHub reproduction command;
6. LessWrong post with one clean three-panel figure.

That is enough signal for a MATS/Goodfire-facing project.
