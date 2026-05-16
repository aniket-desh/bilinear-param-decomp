# Theory Breakdown: A Bilinear Benchmark for Weight-Based Parameter Decomposition

**Working title:** *Do Rank-One Parameter Decompositions Recover Bilinear Mechanisms?*  
**Purpose:** a LessWrong-facing mini-project that tests whether SPD/VPD-style rank-one parameter subcomponents align with exact, probe-conditioned functional mechanisms in a bilinear MLP.

## 0. One-paragraph summary

Current Goodfire-style parameter decomposition methods try to decompose neural network weights into simple, sparsely used parameter subcomponents. Bilinear MLPs are unusually attractive as a benchmark because, unlike GELU/ReLU MLPs, their nonlinear computation can be rewritten exactly as a quadratic form. For a bilinear MLP and a chosen output/logit probe direction \(r\), the scalar output \(r^\top y(x)\) equals \(x^\top M_r x\), where \(M_r\) is a symmetric interaction matrix constructed directly from the weights. The eigendecomposition of \(M_r\) gives a canonical, probe-conditioned functional mechanism basis. The mini-project asks: when we learn or approximate an SPD-style rank-one parameter decomposition of the bilinear weights, do individual rank-one parameter atoms align with these functional eigenspaces, or do meaningful mechanisms only appear after clustering co-active atoms?

## 1. Why this project is conceptually nontrivial

A shallow version of the project would say:

> Bilinear MLPs have eigendecompositions. SPD has rank-one atoms. Let’s compare them.

That is not quite right. The more precise version is:

> SPD/VPD-style atoms live in **parameter space**, while bilinear eigendirections live in **function space**. The relevant question is whether simple parameter-space components induce simple function-space perturbations.

The project therefore studies the map

\[
\text{parameter subcomponent } S
\quad\longmapsto\quad
\Delta M_r(S),
\]

where \(\Delta M_r(S)\) is the change in the probe-conditioned bilinear interaction matrix caused by ablating parameter subcomponents \(S\).

This is the central conceptual move. It avoids the common mistake of comparing vectors in the wrong spaces.

## 2. Background: parameter decomposition

### 2.1 The parameter-first motivation

Activation-based interpretability methods, especially sparse dictionary learning / SAEs, decompose hidden activations into sparse directions. They often expose useful human-interpretable correlates of model behavior, but they do not directly decompose the objects that compute the behavior: the weights and nonlinearities.

The parameter-decomposition agenda starts from a different premise:

> If we want a causal account of the algorithm learned by a network, we should try to decompose the network’s parameters into reusable, sparsely used computational parts.

In the APD/SPD/VPD line, the ideal decomposition satisfies several desiderata:

1. **Parameter faithfulness.** The learned components sum to, or approximately reconstruct, the original parameters.
2. **Behavioral/mechanistic faithfulness.** On any input, ablating components predicted to be unnecessary should not substantially change the model’s output.
3. **Minimality.** Only a small number of components should be necessary on each input.
4. **Simplicity.** Each component should be structurally simple, often rank-one or otherwise low-complexity.

### 2.2 Rank-one matrix subcomponents

For a target weight matrix \(W^\ell\), VPD parameterizes subcomponents as rank-one outer products:

\[
W^\ell \approx \sum_{c=1}^{C_\ell} U_c^\ell (V_c^\ell)^\top + \Delta^\ell,
\]

where

\[
U_c^\ell\in \mathbb R^{d_{\text{out}}},
\qquad
V_c^\ell\in \mathbb R^{d_{\text{in}}},
\qquad
U_c^\ell (V_c^\ell)^\top \in \mathbb R^{d_{\text{out}}\times d_{\text{in}}}.
\]

The extra residual term \(\Delta^\ell\) enforces exact or approximate parameter reconstruction.

For an input \(x\), a causal importance function predicts values

\[
g^\ell_c(x)\in[0,1],
\]

or, in language models,

\[
g^\ell_{b,t,c}\in[0,1],
\]

where \(b\) is the batch index and \(t\) is token position. A small \(g\) should mean the subcomponent is ablatable for that datapoint; a large \(g\) means it is needed.

The masked weight matrix is schematically

\[
W^\ell_g(x)
=
\sum_{c=1}^{C_\ell} m^\ell_c(x) U_c^\ell (V_c^\ell)^\top + \Delta^\ell,
\]

where \(m_c^\ell(x)\) is an ablation mask constrained by the causal importance value \(g_c^\ell(x)\). In VPD, some masks are adversarially chosen to maximize output reconstruction loss subject to the mask constraints.

### 2.3 The ontology question

The mathematical and philosophical question is:

> Are rank-one subcomponents a good ontology for mechanisms?

Rank-one matrices are simple, but mechanisms may be:

- low-rank blocks;
- eigenspaces;
- tensor factors;
- circuits involving multiple matrices;
- temporally extended structures;
- clusters of co-active rank-one atoms.

This mini-project does not try to settle the ontology question in general. It asks a narrow, controlled version:

> In a model where a useful functional mechanism basis is exactly computable, how do rank-one parameter atoms relate to it?

That is a real test of the rank-one-subcomponent ontology.

## 3. Background: bilinear MLPs

### 3.1 Standard MLP difficulty

A standard MLP layer has the form

\[
\operatorname{MLP}(x)=W_{\text{out}}\phi(W_{\text{in}}x),
\]

where \(\phi\) is usually GELU, ReLU, SwiGLU, or another elementwise nonlinearity. The elementwise nonlinearity makes the function hard to analyze as a clean multilinear object.

For example, with GELU:

\[
\phi(z_i)=z_i\Phi(z_i),
\]

and the induced input-output behavior depends on smooth gating regions that are not algebraically simple.

### 3.2 Bilinear MLP definition

A bilinear MLP removes the elementwise scalar nonlinearity but keeps multiplicative interaction:

\[
h(x)=(A^\top x)\odot(B^\top x),
\]

\[
y(x)=C^\top h(x).
\]

Here

\[
x\in \mathbb R^d,
\qquad
A,B\in\mathbb R^{d\times m},
\qquad
C\in\mathbb R^{m\times p},
\qquad
y(x)\in\mathbb R^p.
\]

Let

\[
a_i=A_{:,i},\qquad b_i=B_{:,i},\qquad c_i=C_{i,:}.
\]

Then the \(k\)-th output logit is

\[
y_k(x)
=
\sum_{i=1}^m C_{i,k}(a_i^\top x)(b_i^\top x).
\]

This is nonlinear in \(x\), but exactly quadratic.

### 3.3 Bilinear MLP as a third-order tensor

Define a third-order tensor

\[
T\in\mathbb R^{d\times d\times p}
\]

such that

\[
y_k(x)=x^\top T_{:,:,k}x.
\]

For each output \(k\),

\[
T_{:,:,k}
=
\sum_{i=1}^m C_{i,k}\operatorname{sym}(a_i b_i^\top),
\]

where

\[
\operatorname{sym}(a_i b_i^\top)
=
\frac12(a_i b_i^\top + b_i a_i^\top).
\]

Only the symmetric part matters because

\[
x^\top A x = x^\top \operatorname{sym}(A)x
\]

for any square matrix \(A\). The antisymmetric part vanishes in the quadratic form.

## 4. Probe-conditioned functional mechanisms

### 4.1 Why not analyze raw class logits only?

For classification tasks, raw logits contain common-mode directions. If we take \(r=e_k\), we analyze the scalar \(y_k(x)\), but some of its structure may reflect shared logit offsets rather than the decision boundary of class \(k\).

Better probes:

1. **Centered class probe**
   \[
   r_k=e_k-\frac1p\mathbf 1.
   \]

2. **Pairwise contrast**
   \[
   r_{k,k'}=e_k-e_{k'}.
   \]

The pairwise contrast is especially interpretable: it asks which quadratic interactions separate class \(k\) from class \(k'\).

### 4.2 Constructing \(M_r\)

For any probe \(r\in\mathbb R^p\),

\[
r^\top y(x)
=
r^\top C^\top\left((A^\top x)\odot(B^\top x)\right).
\]

Let

\[
\alpha_i=(Cr)_i.
\]

Then

\[
r^\top y(x)
=
\sum_{i=1}^m \alpha_i(a_i^\top x)(b_i^\top x).
\]

Since

\[
(a_i^\top x)(b_i^\top x)
=
x^\top a_i b_i^\top x
=
x^\top \frac12(a_i b_i^\top+b_i a_i^\top)x,
\]

we get

\[
r^\top y(x)=x^\top M_r x,
\]

where

\[
M_r=
\frac12\sum_{i=1}^m
\alpha_i
(a_i b_i^\top+b_i a_i^\top).
\]

This is the most important object in the project.

### 4.3 Exact numerical check

For every dataset example \(x_n\), verify:

\[
\left|r^\top y(x_n)-x_n^\top M_r x_n\right|\leq \epsilon.
\]

This should hold up to floating point error.

If this check fails, one of these is wrong:

- matrix orientation convention;
- whether \(C\) is stored as \(m\times p\) or \(p\times m\);
- whether \(A^\top x\) or \(xA\) is used;
- the symmetrization factor;
- batch dimension broadcasting.

### 4.4 Eigenspace decomposition

Since \(M_r\) is symmetric,

\[
M_r=Q\Lambda Q^\top.
\]

Then

\[
x^\top M_r x
=
\sum_{j=1}^d \lambda_j(q_j^\top x)^2.
\]

This expresses the probe-conditioned computation as a weighted sum of squared projections.

A naive statement would be:

> The \(q_j\)'s are the ground-truth mechanisms.

A better statement:

> The eigenspaces of \(M_r\) are a canonical functional decomposition of the scalar probe \(r^\top y(x)\).

This is **probe-conditioned** and **basis-dependent only up to eigenspace degeneracy**.

## 5. Degenerate eigenvalues and why eigenspace projectors matter

Modular addition has symmetries. Symmetries often produce repeated eigenvalues. If

\[
\lambda_i=\lambda_j,
\]

then the individual eigenvectors \(q_i,q_j\) are not identifiable: any orthonormal rotation within the degenerate eigenspace gives an equally valid eigendecomposition.

So the invariant object is the projector

\[
P_\lambda=\sum_{j:\lambda_j\approx \lambda} q_jq_j^\top.
\]

The functional contribution from that eigenspace is

\[
x^\top \lambda P_\lambda x.
\]

Alignment metrics should therefore compare subcomponent-induced perturbations to \(P_\lambda\), not to arbitrary \(q_jq_j^\top\), when eigenvalues are degenerate or nearly degenerate.

### 5.1 Practical grouping rule

Sort eigenvalues by absolute value. Define a tolerance

\[
|\lambda_i-\lambda_j|
\leq
\tau\cdot \max_k|\lambda_k|,
\]

with \(\tau\in[10^{-4},10^{-2}]\). Group eigenvalues into equivalence classes under this rule, and form projectors.

Record sensitivity to \(\tau\).

### 5.2 Why this improves the post

This is one of the key credibility moves in the LessWrong post. It demonstrates that you are not just running a heatmap pipeline; you understand the linear algebraic identifiability issue.

## 6. What exactly is an SPD-style decomposition here?

There are two options.

### 6.1 Full Goodfire implementation

Use Goodfire's `param-decomp` repository and configure it for a tiny bilinear MLP. This has the advantage of being closer to the actual method, but it may be too much friction for a short project.

### 6.2 Minimal SPD-style decomposition

Implement the minimal version yourself:

\[
A\approx \sum_{c=1}^{C_A} u^A_c(v^A_c)^\top,
\]

\[
B\approx \sum_{c=1}^{C_B} u^B_c(v^B_c)^\top.
\]

Let

\[
G_A(x)\in[0,1]^{C_A},
\qquad
G_B(x)\in[0,1]^{C_B}
\]

be gating functions.

The masked matrices are

\[
A_g(x)
=
\sum_{c=1}^{C_A}
g^A_c(x)u^A_c(v^A_c)^\top,
\]

\[
B_g(x)
=
\sum_{c=1}^{C_B}
g^B_c(x)u^B_c(v^B_c)^\top.
\]

The masked output is

\[
\hat y_g(x)
=
C^\top\left((A_g(x)^\top x)\odot(B_g(x)^\top x)\right).
\]

A suitable loss:

\[
\mathcal L
=
\mathcal L_{\text{behavior}}
+
\lambda_A\mathcal L_{\text{param},A}
+
\lambda_B\mathcal L_{\text{param},B}
+
\lambda_g\mathcal L_{\text{sparse}}
+
\lambda_f\mathcal L_{\text{freq}}.
\]

Where:

\[
\mathcal L_{\text{behavior}}
=
\mathbb E_x
D_{\mathrm{KL}}
\left(
\operatorname{softmax}(y(x))
\;\|\;
\operatorname{softmax}(\hat y_g(x))
\right),
\]

\[
\mathcal L_{\text{param},A}
=
\left\|
A-\sum_{c=1}^{C_A}u^A_c(v^A_c)^\top
\right\|_F^2,
\]

\[
\mathcal L_{\text{param},B}
=
\left\|
B-\sum_{c=1}^{C_B}u^B_c(v^B_c)^\top
\right\|_F^2,
\]

\[
\mathcal L_{\text{sparse}}
=
\mathbb E_x
\left[
\sum_c |g^A_c(x)|^p+\sum_c |g^B_c(x)|^p
\right],
\]

with \(0<p\leq 1\) or \(p=1\) for a stable first pass.

A frequency penalty analogous in spirit to VPD:

\[
\mathcal L_{\text{freq}}
=
\sum_c \mu_c\log(1+B\mu_c),
\]

where

\[
\mu_c=\mathbb E_x[g_c(x)].
\]

This discourages a single component from being weakly active everywhere.

### 6.3 What to call it

Do **not** call this implementation "VPD."

Call it:

> an SPD-style rank-one gated parameter decomposition.

or

> a minimal ablation-based rank-one parameter decomposition inspired by SPD/VPD.

That honesty is important.

## 7. Functional perturbation induced by parameter atoms

### 7.1 The basic perturbation definition

Let \(S\) be a set of parameter subcomponents. Define ablated matrices:

\[
A_{-S}=A-\Delta A_S,
\qquad
B_{-S}=B-\Delta B_S.
\]

The functional perturbation is

\[
\Delta M_r(S)
=
M_r(A,B,C)-M_r(A_{-S},B_{-S},C).
\]

### 7.2 Expansion and the bilinear interaction term

Because \(M_r\) depends bilinearly on \(A\) and \(B\), the perturbation expands as:

\[
\Delta M_r(S)
=
M_r(\Delta A_S,B,C)
+
M_r(A,\Delta B_S,C)
-
M_r(\Delta A_S,\Delta B_S,C).
\]

The final term is important. It says that the effect of ablating parameter pieces in a bilinear model is not simply additive in the separately ablated \(A\)- and \(B\)-parts. This is exactly why bilinear models are interesting: the parameter-to-function map is structured but not trivial.

### 7.3 What is \(\Delta A_S\)?

If \(S_A\) is the subset of atoms in \(A\), then

\[
\Delta A_S
=
\sum_{c\in S_A}
u^A_c(v^A_c)^\top.
\]

Similarly,

\[
\Delta B_S
=
\sum_{c\in S_B}
u^B_c(v^B_c)^\top.
\]

If \(S\) includes only \(A\)-atoms, then \(\Delta B_S=0\), and the perturbation is linear in \(\Delta A_S\):

\[
\Delta M_r(S)=M_r(\Delta A_S,B,C).
\]

If \(S\) includes both \(A\)- and \(B\)-atoms, include the cross term.

## 8. Alignment metrics

### 8.1 Alignment to a rank-one eigenmechanism

If the eigenvalue is nondegenerate, compare \(\Delta M_r(S)\) to \(q_jq_j^\top\):

\[
\operatorname{align}(S,j)
=
\frac{
|\langle \Delta M_r(S), q_jq_j^\top\rangle_F|
}
{
\|\Delta M_r(S)\|_F
\|q_jq_j^\top\|_F
}.
\]

Because

\[
\|q_jq_j^\top\|_F=1,
\]

this simplifies to

\[
\operatorname{align}(S,j)
=
\frac{
|\langle \Delta M_r(S), q_jq_j^\top\rangle_F|
}
{
\|\Delta M_r(S)\|_F
}.
\]

### 8.2 Alignment to an eigenspace projector

For a degenerate eigenspace:

\[
\operatorname{align}(S,\lambda)
=
\frac{
|\langle \Delta M_r(S), P_\lambda\rangle_F|
}
{
\|\Delta M_r(S)\|_F
\|P_\lambda\|_F
}.
\]

Since \(P_\lambda\) has rank \(r_\lambda\),

\[
\|P_\lambda\|_F=\sqrt{r_\lambda}.
\]

### 8.3 Signed vs unsigned alignment

Use unsigned alignment in the main plots:

\[
|\langle A,B\rangle_F|.
\]

But compute signed alignment for diagnostics:

\[
\operatorname{signed\_align}(S,k)
=
\frac{
\langle \Delta M_r(S), P_k\rangle_F
}
{
\|\Delta M_r(S)\|_F\|P_k\|_F
}.
\]

Signed alignment distinguishes components that add vs subtract a mechanism. The main post can show unsigned alignment and discuss signs in a footnote or appendix.

### 8.4 Energy capture

Another metric:

\[
\operatorname{energy}(S,k)
=
\frac{
\|P_k\Delta M_r(S)P_k\|_F^2
}
{
\|\Delta M_r(S)\|_F^2
}.
\]

This asks how much of the perturbation lives inside an eigenspace, rather than how parallel it is to the projector itself.

For the mini-project, implement both but foreground Frobenius cosine alignment.

## 9. Clustering atoms by causal importance

The Goodfire/VPD intuition is that subcomponents may later be clustered into larger parameter components. This project can test whether clusters are needed.

Let

\[
g_c(x_n)
\]

be the causal importance or gate value for component \(c\) on dataset example \(x_n\). Define a co-activation correlation matrix:

\[
\Sigma_{cc'}
=
\operatorname{corr}_n(g_c(x_n),g_{c'}(x_n)).
\]

Then build a graph:

\[
c\sim c'
\quad\Longleftrightarrow\quad
\Sigma_{cc'}>\rho.
\]

Clusters are connected components.

Try thresholds:

\[
\rho\in\{0.5,0.6,0.7,0.8,0.9\}.
\]

Report the most stable result, and include threshold sensitivity in the appendix.

### 9.1 Why co-activation clustering is meaningful

If two rank-one atoms are repeatedly necessary on the same inputs, they may form one higher-level computational mechanism. In bilinear models, a natural failure mode is:

- one atom carries the cosine mode;
- one atom carries the sine mode;
- neither alone aligns with the full circular computation;
- together they align with a Fourier eigenspace.

This is precisely the kind of case where rank-one subcomponents are too fine-grained but clusters are meaningful.

## 10. Main hypotheses

### H1: Individual atoms align with eigenspaces

\[
\text{MeanMaxAlign}_{\text{atoms}}
\approx
\text{MeanMaxAlign}_{\text{clusters}}.
\]

Interpretation:

Rank-one parameter atoms are already a good functional ontology in this setting.

This would be a positive validation of SPD/VPD-style rank-one decomposition.

### H2: Clusters align but individual atoms do not

\[
\text{MeanMaxAlign}_{\text{clusters}}
\gg
\text{MeanMaxAlign}_{\text{atoms}}.
\]

Interpretation:

Rank-one atoms are useful but too fine-grained. The meaningful functional units are clusters of co-active atoms.

This supports a future direction toward:

\[
W^\ell\approx\sum_j U_j^\ell S_j^\ell(V_j^\ell)^\top,
\]

where a whole low-rank block shares one gate.

### H3: Neither atoms nor clusters align

Interpretation:

The decomposition objective is not recovering the functional mechanisms exposed by \(M_r\). This is not a failed project; it is a useful negative benchmark.

Possible reasons:

- the decomposition optimizes parameter reconstruction too strongly;
- gates learn input partitions rather than functional eigenspaces;
- the chosen probe \(r\) is not aligned with the learned decomposition;
- the bilinear model did not learn a clean spectral mechanism;
- rank-one atoms in \(A,B\) do not correspond to simple perturbations in \(M_r\).

### H4: SVD baseline beats SPD-style atoms

This would be especially interesting. It would mean that, at least for this toy bilinear model, ordinary matrix factorization gives a cleaner functional basis than learned sparse gated parameter components.

If this happens, report it honestly.

## 11. Causal ablation

Alignment alone is geometric. The post needs causal evaluation.

For each eigenspace \(P_k\), choose:

\[
S_k^\star
=
\arg\max_S \operatorname{align}(S,k).
\]

Ablate \(S_k^\star\) and measure:

1. accuracy drop;
2. KL divergence from base logits;
3. reduction in the eigenspace coefficient;
4. effect on examples where the corresponding class/contrast matters.

### 11.1 KL metric

\[
\Delta_{\mathrm{KL}}(S)
=
\mathbb E_x
D_{\mathrm{KL}}
\left(
\operatorname{softmax}(y(x))
\;\|\;
\operatorname{softmax}(y_{-S}(x))
\right).
\]

### 11.2 Probe-specific functional effect

\[
\Delta_{\text{functional}}(S,k)
=
\left|
\langle M_r-M_{r,-S},P_k\rangle_F
\right|.
\]

### 11.3 Random baseline

For each cluster \(S\), sample random atom sets \(R\) with

\[
|R|=|S|
\]

and compute the same metrics. Plot aligned cluster vs random size-matched baseline.

### 11.4 Low-alignment baseline

Choose clusters with similar norm but low alignment:

\[
\|\Delta M_r(R)\|_F\approx\|\Delta M_r(S)\|_F,
\qquad
\operatorname{align}(R,k)\ll\operatorname{align}(S,k).
\]

This controls for perturbation size.

## 12. Why modular addition is a good task

Modular addition over \(\mathbb Z_p\) has a known Fourier structure. If the model learns a clean algorithm, the interaction matrix should expose periodic/circular eigenspaces.

Input:

\[
x=[e_a;e_b]\in\mathbb R^{2p}.
\]

Label:

\[
y=(a+b)\bmod p.
\]

The target function is translation-equivariant:

\[
(a,b)\mapsto(a+s,b-s)
\]

leaves \(a+b\) invariant modulo \(p\). That symmetry is why Fourier-like modes are expected.

### 12.1 Caveat

The bilinear MLP may memorize the table in a non-Fourier way, especially for small \(p\) and overparameterized hidden dimension. This is why the eigenvector visualizations are sanity checks, not decorative plots.

## 13. Relationship to TXC / TempBench research taste

Your temporal crosscoder work uses synthetic data to create known local/global ground truth. There, the benchmark asks whether dictionaries recover local emission directions or hidden temporal states. This mini-project is analogous:

- TXC benchmark:
  \[
  \text{activation dictionary}
  \quad\text{vs}\quad
  \text{known temporal latent structure}.
  \]

- Bilinear-SPD benchmark:
  \[
  \text{parameter decomposition}
  \quad\text{vs}\quad
  \text{exact probe-conditioned functional eigenspaces}.
  \]

That is the deeper reason the project fits you. It is not merely a literature mashup; it is the same methodology applied to Goodfire's current parameter-space agenda.

## 14. Limitations to state in the post

1. **Probe-conditioned ground truth.** \(M_r\) depends on \(r\). There is no single global mechanism basis for the whole vector-valued function \(y(x)\) unless we analyze the full tensor \(T\).
2. **Small model.** A modular-addition bilinear MLP is not a language model.
3. **Bilinear-only.** GELU/SwiGLU MLPs do not admit the same exact quadratic analysis.
4. **Eigenspace ambiguity.** Degenerate eigenvalues force us to analyze projectors rather than individual eigenvectors.
5. **Minimal SPD-style implementation.** If using a custom implementation, it is inspired by SPD/VPD but not a reproduction.
6. **Frobenius geometry.** Frobenius alignment may not capture all causal structure; operator norm and data-weighted metrics may be more appropriate in some regimes.
7. **Dataset weighting.** The same \(M_r\) can be analyzed globally or with data-weighted metrics; the post should be explicit about which is used.
8. **Choice of probe.** Centered class probes and pairwise contrasts may yield different decompositions.

## 15. Future directions

### 15.1 Full VPD on bilinear MLPs

Run the actual Goodfire `param-decomp` implementation on the same trained bilinear model and compare with the minimal implementation.

### 15.2 Rank-\(r\) block components

Replace

\[
u_cv_c^\top
\]

with

\[
U_jS_jV_j^\top.
\]

A single gate controls a low-rank block. Test whether this recovers Fourier/circular eigenspaces more naturally.

### 15.3 Tensor decomposition of the full bilinear tensor

Instead of choosing \(r\), decompose

\[
T\in\mathbb R^{d\times d\times p}
\]

directly using CP/Tucker/Tensor Train decompositions. This would avoid reducing the vector output to scalar probes.

### 15.4 AIS / adversarial mask sampling

VPD uses adversarial masks to enforce mechanistic faithfulness. One could define an energy landscape over ablation masks:

\[
E(m)=D_{\mathrm{KL}}(f(x),f_m(x))
\]

and sample from

\[
\pi_\beta(m)\propto e^{\beta E(m)}.
\]

AIS could estimate whether faithfulness is broad or knife-edge.

### 15.5 Data-weighted function-space alignment

Replace raw Frobenius alignment with data-weighted alignment:

\[
\langle A,B\rangle_{\mathcal D}
=
\mathbb E_{x\sim \mathcal D}
(x^\top A x)(x^\top B x).
\]

This may better reflect causal effect on the task distribution.

## 16. A crisp final thesis for the post

The strongest version of the project is not:

> I applied SPD to bilinear MLPs.

It is:

> Bilinear MLPs give us an exact bridge between weight space and function space. This makes them a useful testbed for asking whether rank-one parameter decompositions recover functional mechanisms, or whether mechanisms only emerge after clustering simple parameter atoms.

That is the thesis to keep intact.
