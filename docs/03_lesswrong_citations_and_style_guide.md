# LessWrong Post: Citation Map, Style Guide, and Drafting Plan

**Working post title:** *Do Rank-One Parameter Decompositions Recover Bilinear Mechanisms?*  
**Alternative title:** *A Toy Benchmark for Weight-Based Interpretability*  
**Audience:** LessWrong / Alignment Forum readers, especially people interested in mechanistic interpretability, Goodfire-style parameter decomposition, MATS, and weight-based interpretability.

## 0. Intended vibe

This should not read like:

> I read some Goodfire papers and made an LLM summary.

It should read like:

> I noticed a concrete gap between two adjacent research lines, built a small benchmark around it, and found a result that clarifies what rank-one parameter atoms are doing.

The post should be technical, honest, and figure-driven. The writing should make clear that you understand:

- why Goodfire cares about parameter decomposition;
- why bilinear MLPs are special;
- why comparing objects across parameter space and function space is subtle;
- why a toy benchmark can still be valuable;
- why the result is preliminary.

## 1. Core citation map

## 1.1 Parameter decomposition lineage

### APD: Interpretability in Parameter Space

**Citation:**  
Dan Braun, Lucius Bushnaq, Stefan Heimersheim, Jake Mendel, Lee Sharkey. *Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition.* arXiv:2501.14926.  
Link: <https://arxiv.org/abs/2501.14926>

**Use in post:**  
Introduce the original parameter-decomposition desiderata: faithful, minimal, simple mechanisms; mechanistic description length; direct decomposition of parameters rather than activations.

**Suggested sentence:**  
"APD framed parameter decomposition as a search for short mechanistic descriptions: components should reconstruct the original parameters, use as few active pieces as possible on each input, and be individually simple."

### SPD: Stochastic Parameter Decomposition

**Citation:**  
Lucius Bushnaq, Dan Braun, Lee Sharkey. *Stochastic Parameter Decomposition.* arXiv:2506.20790.  
Link: <https://arxiv.org/abs/2506.20790>

**Goodfire explainer:**  
<https://www.goodfire.ai/research/stochastic-param-decomp>

**LessWrong linkpost:**  
<https://www.lesswrong.com/posts/yjrpmCmqurDmbMztW/paper-stochastic-parameter-decomposition>

**GitHub / code:**  
<https://github.com/goodfire-ai/param-decomp/tree/spd-paper>

**Use in post:**  
Explain the move from APD to scalable stochastic ablation. SPD is the immediate inspiration for "rank-one parameter-space atoms with sparse causal use."

**Suggested sentence:**  
"SPD replaced APD's brittle attribution machinery with stochastic ablations: a component is treated as unnecessary on an input if the model can tolerate deleting it without changing behavior."

### VPD: Interpreting Language Model Parameters

**Citation/post:**  
Lucius Bushnaq, Dan Braun, Oliver Clive-Griffin, Bart Bussmann, Nathan Hu, Michael Ivanitskiy, Linda Linsefors, Lee Sharkey. *Interpreting Language Model Parameters.* Goodfire technical report/post, 2026.  
Link: <https://www.goodfire.ai/research/interpreting-lm-parameters>

**Explainer:**  
<https://www.goodfire.ai/research/vpd-explainer>

**GitHub / code:**  
<https://github.com/goodfire-ai/param-decomp>

**Use in post:**  
Use VPD as the current frontier: rank-one subcomponents, causal importance values, stochastic/adversarial ablations, clustering subcomponents into larger components, language-model parameter editing.

**Suggested sentence:**  
"VPD scales this agenda to a small language model, decomposing matrices into rank-one subcomponents with causal importance values \(g^\ell_{b,t,c}\), and uses adversarial ablations to enforce a stricter notion of mechanistic faithfulness."

### Open Problems in Mechanistic Interpretability

**Citation:**  
Lee Sharkey et al. *Open Problems in Mechanistic Interpretability.* arXiv:2501.16496.  
Link: <https://arxiv.org/abs/2501.16496>  
LessWrong linkpost: <https://www.lesswrong.com/posts/fqDzevPyw3GGaF5o9/open-problems-in-mechanistic-interpretability>

**Use in post:**  
Mention only lightly. Use it to situate the decomposition problem.

**Suggested sentence:**  
"Choosing the right unit of decomposition remains one of the central open problems in mechanistic interpretability."

## 1.2 Bilinear / weight-based interpretability lineage

### Technical note on bilinear layers

**Citation:**  
Lee Sharkey. *A technical note on bilinear layers for interpretability.* arXiv:2305.03452.  
Link: <https://arxiv.org/abs/2305.03452>

**Use in post:**  
Introduce the reason bilinear layers are analytically attractive: nonlinear in the input but expressible with linear operations and third-order tensors.

**Suggested sentence:**  
"Bilinear layers are nonlinear, but their input-output computation can be rewritten exactly as a third-order tensor contraction."

### Weight-based Decomposition: A Case for Bilinear MLPs

**Citation:**  
Michael T. Pearce, Thomas Dooms, Alice Rigg. *Weight-based Decomposition: A Case for Bilinear MLPs.* arXiv:2406.03947.  
Link: <https://arxiv.org/abs/2406.03947>

**Use in post:**  
Optional citation for the earlier version of the bilinear decomposition agenda.

### Bilinear MLPs enable weight-based mechanistic interpretability

**Citation:**  
Michael T. Pearce, Thomas Dooms, Alice Rigg, Jose M. Oramas, Lee Sharkey. *Bilinear MLPs enable weight-based mechanistic interpretability.* arXiv:2410.08417.  
Link: <https://arxiv.org/abs/2410.08417>

**Use in post:**  
Main bilinear citation. Use it for interaction matrices, eigendecomposition, low-rank structure, and the claim that weight-based interpretability is viable in bilinear layers.

**Suggested sentence:**  
"Pearce et al. showed that bilinear MLPs can be analyzed directly through their weights: for a chosen output direction, the layer induces an interaction matrix whose spectrum can expose low-rank functional structure."

### Decomposing the QK Circuit with Bilinear Sparse Dictionary Learning

**Citation/post:**  
Keith Wynroe and Lee Sharkey. *Decomposing the QK circuit with Bilinear Sparse Dictionary Learning.* LessWrong, 2024.  
Link: <https://www.lesswrong.com/posts/2ep6FGjTQoGDRnhrq/decomposing-the-qk-circuit-with-bilinear-sparse-dictionary>

**Use in post:**  
Optional. This shows that bilinear structure and sparse dictionary learning have already been connected in the QK circuit context.

**Suggested sentence:**  
"This is not the first time bilinear structure and sparse decomposition have met; QK circuits are another natural bilinear object."

## 1.3 Activation / SAE background

Use these sparingly. The post is not about SAEs, but you need to explain why parameter decomposition differs.

### Sparse autoencoders find highly interpretable features

**Citation:**  
Hoagy Cunningham et al. *Sparse autoencoders find highly interpretable features in language models.* arXiv:2309.08600.  
Link: <https://arxiv.org/abs/2309.08600>

### Towards Monosemanticity

**Citation:**  
Bricken et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits, 2023.  
Link: <https://transformer-circuits.pub/2023/monosemantic-features/index.html>

### Scaling Monosemanticity

**Citation:**  
Templeton et al. *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Transformer Circuits, 2024.  
Link: <https://transformer-circuits.pub/2024/scaling-monosemanticity/>

**Use in post:**  
One paragraph max: activation dictionaries are useful but do not directly decompose the parameters doing the computation.

## 1.4 Geometry / multi-dimensional feature context

Use only if discussing rank-\(r\) future work.

### Not All Language Model Features Are Linear

**Citation:**  
Joshua Engels, Isaac Liao, Eric Michaud, Wes Gurnee, Max Tegmark. *Not All Language Model Features Are Linear.* arXiv:2405.14860.  
Link: <https://arxiv.org/abs/2405.14860>

**Use:**  
Motivate the idea that functional units may be multi-dimensional or manifold-like, not one-dimensional.

### Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds

**Citation:**  
Michaud, Gorton, McGrath. *Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds.* arXiv:2509.02565.  
Link: <https://arxiv.org/abs/2509.02565>

**Use:**  
Optional future-work citation.

## 1.5 Your TXC context

Do not over-cite your unpublished paper in the LessWrong post unless it is already public. But the post can borrow the research pattern:

- create a setting with known latent/functional structure;
- compare learned decompositions against ground truth;
- report both positive and negative results.

You can write:

> "This project is also motivated by a general lesson from synthetic interpretability benchmarks: if we do not know the ground truth object, it is hard to tell whether a decomposition method has recovered it or merely produced something interpretable-looking."

## 2. Goodfire / LessWrong style observations

## 2.1 Lee Sharkey's LessWrong account

Relevant account:  
<https://www.lesswrong.com/users/lee_sharkey>

Observed style from posts:

- clear epistemic status when making conceptual claims;
- technically literate but not paper-dense;
- often starts from a field-level framing;
- distinguishes "what I think is true" from "what this work demonstrates";
- comfortable with "mini-paradigm" / "wave" framing, but also marks it as subjective;
- uses toy results as evidence for broader research directions, not as overclaimed proof.

Useful posts:

- *Mech interp is not pre-paradigmatic*  
  <https://www.lesswrong.com/posts/beREnXhBnzxbJtr8k/mech-interp-is-not-pre-paradigmatic>

- *[Paper] Stochastic Parameter Decomposition*  
  <https://www.lesswrong.com/posts/yjrpmCmqurDmbMztW/paper-stochastic-parameter-decomposition>

- *Attribution-based parameter decomposition*  
  Search title on Lee's account / LessWrong; use as style reference if needed.

- *Paper: Open Problems in Mechanistic Interpretability*  
  <https://www.lesswrong.com/posts/fqDzevPyw3GGaF5o9/open-problems-in-mechanistic-interpretability>

## 2.2 Goodfire post style

From Goodfire research posts:

- big framing question at the top;
- "why this matters" before method;
- method section with equations but not too much algebra up front;
- figures carry conceptual load;
- subsections with descriptive titles;
- honest limitations/future work;
- often a public-facing explainer paired with a technical paper/report;
- avoids burying the lede;
- uses concrete examples of components, circuits, or interventions;
- makes the method feel like an instrument, not just a metric.

For your post, mimic:

1. **Question first.**
2. **Small example second.**
3. **Math third.**
4. **Result fourth.**
5. **Limitations before grand claims.**

## 2.3 LessWrong norms

LessWrong readers reward:

- clear epistemic status;
- concrete claims;
- code links;
- explicit limitations;
- negative results;
- careful distinction between speculation and evidence;
- comments inviting corrections;
- "I did X, here is what happened" over "I think field Y should do Z."

They punish:

- vague "agenda" posts with no artifact;
- AI-generated summary vibes;
- overclaiming from toy models;
- unexplained jargon;
- references without synthesis;
- too many citations in a row without a reason;
- hiding failure modes.

## 3. Post framing

## 3.1 One-sentence hook

Option A:

> Parameter decomposition methods search for simple atoms in weight space. Bilinear MLPs give us exact functional mechanisms. I wanted to know whether those two decompositions line up.

Option B:

> If a method decomposes weights into rank-one atoms, what should those atoms be compared against? Bilinear MLPs give a rare setting where we can answer that question exactly.

Option C:

> This post is a small benchmark for weight-based interpretability: do rank-one parameter atoms recover the eigenspaces of an exactly computable bilinear mechanism?

## 3.2 Epistemic status

Suggested:

> **Epistemic status:** Small controlled experiment. I think the benchmark is more important than the particular result. The model is tiny and bilinear, and the "ground truth" is probe-conditioned: for a chosen output direction \(r\), the bilinear MLP induces an exact quadratic form \(x^\top M_r x\). The eigenspaces of \(M_r\) are a canonical functional decomposition of that scalar probe, not an absolute decomposition of the whole model.

This paragraph is important. It will make the post feel serious.

## 3.3 TL;DR

Suggested:

> **TL;DR:** I trained a small bilinear MLP on modular addition and used its weights to construct exact probe-conditioned interaction matrices \(M_r\). I then learned a minimal SPD-style rank-one gated parameter decomposition of the bilinear weights and measured whether its atoms, or clusters of co-active atoms, aligned with the eigenspaces of \(M_r\). This gives a toy benchmark for whether rank-one parameter decompositions recover functional mechanisms or merely produce useful smaller pieces.

After running experiments, replace with actual result:

> In this run, individual atoms [did/did not] align cleanly, while clusters [did/did not] recover the functional eigenspaces.

## 4. Recommended outline

# Do Rank-One Parameter Decompositions Recover Bilinear Mechanisms?

## Epistemic status

Small controlled experiment; probe-conditioned ground truth; code link.

## TL;DR

One paragraph.

## Why I cared about this

Explain:

- SAEs decompose activations;
- Goodfire parameter decomposition decomposes weights;
- bilinear MLPs expose exact quadratic functional structure;
- this gives a benchmark.

## The central problem: parameter space is not function space

Introduce the mismatch:

\[
u_cv_c^\top
\quad\text{versus}\quad
\Delta M_r(S).
\]

This is the conceptual contribution.

## Bilinear MLPs give an exact functional object

Define:

\[
h=(A^\top x)\odot(B^\top x),
\qquad
y=C^\top h.
\]

For probe \(r\):

\[
r^\top y=x^\top M_r x.
\]

Then:

\[
M_r=Q\Lambda Q^\top.
\]

Explain eigenspace/projector issue.

## A small benchmark

Describe:

- modular addition mod \(p\);
- bilinear MLP;
- centered class probes or pairwise probes;
- minimal SPD-style rank-one gated decomposition;
- atom/cluster alignment.

## Results

Use the three-panel figure.

Panel 1: spectrum.  
Panel 2: alignment heatmaps.  
Panel 3: ablation baseline.

Write one headline result.

## Interpretation

Depending on outcome:

### If atoms align

"Rank-one atoms may be a reasonable ontology here."

### If clusters align

"Rank-one subcomponents look like useful pieces, but functional mechanisms emerge after clustering."

### If neither align

"Good negative benchmark: parameter reconstruction/minimality may not be enough to recover this functional basis."

## Limitations

Use the limitations list. Do not hide them.

## What I would try next

- full Goodfire VPD implementation;
- rank-\(r\) block-term components;
- data-weighted alignment;
- full tensor decomposition;
- adversarial mask sampling / AIS.

## Acknowledgements

Mention anyone who gave feedback.

## Code

Link GitHub.

## 5. Writing style tips

## 5.1 Keep the thesis narrow

Avoid:

> This tells us whether SPD works.

Use:

> This is a toy benchmark for one question about the relation between rank-one parameter atoms and exact bilinear functional eigenspaces.

## 5.2 Say "probe-conditioned" repeatedly

This is the single most important caveat.

Good phrasing:

- "For a chosen output probe \(r\)..."
- "The ground truth here is not absolute; it is the exact functional decomposition of \(r^\top y(x)\)."
- "Changing \(r\) changes \(M_r\)."

Bad phrasing:

- "The eigenvectors are the ground truth mechanisms."
- "This reveals the true mechanisms of the model."

## 5.3 Do not over-index on Goodfire flattery

Do not write:

> Goodfire's agenda is the future of mech interp.

Write:

> Parameter decomposition is attractive because it tries to decompose the parameters doing the computation, rather than only the activations produced by that computation.

## 5.4 Be clear where your implementation diverges

If using minimal implementation:

> I did not run the full VPD training stack. I used a small SPD-style rank-one gated decomposition because the point of this post is the benchmark and the parameter/function-space comparison, not a faithful reproduction of Goodfire's system.

This is better than pretending.

## 5.5 Use equations as anchors, not walls

LessWrong readers can handle equations. But every equation should answer a question.

Good sequence:

1. "What is the bilinear model?"
2. Equation.
3. "What scalar probe are we analyzing?"
4. Equation.
5. "What is the functional interaction matrix?"
6. Equation.
7. "How do we compare parameter atoms to it?"
8. Equation.

Do not dump all equations in one block.

## 5.6 Give readers the picture before the details

Before deriving \(M_r\), say:

> The whole trick is that for bilinear MLPs, a logit contrast is a quadratic form. Once we have a quadratic form, we can eigendecompose it. That gives us a clean function-space object to compare parameter decompositions against.

Then derive.

## 5.7 Explain why negative results matter

If the result is messy, write:

> I think this is still useful. In real LMs we usually do not know what the decomposition should recover. Here we do know one reasonable functional object. If a decomposition fails to align with it, that tells us what future benchmarks should measure.

## 5.8 Keep a "methods appendix" section

LessWrong posts can be long, but main body should stay readable. Put details like:

- hyperparameters;
- eigenspace tolerance;
- clustering thresholds;
- exact loss;
- ablation baseline procedure;

near the end under "Methods details."

## 6. Figure caption templates

## 6.1 Spectrum

> **Figure 1:** Spectrum of the probe-conditioned bilinear interaction matrix \(M_r\). For a chosen centered class probe \(r\), the bilinear MLP computes \(r^\top y(x)=x^\top M_r x\). Repeated or near-repeated eigenvalues are grouped into eigenspaces, and subsequent alignment metrics use the corresponding projectors rather than arbitrary individual eigenvectors.

## 6.2 Alignment heatmap

> **Figure 2:** Alignment between parameter-decomposition perturbations and functional eigenspaces. Rows are rank-one atoms or co-activation clusters; columns are eigenspaces of \(M_r\). Values are Frobenius cosine similarities between the induced functional perturbation \(\Delta M_r(S)\) and each eigenspace projector. The comparison tests whether functional mechanisms appear at the individual-atom level or only after clustering.

## 6.3 Ablation

> **Figure 3:** Causal effect of ablating the most-aligned cluster for each functional eigenspace. Aligned clusters are compared against random size-matched atom sets and low-alignment clusters. This distinguishes geometric alignment from merely large perturbation size.

## 7. Suggested titles

Best:

1. **Do Rank-One Parameter Decompositions Recover Bilinear Mechanisms?**
2. **A Toy Benchmark for Weight-Based Interpretability**
3. **Parameter Atoms vs Functional Mechanisms in a Bilinear MLP**
4. **When Weight Decompositions Meet Bilinear Eigenspaces**
5. **A Small Test for the Ontology of Parameter Decomposition**

Avoid:

- "Solving Parameter Decomposition"
- "Ground Truth Mechanisms in Neural Networks"
- "The Future of Mechanistic Interpretability"
- "Bilinear MLPs Prove SPD Works"

## 8. Suggested abstract

> Parameter decomposition methods such as APD, SPD, and VPD try to break neural network weights into simple, sparsely used computational components. But in real language models we usually do not know what the "right" decomposition should be. Bilinear MLPs provide a useful toy setting: for any output probe \(r\), the scalar computation \(r^\top y(x)\) can be written exactly as a quadratic form \(x^\top M_r x\), and the eigenspaces of \(M_r\) give a canonical functional decomposition of that probe. I trained a small bilinear MLP on modular addition, learned a minimal SPD-style rank-one gated parameter decomposition, and compared the functional perturbations induced by its atoms and co-activation clusters to the eigenspaces of \(M_r\). This post describes the benchmark and what it suggests about when rank-one parameter atoms should be interpreted individually versus clustered into larger mechanisms.

After experiments, add result sentence.

## 9. Suggested conclusion templates

### If clusters align

> My tentative conclusion is that rank-one parameter atoms may be better understood as submechanistic pieces than as mechanisms themselves. In this toy bilinear setting, the exact functional decomposition lives in eigenspaces of \(M_r\), and the objects that best recover those eigenspaces are clusters of co-active atoms. That is not a criticism of parameter decomposition; it is closer to a type signature. Rank-one atoms may be the right low-level basis, while mechanisms may live one clustering step above.

### If atoms align

> My tentative conclusion is that, at least in this tiny bilinear setting, rank-one parameter atoms can line up surprisingly well with exact functional structure. This is a useful sanity check for parameter decomposition: when the model admits a clean weight-based functional analysis, the learned parameter atoms recover much of it.

### If neither aligns

> My tentative conclusion is negative but useful: a rank-one gated parameter decomposition can be faithful and sparse without obviously recovering the eigenspaces of the bilinear computation. This suggests that future parameter-decomposition benchmarks should not only measure reconstruction and sparsity, but also compare learned components against exact functional structure when such structure is available.

## 10. Tags for LessWrong / Alignment Forum

Potential tags:

- Mechanistic Interpretability
- Interpretability (ML & AI)
- Sparse Autoencoders
- AI Alignment
- Technical AI Alignment
- Machine Learning
- Deep Learning
- MATS Program
- Research
- LessWrong frontpage if accessible

Use restraint. Pick only the tags that are real.

## 11. Comment strategy

At the end:

> I would be especially interested in feedback on:
>
> 1. whether the \(M_r\) eigenspace basis is the right functional object to compare against;
> 2. whether a data-weighted inner product would be preferable to Frobenius alignment;
> 3. whether this benchmark should use full VPD rather than a minimal SPD-style implementation;
> 4. whether there are better toy tasks than modular addition for this comparison.

This invites useful technical comments rather than generic praise.

## 12. Avoiding "LLM-generated summary BS"

Concrete ways:

1. **Include a real result.** Even if preliminary.
2. **Use one original mathematical object.** Here: \(\Delta M_r(S)\) as the map from parameter atom sets to functional perturbations.
3. **State a falsifiable question.** Atom alignment vs cluster alignment.
4. **Share code.**
5. **Be explicit about implementation shortcuts.**
6. **Report failures.**
7. **Don't summarize every related paper.** Cite only what the reader needs.
8. **Use a figure with your own data.**
9. **End with next experiments, not grand narrative.**

## 13. Citation formatting in the post

Use inline links in prose. Example:

> Goodfire's [Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790) decomposes weight matrices into sparsely used parameter-space vectors, while [VPD](https://www.goodfire.ai/research/interpreting-lm-parameters) adds adversarial ablations and scales the method to a small language model.

For references section:

```markdown
## References

- Braun et al., *Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition*, 2025. <https://arxiv.org/abs/2501.14926>
- Bushnaq, Braun, Sharkey, *Stochastic Parameter Decomposition*, 2025. <https://arxiv.org/abs/2506.20790>
- Bushnaq et al., *Interpreting Language Model Parameters*, Goodfire, 2026. <https://www.goodfire.ai/research/interpreting-lm-parameters>
- Sharkey, *A technical note on bilinear layers for interpretability*, 2023. <https://arxiv.org/abs/2305.03452>
- Pearce et al., *Bilinear MLPs enable weight-based mechanistic interpretability*, 2024. <https://arxiv.org/abs/2410.08417>
```

## 14. Final pre-post checklist

Before posting:

- [ ] The first screen contains the question, the result, and why it matters.
- [ ] The phrase "probe-conditioned" appears before any "ground truth" claim.
- [ ] \(M_r\) derivation is correct and tested in code.
- [ ] Degenerate eigenspaces are handled with projectors.
- [ ] The alignment metric is defined before the heatmap.
- [ ] The implementation is not misrepresented as full VPD unless it is full VPD.
- [ ] There is a GitHub link.
- [ ] Limitations are real, not perfunctory.
- [ ] The post has one clear ask for feedback.
- [ ] The title is modest.
- [ ] The result is not overclaimed.

## 15. What to send Dmitry before posting

Suggested message:

> I drafted a small LW post around the bilinear-SPD benchmark idea. The core object is \(\Delta M_r(S)\): for a set of rank-one parameter atoms \(S\), I compute the induced perturbation to the bilinear interaction matrix for a probe \(r\), then compare it to eigenspace projectors of \(M_r\). The post is trying to be a benchmark for whether rank-one parameter atoms align with exact functional mechanisms or only become meaningful after clustering. Could you sanity-check whether this framing is useful for Goodfire/MATS, and especially whether I am being too casual about the "ground truth" language?

## 16. Why this post should work for MATS/Goodfire signaling

It shows:

1. you understand the parameter-decomposition agenda;
2. you understand the bilinear-weight-based interpretability line;
3. you can identify a missing benchmark;
4. you can build a small experiment;
5. you can reason carefully about spaces and invariances;
6. you can write with honest limitations;
7. you have a tensor/numerical-linear-algebra angle that is genuinely yours.

That is much stronger than a literature summary.
