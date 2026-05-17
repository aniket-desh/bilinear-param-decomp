"""Causal ablations: how much does removing atom set $S$ change behaviour?

Given a `BilinearComponentMLP` with atoms $\\{(u^A_c, v^A_c)\\}_c$ and
$\\{(u^B_c, v^B_c)\\}_c$, an ablation removes a set $S$ of atom indices by
subtracting their rank-one contribution from the *weights themselves*:

    $A_{-S} = A - \\sum_{c \\in S \\cap A} u^A_c (v^A_c)^\\top, \\quad
     B_{-S} = B - \\sum_{c \\in S \\cap B} u^B_c (v^B_c)^\\top.$

The model is then evaluated as a *plain bilinear MLP* with these surgical
weights — no gates, no stochastic masks — and compared against the frozen
target's behaviour. Per spec § 5.9 we report KL, accuracy change, mean
logit margin change, plus the probe-scalar coefficient $r^\\top y(x)$
change as the "functional" metric.

The flat atom-index convention (rows 0..C_A−1 are A-atoms, rows C_A..C_A+C_B−1
are B-atoms) matches `analysis.py`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .decomposition import BilinearComponentMLP


@torch.no_grad()
def ablate_weights(
    decomp: BilinearComponentMLP,
    atom_set: list[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build $(A_{-S}, B_{-S})$: target weights with the listed atoms subtracted.

    `atom_set` is a list/tensor of flat indices in $[0, C_A + C_B)$ — A-atoms
    first, then B-atoms — matching `analysis.atom_perturbations` / the saved
    alignment matrices.
    """
    C_A = decomp.C_A
    A = decomp.A_target.clone()
    B = decomp.B_target.clone()
    if isinstance(atom_set, torch.Tensor):
        atom_set = atom_set.tolist()
    for atom_idx in atom_set:
        if atom_idx < C_A:
            c = atom_idx
            A -= torch.outer(decomp.U_A[c], decomp.V_A[c])
        else:
            c = atom_idx - C_A
            B -= torch.outer(decomp.U_B[c], decomp.V_B[c])
    return A, B


@torch.no_grad()
def bilinear_forward(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    b_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """y = ((x A) ⊙ (x B)) C [+ b]. Pure forward, no gates."""
    h = (x @ A) * (x @ B)
    y = h @ C
    if b_out is not None:
        y = y + b_out
    return y


@torch.no_grad()
def evaluate_ablation(
    decomp: BilinearComponentMLP,
    x: torch.Tensor,
    y_target: torch.Tensor,
    A_minus: torch.Tensor,
    B_minus: torch.Tensor,
    r: torch.Tensor | None = None,
    targets_for_acc: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute ablation metrics for the perturbed-weight forward.

    Returns a dict with:
      kl: $\\mathrm{KL}(\\mathrm{softmax}(y_\\text{target}) \\,\\|\\, \\mathrm{softmax}(y_\\text{abl}))$
      acc_target / acc_abl / d_acc: accuracy (argmax) on `targets_for_acc`
      margin_target / margin_abl / d_margin: mean (top1 - top2) logit margin
      probe_scalar_target / probe_scalar_abl / d_probe_scalar:
          mean $r^\\top y(x)$ if `r` is provided, else 0
    """
    y_abl = bilinear_forward(x, A_minus, B_minus, decomp.C_target,
                             b_out=decomp.b_out)
    log_q = F.log_softmax(y_abl, dim=-1)
    p = F.softmax(y_target.detach(), dim=-1)
    kl = F.kl_div(log_q, p, reduction="batchmean").item()

    def _top2_margin(y):
        sorted_y, _ = y.sort(dim=-1, descending=True)
        return float((sorted_y[..., 0] - sorted_y[..., 1]).mean().item())

    margin_target = _top2_margin(y_target)
    margin_abl = _top2_margin(y_abl)

    if targets_for_acc is not None:
        acc_target = float((y_target.argmax(dim=-1) == targets_for_acc).float().mean().item())
        acc_abl = float((y_abl.argmax(dim=-1) == targets_for_acc).float().mean().item())
    else:
        acc_target = float((y_target.argmax(dim=-1) == y_target.argmax(dim=-1)).float().mean().item())
        acc_abl = float((y_abl.argmax(dim=-1) == y_target.argmax(dim=-1)).float().mean().item())

    out = {
        "kl": kl,
        "acc_target": acc_target,
        "acc_abl": acc_abl,
        "d_acc": acc_abl - acc_target,
        "margin_target": margin_target,
        "margin_abl": margin_abl,
        "d_margin": margin_abl - margin_target,
    }
    if r is not None:
        ps_target = float((y_target @ r).mean().item())
        ps_abl = float((y_abl @ r).mean().item())
        out.update({
            "probe_scalar_target": ps_target,
            "probe_scalar_abl": ps_abl,
            "d_probe_scalar": ps_abl - ps_target,
        })
    return out


def random_size_matched_sets(
    n_atoms: int,
    size: int,
    n_samples: int,
    exclude: list[int] | set[int] | None = None,
    rng: torch.Generator | None = None,
) -> list[list[int]]:
    """`n_samples` random atom-index sets of the requested size.

    `exclude` atoms are guaranteed not to appear in any of the random sets,
    which lets you make the random baseline genuinely "the aligned cluster
    isn't included." If `n_atoms - |exclude| < size`, an exception is raised
    rather than degrading silently — we'd rather know the experiment is
    ill-specified.
    """
    if rng is None:
        rng = torch.Generator(device="cpu").manual_seed(0)
    exclude_set = set(exclude) if exclude is not None else set()
    candidates = [i for i in range(n_atoms) if i not in exclude_set]
    if len(candidates) < size:
        raise ValueError(
            f"cannot draw size-{size} sets from {len(candidates)} candidates "
            f"({n_atoms} atoms minus {len(exclude_set)} excluded)"
        )
    out: list[list[int]] = []
    cand_t = torch.tensor(candidates)
    for _ in range(n_samples):
        perm = torch.randperm(len(candidates), generator=rng)
        out.append(sorted(cand_t[perm[:size]].tolist()))
    return out


def low_alignment_atom_set(
    align_to_eigenspace: torch.Tensor,
    delta_norms: torch.Tensor,
    target_size: int,
    aligned_set: list[int] | set[int],
    norm_tolerance: float = 0.25,
) -> list[int]:
    """Choose a low-alignment atom set with comparable Frobenius mass.

    Pick `target_size` atoms whose total $\\|\\Delta M_r\\|_F$ is within
    `norm_tolerance` of the aligned set's total, subject to having the
    bottom-quartile alignment to the target eigenspace. Greedy heuristic:
    sort by alignment ascending, take the lowest-alignment atoms until the
    cumulative norm reaches the target window.

    Returns at most `target_size` indices; if no combination fits the
    tolerance, returns the lowest-alignment `target_size` atoms anyway with
    a fallback.
    """
    if isinstance(aligned_set, list):
        aligned_set = set(aligned_set)
    aligned_norm = float(delta_norms[list(aligned_set)].sum().item()) if aligned_set else 1.0

    # candidates: not in aligned set, alignment in bottom 25% to this eigenspace
    n = align_to_eigenspace.numel()
    q25 = float(torch.quantile(align_to_eigenspace, 0.25).item())
    candidates = [
        i for i in range(n)
        if i not in aligned_set and float(align_to_eigenspace[i].item()) <= q25
    ]
    if not candidates:
        # fallback: take lowest-alignment outside aligned set
        order = torch.argsort(align_to_eigenspace)
        candidates = [int(i.item()) for i in order if int(i.item()) not in aligned_set]

    # sort candidates by alignment ascending, then greedily build up to size
    candidates.sort(key=lambda i: float(align_to_eigenspace[i].item()))
    chosen: list[int] = []
    cur_norm = 0.0
    for i in candidates:
        chosen.append(i)
        cur_norm += float(delta_norms[i].item())
        if len(chosen) >= target_size:
            # if too low, keep one more if it brings us closer to target
            if cur_norm < (1.0 - norm_tolerance) * aligned_norm and i != candidates[-1]:
                continue
            break
    # if we overshot in count, trim to target_size; the closeness-of-norm check
    # was best-effort.
    if len(chosen) > target_size:
        chosen = chosen[:target_size]
    return sorted(chosen)
