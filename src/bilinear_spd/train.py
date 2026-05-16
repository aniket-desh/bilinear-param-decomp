"""Training loops. Day 1: bilinear MLP. Day 2.5: grok-metric tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .functional import construct_M_r, eigendecompose_M, group_eigenspaces
from .models import BilinearMLP


@dataclass
class BilinearTrainMetrics:
    final_step: int
    final_loss: float
    final_accuracy: float
    final_logit_margin: float
    history: list[dict] = field(default_factory=list)
    early_stopped: bool = False


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def _logit_margin(logits: torch.Tensor, y: torch.Tensor) -> float:
    correct = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    others = logits.clone()
    others.scatter_(1, y.unsqueeze(1), float("-inf"))
    runner_up = others.max(dim=-1).values
    return (correct - runner_up).mean().item()


def _param_l2(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        s += float(p.detach().pow(2).sum().item())
    return s


def _grok_snapshot(
    model: BilinearMLP,
    probe_r: torch.Tensor,
    rel_tols: tuple[float, ...],
    min_abs_eval: float,
) -> dict:
    with torch.no_grad():
        M = construct_M_r(model.A, model.B, model.C, probe_r)
        evals, evecs = eigendecompose_M(M)
        top_abs = float(evals[0].abs().item()) if evals.numel() > 0 else 0.0
    snap: dict = {
        "param_l2_norm": _param_l2(model),
        "top_abs_eval": top_abs,
    }
    for tol in rel_tols:
        groups = group_eigenspaces(evals, evecs, rel_tol=tol, min_abs_eval=min_abs_eval)
        snap[f"n_degenerate_tau_{tol:g}"] = sum(1 for g in groups if g.rank >= 2)
        snap[f"n_groups_tau_{tol:g}"] = len(groups)
    return snap


def train_bilinear(
    model: BilinearMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: dict,
    probe_for_grok: torch.Tensor | None = None,
    grok_eval_every: int | None = None,
    grok_rel_tols: tuple[float, ...] = (0.01, 0.05),
    grok_min_abs_eval: float = 1e-6,
    verbose: bool = True,
) -> BilinearTrainMetrics:
    """Train a bilinear MLP to fit (x, y) under cross-entropy.

    Dataset fits in one batch; we use the full table per step.

    Grokking tracking: if `probe_for_grok` is provided and `grok_eval_every` is
    set, every N steps we additionally log param-L2 norm, top |λ| of M_r for
    the probe, and the number of degenerate-eigenspace groups at each of
    `grok_rel_tols`. These are the signals that distinguish memorization
    (constant degeneracy count, growing weight norm) from grokking
    (sudden jump in degeneracy count, weight norm decreasing).
    """
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    n_steps = int(cfg["steps"])
    log_every = int(cfg.get("log_every", 100))
    target_acc = float(cfg.get("target_accuracy", 0.99))
    patience = int(cfg.get("target_acc_patience", 500))
    track_grok = probe_for_grok is not None and grok_eval_every is not None

    history: list[dict] = []
    above_target = 0
    early_stopped = False

    last_loss = float("nan")
    last_acc = 0.0
    last_margin = float("nan")
    last_grok: dict = {}

    for step in range(n_steps):
        optim.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()

        last_loss = loss.item()
        last_acc = _accuracy(logits.detach(), y)
        last_margin = _logit_margin(logits.detach(), y)

        if last_acc >= target_acc:
            above_target += 1
        else:
            above_target = 0

        if track_grok and (step % grok_eval_every == 0 or step == n_steps - 1):
            last_grok = _grok_snapshot(
                model, probe_for_grok, grok_rel_tols, grok_min_abs_eval
            )

        if step % log_every == 0 or step == n_steps - 1:
            entry = {
                "step": step,
                "loss": last_loss,
                "accuracy": last_acc,
                "logit_margin": last_margin,
            }
            entry.update(last_grok)
            history.append(entry)
            if verbose:
                grok_line = ""
                if track_grok and last_grok:
                    n_deg_keys = [k for k in last_grok if k.startswith("n_degenerate_")]
                    grok_line = (
                        f"  ‖w‖²={last_grok['param_l2_norm']:7.2f}"
                        f"  top|λ|={last_grok['top_abs_eval']:6.2f}"
                    )
                    for k in n_deg_keys:
                        tau = k.split("_")[-1]
                        grok_line += f"  n_deg(τ={tau})={last_grok[k]:2d}"
                print(
                    f"step {step:6d}  loss {last_loss:.4f}  acc {last_acc:.3f}"
                    f"  margin {last_margin:+.3f}{grok_line}"
                )

        if above_target >= patience:
            early_stopped = True
            if verbose:
                print(f"early stop at step {step} (acc >= {target_acc} for {patience} steps)")
            break

    return BilinearTrainMetrics(
        final_step=step,
        final_loss=last_loss,
        final_accuracy=last_acc,
        final_logit_margin=last_margin,
        history=history,
        early_stopped=early_stopped,
    )
