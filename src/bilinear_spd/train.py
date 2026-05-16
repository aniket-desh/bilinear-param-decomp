"""Training loops. Day 1: bilinear MLP only."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def train_bilinear(
    model: BilinearMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: dict,
    verbose: bool = True,
) -> BilinearTrainMetrics:
    """Train a bilinear MLP to fit (x, y) under cross-entropy.

    The dataset is small enough to fit in one batch; we ignore batch_size and
    use the full table per step. This matches the spec's exact-table-learning
    intent for modular addition.
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

    history: list[dict] = []
    above_target = 0
    early_stopped = False

    last_loss = float("nan")
    last_acc = 0.0
    last_margin = float("nan")

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

        if step % log_every == 0 or step == n_steps - 1:
            history.append(
                {
                    "step": step,
                    "loss": last_loss,
                    "accuracy": last_acc,
                    "logit_margin": last_margin,
                }
            )
            if verbose:
                print(
                    f"step {step:6d}  loss {last_loss:.4f}  acc {last_acc:.3f}  "
                    f"margin {last_margin:+.3f}"
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
