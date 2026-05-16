"""Seeding, device selection, config loading, checkpoint IO."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    """Resolve a config-requested device against what's actually available.

    'auto' picks cuda > mps > cpu. Explicit requests fall back to cpu if unavailable.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def run_dir(cfg: dict[str, Any], root: str | Path = "runs") -> Path:
    """`runs/<run_name>/` with checkpoints/ artifacts/ figures/ reports/ subdirs."""
    d = Path(root) / cfg["run_name"]
    for sub in ("checkpoints", "artifacts", "figures", "reports"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


@dataclass
class Checkpoint:
    state_dict: dict[str, torch.Tensor]
    config: dict[str, Any]
    metrics: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "config": config, "metrics": metrics},
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Checkpoint:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    return Checkpoint(
        state_dict=blob["state_dict"],
        config=blob["config"],
        metrics=blob.get("metrics", {}),
    )
