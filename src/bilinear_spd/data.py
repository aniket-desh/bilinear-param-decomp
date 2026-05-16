"""Datasets for the benchmark.

Encoding: each scalar input slot becomes a one-hot vector and the two slots are
concatenated, so x has shape [2*p] for modular addition and [2*2] for XOR.
The bilinear analysis treats the first p (or 2) coordinates as the 'a-slot'
and the remainder as the 'b-slot'.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_modular_addition_dataset(p: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Full p x p table for (a + b) mod p with concat-onehot inputs.

    Returns
    -------
    x : [p*p, 2*p]   concat one-hot
    a, b, y : [p*p] int64
    """
    a_idx = torch.arange(p, device=device).repeat_interleave(p)  # [p*p]
    b_idx = torch.arange(p, device=device).repeat(p)             # [p*p]
    y = (a_idx + b_idx) % p
    x_a = F.one_hot(a_idx, num_classes=p).float()
    x_b = F.one_hot(b_idx, num_classes=p).float()
    x = torch.cat([x_a, x_b], dim=-1)
    return {"x": x, "a": a_idx, "b": b_idx, "y": y}


def make_xor_dataset(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """4-example XOR with concat-onehot 2-bit inputs.

    x has shape [4, 4] (two 2-dim one-hots concatenated).
    """
    a_idx = torch.tensor([0, 0, 1, 1], device=device)
    b_idx = torch.tensor([0, 1, 0, 1], device=device)
    y = a_idx ^ b_idx
    x_a = F.one_hot(a_idx, num_classes=2).float()
    x_b = F.one_hot(b_idx, num_classes=2).float()
    x = torch.cat([x_a, x_b], dim=-1)
    return {"x": x, "a": a_idx, "b": b_idx, "y": y}


def make_dataset(task_cfg: dict, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    name = task_cfg["name"]
    if name == "modular_addition":
        return make_modular_addition_dataset(p=task_cfg["p"], device=device)
    if name == "xor":
        return make_xor_dataset(device=device)
    raise ValueError(f"unknown task: {name}")


def input_dim(task_cfg: dict) -> int:
    if task_cfg["name"] == "modular_addition":
        return 2 * task_cfg["p"]
    if task_cfg["name"] == "xor":
        return 4
    raise ValueError(f"unknown task: {task_cfg['name']}")


def output_dim(task_cfg: dict) -> int:
    if task_cfg["name"] == "modular_addition":
        return task_cfg["p"]
    if task_cfg["name"] == "xor":
        return 2
    raise ValueError(f"unknown task: {task_cfg['name']}")
