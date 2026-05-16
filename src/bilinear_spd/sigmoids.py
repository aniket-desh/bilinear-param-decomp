"""Leaky-hard sigmoids with custom backward.

These two functions are cherry-picked (with attribution) from Goodfire's
single-file SPD reference at
https://github.com/goodfire-ai/param-decomp/blob/main/nano_param_decomp/run.py
§ B "Leaky-hard sigmoids", commit at clone time was the public main.
Verbatim port; renamed for clarity, no logic changes.

Why we need them
----------------
The causal-importance ("gate") values for a parameter component should live
in [0, 1]. A vanilla `torch.sigmoid` works for the forward but lets dead
components stay dead — once a logit is pushed deep negative, gradient vanishes
and the component never resurrects.

`lower_leaky` is `clamp(x, 0, 1)` in the forward, but its backward returns
`alpha * grad_output` in the `x <= 0` region *only when the gradient wants to
move y up*. That selectively passes gradient to resurrect dead components
while keeping the forward saturated at 0.

`upper_leaky` does the symmetric thing past 1 (a linear continuation), which
lets the frequency-penalty loss push values down even when the forward clamp
has hidden them at 1.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor


class _LowerLeakyHardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        grad_output = grad_outputs[0]
        (x,) = ctx.saved_tensors
        alpha: float = ctx.alpha
        zero = torch.zeros_like(grad_output)
        grad = torch.where(
            x <= 0,
            torch.where(grad_output < 0, alpha * grad_output, zero),
            torch.where(x <= 1, grad_output, zero),
        )
        return grad, None


def lower_leaky(x: Tensor, alpha: float = 0.01) -> Tensor:
    """`clamp(x, 0, 1)` forward; resurrection-friendly leaky backward at x <= 0."""
    return cast(Tensor, _LowerLeakyHardSigmoid.apply(x, alpha))


def upper_leaky(x: Tensor, alpha: float = 0.01) -> Tensor:
    """For x > 1: 1 + alpha*(x-1) (linear continuation, grad = alpha); else clamp.

    Native autograd handles this correctly without a custom Function.
    """
    return torch.where(x > 1, 1 + alpha * (x - 1), x.clamp(0.0, 1.0))
