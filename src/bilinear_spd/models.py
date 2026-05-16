"""Bilinear MLP and (later) the gated rank-one decomposition wrapper.

Shape conventions (matching docs/01 and docs/02):

    A, B : [d_in, m]
    C    : [m,    d_out]

Forward (batched):

    z_a = X @ A          # [B, m]
    z_b = X @ B          # [B, m]
    h   = z_a * z_b      # [B, m]
    y   = h @ C          # [B, d_out]

Equivalent to h_i = (a_i^T x)(b_i^T x) with a_i = A[:, i], b_i = B[:, i].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class BilinearMLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        use_bias: bool = False,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.use_bias = use_bias

        self.A = nn.Parameter(torch.empty(d_in, d_hidden))
        self.B = nn.Parameter(torch.empty(d_in, d_hidden))
        self.C = nn.Parameter(torch.empty(d_hidden, d_out))
        if use_bias:
            self.b_out = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("b_out", None)

        self._init_weights(init_scale)

    def _init_weights(self, init_scale: float) -> None:
        # Small Gaussian init. The bilinear product squares the activation
        # magnitude, so we keep scales conservative.
        for w in (self.A, self.B):
            nn.init.normal_(w, std=init_scale)
        nn.init.normal_(self.C, std=init_scale / math.sqrt(self.d_hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_a = x @ self.A
        z_b = x @ self.B
        h = z_a * z_b
        y = h @ self.C
        if self.b_out is not None:
            y = y + self.b_out
        return y
