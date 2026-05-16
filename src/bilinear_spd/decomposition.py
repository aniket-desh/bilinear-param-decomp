"""Gated rank-one decomposition of the bilinear weights A, B.

We replace each of the trained bilinear factors with a sum of rank-one atoms
plus a residual:

    A ≈ sum_{c=1..C_A} u^A_c (v^A_c)^T            with residual  ΔA = A − Â
    B ≈ sum_{c=1..C_B} u^B_c (v^B_c)^T            with residual  ΔB = B − B̂

On each input x, a small gate network produces per-atom causal-importance
values g_A(x) ∈ [0,1]^{C_A} and g_B(x) ∈ [0,1]^{C_B}. The forward
substitutes per-sample masked weights

    A_g(x) = sum_c m_c^A(x) u^A_c (v^A_c)^T + ΔA       (ΔA unmasked)
    B_g(x) = sum_c m_c^B(x) u^B_c (v^B_c)^T + ΔB

into the original bilinear computation, with the frozen output map C reused:

    y_hat(x) = ((x A_g(x)) ⊙ (x B_g(x))) C.

The mask m can be either (i) deterministic — set to the gate value g — or
(ii) stochastic with `mask = g + (1−g)·U(0,1)` (cherry-picked from Goodfire
nano § E.`sample_continuous_masks`). Stochastic masks are what make the
gate values *causal-importance* rather than just attention weights: a low g
on an input that ablates cleanly means "this component is dispensable here."

Shape convention matches BilinearMLP: A, B are [d_in, m]; the per-atom
factors are stored as U: [C, d_in], V: [C, d_hidden].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .models import BilinearMLP
from .sigmoids import lower_leaky, upper_leaky


class GateNet(nn.Module):
    """Small MLP mapping x in R^{d_in} to per-atom raw causal-importance logits."""

    def __init__(self, d_in: int, gate_hidden: int, C: int, init_bias: float = 0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, C),
        )
        # Initialize the final layer so raw logits start near `init_bias`,
        # giving lower_leaky(0.5) = 0.5 — i.e., every atom contributes ~half
        # at step 0, rather than all atoms being dead.
        last: nn.Linear = self.layers[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)  # raw, pre-clamp


@dataclass
class DecompForwardAux:
    """Auxiliary tensors returned alongside the gated forward output."""

    g_A: torch.Tensor                 # [B, C_A]  lower-leaky gates
    g_B: torch.Tensor                 # [B, C_B]
    g_A_upper: torch.Tensor           # [B, C_A]  upper-leaky (for frequency penalty)
    g_B_upper: torch.Tensor           # [B, C_B]
    m_A: torch.Tensor                 # [B, C_A]  actually-applied masks (deterministic or stochastic)
    m_B: torch.Tensor                 # [B, C_B]
    A_hat: torch.Tensor               # [d_in, d_hidden]  sum_c U_A[c] V_A[c]^T (unmasked atom-only recon)
    B_hat: torch.Tensor               # [d_in, d_hidden]
    delta_A: torch.Tensor             # [d_in, d_hidden]  A_target − A_hat
    delta_B: torch.Tensor             # [d_in, d_hidden]


MaskMode = Literal["deterministic", "stochastic"]


class BilinearComponentMLP(nn.Module):
    """Wraps a frozen BilinearMLP with gated rank-one atoms over A and B.

    Use `forward(x, mask_mode="stochastic")` during training; switch to
    `"deterministic"` for evaluation. Direct atom ablation is supported via
    `forward_with_masks(x, m_A, m_B, include_delta_A, include_delta_B)`.
    """

    def __init__(
        self,
        target: BilinearMLP,
        C_A: int,
        C_B: int,
        gate_hidden: int,
        leaky_alpha: float = 0.01,
        init_atom_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.d_in = target.d_in
        self.d_hidden = target.d_hidden
        self.d_out = target.d_out
        self.C_A = C_A
        self.C_B = C_B
        self.leaky_alpha = leaky_alpha

        # Frozen target weights (kept as buffers so they live with the module).
        self.register_buffer("A_target", target.A.detach().clone())
        self.register_buffer("B_target", target.B.detach().clone())
        self.register_buffer("C_target", target.C.detach().clone())
        if target.b_out is not None:
            self.register_buffer("b_out", target.b_out.detach().clone())
        else:
            self.register_buffer("b_out", None)
        for p in self.buffers():
            if p is not None:
                p.requires_grad_(False)

        # Atom parameters. Initialize at scale matching the target factors so
        # parameter reconstruction can start moving downward immediately.
        if init_atom_scale is None:
            init_atom_scale = float(target.A.detach().pow(2).mean().sqrt())
        self.U_A = nn.Parameter(torch.empty(C_A, self.d_in).normal_(0.0, init_atom_scale))
        self.V_A = nn.Parameter(torch.empty(C_A, self.d_hidden).normal_(0.0, init_atom_scale))
        self.U_B = nn.Parameter(torch.empty(C_B, self.d_in).normal_(0.0, init_atom_scale))
        self.V_B = nn.Parameter(torch.empty(C_B, self.d_hidden).normal_(0.0, init_atom_scale))

        self.gate_A = GateNet(self.d_in, gate_hidden, C_A)
        self.gate_B = GateNet(self.d_in, gate_hidden, C_B)

    # --- helpers ---

    def reconstruct_A(self) -> torch.Tensor:
        return torch.einsum("cd,ch->dh", self.U_A, self.V_A)

    def reconstruct_B(self) -> torch.Tensor:
        return torch.einsum("cd,ch->dh", self.U_B, self.V_B)

    def _gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_A = self.gate_A(x)
        raw_B = self.gate_B(x)
        return (
            lower_leaky(raw_A, self.leaky_alpha),
            lower_leaky(raw_B, self.leaky_alpha),
            upper_leaky(raw_A, self.leaky_alpha),
            upper_leaky(raw_B, self.leaky_alpha),
        )

    @staticmethod
    def _stochastic_mask(g: torch.Tensor) -> torch.Tensor:
        """mask = g + (1 - g) * U(0, 1) — cherry-picked from Goodfire nano § E."""
        return g + (1.0 - g) * torch.rand_like(g)

    def _bilinear_forward_with_weights(
        self,
        x: torch.Tensor,
        A_eff: torch.Tensor,
        B_eff: torch.Tensor,
    ) -> torch.Tensor:
        """y = ((x A) ⊙ (x B)) C, with optional output bias.

        A_eff and B_eff may be per-sample ([B, d_in, d_hidden]) or shared
        ([d_in, d_hidden]). We dispatch on rank.
        """
        if A_eff.ndim == 3:
            z_a = torch.einsum("bd,bdh->bh", x, A_eff)
            z_b = torch.einsum("bd,bdh->bh", x, B_eff)
        else:
            z_a = x @ A_eff
            z_b = x @ B_eff
        h = z_a * z_b
        y = h @ self.C_target
        if self.b_out is not None:
            y = y + self.b_out
        return y

    # --- public forwards ---

    def forward(
        self,
        x: torch.Tensor,
        mask_mode: MaskMode = "stochastic",
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DecompForwardAux]:
        """Gated forward.

        mask_mode:
          'deterministic' — masks are the gate values themselves.
          'stochastic'    — masks = g + (1-g)·U(0,1) [as in Goodfire SPD].

        The Δ residuals are always included unmasked: they represent whatever
        the atoms haven't reconstructed yet, so masking them would change the
        effective target and break parameter-reconstruction interpretation.
        """
        g_A, g_B, g_A_upper, g_B_upper = self._gates(x)
        if mask_mode == "stochastic":
            m_A = self._stochastic_mask(g_A)
            m_B = self._stochastic_mask(g_B)
        else:
            m_A = g_A
            m_B = g_B

        A_hat = self.reconstruct_A()                   # [d_in, d_hidden]
        B_hat = self.reconstruct_B()
        delta_A = self.A_target - A_hat
        delta_B = self.B_target - B_hat

        # Per-sample masked weight: A_g[b] = sum_c m_A[b,c] U_A[c] V_A[c]^T + ΔA
        A_g = torch.einsum("bc,cd,ch->bdh", m_A, self.U_A, self.V_A) + delta_A
        B_g = torch.einsum("bc,cd,ch->bdh", m_B, self.U_B, self.V_B) + delta_B
        y = self._bilinear_forward_with_weights(x, A_g, B_g)

        if return_aux:
            aux = DecompForwardAux(
                g_A=g_A, g_B=g_B,
                g_A_upper=g_A_upper, g_B_upper=g_B_upper,
                m_A=m_A, m_B=m_B,
                A_hat=A_hat, B_hat=B_hat,
                delta_A=delta_A, delta_B=delta_B,
            )
            return y, aux
        return y

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen target forward — used as the behavior-loss target during training."""
        return self._bilinear_forward_with_weights(x, self.A_target, self.B_target)

    @torch.no_grad()
    def forward_with_masks(
        self,
        x: torch.Tensor,
        m_A: torch.Tensor | None = None,
        m_B: torch.Tensor | None = None,
        include_delta_A: bool = True,
        include_delta_B: bool = True,
    ) -> torch.Tensor:
        """Eval-time forward with explicitly supplied masks.

        m_A: [B, C_A] or [C_A]. None → all-ones (every atom on).
        Use this for atom ablations during the analysis pass (Day 4).
        """
        A_hat = self.reconstruct_A()
        B_hat = self.reconstruct_B()
        delta_A = self.A_target - A_hat
        delta_B = self.B_target - B_hat

        if m_A is None:
            m_A = torch.ones(self.C_A, device=x.device, dtype=x.dtype)
        if m_B is None:
            m_B = torch.ones(self.C_B, device=x.device, dtype=x.dtype)

        if m_A.ndim == 1:
            A_g_static = torch.einsum("c,cd,ch->dh", m_A, self.U_A, self.V_A)
            if include_delta_A:
                A_g_static = A_g_static + delta_A
            A_eff: torch.Tensor = A_g_static
        else:
            A_g = torch.einsum("bc,cd,ch->bdh", m_A, self.U_A, self.V_A)
            if include_delta_A:
                A_g = A_g + delta_A
            A_eff = A_g

        if m_B.ndim == 1:
            B_g_static = torch.einsum("c,cd,ch->dh", m_B, self.U_B, self.V_B)
            if include_delta_B:
                B_g_static = B_g_static + delta_B
            B_eff: torch.Tensor = B_g_static
        else:
            B_g = torch.einsum("bc,cd,ch->bdh", m_B, self.U_B, self.V_B)
            if include_delta_B:
                B_g = B_g + delta_B
            B_eff = B_g

        return self._bilinear_forward_with_weights(x, A_eff, B_eff)
