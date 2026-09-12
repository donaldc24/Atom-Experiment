"""E10 Reynolds Transport Network.

This is a separate model family, not an AtomModel variant.  A token advances
a six-cell feature field by transport, optional periodic heat diffusion, and
a pointwise token-conditioned force.  The Reynolds arms parameterize transport
logits by the cross-covariance of centered signed pulse profiles and project
them to a doubly stochastic matrix with log-Sinkhorn normalization.

The construction is only a neural design analogy.  It is not a numerical
Navier-Stokes solver and no fluid theorem is assumed here.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import registered as R


def log_sinkhorn(logits: torch.Tensor, n_iters: int) -> torch.Tensor:
    """Positive doubly stochastic projection in log space.

    Alternating row/column log normalizations preserve differentiability and
    avoid the overflow of exponentiating unrestricted learned logits first.
    The final row normalization makes the forward convex-combination contract
    especially tight; the registered 32 iterations make column error tiny at
    the six-cell scale.
    """
    z = logits
    for _ in range(n_iters):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
    z = z - torch.logsumexp(z, dim=-1, keepdim=True)
    return z.exp()


class ReynoldsFlowModel(nn.Module):
    """Sequential opaque-token operator over a six-cell feature field."""

    def __init__(self, arm: str):
        super().__init__()
        if arm not in R.E10_ARMS:
            raise ValueError(f"unknown E10 arm {arm!r}")
        self.arm = arm
        self.transport_kind = R.E10_TRANSPORT[arm]
        self.incompressible = R.E10_INCOMPRESSIBLE[arm]
        self.viscosity = R.E10_VISCOSITY[arm]
        self.width = R.E10_WIDTH
        self.n_positions = R.SEQ_LEN

        self.digit_emb = nn.Embedding(R.VOCAB, self.width)
        self.state_pos = nn.Parameter(
            torch.randn(self.n_positions, self.width) * 0.02)
        self.token_emb = nn.Embedding(R.N_SURFACE, R.E10_TOKEN_DIM)
        self.force_pos = nn.Parameter(
            torch.randn(self.n_positions, R.E10_POS_DIM) * 0.02)
        force_in = self.width + R.E10_TOKEN_DIM + R.E10_POS_DIM
        self.force = nn.Sequential(
            nn.Linear(force_in, R.E10_FORCE_HIDDEN),
            nn.GELU(),
            nn.Linear(R.E10_FORCE_HIDDEN, self.width),
        )
        self.input_norm = nn.LayerNorm(self.width)
        self.state_norm = nn.LayerNorm(self.width)
        self.decoder = nn.Linear(self.width, R.VOCAB)

        # A29 and A30/A31 each receive exactly 36 learned transport scalars
        # per token: 6x6 direct logits versus 2 * rank(3) * 6 pulse values.
        if self.transport_kind == "direct":
            self.transport_logits = nn.Parameter(torch.zeros(
                R.N_SURFACE, self.n_positions, self.n_positions))
        elif self.transport_kind == "reynolds":
            scale = 0.20
            self.pulse_a = nn.Parameter(scale * torch.randn(
                R.N_SURFACE, R.E10_PULSE_RANK, self.n_positions))
            self.pulse_b = nn.Parameter(scale * torch.randn(
                R.N_SURFACE, R.E10_PULSE_RANK, self.n_positions))

        self.register_buffer("identity", torch.eye(self.n_positions))

    @property
    def transport_parameter_count(self) -> int:
        if self.transport_kind == "local":
            return 0
        if self.transport_kind == "direct":
            return self.transport_logits.numel()
        return self.pulse_a.numel() + self.pulse_b.numel()

    def pulse_profiles(self, token_ids: torch.Tensor) -> tuple[torch.Tensor,
                                                                torch.Tensor]:
        """Centered profiles; the implicit +/- pulse ensemble has zero mean."""
        if self.transport_kind != "reynolds":
            raise RuntimeError("pulse profiles exist only for Reynolds arms")
        a = self.pulse_a[token_ids]
        b = self.pulse_b[token_ids]
        return a - a.mean(dim=-1, keepdim=True), \
            b - b.mean(dim=-1, keepdim=True)

    def transport_matrix(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return P[b, output_position, input_position]."""
        batch = token_ids.shape[0]
        eye = self.identity.to(dtype=self.digit_emb.weight.dtype)
        if self.transport_kind == "local":
            return eye.unsqueeze(0).expand(batch, -1, -1)
        if self.transport_kind == "direct":
            raw = self.transport_logits[token_ids]
        else:
            a, b = self.pulse_profiles(token_ids)
            raw = torch.einsum("brj,brk->bjk", a, b) / math.sqrt(
                R.E10_PULSE_RANK)
        logits = (raw + R.E10_IDENTITY_BIAS * eye) / R.E10_TRANSPORT_TEMP
        if self.incompressible:
            return log_sinkhorn(logits, R.E10_SINKHORN_ITERS)
        return F.softmax(logits, dim=-1)

    def code(self, digits: torch.Tensor) -> torch.Tensor:
        return self.input_norm(self.digit_emb(digits) + self.state_pos)

    def advance(self, state: torch.Tensor,
                token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
        """One Lie-split transport/diffusion/forcing token update."""
        p = self.transport_matrix(token_ids)
        transported = torch.einsum("bij,bjd->bid", p, state)
        if self.viscosity:
            lap = (torch.roll(transported, 1, dims=1)
                   + torch.roll(transported, -1, dims=1)
                   - 2.0 * transported)
            transported = transported + self.viscosity * lap

        tok = self.token_emb(token_ids)[:, None, :].expand(
            -1, self.n_positions, -1)
        pos = self.force_pos[None, :, :].expand(state.shape[0], -1, -1)
        force = self.force(torch.cat([transported, tok, pos], dim=-1))
        return self.state_norm(transported + force), p, force

    def forward(self, digits: torch.Tensor, tokens: torch.Tensor,
                n_tokens: torch.Tensor | None = None,
                handoff_noise: torch.Tensor | None = None) -> dict:
        """Apply an arbitrary-length opaque-token sequence.

        `handoff_noise[:, k]` is injected after token k and before k+1.  It is
        evaluation-only in E10; callers provide a fixed indexed tensor so no
        model RNG stream is consumed.
        """
        state = self.code(digits)
        states = [state]
        matrices, forces = [], []
        if n_tokens is None:
            n_tokens = torch.full((digits.shape[0],), tokens.shape[1],
                                  dtype=torch.int64, device=digits.device)
        for k in range(tokens.shape[1]):
            live = k < n_tokens
            safe_token = tokens[:, k].clamp(0, R.N_SURFACE - 1)
            candidate, p, force = self.advance(state, safe_token)
            state = torch.where(live[:, None, None], candidate, state)
            matrices.append(p)
            forces.append(force)
            states.append(state)
            if handoff_noise is not None and k < handoff_noise.shape[1]:
                nonterminal = (k + 1) < n_tokens
                noisy = self.state_norm(state + handoff_noise[:, k])
                state = torch.where(nonterminal[:, None, None], noisy, state)
        return {
            "logits": self.decoder(state),
            "states": states,
            "transport_matrices": matrices,
            "forces": forces,
        }
