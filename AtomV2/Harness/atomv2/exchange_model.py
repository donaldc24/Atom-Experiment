"""E11 exact conservative exchange model over categorical digit fields.

This is a discrete neural construction inspired by conservative transport. It
is neither a Navier-Stokes discretization nor a consequence of a fluid theorem.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import registered as R


EVEN_MATCHING = ((0, 1), (2, 3), (4, 5))
ODD_MATCHING = ((1, 2), (3, 4), (5, 0))
MATCHINGS = tuple(EVEN_MATCHING if s % 2 == 0 else ODD_MATCHING
                  for s in range(R.E11_EXCHANGE_STAGES))


class ConservativeExchangeModel(nn.Module):
    """Opaque-token operators with exact pair transport and digit interfaces."""

    def __init__(self, arm: str):
        super().__init__()
        if arm not in R.E11_ARMS:
            raise ValueError(f"unknown E11 arm {arm!r}")
        self.arm = arm
        self.transport_kind = R.E11_TRANSPORT[arm]
        self.viscosity = R.E11_VISCOSITY[arm]

        reaction = R.E11_REACTION_INIT * torch.randn(
            R.N_SURFACE, R.SEQ_LEN, R.VOCAB, R.VOCAB)
        self.reaction_logits = nn.Parameter(reaction)

        if self.transport_kind == "direct":
            self.gate_logits = nn.Parameter(torch.full(
                (R.N_SURFACE, R.E11_EXCHANGE_STAGES, 3),
                R.E11_GATE_BIAS))
        elif self.transport_kind == "reynolds":
            shape = (R.N_SURFACE, R.E11_EXCHANGE_STAGES, R.SEQ_LEN)
            self.pulse_a = nn.Parameter(R.E11_PULSE_INIT * torch.randn(shape))
            self.pulse_b = nn.Parameter(R.E11_PULSE_INIT * torch.randn(shape))
        else:
            raise ValueError(f"unknown transport {self.transport_kind!r}")

        left = [[a for a, _ in pairs] for pairs in MATCHINGS]
        right = [[b for _, b in pairs] for pairs in MATCHINGS]
        self.register_buffer("left_index", torch.tensor(left, dtype=torch.long))
        self.register_buffer("right_index", torch.tensor(right, dtype=torch.long))
        self.register_buffer("digit_identity", torch.eye(R.VOCAB))

    @property
    def transport_parameter_count(self) -> int:
        if self.transport_kind == "direct":
            return self.gate_logits.numel()
        return self.pulse_a.numel() + self.pulse_b.numel()

    def pulse_profiles(self, token_ids: torch.Tensor) -> tuple[torch.Tensor,
                                                                torch.Tensor]:
        if self.transport_kind != "reynolds":
            raise RuntimeError("pulse profiles exist only for Reynolds arms")
        a = self.pulse_a[token_ids]
        b = self.pulse_b[token_ids]
        return (a - a.mean(dim=-1, keepdim=True),
                b - b.mean(dim=-1, keepdim=True))

    def exchange_gates(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return gates with shape [batch, stage, three disjoint edges]."""
        if self.transport_kind == "direct":
            return torch.sigmoid(self.gate_logits[token_ids])
        a, b = self.pulse_profiles(token_ids)
        left = self.left_index[None].expand(token_ids.shape[0], -1, -1)
        right = self.right_index[None].expand(token_ids.shape[0], -1, -1)
        ai, aj = a.gather(2, left), a.gather(2, right)
        bi, bj = b.gather(2, left), b.gather(2, right)
        stress = 0.5 * (ai * bj + bi * aj)
        return torch.sigmoid(R.E11_GATE_BIAS + R.E11_STRESS_SCALE * stress)

    def transport(self, state: torch.Tensor,
                  token_ids: torch.Tensor) -> tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     list[torch.Tensor]]:
        """Apply exact conservative pair blocks, retaining stage audit states."""
        gates = self.exchange_gates(token_ids)
        current = state
        stage_states = []
        for stage, pairs in enumerate(MATCHINGS):
            nxt = current.clone()
            for edge, (i, j) in enumerate(pairs):
                g = gates[:, stage, edge, None]
                qi, qj = current[:, i, :], current[:, j, :]
                nxt[:, i, :] = (1.0 - g) * qi + g * qj
                nxt[:, j, :] = g * qi + (1.0 - g) * qj
            current = nxt
            stage_states.append(current)
        return current, gates, stage_states

    def reaction_kernel(self, token_ids: torch.Tensor) -> torch.Tensor:
        eye = self.digit_identity.to(dtype=self.reaction_logits.dtype)
        logits = (self.reaction_logits[token_ids]
                  + R.E11_REACTION_IDENTITY_BIAS * eye[None, None])
        return F.softmax(logits, dim=-1)

    @staticmethod
    def normalize(state: torch.Tensor) -> torch.Tensor:
        nonnegative = state.clamp_min(0.0)
        return nonnegative / nonnegative.sum(dim=-1, keepdim=True).clamp_min(
            1e-12)

    def canonicalize(self, state: torch.Tensor) -> torch.Tensor:
        """One-hot forward boundary with the normalized soft backward path."""
        soft = self.normalize(state)
        hard = F.one_hot(soft.argmax(dim=-1), R.VOCAB).to(soft.dtype)
        if self.training:
            return hard + soft - soft.detach()
        return hard

    def advance(self, state: torch.Tensor,
                token_ids: torch.Tensor) -> dict[str, torch.Tensor | list]:
        transported, gates, stage_states = self.transport(state, token_ids)
        if self.viscosity:
            transported = ((1.0 - 2.0 * self.viscosity) * transported
                           + self.viscosity * torch.roll(transported, 1, dims=1)
                           + self.viscosity * torch.roll(transported, -1, dims=1))
        kernel = self.reaction_kernel(token_ids)
        reacted = torch.einsum("bpi,bpio->bpo", transported, kernel)
        return {"state": self.normalize(reacted), "gates": gates,
                "stage_states": stage_states, "kernel": kernel,
                "transported": transported}

    def _perturb_boundary(self, state: torch.Tensor, k: int,
                          handoff_noise: torch.Tensor | None,
                          drop_positions: torch.Tensor | None,
                          corrupt_positions: torch.Tensor | None,
                          corrupt_digits: torch.Tensor | None) -> torch.Tensor:
        out = state
        if handoff_noise is not None and k < handoff_noise.shape[1]:
            out = self.normalize(out + handoff_noise[:, k])
        if drop_positions is not None and k < drop_positions.shape[1]:
            pos = drop_positions[:, k]
            mask = F.one_hot(pos.clamp(0, R.SEQ_LEN - 1), R.SEQ_LEN).bool()
            mask = mask & (pos >= 0)[:, None]
            uniform = torch.full_like(out, 1.0 / R.VOCAB)
            out = torch.where(mask[:, :, None], uniform, out)
        if corrupt_positions is not None and k < corrupt_positions.shape[1]:
            if corrupt_digits is None:
                raise ValueError("corrupt_digits required with corrupt_positions")
            pos = corrupt_positions[:, k]
            digit = corrupt_digits[:, k].clamp(0, R.VOCAB - 1)
            mask = F.one_hot(pos.clamp(0, R.SEQ_LEN - 1), R.SEQ_LEN).bool()
            mask = mask & (pos >= 0)[:, None]
            replacement = F.one_hot(digit, R.VOCAB).to(out.dtype)
            replacement = replacement[:, None, :].expand(-1, R.SEQ_LEN, -1)
            out = torch.where(mask[:, :, None], replacement, out)
        return self.normalize(out)

    def forward(self, digits: torch.Tensor, tokens: torch.Tensor,
                n_tokens: torch.Tensor | None = None,
                handoff_noise: torch.Tensor | None = None,
                drop_positions: torch.Tensor | None = None,
                corrupt_positions: torch.Tensor | None = None,
                corrupt_digits: torch.Tensor | None = None) -> dict:
        """Apply arbitrary-length sequences with canonical nonterminal states."""
        state = F.one_hot(digits, R.VOCAB).to(self.reaction_logits.dtype)
        if n_tokens is None:
            n_tokens = torch.full((digits.shape[0],), tokens.shape[1],
                                  dtype=torch.long, device=digits.device)
        final = state
        boundaries = [state]
        gates, kernels = [], []
        for k in range(tokens.shape[1]):
            live = k < n_tokens
            safe_token = tokens[:, k].clamp(0, R.N_SURFACE - 1)
            step = self.advance(state, safe_token)
            candidate = step["state"]
            finishes = live & ((k + 1) == n_tokens)
            final = torch.where(finishes[:, None, None], candidate, final)
            continues = live & ((k + 1) < n_tokens)
            handoff = self._perturb_boundary(
                candidate, k, handoff_noise, drop_positions,
                corrupt_positions, corrupt_digits)
            canonical = self.canonicalize(handoff)
            state = torch.where(continues[:, None, None], canonical, state)
            boundaries.append(state)
            gates.append(step["gates"])
            kernels.append(step["kernel"])
        return {"probs": final,
                "logits": final.clamp_min(1e-12).log(),
                "boundaries": boundaries,
                "gates": gates,
                "kernels": kernels}
