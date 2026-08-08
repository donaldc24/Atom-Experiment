"""Function-space atom architecture for E1.

Atoms are invoked subnetworks composed by routing and sequencing. There is no
summation of weight deltas anywhere in this file (prospectus 2.1).

    tokens -> Encoder -> h0
    for t in 1..T:
        q_t = Composer(h_{t-1}, instruction_t, t)
        w_t = route(q_t . K_atoms / sqrt(d_k))
        h_t = h_{t-1} + sum_i w_t[i] * Atom_i(h_{t-1})
    h_T -> Decoder -> logits over V^L

Keys live in the atoms, not the composer (H5 content-based routing), so composer
parameter count is independent of N.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens [B, L] -> state [B, L*d]
        h = self.emb(tokens) + self.pos
        h = self.layer(h)
        return h.flatten(1)

    def forward_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Re-encode a distribution over tokens, differentiably.

        E1b's code-consistency and bottleneck rungs need enc(dec(h)), and dec emits
        logits rather than token ids. Weighting the embedding table by the token
        distribution keeps that path differentiable; a hard one-hot input reproduces
        forward() exactly.  probs [B, L, V] -> state [B, L*d].
        """
        h = probs @ self.emb.weight + self.pos
        h = self.layer(h)
        return h.flatten(1)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state [B, L*d] -> logits [B, L, V]
        h = state.view(-1, self.cfg.seq_len, self.cfg.d_model)
        h = self.layer(h)
        return self.head(self.norm(h))


class AtomBank(nn.Module):
    """N independent 2-layer MLPs over the flattened state, evaluated as one batched op.

    Each atom carries its own key vector k_i in R^{d_k}.
    """

    def __init__(self, cfg):
        super().__init__()
        n, s, hid, dk = cfg.n_atoms, cfg.state_dim, cfg.atom_hidden, cfg.d_key
        self.n_atoms, self.state_dim, self.hidden, self.d_key = n, s, hid, dk

        self.w1 = nn.Parameter(torch.empty(n, s, hid))
        self.b1 = nn.Parameter(torch.zeros(n, hid))
        self.w2 = nn.Parameter(torch.empty(n, hid, s))
        self.b2 = nn.Parameter(torch.zeros(n, s))
        self.keys = nn.Parameter(torch.empty(n, dk))

        with torch.no_grad():
            for i in range(n):
                nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
            # Start each atom close to a no-op so the residual stack begins near identity.
            self.w2.mul_(0.1)
            nn.init.normal_(self.keys, std=1.0 / math.sqrt(dk))

    def outputs(self, state: torch.Tensor) -> torch.Tensor:
        """All atom outputs for a batch: [B, N, state_dim]."""
        z = torch.einsum("bs,nsh->bnh", state, self.w1) + self.b1
        z = F.gelu(z)
        return torch.einsum("bnh,nhs->bns", z, self.w2) + self.b2

    def logits(self, query: torch.Tensor) -> torch.Tensor:
        return query @ self.keys.t() / math.sqrt(self.d_key)


class Composer(nn.Module):
    """Small, and does NOT grow with N -- H5 instrumentation depends on this."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.instr_emb = nn.Embedding(cfg.n_primitives, cfg.composer_instr_dim)
        self.state_proj = nn.Linear(cfg.d_model, cfg.composer_state_dim)
        self.gru = nn.GRUCell(
            cfg.composer_instr_dim + cfg.composer_state_dim, cfg.composer_hidden
        )
        self.to_query = nn.Linear(cfg.composer_hidden, cfg.d_key)

    def init_hidden(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.composer_hidden, device=device, dtype=dtype)

    def forward(self, state: torch.Tensor, instr_token: torch.Tensor,
                hidden: torch.Tensor):
        pooled = state.view(-1, self.cfg.seq_len, self.cfg.d_model).mean(dim=1)
        x = torch.cat([self.instr_emb(instr_token), self.state_proj(pooled)], dim=-1)
        hidden = self.gru(x, hidden)
        return self.to_query(hidden), hidden


class AtomNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.atoms = AtomBank(cfg)
        self.composer = Composer(cfg)
        self.decoder = Decoder(cfg)
        # E1b R1+: normalise the state after each atom application, so composition
        # cannot drift off the shell the encoder occupies. Built only when enabled,
        # so E1 checkpoints still load into this class unchanged.
        self.state_norm = (
            nn.LayerNorm(cfg.d_model) if getattr(cfg, "atom_layernorm", False) else None
        )

    def _apply_state_norm(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_norm is None:
            return state
        h = state.view(-1, self.cfg.seq_len, self.cfg.d_model)
        return self.state_norm(h).flatten(1)

    # -- canonical state space -------------------------------------------
    # Every probe must go through these two, never reconstruct a transition by
    # hand. With LayerNorm in the path "the code" is LN(enc(.)), not enc(.), so a
    # hand-built h0 + atom(h0) would be measured against a target the model is not
    # trying to hit, and a hand-built probe would feed the decoder un-normed states
    # it never saw in training. See D26.

    def code(self, tokens: torch.Tensor) -> torch.Tensor:
        """The canonical encoding of a token sequence, in the space states live in."""
        return self._apply_state_norm(self.encoder(tokens))

    def encode_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Canonical encoding of a token distribution (differentiable)."""
        return self._apply_state_norm(self.encoder.forward_from_probs(probs))

    def recode(self, state: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """enc(dec(h)) -- the round trip the code-consistency constraint asks for.

        `hard=True` uses a straight-through one-hot: the forward pass re-encodes the
        ARGMAX token, exactly what `code_residual` measures at eval, while gradients
        flow through the softmax.

        A soft round trip is not equivalent and must not be used as the training
        form. With softmax probabilities the model can satisfy the constraint by
        making the intermediate decode *uninformative*: a near-uniform distribution
        re-encodes to one fixed point, so h_t can sit at that point without ever
        being a valid code. Measured on R2 w=10 seed 0 -- soft residual 0.028, hard
        residual 0.802, decoder entropy 2.203 of a maximum 2.303 nats. See D27.
        """
        logits = self.decoder(state)
        soft = F.softmax(logits, dim=-1)
        if not hard:
            return self.encode_probs(soft)
        onehot = F.one_hot(logits.argmax(dim=-1), self.cfg.vocab).to(soft.dtype)
        return self.encode_probs(onehot + soft - soft.detach())

    def step(self, state: torch.Tensor, atom_idx, *, project: bool = False,
             tau: float = 1.0, generator=None) -> torch.Tensor:
        """One composition step, exactly as forward() performs it.

        `atom_idx` is an int (same atom for the batch) or a [B] tensor. `project`
        applies the R3 bottleneck; forward() only projects at intermediate steps, so
        callers must mirror whichever step they are reproducing.
        """
        atom_out = self.atoms.outputs(state)
        if isinstance(atom_idx, int):
            delta = atom_out[:, atom_idx, :]
        else:
            idx = atom_idx.view(-1, 1, 1).expand(-1, 1, atom_out.shape[2])
            delta = atom_out.gather(1, idx).squeeze(1)
        state = self._apply_state_norm(state + delta)
        if project and getattr(self.cfg, "code_bottleneck", False):
            state = self.code_project(state, tau=tau, generator=generator)
        return state

    def code_project(self, state: torch.Tensor, tau: float = 1.0,
                     hard: bool = True, stochastic: bool = True,
                     generator=None) -> torch.Tensor:
        """E1b R3: push the state through the discrete code and back.

        h -> dec -> Gumbel-softmax over the vocabulary -> enc -> h'. Gumbel-softmax
        with hard=True is used rather than a plain straight-through argmax, to match
        the routing path's existing discretisation (spec 2) so the two discrete
        choices in the model behave the same way and share one temperature schedule.
        """
        logits = self.decoder(state)
        if not hard:
            probs = F.softmax(logits / tau, dim=-1)
        elif stochastic:
            probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            # Deterministic straight-through argmax: same forward value every pass.
            soft = F.softmax(logits / tau, dim=-1)
            onehot = F.one_hot(logits.argmax(dim=-1), self.cfg.vocab).to(soft.dtype)
            probs = onehot + soft - soft.detach()
        return self.encode_probs(probs)

    def code_residual(self, state: torch.Tensor) -> torch.Tensor:
        """E1b metric: relative L2 distance from the state to its own re-encoding.

        Measures whether the state is a valid code at all -- low means h sits where
        enc puts things. Uses a hard argmax decode: this is a measurement, not a
        training signal, so it needs no gradient path.
        """
        probs = F.one_hot(self.decoder(state).argmax(dim=-1),
                          self.cfg.vocab).to(state.dtype)
        recoded = self.encode_probs(probs)
        return (state - recoded).norm(dim=1) / state.norm(dim=1).clamp_min(1e-6)

    # -- routing -----------------------------------------------------------
    def _route(self, logits, mode, tau, forced_idx, atom_mask):
        if atom_mask is not None:
            logits = logits.masked_fill(~atom_mask, float("-inf"))

        if mode == "forced":
            weights = F.one_hot(forced_idx, self.cfg.n_atoms).to(logits.dtype)
        elif mode == "gumbel":
            weights = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        elif mode == "hard":
            weights = F.one_hot(logits.argmax(dim=-1), self.cfg.n_atoms).to(logits.dtype)
        elif mode == "soft":
            weights = F.softmax(logits, dim=-1)
        else:
            raise ValueError(f"unknown routing mode {mode!r}")
        return weights, logits

    def forward(
        self,
        tokens: torch.Tensor,
        instruction: torch.Tensor,      # [B, T] primitive-id tokens
        mode: str = "gumbel",
        tau: float = 1.0,
        forced: torch.Tensor | None = None,   # [B, T] atom indices (A0)
        atom_mask: torch.Tensor | None = None,  # [N] bool, True = selectable
        ablate: torch.Tensor | None = None,     # [N] bool, True = output zeroed
        atom_dropout: float = 0.0,
        generator: torch.Generator | None = None,
    ):
        B = tokens.shape[0]
        # Normalise the encoder output too: otherwise h0 is un-normed while every
        # later state is normed, so the same atom weights see two different input
        # distributions and "the code" is ill-defined. See D26.
        state = self.code(tokens)
        hidden = self.composer.init_hidden(B, state.device, state.dtype)

        route_logits, hard_choices, soft_weights, states = [], [], [], []
        for t in range(self.cfg.depth):
            query, hidden = self.composer(state, instruction[:, t], hidden)
            logits = self.atoms.logits(query)
            weights, masked_logits = self._route(
                logits, mode, tau, None if forced is None else forced[:, t], atom_mask
            )

            keep = torch.ones(B, self.cfg.n_atoms, device=state.device, dtype=state.dtype)
            if atom_dropout > 0.0:
                drop = torch.rand(
                    (B, self.cfg.n_atoms), device=state.device, generator=generator
                ) < atom_dropout
                keep = keep.masked_fill(drop, 0.0)
            if ablate is not None:
                keep = keep * (~ablate).to(state.dtype)

            atom_out = self.atoms.outputs(state)                 # [B, N, S]
            state = state + torch.einsum("bn,bns->bs", weights * keep, atom_out)
            state = self._apply_state_norm(state)                # E1b R1+
            if getattr(self.cfg, "code_bottleneck", False) and t < self.cfg.depth - 1:
                # E1b R3: intermediate states are forced through the discrete code.
                # Gumbel sampling ONLY while training. At eval the projection must be
                # a deterministic argmax, or every pass realises a different
                # intermediate and predictions/ablations/diagnostics are each measured
                # on a different model. See D32.
                state = self.code_project(
                    state, tau=tau, stochastic=self.training, generator=generator)

            route_logits.append(masked_logits)
            hard_choices.append(masked_logits.argmax(dim=-1))
            soft_weights.append(F.softmax(masked_logits, dim=-1))
            states.append(state)

        logits_out = self.decoder(state)
        return {
            "logits": logits_out,
            "route_logits": torch.stack(route_logits, dim=1),   # [B, T, N]
            "routing_hard": torch.stack(hard_choices, dim=1),   # [B, T]
            "routing_soft": torch.stack(soft_weights, dim=1),   # [B, T, N]
            "states": torch.stack(states, dim=1),               # [B, T, state_dim]
        }

    def probe_atom(self, tokens: torch.Tensor, atom_idx: int) -> torch.Tensor:
        """M3 as literally specified: encoder -> single atom -> decoder.

        Kept as a diagnostic only. It is off-distribution for this architecture:
        T is always 2, so the decoder never sees a state with one residual add.
        See probe_atom_depth_matched and DECISIONS.md D11.
        """
        return self.decoder(self.step(self.code(tokens), int(atom_idx)))

    @torch.no_grad()
    def identity_atom(self, tokens: torch.Tensor) -> int:
        """The atom this model itself uses for the identity slot of a length-1 task.

        Determined by the model's own routing on the identity instruction, so it is
        model-agnostic -- no arm is assumed to have put identity in atom 0.
        """
        instr = torch.zeros(tokens.shape[0], self.cfg.depth, dtype=torch.long)
        out = self(tokens, instr, mode="hard")
        choices = out["routing_hard"][:, -1]
        return int(torch.bincount(choices, minlength=self.cfg.n_atoms).argmax())

    def probe_atom_depth_matched(self, tokens: torch.Tensor, atom_idx: int,
                                 id_atom: int) -> torch.Tensor:
        """M3, depth-matched: apply atom_i, then the model's identity atom, then decode.

        This is exactly the length-1 task (p, identity) that spec 1 places in training
        so that standalone probing is in-distribution.
        """
        b = tokens.shape[0]
        instr = torch.zeros(b, self.cfg.depth, dtype=torch.long)
        forced = torch.full((b, self.cfg.depth), id_atom, dtype=torch.long)
        forced[:, 0] = atom_idx
        # Routing is forced, so the instruction cannot influence the output.
        return self(tokens, instr, mode="forced", forced=forced)["logits"]

    # -- parameter groups --------------------------------------------------
    def atom_parameters(self):
        return list(self.atoms.parameters())

    def non_atom_parameters(self):
        # state_norm MUST be included. It is a learnable LayerNorm; omitting it left
        # its gains frozen at init while its gradients still entered the global norm
        # used by clip_grad_norm_, shrinking the clip coefficient and suppressing
        # every other parameter's update. See D32.
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
            list(self.composer.parameters())
        if self.state_norm is not None:
            params += list(self.state_norm.parameters())
        return params
