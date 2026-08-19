"""The V2 model: Encoder -> (Composer picks Atom | pass) x micro-steps -> Decoder.

Architectural invariants (H1Experiments.md "The Model", E0 amendment R8):
  - The 384-dim state is the ONLY inter-atom DATA channel and is a canonical,
    task-independent code of the digit list. Opaque task identity is exogenous
    router control; atoms and the decoder never receive it.
  - The composer is memoryless: its query is a function of (current state,
    active opaque surface token, micro-step within that token) and nothing
    else - no partner token, absolute token position, GRU, or carried state.
  - Atoms are fully independent function-space MLPs applied as residuals;
    each atom owns its 32-dim routing key. Keys live here, not in the composer.
  - Routing: 17 options = 16 atom keys + 1 dedicated pass key. Pass is a real
    routing choice, inspectable and logged - not an atom that learned nothing.
  - Free routing only. Forced routing exists as a MODE for the quarantined E0
    oracle and for panel diagnostics; nothing in the E1 training path sets it.
  - The state norm is a per-position LayerNorm WITHOUT affine parameters, so
    norm(norm(x)) == norm(x) exactly: a pass step is a bit-exact no-op, and
    zero-delta ablation is exactly "this atom becomes pass".
  - The composer's parameter count must not grow with atom count (checked in
    tests and logged per run in param_counts.json, composer and library always
    separate line items).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import registered as R

MAX_TOKENS = 2
N_STEPS = MAX_TOKENS * R.MICRO_STEPS      # 6 micro-steps for a pair task


def _encoder_layer(cfg) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ff_dim,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=True)


class Encoder(nn.Module):
    """Digit-only canonical encoder: digits [B,6] -> state [B,384]."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.digit_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.seq_len, cfg.d_model) * 0.02)
        self.layer = _encoder_layer(cfg)

    def forward(self, digits: torch.Tensor) -> torch.Tensor:
        h = self.digit_emb(digits) + self.pos            # [B,6,64]
        return self.layer(h)


class Atoms(nn.Module):
    """16 fully independent residual MLPs 384 -> 192 -> 384 (GELU), batched.

    No weight sharing between slots. Each atom owns its routing key; the
    dedicated pass key also lives here (it is a key, not an atom).
    """

    def __init__(self, cfg):
        super().__init__()
        n, s, h = cfg.n_atoms, cfg.state_dim, cfg.atom_hidden
        self.w1 = nn.Parameter(torch.empty(n, s, h))
        self.b1 = nn.Parameter(torch.zeros(n, h))
        self.w2 = nn.Parameter(torch.empty(n, h, s))
        self.b2 = nn.Parameter(torch.zeros(n, s))
        for i in range(n):  # nn.Linear-default init, per independent atom
            nn.init.kaiming_uniform_(self.w1[i].T, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i].T, a=math.sqrt(5))
        bound1 = 1 / math.sqrt(s)
        bound2 = 1 / math.sqrt(h)
        nn.init.uniform_(self.b1, -bound1, bound1)
        nn.init.uniform_(self.b2, -bound2, bound2)
        self.keys = nn.Parameter(torch.randn(n, cfg.key_dim) * 0.02)
        self.pass_key = nn.Parameter(torch.randn(cfg.key_dim) * 0.02)

    def outputs(self, state: torch.Tensor) -> torch.Tensor:
        """Residual deltas of ALL atoms on the state: [B, n_atoms, 384]."""
        h = torch.einsum("bs,nsh->bnh", state, self.w1) + self.b1
        h = F.gelu(h)
        return torch.einsum("bnh,nhs->bns", h, self.w2) + self.b2

    def all_keys(self) -> torch.Tensor:
        return torch.cat([self.keys, self.pass_key[None, :]], dim=0)  # [17,32]


class Composer(nn.Module):
    """Memoryless router: (state, active token, micro-step) -> 32-dim query.

    Control resets for every token: micro-step is 0..2, never flat position
    0..5. This makes a singleton-learned program callable in either token
    position (load-bearing for the P3/Dax split). The partner token is never
    exposed. Parameter count remains independent of atom count.
    """

    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(R.N_SURFACE + 1, cfg.key_dim)  # +1 PAD
        self.micro_emb = nn.Embedding(R.MICRO_STEPS, cfg.key_dim)
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.key_dim, 64),
            nn.GELU(),
            nn.Linear(64, cfg.key_dim),
        )

    def query(self, state: torch.Tensor, active_token: torch.Tensor,
              micro_step: int) -> torch.Tensor:
        control = self.token_emb(active_token) + self.micro_emb.weight[micro_step]
        return self.net(torch.cat([state, control], dim=-1))   # [B,32]


class Decoder(nn.Module):
    """state [B,384] -> per-position digit logits [B,6,10]."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layer = _encoder_layer(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = state.view(-1, self.cfg.seq_len, self.cfg.d_model)
        return self.head(self.layer(h))


class AtomModel(nn.Module):
    ROUTING_MODES = ("gumbel", "hard", "soft", "forced")

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.atoms = Atoms(cfg)
        self.composer = Composer(cfg)
        self.decoder = Decoder(cfg)
        # Non-affine per-position LayerNorm: idempotent, so pass is a no-op.
        self.state_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)

    # -- canonical accessors (D26 lesson: every probe goes through these) -----
    def _norm(self, state: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        h = state.view(b, self.cfg.seq_len, self.cfg.d_model)
        return self.state_norm(h).reshape(b, -1)

    def code(self, digits: torch.Tensor) -> torch.Tensor:
        """The canonical, task-independent state of a digit list: [B,384]."""
        h = self.encoder(digits)
        return self._norm(h.reshape(h.shape[0], -1))

    def step_once(self, state: torch.Tensor, atom_idx: int) -> torch.Tensor:
        """One hard application of a single atom (panel probes)."""
        delta = self.atoms.outputs(state)[:, atom_idx]
        return self._norm(state + delta)

    # -- routing --------------------------------------------------------------
    def _route(self, state, active_token, micro_step, mode, tau,
               forced_idx=None, atom_mask=None, generator=None):
        """Returns (weights [B,17], logits [B,17], soft [B,17]).

        Routers (cfg.router):
          'scaled_dot' - certified E0/E1: q.k / sqrt(key_dim), annealed tau.
          'cosine'     - E1b anti-saturation stack: L2-normalized q and k,
                         logits = alpha * cos(q, k) in [-alpha, alpha]; in
                         gumbel mode the forward pick is argmax(z + sigma*g)
                         and the straight-through surrogate is
                         softmax((z + sigma*g) / tau_backward). Learned norm
                         growth can no longer widen the logit gap.
        """
        q = self.composer.query(state, active_token, micro_step)
        keys = self.atoms.all_keys()
        if self.cfg.router == "cosine":
            q = q / (q.norm(dim=-1, keepdim=True) + self.cfg.router_norm_eps)
            keys = keys / (keys.norm(dim=-1, keepdim=True)
                           + self.cfg.router_norm_eps)
            logits = self.cfg.router_alpha * (q @ keys.T)
        else:
            logits = (q @ keys.T) / math.sqrt(self.cfg.key_dim)
        if atom_mask is not None:  # compensation probe: atom removed from routing
            masked = torch.zeros_like(logits)
            masked[:, : self.cfg.n_atoms] = torch.where(
                atom_mask[None, :], torch.finfo(logits.dtype).min, 0.0)
            logits = logits + masked
        if mode == "forced":
            weights = F.one_hot(forced_idx, self.cfg.n_atoms + 1).to(logits.dtype)
            return weights, logits, weights
        if mode == "hard":
            idx = logits.argmax(dim=-1)
            weights = F.one_hot(idx, self.cfg.n_atoms + 1).to(logits.dtype)
            return weights, logits, weights
        if mode == "soft":
            soft = F.softmax(logits / tau, dim=-1)
            return soft, logits, soft
        # gumbel: straight-through top-1. The unit-Gumbel stream is drawn
        # identically in both routers; the cosine router only SCALES it by
        # sigma (E1b: same underlying draws across arms within a seed).
        u = torch.rand(logits.shape, generator=generator, dtype=logits.dtype)
        neg_log_u = -torch.log(u.clamp(1e-20, 1.0))          # >= 0
        gumbel = -torch.log(neg_log_u.clamp_min(1e-20))
        if self.cfg.router == "cosine":
            noisy = logits + self.cfg.router_sigma * gumbel
            soft = F.softmax(noisy / self.cfg.router_tau_backward, dim=-1)
            idx = noisy.argmax(dim=-1)
        else:
            soft = F.softmax((logits + gumbel) / tau, dim=-1)
            idx = soft.argmax(dim=-1)
        hard = F.one_hot(idx, self.cfg.n_atoms + 1).to(logits.dtype)
        weights = hard + soft - soft.detach()
        return weights, logits, soft

    # -- full forward ---------------------------------------------------------
    def forward(self, digits, tokens, n_tokens, mode="gumbel", tau=1.0,
                forced=None, ablate=None, atom_mask=None, generator=None,
                noise_sigma=0.0, noise_generator=None):
        """
        digits [B,6], tokens [B,2] (PAD-filled), n_tokens [B] in {1,2}.
        mode: 'gumbel' (train) | 'hard'/'soft' (eval) | 'forced' (oracle/panel).
        forced: [B, N_STEPS] int64 routing indices (0..16), used when
            mode='forced'.
        ablate: bool [n_atoms] - zero-delta INTERCEPT at application; routing
            untouched (registered F2 mechanism).
        atom_mask: bool [n_atoms] - remove atoms from routing (compensation
            probe, final checkpoint only, non-gating).
        noise_sigma / noise_generator: E2 interface noise. When sigma > 0,
            every live NONTERMINAL handoff transmits
            LayerNorm(s_clean + Normal(0, sigma^2 I)) to the next composer
            decision and atom; the final live state reaches the decoder clean;
            dead steps receive no noise; pass gets the same channel noise as
            atoms. sigma == 0 bypasses noise generation completely and never
            touches noise_generator (registered no-noise equivalence).
        Number of composition steps = n_tokens * 3; steps beyond an example's
        budget are dead: state frozen, no rent, choice logged as -1.
        """
        assert mode in self.ROUTING_MODES
        state = self.code(digits)
        live_all = (torch.arange(N_STEPS)[None, :]
                    < (n_tokens * R.MICRO_STEPS)[:, None])       # [B,6]
        return self._run_steps(state, tokens, live_all, mode, tau, forced,
                               ablate, atom_mask, generator, start_step=0,
                               noise_sigma=noise_sigma,
                               noise_generator=noise_generator)

    def _run_steps(self, state, tokens, live_all, mode, tau, forced, ablate,
                   atom_mask, generator, start_step=0, noise_sigma=0.0,
                   noise_generator=None):
        """The composition loop. THE single stepping code path.

        forward() enters it at step 0 from code(digits); execute_from_state()
        enters it at a token boundary from an injected state. Steps before
        start_step are not executed at all - they contribute placeholder
        entries (state unchanged, choice -1, zero logits/mass) so every
        returned tensor keeps full [B, N_STEPS] indexing.
        """
        states = [state]
        route_logits, choices, soft_atom_mass = [], [], []
        transmitted = []       # E2: what the next step actually received
        for k in range(N_STEPS):
            if k < start_step:
                states.append(state)
                route_logits.append(torch.zeros(
                    state.shape[0], self.cfg.n_atoms + 1, dtype=state.dtype))
                choices.append(torch.full((state.shape[0],), -1,
                                          dtype=torch.int64))
                soft_atom_mass.append(torch.zeros(state.shape[0],
                                                  dtype=state.dtype))
                continue
            live = live_all[:, k]
            token_pos = k // R.MICRO_STEPS
            micro_step = k % R.MICRO_STEPS
            active_token = tokens[:, token_pos]
            weights, logits, soft = self._route(
                state, active_token, micro_step, mode, tau,
                forced_idx=forced[:, k] if forced is not None else None,
                atom_mask=atom_mask, generator=generator)
            atom_w = weights[:, : self.cfg.n_atoms]              # [B,16]
            if ablate is not None:
                atom_w = atom_w * (~ablate)[None, :].to(atom_w.dtype)
            delta = torch.einsum("bn,bns->bs", atom_w, self.atoms.outputs(state))
            new_state = self._norm(state + delta)
            # In hard/forced modes a step contributing no delta (pass pick,
            # ablated pick) is an EXACT no-op: bypass the norm so pass ==
            # identity bitwise and zero-delta ablation is exactly "atom
            # becomes pass". Gumbel mode keeps the normed path everywhere so
            # the straight-through gradient of the routing weights survives
            # (the value-zero soft residual still carries gradient).
            if mode in ("hard", "forced"):
                noop = (delta.abs().amax(dim=-1) == 0) | ~live
            else:
                noop = ~live
            state = torch.where(noop[:, None], state, new_state)

            states.append(state)
            route_logits.append(logits)
            hard_choice = (weights.detach().argmax(dim=-1)
                           .masked_fill(~live, -1))
            choices.append(hard_choice)
            soft_atom_mass.append(
                (soft[:, : self.cfg.n_atoms].sum(dim=-1) * live.to(soft.dtype)))

            # E2 interface noise: corrupt the handoff into step k+1 for every
            # example whose NEXT step is live (prefix liveness makes that
            # exactly "live nonterminal", token boundaries included). The
            # clean producer state stays in `states` for diagnostics only;
            # the carry - what composer and atom see next - is transmitted.
            # One fixed-shape draw per handoff keeps the dedicated noise
            # stream's consumption independent of batch masks.
            if noise_sigma > 0 and k + 1 < N_STEPS:
                nonterminal = live_all[:, k + 1]
                eps = noise_sigma * torch.randn(
                    state.shape, generator=noise_generator, dtype=state.dtype)
                noisy = self._norm(state + eps)
                state = torch.where(nonterminal[:, None], noisy, state)
                transmitted.append(state)

        out = {
            "logits": self.decoder(state),                       # [B,6,10]
            "states": states,                                    # 7 x [B,384]
            "route_logits": torch.stack(route_logits, dim=1),    # [B,6,17]
            "choices": torch.stack(choices, dim=1),              # [B,6], -1 dead
            "soft_atom_mass": torch.stack(soft_atom_mass, dim=1),  # [B,6]
            "live": live_all,                                    # [B,6]
        }
        if noise_sigma > 0:
            out["states_transmitted"] = transmitted  # entry k -> into step k+1
        return out

    def execute_from_state(self, state, tokens, n_tokens, start_token_idx,
                           mode="hard", tau=1.0, forced=None, ablate=None,
                           atom_mask=None):
        """PANEL-SIDE ONLY. Continue execution from an injected state.

        Used by the canonical substitution test: replace the state at a token
        boundary with the encoding of the ground-truth partial result, then run
        the remaining token's micro-steps through the SAME loop forward() uses.

        state: [B, state_dim] injected at the boundary before token
            start_token_idx. start_token_idx=0 with state=code(x) reproduces
            forward() exactly (asserted in tests).
        Returns forward()'s dict shape; entries for steps before the boundary
        are placeholders, not executed.
        """
        assert mode in self.ROUTING_MODES
        assert not self.training, (
            "execute_from_state is a read-only panel probe; the model must be "
            "in eval mode (probes read, never write)")
        live_all = (torch.arange(N_STEPS)[None, :]
                    < (n_tokens * R.MICRO_STEPS)[:, None])
        return self._run_steps(state, tokens, live_all, mode, tau, forced,
                               ablate, atom_mask, generator=None,
                               start_step=start_token_idx * R.MICRO_STEPS)


def param_counts(model: AtomModel) -> dict:
    def n(mod) -> int:
        return sum(p.numel() for p in mod.parameters())

    keys = model.atoms.keys.numel() + model.atoms.pass_key.numel()
    atoms_total = n(model.atoms) - keys
    counts = {
        "encoder": n(model.encoder),
        "decoder": n(model.decoder),
        "composer": n(model.composer),          # separate line item, ALWAYS
        "atoms_total": atoms_total,             # never summed with composer
        "atoms_each": atoms_total // model.cfg.n_atoms,
        "keys": keys,
        "state_norm": sum(p.numel() for p in model.state_norm.parameters()),
        "n_atoms": model.cfg.n_atoms,
        "total": n(model),
    }
    return counts
