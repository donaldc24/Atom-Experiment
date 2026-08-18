"""E3 atom sandbox (H1-Experiment3.md): three content-agnostic pressures on
the atom MLPs, trained alongside the UNTOUCHED A6 task path.

    standalone + closure + functional uniqueness

Non-negotiable invariants, enforced here and unit-tested in tests/test_e3.py:

  1. Gradient boundary. The sandbox receives z0 = stopgrad(code(x)) and its
     losses update ONLY the atom MLP parameters (w1/b1/w2/b2). Encoder,
     decoder, composer, routing keys and the pass key are frozen for the
     construction of every sandbox graph (sandbox_grad_boundary). The decoder
     in particular is the frozen validity evaluator and fingerprint reader -
     it must never learn to interpret bad atom states; the atoms must become
     independently readable.
  2. Exact same atom update. Every sandbox application goes through
     model.step_once (Apply_i(z) = LN(z + A_i(z))) via apply_chain. There is
     no separate sandbox atom implementation.
  3. No gradient into routing. The composer is never called; keys never enter
     any sandbox computation; usage weights are detached buffers.
  4. Dedicated randomness. All sandbox sampling comes from the 'e3_sandbox'
     numpy stream (telemetry/init calibration from the indexed
     'e3_sandbox_eval' stream). No torch RNG is consumed: a sandbox-bearing
     run's routing/noise/init draws are bit-identical to its A6 pair.

The three branches, computed per training step on the SAME batch's z0:

  standalone   z_i = Apply_i(z0) for one uniformly drawn atom i;
               L_standalone = L_valid(z_i). Every atom is explicitly trained
               on a clean encoder state with no predecessor.
  closure      a no-grad predecessor chain of K ~ U{1..E3_CHAIN_MAX} random
               atoms (self-predecessors included, no exclusions) produces
               z_ctx; a fresh target atom is tested on stopgrad(z_ctx):
               L_closure = L_valid(Apply_i(sg(z_ctx))). Predecessors cannot
               adapt themselves to help the target.
  uniqueness   fingerprints F_i = softmax(Dec_frozen(Apply_i(z0))) on CLEAN
               standalone states for E3_UNIQUE_ATOMS_PER_STEP sampled atoms,
               plus F_pass = softmax(Dec_frozen(z0)); mean-total-variation
               distance across the batch; hinge at E3_UNIQUE_MARGIN, each
               term weighted by detached usage weights w_i (w_i*w_j for
               pairs) so unused slots may remain unused.

L_valid(z) (registered decision, see registered.py):
  READ  - per-position CE of Dec_frozen(z) against its own hard readout
          (-log p_max: the state must read as a DEFINITE digit list), plus
  CYCLE - relative MSE from z to the frozen encoder's canonical code of that
          readout, stop-grad target (same relative-MSE form as the oracle's
          state supervision): the state must BE the code of what it reads as.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from . import registered as R
from .utils import stream_rng


class SandboxState:
    """Per-run sandbox state: the usage EMA and the dedicated sampling stream.

    Lives outside the model on purpose - nothing here is a parameter, nothing
    is part of the A6 path, and a zero-lambda run never constructs one.
    """

    def __init__(self, cfg, stream: str = "e3_sandbox", stream_index: int = 0):
        self.cfg = cfg
        self.rng = stream_rng(cfg.seed, stream, stream_index)
        self.usage_ema = torch.full((R.N_ATOMS,), R.E3_USAGE_EMA_INIT)

    def update_usage(self, choices: torch.Tensor) -> None:
        """EMA of the REAL hard-routing atom-selection frequency, from the A6
        forward's recorded choices [B, N_STEPS] (-1 dead, 16 pass). Pass picks
        are excluded from the denominator, matching the registered census. A
        batch with no atom picks at all leaves the EMA untouched."""
        c = choices[(choices >= 0) & (choices < R.N_ATOMS)]
        if c.numel() == 0:
            return
        freq = torch.bincount(c, minlength=R.N_ATOMS).to(
            self.usage_ema.dtype) / c.numel()
        self.usage_ema.mul_(R.E3_USAGE_EMA_DECAY).add_(
            (1.0 - R.E3_USAGE_EMA_DECAY) * freq)

    def usage_weights(self) -> torch.Tensor:
        """w_i = clamp(usage_ema_i / CENSUS_EPS, 0, 1). A detached buffer by
        construction: uniqueness pressure follows real routing usage but no
        gradient ever flows back into routing through it."""
        return (self.usage_ema / R.E3_USAGE_WEIGHT_EPS).clamp(0.0, 1.0)

    def draw(self) -> dict:
        """One step's sandbox sample. Consumption is FIXED per call (the chain
        is always drawn at full length; only the first chain_len entries are
        applied) so the stream position is a function of the step count."""
        r = self.rng
        return {
            "standalone_atom": int(r.integers(R.N_ATOMS)),
            "chain_len": int(r.integers(1, R.E3_CHAIN_MAX + 1)),
            "chain_atoms": [int(a) for a in
                            r.integers(0, R.N_ATOMS, size=R.E3_CHAIN_MAX)],
            "closure_atom": int(r.integers(R.N_ATOMS)),
            "unique_atoms": [int(a) for a in
                             r.permutation(R.N_ATOMS)
                             [:R.E3_UNIQUE_ATOMS_PER_STEP]],
        }


@contextmanager
def sandbox_grad_boundary(model):
    """Freeze everything except the atom MLPs while a sandbox graph is built.

    requires_grad is captured at graph-construction time, so flipping it off
    here makes encoder/decoder/composer/keys constants of the sandbox
    subgraph even though the total loss is backpropagated in one pass
    together with the task loss (whose graph was built unfrozen)."""
    frozen = (list(model.encoder.parameters())
              + list(model.decoder.parameters())
              + list(model.composer.parameters())
              + [model.atoms.keys, model.atoms.pass_key])
    prior = [p.requires_grad for p in frozen]
    for p in frozen:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in zip(frozen, prior):
            p.requires_grad_(r)


def apply_chain(model, state: torch.Tensor, atom_ids) -> torch.Tensor:
    """Sequential hard atom applications through model.step_once - THE same
    update normal execution uses (Apply_i(z) = LN(z + A_i(z))). No separate
    sandbox atom implementation exists."""
    for a in atom_ids:
        state = model.step_once(state, int(a))
    return state


def validity_terms(model, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(read, cycle) components of L_valid(z); see the module docstring.
    Call under sandbox_grad_boundary: the decoder/encoder must be frozen."""
    logits = model.decoder(z)                                  # [B,6,10]
    readout = logits.argmax(dim=-1)                            # [B,6] int64
    read = F.cross_entropy(logits.reshape(-1, model.cfg.vocab),
                           readout.reshape(-1))
    with torch.no_grad():
        target = model.code(readout)
    cycle = F.mse_loss(z, target) / target.pow(2).mean().clamp_min(1e-6)
    return read, cycle


def fingerprint(model, z: torch.Tensor) -> torch.Tensor:
    """Behavioral fingerprint: frozen-decoder softmax output, [B,6,10]."""
    return F.softmax(model.decoder(z), dim=-1)


def tv_distance(fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    """Mean total variation across the batch and positions (scalar). Distance
    is always an expectation over a batch of inputs, never one example."""
    return 0.5 * (fa - fb).abs().sum(dim=-1).mean()


def sandbox_losses(model, z0: torch.Tensor, sb: SandboxState,
                   draws: dict | None = None) -> dict:
    """All sandbox loss terms for one step. z0 MUST be detached (asserted).

    Returns tensors; the caller weights loss_sandbox_valid /
    loss_sandbox_unique into the total loss and logs the rest."""
    assert not z0.requires_grad, "sandbox must receive stopgrad(code(x))"
    if draws is None:
        draws = sb.draw()
    w = sb.usage_weights()          # detached buffer

    with sandbox_grad_boundary(model):
        # -- standalone: clean encoder state, no predecessor required -------
        z_alone = model.step_once(z0, draws["standalone_atom"])
        read_s, cycle_s = validity_terms(model, z_alone)
        loss_standalone = (R.E3_VALID_READ_WEIGHT * read_s
                           + R.E3_VALID_CYCLE_WEIGHT * cycle_s)

        # -- closure: arbitrary no-grad predecessors, stop-grad handoff -----
        with torch.no_grad():
            z_ctx = apply_chain(
                model, z0, draws["chain_atoms"][:draws["chain_len"]])
        # detach() is redundant under no_grad but registered explicitly:
        # predecessors can NEVER adapt themselves to help the target atom.
        z_close = model.step_once(z_ctx.detach(), draws["closure_atom"])
        read_c, cycle_c = validity_terms(model, z_close)
        loss_closure = (R.E3_VALID_READ_WEIGHT * read_c
                        + R.E3_VALID_CYCLE_WEIGHT * cycle_c)

        # -- functional uniqueness: clean standalone states + pass ----------
        atoms = draws["unique_atoms"]
        fps = {i: fingerprint(model, model.step_once(z0, i)) for i in atoms}
        fp_pass = fingerprint(model, z0)
        pair_terms, nonid_terms = [], []
        for pos, i in enumerate(atoms):
            for j in atoms[pos + 1:]:
                d = tv_distance(fps[i], fps[j])
                pair_terms.append(w[i] * w[j]
                                  * F.relu(R.E3_UNIQUE_MARGIN - d))
            d_pass = tv_distance(fps[i], fp_pass)
            nonid_terms.append(w[i] * F.relu(R.E3_UNIQUE_MARGIN - d_pass))
        loss_pair = torch.stack(pair_terms).mean()
        loss_nonidentity = torch.stack(nonid_terms).mean()

    return {
        "loss_sandbox_valid": 0.5 * (loss_standalone + loss_closure),
        "loss_sandbox_unique": loss_pair + loss_nonidentity,
        "loss_sandbox_standalone": loss_standalone,
        "loss_sandbox_closure": loss_closure,
        "loss_sandbox_read_standalone": read_s,
        "loss_sandbox_cycle_standalone": cycle_s,
        "loss_sandbox_read_closure": read_c,
        "loss_sandbox_cycle_closure": cycle_c,
        "loss_sandbox_unique_pair": loss_pair,
        "loss_sandbox_nonidentity": loss_nonidentity,
        "sandbox_chain_len": draws["chain_len"],
    }


# ---------------------------------------------------------------------------
# Telemetry (eval cadence). Measurement only: no optimizer step, nothing
# written back to any model, the training sandbox stream is never touched.
# ---------------------------------------------------------------------------

def measure(model, cfg, diag: dict, sb: SandboxState, step: int) -> dict:
    """Full sandbox picture on the fixed diagnostic batch: per-atom standalone
    validity, the complete 16x16 fingerprint-distance matrix (+ vs pass),
    random-chain closure validity, and the usage EMA snapshot."""
    was_training = model.training
    model.eval()
    x = torch.from_numpy(diag["x"])
    w = sb.usage_weights()
    weighted = w > 0.5              # telemetry summarization only, NOT the
                                    # census; the census stays authoritative
    with torch.no_grad():
        z0 = model.code(x)
        fps, reads, cycles = [], [], []
        for i in range(R.N_ATOMS):
            z_i = model.step_once(z0, i)
            r, c = validity_terms(model, z_i)
            reads.append(float(r))
            cycles.append(float(c))
            fps.append(fingerprint(model, z_i))
        fp_pass = fingerprint(model, z0)
        dist = np.zeros((R.N_ATOMS, R.N_ATOMS))
        for i in range(R.N_ATOMS):
            for j in range(i + 1, R.N_ATOMS):
                dist[i, j] = dist[j, i] = float(tv_distance(fps[i], fps[j]))
        pass_dist = [float(tv_distance(fps[i], fp_pass))
                     for i in range(R.N_ATOMS)]

        rng = stream_rng(cfg.seed, "e3_sandbox_eval", 1 + step)
        closure_reads, closure_cycles, chain_lens = [], [], []
        for _ in range(R.E3_TELEMETRY_CHAINS):
            k = int(rng.integers(1, R.E3_CHAIN_MAX + 1))
            chain = [int(a) for a in rng.integers(0, R.N_ATOMS,
                                                  size=R.E3_CHAIN_MAX)][:k]
            target = int(rng.integers(R.N_ATOMS))
            z_c = model.step_once(apply_chain(model, z0, chain), target)
            r, c = validity_terms(model, z_c)
            closure_reads.append(float(r))
            closure_cycles.append(float(c))
            chain_lens.append(k)
    if was_training:
        model.train()

    wi = [i for i in range(R.N_ATOMS) if bool(weighted[i])]
    pair_vals = [dist[i, j] for a, i in enumerate(wi) for j in wi[a + 1:]]
    pass_vals = [pass_dist[i] for i in wi]
    n_pairs = len(pair_vals)
    return {
        "step": step,
        "arm": cfg.arm,
        "lambda_sandbox_valid": cfg.lambda_sandbox_valid,
        "lambda_sandbox_unique": cfg.lambda_sandbox_unique,
        "usage": {
            "ema": [float(v) for v in sb.usage_ema],
            "weights": [float(v) for v in w],
            "n_weighted": int(sum(1 for i in wi)),
            "weighted_atoms": wi,
        },
        "standalone": {
            "read_per_atom": reads,
            "cycle_per_atom": cycles,
            "read_mean": float(np.mean(reads)),
            "cycle_mean": float(np.mean(cycles)),
            "read_mean_weighted": (float(np.mean([reads[i] for i in wi]))
                                   if wi else None),
            "cycle_mean_weighted": (float(np.mean([cycles[i] for i in wi]))
                                    if wi else None),
        },
        "closure": {
            "read_per_chain": closure_reads,
            "cycle_per_chain": closure_cycles,
            "chain_lens": chain_lens,
            "read_mean": float(np.mean(closure_reads)),
            "cycle_mean": float(np.mean(closure_cycles)),
        },
        "uniqueness": {
            "pair_dist_matrix": dist.tolist(),
            "pass_dist_per_atom": pass_dist,
            "pair_dist_mean_weighted": (float(np.mean(pair_vals))
                                        if pair_vals else None),
            "pair_dist_min_weighted": (float(np.min(pair_vals))
                                       if pair_vals else None),
            "pass_dist_min_weighted": (float(np.min(pass_vals))
                                       if pass_vals else None),
            "margin": R.E3_UNIQUE_MARGIN,
            "margin_satisfied_frac_weighted": (
                float(np.mean([v >= R.E3_UNIQUE_MARGIN for v in pair_vals]))
                if pair_vals else None),
            "n_weighted_pairs": n_pairs,
        },
    }
