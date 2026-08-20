"""E5 producer branch (H1-Experiment5.md): training atoms to EMIT states
that arbitrary frozen strangers can use, trained alongside the UNTOUCHED
task path and the inherited E4 pressures.

    z_out = Apply_i(z0)                       # trainable target atom i EMITS
    for k in 1..K (independent branches):
        chain_k = random frozen atoms, length ~ U{1..E5_CHAIN_MAX}
        L_k     = READ(chain_k(z_out))        # frozen-decoder validity
    L_producer = mean_k(L_k)

Non-negotiable invariants, enforced here and unit-tested in tests/test_e5.py:

  1. Gradient path. Unlike the receiver/closure branch (stopgrad BEFORE the
     target), gradient flows THROUGH the frozen downstream activations back
     into the producer atom ONLY. Frozen parameters, live activations - the
     same mechanism as training through the frozen decoder. Mechanically:
     the emission is built under the sandbox gradient boundary (atom MLPs
     trainable, encoder/decoder/composer/keys frozen), then the K chains and
     their READs are built with EVERY parameter frozen, so z_out is the only
     gradient-bearing leaf in the continuation graph. The producer atom gets
     gradient; chain atoms and the decoder get exactly none.
  2. Exact same atom update. Every application goes through model.step_once
     (Apply_i(z) = LN(z + A_i(z))) via sandbox.apply_chain. There is no
     separate producer atom implementation.
  3. No routing involvement. The composer is never called; keys never enter
     any producer computation.
  4. Dedicated randomness. All producer sampling comes from the
     'e5_producer' numpy stream (telemetry/init calibration from the indexed
     'e5_producer_eval' stream). No torch RNG is consumed, and consumption
     is FIXED per step (chains drawn at full length, only the first k
     applied): a producer-bearing run's routing/noise/sandbox/init draws are
     bit-identical to its A14 pair.

Gaming audit (registered): the degenerate exit is one fixed survives-
anything state, and K branches INCREASE the pull toward it. Beyond the task
loss and the inherited uniqueness fingerprints, measure() logs the per-atom
producer-output variance across inputs (a collapsing producer must announce
itself) and the spread of READ across the K branches for the same emitted
state (the registered free metric: remaining context-sensitivity in
production).
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from . import registered as R
from .sandbox import apply_chain, sandbox_grad_boundary, validity_terms
from .utils import stream_rng


class ProducerState:
    """Per-run producer state: nothing but the dedicated sampling stream.

    Lives outside the model on purpose - nothing here is a parameter,
    nothing is part of the task path, and a zero-lambda run never
    constructs one.
    """

    def __init__(self, cfg, stream: str = "e5_producer", stream_index: int = 0):
        self.cfg = cfg
        self.rng = stream_rng(cfg.seed, stream, stream_index)

    def draw(self) -> dict:
        """One step's producer sample. Consumption is FIXED per call (every
        chain is drawn at full length; only the first chain_len entries are
        applied) so the stream position is a function of the step count."""
        r = self.rng
        return {
            "producer_atom": int(r.integers(R.N_ATOMS)),
            "branches": [
                {"chain_len": int(r.integers(1, R.E5_CHAIN_MAX + 1)),
                 "chain_atoms": [int(a) for a in
                                 r.integers(0, R.N_ATOMS,
                                            size=R.E5_CHAIN_MAX)]}
                for _ in range(R.E5_PRODUCER_BRANCHES)],
        }


@contextmanager
def frozen_continuation_boundary(model):
    """Freeze EVERY parameter while the continuation chains + READ are built.

    requires_grad is captured at graph-construction time: with all
    parameters frozen, the chains and the decoder become constants of the
    continuation subgraph - frozen parameters, LIVE activations - and the
    only gradient-bearing leaf reachable from L_producer is the emitted
    state z_out (built earlier with the atom MLPs trainable). Gradient
    therefore flows THROUGH the frozen chains into the producer atom only.
    """
    params = list(model.parameters())
    prior = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in zip(params, prior):
            p.requires_grad_(r)


def producer_losses(model, z0: torch.Tensor, ps: ProducerState,
                    draws: dict | None = None) -> dict:
    """The producer loss for one step. z0 MUST be detached (asserted): the
    branch emits from the SAME stopgrad(code(x)) the sandbox uses.

    Returns tensors; the caller weights loss_producer into the total loss
    and logs the rest."""
    assert not z0.requires_grad, "producer must receive stopgrad(code(x))"
    if draws is None:
        draws = ps.draw()

    # -- emission: atom MLPs trainable, everything else frozen --------------
    with sandbox_grad_boundary(model):
        z_out = model.step_once(z0, draws["producer_atom"])

    # -- K frozen continuations: frozen parameters, live activations --------
    with frozen_continuation_boundary(model):
        reads = []
        for br in draws["branches"]:
            z_end = apply_chain(model, z_out,
                                br["chain_atoms"][:br["chain_len"]])
            read, _cycle = validity_terms(model, z_end)  # cycle: telemetry
            reads.append(read)                           # elsewhere, unused
    reads_t = torch.stack(reads)

    return {
        "loss_producer": reads_t.mean(),
        # Free per-step gauge (detached): spread of READ across the K
        # branches for the same emitted state. The registered per-eval form
        # lives in measure(); this is the zero-cost training-time shadow.
        "producer_read_spread": reads_t.detach().std(correction=0),
        "producer_chain_len_mean": float(np.mean(
            [br["chain_len"] for br in draws["branches"]])),
    }


# ---------------------------------------------------------------------------
# Telemetry (eval cadence). Measurement only: no optimizer step, nothing
# written back to any model, the training producer stream is never touched.
# ---------------------------------------------------------------------------

def measure(model, cfg, diag: dict, step: int) -> dict:
    """The two registered producer telemetry rows on the fixed diagnostic
    batch:

      1. Per-atom producer-output variance across inputs (gaming audit: a
         collapsing producer - one fixed survives-anything state - must
         announce itself). z0's own variance is logged as the reference
         floor for an input-following state.
      2. Spread of READ across the K branches for the same emitted state
         (the registered free metric: remaining context-sensitivity in
         production), over E5_TELEMETRY_PRODUCERS sampled emissions from the
         indexed 'e5_producer_eval' stream.
    """
    was_training = model.training
    model.eval()
    x = torch.from_numpy(diag["x"])
    with torch.no_grad():
        z0 = model.code(x)
        var_per_atom = []
        for i in range(R.N_ATOMS):
            z_i = model.step_once(z0, i)
            var_per_atom.append(float(z_i.var(dim=0, correction=0).mean()))
        z0_var = float(z0.var(dim=0, correction=0).mean())

        rng = stream_rng(cfg.seed, "e5_producer_eval", 1 + step)
        producers, read_means, read_spreads, all_reads = [], [], [], []
        for _ in range(R.E5_TELEMETRY_PRODUCERS):
            i = int(rng.integers(R.N_ATOMS))
            z_out = model.step_once(z0, i)
            reads = []
            for _k in range(R.E5_PRODUCER_BRANCHES):
                klen = int(rng.integers(1, R.E5_CHAIN_MAX + 1))
                chain = [int(a) for a in
                         rng.integers(0, R.N_ATOMS,
                                      size=R.E5_CHAIN_MAX)][:klen]
                read, _cycle = validity_terms(
                    model, apply_chain(model, z_out, chain))
                reads.append(float(read))
            producers.append(i)
            read_means.append(float(np.mean(reads)))
            read_spreads.append(float(np.std(reads)))
            all_reads.append(reads)
    if was_training:
        model.train()

    return {
        "step": step,
        "arm": cfg.arm,
        "lambda_producer": cfg.lambda_producer,
        "output_variance": {
            "per_atom": var_per_atom,
            "min": float(np.min(var_per_atom)),
            "mean": float(np.mean(var_per_atom)),
            "z0_reference": z0_var,
        },
        "branch_read": {
            "producers": producers,
            "reads_per_producer": all_reads,
            "read_mean_per_producer": read_means,
            "spread_per_producer": read_spreads,
            "read_mean": float(np.mean(read_means)),
            "spread_mean": float(np.mean(read_spreads)),
            "n_branches": R.E5_PRODUCER_BRANCHES,
        },
    }
