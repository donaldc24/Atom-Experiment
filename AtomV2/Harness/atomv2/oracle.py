"""QUARANTINED oracle machinery for Experiment 0 calibration ONLY.

This module is imported exclusively when cfg.forced_routing is True, which
config_for_arm sets only for arm 'A0-oracle' in experiment 'e0'. Nothing in the
Experiment 1 training path touches anything defined here; the E1 loss is task
cross-entropy + rent, full stop (round-trip/manifold/intermediate supervision
were gamed twice in V1 and are permanently retired FROM THE FREE ARMS - the
oracle uses intermediate supervision precisely because it is scaffolding whose
job is to demonstrate a reachable ceiling, then get locked back up).

Oracle mechanics (registered):
  - Forced routing [atom, pass, pass] per task token, atom index = surface-op
    index (P1 -> atom 0 ... P8 -> atom 7). Atoms 8..15 stay idle - the
    overcomplete budget is preserved, the oracle just never calls on it.
  - Ground-truth intermediate state supervision: after each token's micro-step
    block, the state must match the CURRENT digit-only encoder's canonical
    embedding of the partial composition, stop-grad on the target (shapes the
    atoms, not the encoder). Relative MSE, weight 40.0
    (V1 D18 lineage).
  - Intermediate decode CE (pairs only): the decoder must read the mid-task
    state as the partial result. Weight 1.0.
  - Routing CE: the composer is trained to imitate the forced schedule, weight
    1.0, so A0-oracle's composer is meaningful even though routing is
    overridden during its forward passes.

  E0 also uses the oracle's ground truth to audit the instruments. The 2%
  headline census mechanically reads 7 because singleton-only P3 has share
  1/76; the audit separately checks that all 8 forced atoms are observed and
  that learned hard usage matches the tasks containing P(i+1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import ops
from . import registered as R
from .model import N_STEPS


def forced_schedule(tokens: np.ndarray, n_tokens: np.ndarray) -> np.ndarray:
    """[B, N_STEPS] routing indices: token t -> [surface_idx, PASS, PASS].

    Dead steps (beyond a singleton's budget) are filled with PASS; the model
    masks them anyway.
    """
    b = tokens.shape[0]
    sched = np.full((b, N_STEPS), R.PASS_INDEX, dtype=np.int64)
    for t in range(tokens.shape[1]):
        live = n_tokens > t
        sched[live, t * R.MICRO_STEPS] = tokens[live, t]
    return sched


def partial_targets(task_id: str, x: np.ndarray) -> list[np.ndarray]:
    """Ground-truth digit lists after each surface token: [after_tok1(, after_tok2)]."""
    outs = []
    cur = x
    for p in ops.task_surface_ops(task_id):
        cur = ops.SURFACE_FNS[p](cur)
        outs.append(cur)
    return outs


def oracle_losses(model, out, digits, tokens, n_tokens, y_partial, cfg):
    """The three oracle loss terms. y_partial: [B, MAX_TOKENS, 6] int64 with
    the partial-composition digits (token 2 slot ignored for singletons)."""
    losses = {}

    # (1) state supervision at token-block boundaries, relative MSE, stop-grad
    # target computed by the CURRENT task-independent encoder.
    total_state = out["states"][0].new_zeros(())
    n_terms = 0
    for t in range(y_partial.shape[1]):
        live = n_tokens > t
        if not bool(live.any()):
            continue
        boundary_state = out["states"][(t + 1) * R.MICRO_STEPS]     # [B,384]
        with torch.no_grad():
            target = model.code(y_partial[:, t])
        rel = F.mse_loss(boundary_state[live], target[live]) / (
            target[live].pow(2).mean().clamp_min(1e-6))
        total_state = total_state + rel
        n_terms += 1
    losses["loss_state_rel"] = total_state / max(n_terms, 1)

    # (2) intermediate decode CE (pairs only: the state after token 1 must
    # decode to the partial result).
    pairs = n_tokens > 1
    if bool(pairs.any()):
        mid_state = out["states"][R.MICRO_STEPS]
        logits = model.decoder(mid_state[pairs])
        losses["loss_intermediate_ce"] = F.cross_entropy(
            logits.reshape(-1, cfg.vocab), y_partial[pairs, 0].reshape(-1))
    else:
        losses["loss_intermediate_ce"] = total_state.new_zeros(())

    # (3) routing CE: composer imitates the forced schedule on live steps.
    sched = torch.from_numpy(
        forced_schedule(tokens.numpy(), n_tokens.numpy()))
    live = out["live"]
    logits = out["route_logits"][live]
    losses["loss_route_ce"] = F.cross_entropy(logits, sched[live])

    losses["oracle_total"] = (
        cfg.oracle_state_sup_weight * losses["loss_state_rel"]
        + cfg.oracle_intermediate_ce_weight * losses["loss_intermediate_ce"]
        + cfg.oracle_routing_ce_weight * losses["loss_route_ce"])
    return losses
