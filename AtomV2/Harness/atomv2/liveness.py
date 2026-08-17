"""E1b liveness telemetry (H1-E1bExperiment.md, "Liveness telemetry and
validity gates"). Preregistered measurement, never training: gradients are
read with torch.autograd.grad and discarded; no optimizer step is ever taken
and nothing is written back to any model.

Three families, all on ONE fixed diagnostic batch at every scheduled eval:

  1. Base-router geometry (deterministic, no noise): quantiles of the base
     hard-choice probability softmax(z / sigma) - by the Gumbel-max property
     this IS the marginal probability of each route under forward noise
     sigma - plus saturation fractions, top1-top2 logit gap, base-softmax
     Jacobian Frobenius norm, raw (pre-normalization) query/key norms, and
     the max |z|. Implementation invariant: max base probability must not
     exceed the arm's registered p_max + tolerance.
  2. Actual learning signal: task-loss gradient norms into composer params
     and route keys (separately and combined) and into the atoms, plus the
     router/atom ratio, over E1B_DIAG_DRAWS independently seeded Gumbel
     draws. Median and full range are logged - a single draw was observed
     (routing_health) to swing by orders of magnitude.
  3. Routing diversity: stochastic execution diversity across the draws on
     identical inputs, and deterministic content-conditional diversity
     (programs per task) computed from the eval pass's hard choices.

The deafness rule and the p_max invariant are EVALUATED in analyze.py from
these artifacts; this module only measures.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import registered as R
from .utils import stream_rng, stream_seed

_QUANTS = (0.10, 0.50, 0.90, 0.99)


def registered_p_max(cfg) -> float:
    """The arm's p_max, re-derived from (alpha, sigma) rather than transcribed:
    p_max = exp(2a/s) / (exp(2a/s) + 16)."""
    e = math.exp(2.0 * cfg.router_alpha / cfg.router_sigma)
    return e / (e + R.E1B_N_ALTERNATIVES)


def diag_batch(arrays: dict, cfg) -> dict:
    """One fixed diagnostic batch, identical at every eval of a run and across
    arms within a seed (dedicated stream; never consumes the training stream)."""
    idx = stream_rng(cfg.seed, "e1b_diag").choice(
        len(arrays["x"]), size=cfg.batch_size, replace=False)
    idx = np.sort(idx)
    return {k: arrays[k][idx] for k in ("x", "y", "tokens", "n_tokens")}


def _q(t: torch.Tensor) -> dict:
    v = t.detach().float()
    out = {f"p{int(q * 100):02d}": float(torch.quantile(v, q)) for q in _QUANTS}
    out["max"] = float(v.max())
    return out


def _group_norm(grads) -> float:
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum())
    return math.sqrt(total)


def _live_queries(model, out, toks) -> torch.Tensor:
    """Raw (pre-normalization) composer queries, recomputed from the recorded
    per-step states - states[k] is exactly the state routing saw at step k."""
    qs = []
    live = out["live"]
    for k in range(live.shape[1]):
        mask = live[:, k]
        if not bool(mask.any()):
            continue
        q = model.composer.query(out["states"][k][mask],
                                 toks[mask, k // R.MICRO_STEPS],
                                 k % R.MICRO_STEPS)
        qs.append(q)
    return torch.cat(qs, dim=0)


@torch.no_grad()
def _base_geometry(model, cfg, x, toks, ntok) -> tuple[dict, dict]:
    out = model(x, toks, ntok, mode="hard")
    live = out["live"]
    z = out["route_logits"][live]                        # [n_live, 17]
    p = F.softmax(z / cfg.router_sigma, dim=-1)          # marginal hard-choice
    pmax = p.max(dim=-1).values
    top2 = z.topk(2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    jac = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    jac_fro = jac.flatten(1).norm(dim=-1)

    queries = _live_queries(model, out, toks)
    key_norms = model.atoms.all_keys().norm(dim=-1)

    p_pass = p[:, R.PASS_INDEX]
    pass_rank = (z > z[:, R.PASS_INDEX: R.PASS_INDEX + 1]).sum(dim=-1) + 1
    pass_gap = z.max(dim=-1).values - z[:, R.PASS_INDEX]

    geometry = {
        "n_live_decisions": int(live.sum()),
        "base_prob": _q(pmax),
        "frac_above_0999": float((pmax > 0.999).float().mean()),
        "frac_above_0999999": float((pmax > 0.999999).float().mean()),
        "frac_numerically_one": float((pmax == 1.0).float().mean()),
        "top1_top2_logit_gap": _q(gap),
        "softmax_jacobian_fro": _q(jac_fro),
        "max_abs_base_logit": float(z.abs().max()),
        "raw_query_norm": _q(queries.norm(dim=-1)),
        "raw_key_norm": _q(key_norms),
        "p_max_registered": registered_p_max(cfg),
        "max_base_prob_observed": float(pmax.max()),
        "pmax_invariant_ok": bool(
            float(pmax.max()) <= registered_p_max(cfg) + R.E1B_PMAX_TOL),
        "pass": {"prob": _q(p_pass),
                 "rank_median": float(pass_rank.float().median()),
                 "to_winner_logit_gap": _q(pass_gap)},
    }
    det_choices = out["choices"]                         # [B, 6], -1 dead
    return geometry, {"det_choices": det_choices, "live": live}


def _learning_signal(model, cfg, x, y, toks, ntok, step: int) -> tuple[dict, np.ndarray]:
    """Task-loss gradient norms over E1B_DIAG_DRAWS seeded Gumbel draws.
    Uses torch.autograd.grad: nothing is accumulated into .grad."""
    composer_params = list(model.composer.parameters())
    key_params = [model.atoms.keys, model.atoms.pass_key]
    atom_params = [model.atoms.w1, model.atoms.b1,
                   model.atoms.w2, model.atoms.b2]
    per_draw = {"composer": [], "route_keys": [], "router_total": [],
                "atoms": [], "router_atom_ratio": []}
    draw_choices = []
    for d in range(R.E1B_DIAG_DRAWS):
        gen = torch.Generator().manual_seed(int(
            stream_seed(cfg.seed, "e1b_liveness",
                        step * R.E1B_DIAG_DRAWS + d).generate_state(1)[0]))
        with torch.enable_grad():
            out = model(x, toks, ntok, mode="gumbel", tau=cfg.tau_end,
                        generator=gen)
            loss = F.cross_entropy(out["logits"].reshape(-1, cfg.vocab),
                                   y.reshape(-1))
            grads = torch.autograd.grad(
                loss, composer_params + key_params + atom_params,
                allow_unused=True)
        n_c = len(composer_params)
        n_k = len(key_params)
        g_comp = _group_norm(grads[:n_c])
        g_keys = _group_norm(grads[n_c:n_c + n_k])
        g_router = math.sqrt(g_comp ** 2 + g_keys ** 2)
        g_atoms = _group_norm(grads[n_c + n_k:])
        per_draw["composer"].append(g_comp)
        per_draw["route_keys"].append(g_keys)
        per_draw["router_total"].append(g_router)
        per_draw["atoms"].append(g_atoms)
        per_draw["router_atom_ratio"].append(
            g_router / g_atoms if g_atoms > 0 else None)
        draw_choices.append(out["choices"].detach().numpy())
    signal = {}
    for name, vals in per_draw.items():
        known = [v for v in vals if v is not None]
        signal[name] = {
            "per_draw": vals,
            "median": float(np.median(known)) if known else None,
            "min": float(np.min(known)) if known else None,
            "max": float(np.max(known)) if known else None,
            "n_undefined": len(vals) - len(known),
        }
    return signal, np.stack(draw_choices)                # [draws, B, 6]


def _stochastic_diversity(draw_choices: np.ndarray, det_choices: np.ndarray,
                          live: np.ndarray) -> dict:
    """Across the seeded draws, on identical inputs: unique hard route
    sequences per input, and disagreement with the deterministic choice."""
    n_draws, b, _ = draw_choices.shape
    uniq = [len({tuple(draw_choices[d, i].tolist()) for d in range(n_draws)})
            for i in range(b)]
    live_b = np.asarray(live, dtype=bool)
    dis = (draw_choices != det_choices[None]) & live_b[None]
    return {
        "n_draws": int(n_draws),
        "unique_sequences_per_input_mean": float(np.mean(uniq)),
        "unique_sequences_per_input_max": int(np.max(uniq)),
        "route_disagreement_rate": float(dis.sum() / (live_b.sum() * n_draws)),
    }


def deterministic_programs(choices_by_task: dict) -> dict:
    """Deterministic content-conditional diversity from the eval pass's hard
    (no-noise) choices: unique route sequences per task over its fixed eval
    examples, plus per-task routing entropy over those sequences (nats)."""
    per_task = {}
    for tid, ch in sorted(choices_by_task.items()):
        rows = [tuple(r.tolist()) for r in np.asarray(ch)]
        counts = np.array(list(
            {r: rows.count(r) for r in set(rows)}.values()), dtype=float)
        freq = counts / counts.sum()
        per_task[tid] = {
            "n_examples": len(rows),
            "n_programs": int(len(counts)),
            "routing_entropy_nats": float(-(freq * np.log(freq)).sum()),
        }
    n_programs = [v["n_programs"] for v in per_task.values()]
    return {
        "per_task": per_task,
        "programs_per_task_mean": float(np.mean(n_programs)),
        "programs_per_task_max": int(np.max(n_programs)),
        "routing_entropy_nats_mean": float(np.mean(
            [v["routing_entropy_nats"] for v in per_task.values()])),
    }


def measure(model, cfg, diag: dict, step: int) -> dict:
    """The full per-eval liveness record. Read-only; restores model mode."""
    was_training = model.training
    model.eval()
    x = torch.from_numpy(diag["x"])
    y = torch.from_numpy(diag["y"])
    toks = torch.from_numpy(diag["tokens"])
    ntok = torch.from_numpy(diag["n_tokens"])

    geometry, det = _base_geometry(model, cfg, x, toks, ntok)
    signal, draw_choices = _learning_signal(model, cfg, x, y, toks, ntok, step)
    diversity = _stochastic_diversity(
        draw_choices, det["det_choices"].numpy(), det["live"].numpy())

    if was_training:
        model.train()
    return {
        "step": step,
        "arm": cfg.arm,
        "router": {"alpha": cfg.router_alpha, "sigma": cfg.router_sigma,
                   "tau_backward": cfg.router_tau_backward,
                   "norm_eps": cfg.router_norm_eps},
        "base_geometry": geometry,
        "learning_signal": signal,
        "stochastic_diversity": diversity,
    }
