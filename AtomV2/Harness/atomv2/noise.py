"""E2 interface-noise telemetry + the registered robustness sweep
(H1-Experiment2.md). Measurement only: no optimizer step, nothing written
back to any model; the training noise stream is never touched (telemetry
draws come from the dedicated indexed 'e2_noise_eval' stream).

Clean producer metrics are kept separate from transmitted-state metrics
throughout - otherwise an arm could appear to produce worse states merely
because the diagnostic measured the injected noise instead of the atom's
output.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import data as data_mod
from . import registered as R
from .evaluate import _closed_map_task
from .utils import stream_seed

_QUANTS = (0.10, 0.50, 0.90)


def target_cosine(sigma: float) -> float:
    """Nominal expected clean/noisy-renormalized cosine (registration receipt):
    1 / sqrt(1 + sigma^2)."""
    return 1.0 / math.sqrt(1.0 + sigma * sigma)


def sigma_for_cosine(c: float) -> float:
    return 0.0 if c >= 1.0 else math.sqrt(1.0 / (c * c) - 1.0)


def _gen(master_seed: int, index: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(
        stream_seed(master_seed, "e2_noise_eval", index).generate_state(1)[0]))


def _q(t: torch.Tensor) -> dict:
    v = t.detach().float()
    out = {f"p{int(q * 100):02d}": float(torch.quantile(v, q)) for q in _QUANTS}
    out["min"] = float(v.min())
    out["max"] = float(v.max())
    return out


def _clean_stack(out) -> np.ndarray:
    return np.stack([s.detach().numpy() for s in out["states"]], axis=1)


def _transmitted_stack(out) -> np.ndarray:
    """[n, 7, 384] trajectory of what consumers actually received: col 0 the
    encoder state, cols 1..5 the transmitted handoffs (equal to clean where no
    noise applied), col 6 the final clean state that reaches the decoder."""
    states = [s.detach() for s in out["states"]]
    cols = [states[0]] + [t.detach() for t in out["states_transmitted"]] \
        + [states[-1]]
    return np.stack([c.numpy() for c in cols], axis=1)


def _pairwise_pred_disagreement(preds: list[np.ndarray]) -> float:
    n = len(preds)
    if n < 2:
        return 0.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(float((preds[i] != preds[j]).any(axis=1).mean()))
    return float(np.mean(vals))


def measure(model, cfg, diag: dict, seen_tds, step: int) -> dict:
    """Per-eval noise telemetry on the fixed diagnostic batch (+ the noisy
    producer/transmitted closed-map curves on subsampled seen tasks)."""
    was_training = model.training
    model.eval()
    sigma = cfg.state_noise_sigma
    x = torch.from_numpy(diag["x"])
    y = torch.from_numpy(diag["y"])
    toks = torch.from_numpy(diag["tokens"])
    ntok = torch.from_numpy(diag["n_tokens"])

    with torch.no_grad():
        clean = model(x, toks, ntok, mode="hard")
    live = clean["live"]

    cos_all, cos_by_handoff = [], {}
    rel_all = []
    flip_rates, preds = [], []
    pos_mean_abs, pos_var = [], []
    for d in range(R.E2_NOISE_DRAWS):
        with torch.no_grad():
            noisy = model(x, toks, ntok, mode="hard", noise_sigma=sigma,
                          noise_generator=_gen(cfg.seed,
                                               step * R.E2_NOISE_DRAWS + d))
        for k, t in enumerate(noisy["states_transmitted"]):
            mask = live[:, k + 1] if k + 1 < live.shape[1] else None
            if mask is None or not bool(mask.any()):
                continue
            prod = noisy["states"][k + 1][mask]      # clean producer output
            recv = t[mask]                           # what step k+1 received
            cos = F.cosine_similarity(prod, recv, dim=-1)
            rel = ((recv - prod).norm(dim=-1)
                   / prod.norm(dim=-1).clamp_min(1e-6))
            cos_all.append(cos)
            rel_all.append(rel)
            cos_by_handoff.setdefault(k, []).append(cos)
            h = recv.view(-1, cfg.seq_len, cfg.d_model)
            pos_mean_abs.append(h.mean(dim=-1).abs().mean())
            pos_var.append(h.var(dim=-1, unbiased=False).mean())
        flips = ((noisy["choices"] != clean["choices"]) & live)
        flip_rates.append(float(flips.sum()) / float(live.sum()))
        preds.append(noisy["logits"].argmax(-1).numpy())

    clean_preds = clean["logits"].argmax(-1).numpy()
    vs_clean = [float((p != clean_preds).any(axis=1).mean()) for p in preds]

    # Noisy producer vs transmitted closed-map, one registered draw, on
    # subsampled seen tasks. Producer error reads the atoms' clean outputs
    # under noisy inputs; transmitted error additionally contains the
    # injected perturbation. Report both, never conflated.
    cm_gen_index = step * R.E2_NOISE_DRAWS       # draw 0's index
    prod_errs, trans_errs, prod_tgt, trans_tgt = [], [], [], []
    for td in seen_tds:
        m = min(R.E2_CM_EXAMPLES, len(td.x))
        sub = data_mod.TaskData(td.task, td.x[:m], td.y[:m])
        toks_t = torch.from_numpy(np.tile(sub.task.tokens, (m, 1)))
        ntok_t = torch.full((m,), sub.task.n_tokens, dtype=torch.int64)
        with torch.no_grad():
            out = model(torch.from_numpy(sub.x), toks_t, ntok_t, mode="hard",
                        noise_sigma=sigma,
                        noise_generator=_gen(cfg.seed, cm_gen_index))
        cm_p = _closed_map_task(model, sub, _clean_stack(out))
        cm_t = _closed_map_task(model, sub, _transmitted_stack(out))
        prod_errs.append(cm_p["error"])
        trans_errs.append(cm_t["error"])
        prod_tgt.append(cm_p["final_dist_to_target"])
        trans_tgt.append(cm_t["final_dist_to_target"])

    # Gradient norms into the state producer under interface noise.
    encoder_params = list(model.encoder.parameters())
    atom_params = [model.atoms.w1, model.atoms.b1,
                   model.atoms.w2, model.atoms.b2]
    composer_params = list(model.composer.parameters())
    grad = {"encoder": [], "atoms": [], "composer": []}
    for d in range(R.E2_GRAD_DRAWS):
        route_gen = _gen(cfg.seed, 10_000_000 + step * R.E2_GRAD_DRAWS + d)
        ngen = _gen(cfg.seed, 20_000_000 + step * R.E2_GRAD_DRAWS + d)
        with torch.enable_grad():
            out = model(x, toks, ntok, mode="gumbel", tau=cfg.tau_end,
                        generator=route_gen, noise_sigma=sigma,
                        noise_generator=ngen)
            loss = F.cross_entropy(out["logits"].reshape(-1, cfg.vocab),
                                   y.reshape(-1))
            grads = torch.autograd.grad(
                loss, encoder_params + atom_params + composer_params,
                allow_unused=True)
        n_e, n_a = len(encoder_params), len(atom_params)

        def _norm(gs):
            return math.sqrt(sum(float(g.pow(2).sum())
                                 for g in gs if g is not None))
        grad["encoder"].append(_norm(grads[:n_e]))
        grad["atoms"].append(_norm(grads[n_e:n_e + n_a]))
        grad["composer"].append(_norm(grads[n_e + n_a:]))

    if was_training:
        model.train()
    cos_cat = torch.cat(cos_all)
    return {
        "step": step,
        "arm": cfg.arm,
        "state_noise_sigma": sigma,
        "target_cosine": target_cosine(sigma),
        "n_draws": R.E2_NOISE_DRAWS,
        "handoff": {
            "cosine": _q(cos_cat),
            "cosine_by_handoff": {str(k): _q(torch.cat(v))
                                  for k, v in sorted(cos_by_handoff.items())},
            "rel_l2_postnorm": _q(torch.cat(rel_all)),
            "cosine_within_tol": bool(
                abs(float(cos_cat.median()) - target_cosine(sigma))
                <= R.E2_COSINE_TOL),
            "transmitted_pos_mean_abs": float(np.mean(
                [float(v) for v in pos_mean_abs])),
            "transmitted_pos_var": float(np.mean(
                [float(v) for v in pos_var])),
        },
        "route_flip_rate": float(np.mean(flip_rates)),
        "pred_disagreement_mean": _pairwise_pred_disagreement(preds),
        "pred_disagreement_vs_clean": float(np.mean(vs_clean)),
        "closed_map_noisy_forward": {
            "producer_error_mean": float(np.mean(prod_errs)),
            "transmitted_error_mean": float(np.mean(trans_errs)),
            "producer_target_dist_mean": float(np.mean(prod_tgt)),
            "transmitted_target_dist_mean": float(np.mean(trans_tgt)),
            "n_examples_per_task": R.E2_CM_EXAMPLES,
            "note": "producer = clean atom outputs under noisy inputs; "
                    "transmitted additionally contains the injected noise",
        },
        "producer_grads_under_noise": {
            k: {"per_draw": v, "median": float(np.median(v))}
            for k, v in grad.items()},
    }


def robustness_sweep(run_dir) -> dict:
    """Registered evaluation-only sweep on the FINAL checkpoint: for every
    target cosine, 8 fixed noise draws on the fixed eval examples; report mean
    accuracy, route-flip rate vs the clean forward, and prediction
    disagreement across draws. Clean hard accuracy at cosine 1.000 remains the
    only headline task result."""
    from .panel import load_checkpoint  # late import: panel imports us not
    model, cfg, _step = load_checkpoint(run_dir, "final.pt")
    bundle = data_mod.build_bundle(cfg)
    eval_sets = [("seen_heldout", bundle.seen_heldout)]
    for level in ("L1", "L2", "L3"):
        eval_sets.append((f"unseen_{level}", bundle.unseen[level]))

    result = {"arm": cfg.arm, "seed": cfg.seed,
              "target_cosines": list(R.E2_ROBUST_COSINES),
              "n_draws": R.E2_NOISE_DRAWS, "sets": {}}
    for set_name, tds in eval_sets:
        per_cos = {}
        for c in R.E2_ROBUST_COSINES:
            sigma = sigma_for_cosine(c)
            accs, flips, all_preds = [], [], []
            for td in tds:
                n = len(td.x)
                toks = torch.from_numpy(np.tile(td.task.tokens, (n, 1)))
                ntok = torch.full((n,), td.task.n_tokens, dtype=torch.int64)
                x = torch.from_numpy(td.x)
                with torch.no_grad():
                    clean = model(x, toks, ntok, mode="hard")
                live = clean["live"]
                if sigma == 0.0:
                    preds = clean["logits"].argmax(-1).numpy()
                    accs.append(float((preds == td.y).all(1).mean()))
                    flips.append(0.0)
                    all_preds.append([preds])
                    continue
                task_preds = []
                for d in range(R.E2_NOISE_DRAWS):
                    gen = _gen(cfg.seed,
                               30_000_000 + int(round(c * 1000)) * 100 + d)
                    with torch.no_grad():
                        noisy = model(x, toks, ntok, mode="hard",
                                      noise_sigma=sigma, noise_generator=gen)
                    preds = noisy["logits"].argmax(-1).numpy()
                    task_preds.append(preds)
                    accs.append(float((preds == td.y).all(1).mean()))
                    flips.append(float(((noisy["choices"] != clean["choices"])
                                        & live).sum()) / float(live.sum()))
                all_preds.append(task_preds)
            per_cos[f"{c:.3f}"] = {
                "sigma": sigma,
                "mean_acc": float(np.mean(accs)),
                "route_flip_rate": float(np.mean(flips)),
                "pred_disagreement": float(np.mean(
                    [_pairwise_pred_disagreement(tp) for tp in all_preds])),
            }
        result["sets"][set_name] = per_cos
    return result
