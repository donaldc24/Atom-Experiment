"""E7 audit (H1-Experiment7.md): the registered per-run diagnostics, all
read-only on the final checkpoint. Never used during training.

  1. D1-style boundary panel (two-token cells): boundary answer
     decodability, distance to canonical, raw continuation accuracy, and
     the self-bottleneck continuation (decode -> re-encode -> continue).
  2. D2-style atom audit: ground-truth-referenced signatures (D2-A1
     lesson), discovered op -> atom mapping, then for every ordered pair
     of discovered competent atoms
        raw composition:            code(x) -> Ai -> Aj -> decode
        canonicalized composition:  code(x) -> Ai -> decode -> encode
                                             -> Aj -> decode
     and the primary E7 metric over qualified pairs (canonicalized
     accuracy >= E7_CANON_CHAIN_MIN):
        Interface Closure Ratio = raw acc / canonicalized acc.
     The ratio never conceals absolutes: both raw numbers are reported.
  3. D3-style state content: linear (ridge) recoverability of the
     transformed answer and the original input from boundary states.

Run:  python -m atomv2.e7_audit --run-dir <dir>     (also called by run_e7)
"""
from __future__ import annotations

import numpy as np
import torch

from . import data as data_mod
from . import ops
from . import registered as R
from .d2_factorization import CANDIDATES, _match
from .d3_position_probe import _ridge_probe_acc
from .panel import load_checkpoint
from .utils import read_json, write_json


@torch.no_grad()
def _forward(model, task, x: np.ndarray) -> dict:
    n = len(x)
    toks = torch.from_numpy(np.tile(task.tokens, (n, 1)))
    ntok = torch.full((n,), task.n_tokens, dtype=torch.int64)
    out = model(torch.from_numpy(x), toks, ntok, mode="hard")
    return out | {"_toks": toks, "_ntok": ntok}


@torch.no_grad()
def boundary_panel(model, bundle) -> dict:
    ms = model.cfg.micro_steps
    cells = {}
    groups = ([("trained", td) for td in bundle.seen_heldout
               if td.task.n_tokens == 2]
              + [(lvl, td) for lvl in ("L1", "L2", "L3")
                 for td in bundle.unseen[lvl]])
    for group, td in groups:
        pa = td.task.surface_ops[0]
        x, y = td.x, td.y
        out = _forward(model, td.task, x)
        boundary = out["states"][ms]
        truth_mid = ops.SURFACE_FNS[pa](x)
        readout = model.decoder(boundary).argmax(-1)
        canon_mid = model.code(torch.from_numpy(truth_mid))
        rel = (torch.linalg.norm(boundary - canon_mid, dim=-1)
               / torch.linalg.norm(canon_mid, dim=-1).clamp_min(1e-6))
        # self-bottleneck: model's OWN readout, re-encoded, continue tok 2
        z_bn = model.code(readout)
        bn = model.execute_from_state(z_bn, out["_toks"], out["_ntok"], 1,
                                      mode="hard")
        cells[td.task.task_id] = {
            "group": group,
            "boundary_decode_acc": _match(readout.numpy(), truth_mid),
            "boundary_rel_dist_to_canonical": float(rel.mean()),
            "raw_acc": _match(out["logits"].argmax(-1).numpy(), y),
            "self_bottleneck_acc": _match(bn["logits"].argmax(-1).numpy(), y),
        }
    by_group: dict[str, dict] = {}
    for c in cells.values():
        g = by_group.setdefault(c["group"], {k: [] for k in
                                             ("boundary_decode_acc",
                                              "raw_acc",
                                              "self_bottleneck_acc",
                                              "boundary_rel_dist_to_canonical")})
        for k in g:
            g[k].append(c[k])
    return {"cells": cells,
            "by_group": {g: {k: float(np.mean(v)) for k, v in d.items()}
                         for g, d in sorted(by_group.items())}}


@torch.no_grad()
def atom_audit(model, bundle) -> dict:
    singles = [td for td in bundle.seen_heldout if td.task.n_tokens == 1]
    x = np.concatenate([td.x[:400 // len(singles)] for td in singles])
    z0 = model.code(torch.from_numpy(x))

    sig = {}
    for i in range(R.N_ATOMS):
        di = model.decoder(model.step_once(z0, i)).argmax(-1).numpy()
        rates = {name: _match(di, fn(x)) for name, fn in CANDIDATES.items()}
        best = max(rates, key=rates.get)
        sig[i] = {"best": best, "best_rate": rates[best],
                  "rates_top3": dict(sorted(rates.items(),
                                            key=lambda kv: -kv[1])[:3])}
    mapping = {}
    for i, s in sig.items():
        if s["best"] == "identity" or s["best_rate"] < R.E7_SIG_THRESHOLD:
            continue
        op = s["best"]
        if op not in mapping or sig[mapping[op]]["best_rate"] < s["best_rate"]:
            mapping[op] = i

    pairs = {}
    for op_a, ia in sorted(mapping.items()):
        fa = CANDIDATES[op_a]
        for op_b, ib in sorted(mapping.items()):
            fb = CANDIDATES[op_b]
            target = fb(fa(x))
            z_raw = model.step_once(model.step_once(z0, ia), ib)
            raw = _match(model.decoder(z_raw).argmax(-1).numpy(), target)
            mid = model.decoder(model.step_once(z0, ia)).argmax(-1)
            z_can = model.step_once(model.code(mid), ib)
            canon = _match(model.decoder(z_can).argmax(-1).numpy(), target)
            pairs[f"{op_a}->{op_b}"] = {
                "atoms": [int(ia), int(ib)], "raw_acc": raw,
                "canon_acc": canon,
                "icr": raw / canon if canon > 0 else None,
                "qualified": canon >= R.E7_CANON_CHAIN_MIN,
            }
    qual = [p for p in pairs.values() if p["qualified"]]
    raw_m = float(np.mean([p["raw_acc"] for p in qual])) if qual else None
    can_m = float(np.mean([p["canon_acc"] for p in qual])) if qual else None
    return {
        "signatures": {str(i): s for i, s in sig.items()},
        "mapping": {k: int(v) for k, v in mapping.items()},
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_qualified_pairs": len(qual),
        "raw_chain_acc_qualified": raw_m,
        "canon_chain_acc_qualified": can_m,
        "interface_closure_ratio": (raw_m / can_m
                                    if raw_m is not None and can_m else None),
    }


@torch.no_grad()
def state_content(model, bundle) -> dict:
    """Ridge recoverability of answer vs original input from boundary
    states of trained pairs (whole-state probes; works for positional and
    flat states alike)."""
    ms = model.cfg.micro_steps
    hs, xs, ans = [], [], []
    for td in bundle.seen_heldout:
        if td.task.n_tokens != 2:
            continue
        x = td.x[:64]
        out = _forward(model, td.task, x)
        hs.append(out["states"][ms].numpy().astype(np.float64))
        xs.append(x)
        ans.append(ops.SURFACE_FNS[td.task.surface_ops[0]](x))
    H = np.concatenate(hs)
    X, A = np.concatenate(xs), np.concatenate(ans)
    answer = [_ridge_probe_acc(H, A[:, j]) for j in range(R.SEQ_LEN)]
    orig = [_ridge_probe_acc(H, X[:, j]) for j in range(R.SEQ_LEN)]
    return {"answer_recoverability": float(np.mean(answer)),
            "orig_recoverability": float(np.mean(orig)),
            "answer_per_pos": answer, "orig_per_pos": orig,
            "n_states": len(H)}


@torch.no_grad()
def run_audit(run_dir) -> dict:
    model, cfg, step = load_checkpoint(run_dir, "final.pt")
    model.eval()
    bundle = data_mod.build_bundle(cfg)

    bp = boundary_panel(model, bundle)
    aa = atom_audit(model, bundle)
    sc = state_content(model, bundle)

    try:
        m = read_json(run_dir / "metrics.json")
        seen = m.get("acc_seen_hard")
    except FileNotFoundError:
        seen = None
    icr = aa["interface_closure_ratio"]
    audit = {
        "arm": cfg.arm, "seed": cfg.seed, "checkpoint_step": step,
        "healthy": (seen is not None and seen >= R.E7_HEALTH_SEEN_MIN),
        "acc_seen_hard": seen,
        "boundary": bp, "atoms": aa, "state_content": sc,
        "verdict": {
            "n_mapped_atoms": len(aa["mapping"]),
            "n_qualified_pairs": aa["n_qualified_pairs"],
            "raw_chain_acc": aa["raw_chain_acc_qualified"],
            "canon_chain_acc": aa["canon_chain_acc_qualified"],
            "interface_closure_ratio": icr,
            "closure_band": (None if icr is None else
                             "strong" if icr >= R.E7_ICR_STRONG else
                             "severe" if icr < R.E7_ICR_SEVERE else
                             "partial"),
        },
    }
    write_json(run_dir / "e7_audit.json", audit)
    return audit


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(description="E7 read-only audit")
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    res = run_audit(Path(a.run_dir))
    print(json.dumps(res["verdict"], indent=2))
