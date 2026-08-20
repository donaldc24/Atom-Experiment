"""D2 - Atom Factorization Audit (read-only). Registered in DECISIONS.md
(2026-08-20) BEFORE any probe ran.

The question is causal atom factorization, not task accuracy:

> An atom is factorized if it implements the same primitive transformation
> across inputs and contexts, and those atoms can be recombined to execute
> novel programs without relying on surface-specific co-adaptation.

Semantics are readout-channel, dialect-independent: atom i implements op f
on state h iff decode(A_i(h)) == f(decode(h)) with the model's OWN frozen
decoder. Five probes (P-1 signature, P-2 context invariance, P-3
composer-free recomposition - the guillotine, P-4 swaps, P-5 selective
ablation), all @no_grad on final checkpoints. No training, no new losses;
ground-truth generators are legal here (quarantine covers training only).

Run:  python -m atomv2.d2_factorization      -> results/d2/
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

from . import data as data_mod
from . import ops
from . import registered as R
from .panel import load_checkpoint
from .utils import RESULTS_DIR, RUNS_DIR, write_json

# Registered thresholds (DECISIONS.md D2 registration).
D2_SHARP = 0.95            # P-1: sharp operator identity
D2_MAP_THRESHOLD = 0.90    # P-3: minimum rate for an op->atom mapping
D2_INVARIANT_MIN = 0.80    # P-2: every context group at or above this
D2_RECOMB_WORKS = 0.50     # P-3: novel recomposition "works" at a level
D2_SELECTIVE_RATIO = 3.0   # P-5: in/out mean-damage ratio

N_SIG = 400                # canonical inputs for P-1
N_CTX = 64                 # examples per task per context group (P-2)
N_FORCED = 400             # examples per cell for P-3

CANDIDATES = {}
for _n, _f in ops.SUBOPS.items():
    CANDIDATES[f"sub:{_n}"] = _f
for _p, _f in ops.SURFACE_FNS.items():
    CANDIDATES[f"surf:{_p}"] = _f
CANDIDATES["identity"] = lambda x: x.copy()


def _readout(model, z: torch.Tensor) -> np.ndarray:
    return model.decoder(z).argmax(-1).numpy()


def _match(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).all(axis=1).mean())


# ---------------------------------------------------------------------------
# P-1 operator signature
# ---------------------------------------------------------------------------

@torch.no_grad()
def signature(model, x: np.ndarray) -> dict:
    """P-1. Amendment D2-A1: sharpness is judged GROUND-TRUTH-referenced
    (decode(A_i(z0)) == f(x)) because decode(code(x)) != x on ~14% of
    inputs (the decoder never trains on raw z0) - the readout-referenced
    rate is confounded by round-trip infidelity and is reported beside."""
    z0 = model.code(torch.from_numpy(x))
    base = _readout(model, z0)
    roundtrip = _match(base, x)
    out = {}
    for i in range(R.N_ATOMS):
        di = _readout(model, model.step_once(z0, i))
        rates = {name: _match(di, fn(x))
                 for name, fn in CANDIDATES.items()}
        rates_readout = {name: _match(di, fn(base))
                         for name, fn in CANDIDATES.items()}
        best = max(rates, key=rates.get)
        out[i] = {"best": best, "best_rate": rates[best],
                  "sharp": rates[best] >= D2_SHARP,
                  "best_rate_readout": rates_readout[best],
                  "identity_rate": rates["identity"],
                  "roundtrip_decode_acc": roundtrip,
                  "rates": rates, "rates_readout": rates_readout}
    return out


# ---------------------------------------------------------------------------
# P-2 context invariance
# ---------------------------------------------------------------------------

@torch.no_grad()
def _states_for(model, task, x: np.ndarray) -> list:
    n = len(x)
    toks = torch.from_numpy(np.tile(task.tokens, (n, 1)))
    ntok = torch.full((n,), task.n_tokens, dtype=torch.int64)
    return model(torch.from_numpy(x), toks, ntok, mode="hard")["states"]


@torch.no_grad()
def context_invariance(model, bundle, sig: dict) -> dict:
    """P-2. Amendment D2-A1: the invariance verdict is judged on the
    truth-referenced groups (canonical + pair boundaries, where the true
    content is x resp. Pa(x)). Mid-token states carry no ground-truth
    content (programs stage freely), so their readout-referenced rates are
    descriptive only."""
    sharp_atoms = [i for i, s in sig.items() if s["sharp"]]
    # groups: name -> list of (states h, truth content or None)
    groups: dict[str, list] = {"mid_token_1": [], "mid_token_2": [],
                               "boundary_trained": [], "boundary_L1": [],
                               "boundary_L2": [], "boundary_L3": []}
    for td in bundle.seen_heldout:
        st = _states_for(model, td.task, td.x[:N_CTX])
        if td.task.n_tokens == 1:
            groups["mid_token_1"].append((st[1], None))
            groups["mid_token_2"].append((st[2], None))
        else:
            truth = ops.SURFACE_FNS[td.task.surface_ops[0]](td.x[:N_CTX])
            groups["boundary_trained"].append((st[R.MICRO_STEPS], truth))
    for level in ("L1", "L2", "L3"):
        for td in bundle.unseen[level]:
            st = _states_for(model, td.task, td.x[:N_CTX])
            truth = ops.SURFACE_FNS[td.task.surface_ops[0]](td.x[:N_CTX])
            groups[f"boundary_{level}"].append((st[R.MICRO_STEPS], truth))

    out = {}
    for i in sharp_atoms:
        fn = CANDIDATES[sig[i]["best"]]
        per_group = {"canonical": sig[i]["best_rate"]}
        per_group_readout = {"canonical": sig[i]["best_rate_readout"]}
        for gname, entries in groups.items():
            truth_vals, readout_vals = [], []
            for h, truth in entries:
                base = _readout(model, h)
                di = _readout(model, model.step_once(h, i))
                readout_vals.append(_match(di, fn(base)))
                if truth is not None:
                    truth_vals.append(_match(di, fn(truth)))
            per_group[gname] = (float(np.mean(truth_vals))
                                if truth_vals else None)
            per_group_readout[gname] = (float(np.mean(readout_vals))
                                        if readout_vals else None)
        rates = [v for v in per_group.values() if v is not None]
        out[i] = {"op": sig[i]["best"],
                  "per_group_truth": per_group,
                  "per_group_readout": per_group_readout,
                  "min_group_rate": float(np.min(rates)),
                  "invariant": bool(np.min(rates) >= D2_INVARIANT_MIN)}
    return out


# ---------------------------------------------------------------------------
# P-3 composer-free recomposition (the guillotine)
# ---------------------------------------------------------------------------

def build_mapping(sig: dict, kind: str) -> dict:
    """op name -> atom id, best rate wins, threshold D2_MAP_THRESHOLD."""
    mapping = {}
    for i, s in sig.items():
        name = s["best"]
        if not name.startswith(kind) or s["best_rate"] < D2_MAP_THRESHOLD:
            continue
        op = name.split(":")[1]
        if op not in mapping or sig[mapping[op]]["best_rate"] < s["best_rate"]:
            mapping[op] = i
    return mapping


def _chain_for(task_id: str, kind: str) -> tuple:
    if kind == "surf":
        return ops.task_surface_ops(task_id)
    return ops.task_subop_chain(task_id)


@torch.no_grad()
def forced_programs(model, bundle, mapping: dict, kind: str) -> dict:
    cells, skipped = {}, []
    tds = ([("trained_singleton" if td.task.n_tokens == 1 else
             "trained_pair", td) for td in bundle.seen_heldout]
           + [(lvl, td) for lvl in ("L1", "L2", "L3")
              for td in bundle.unseen[lvl]])
    for group, td in tds:
        chain = _chain_for(td.task.task_id, kind)
        if any(op not in mapping for op in chain):
            skipped.append(td.task.task_id)
            continue
        x, y = td.x[:N_FORCED], td.y[:N_FORCED]
        z = model.code(torch.from_numpy(x))
        for op in chain:
            z = model.step_once(z, mapping[op])
        acc = _match(_readout(model, z), y)
        cells[td.task.task_id] = {"group": group, "chain": list(chain),
                                  "atoms": [mapping[op] for op in chain],
                                  "acc": acc}
    by_group: dict[str, list] = {}
    for c in cells.values():
        by_group.setdefault(c["group"], []).append(c["acc"])
    return {"kind": kind, "mapping": {k: int(v) for k, v in mapping.items()},
            "cells": cells, "skipped_unmapped": skipped,
            "coverage": len(cells) / max(len(cells) + len(skipped), 1),
            "acc_by_group": {g: float(np.mean(v))
                             for g, v in sorted(by_group.items())}}


# ---------------------------------------------------------------------------
# P-4 atom swaps
# ---------------------------------------------------------------------------

@torch.no_grad()
def _routed_acc(model, td, swap: dict | None = None) -> float:
    x, y = td.x[:N_FORCED], td.y[:N_FORCED]
    n = len(x)
    toks = torch.from_numpy(np.tile(td.task.tokens, (n, 1)))
    ntok = torch.full((n,), td.task.n_tokens, dtype=torch.int64)
    xb = torch.from_numpy(x)
    out = model(xb, toks, ntok, mode="hard")
    if swap is None:
        return _match(out["logits"].argmax(-1).numpy(), y)
    ch = out["choices"].numpy().copy()
    for a, b in swap.items():
        ia, ib = ch == a, ch == b
        ch[ia], ch[ib] = b, a
    ch[ch == -1] = R.PASS_INDEX
    forced = model(xb, toks, ntok, mode="forced",
                   forced=torch.from_numpy(ch))
    return _match(forced["logits"].argmax(-1).numpy(), y)


@torch.no_grad()
def atom_swaps(model, bundle, sig: dict) -> dict:
    by_op: dict[str, list] = {}
    for i, s in sig.items():
        if s["sharp"] and s["best"] != "identity":
            by_op.setdefault(s["best"], []).append(i)
    pairs = [(op, ids) for op, ids in by_op.items() if len(ids) >= 2]
    if not pairs:
        return {"duplicate_pairs": [],
                "note": "no two atoms share a sharp signature; "
                        "interchangeability untestable in this model "
                        "(recorded, not scored)"}
    results = []
    tds = [td for td in bundle.seen_heldout] + list(bundle.unseen["L1"])
    for op, ids in pairs:
        a, b = ids[0], ids[1]
        rows = {}
        for td in tds:
            base = _routed_acc(model, td)
            swapped = _routed_acc(model, td, swap={a: b})
            if base > 0:
                rows[td.task.task_id] = {"routed": base, "swapped": swapped}
        drops = [r["routed"] - r["swapped"] for r in rows.values()]
        results.append({"op": op, "atoms": [int(a), int(b)],
                        "mean_drop": float(np.mean(drops)),
                        "max_drop": float(np.max(drops)),
                        "per_task": rows})
    return {"duplicate_pairs": results}


# ---------------------------------------------------------------------------
# P-5 selective ablation (regroups the existing panel damage matrix)
# ---------------------------------------------------------------------------

def selective_ablation(run_dir, sig: dict) -> dict:
    path = os.path.join(run_dir, "panel", "final", "ablation.npz")
    npz = np.load(path, allow_pickle=True)
    damage, task_ids = npz["damage"], [str(t) for t in npz["task_ids"]]

    def contains(tid, name):
        if name.startswith("surf:"):
            return name.split(":")[1] in ops.task_surface_ops(tid)
        return name.split(":")[1] in ops.task_subop_chain(tid)

    out = {}
    for i, s in sig.items():
        if not s["sharp"] or s["best"] == "identity":
            continue
        inside = [j for j, t in enumerate(task_ids)
                  if contains(t, s["best"])]
        outside = [j for j in range(len(task_ids)) if j not in inside]
        d_in = damage[i, inside]
        d_out = damage[i, outside]
        d_in, d_out = d_in[~np.isnan(d_in)], d_out[~np.isnan(d_out)]
        m_in = float(d_in.mean()) if len(d_in) else None
        m_out = float(d_out.mean()) if len(d_out) else None
        ratio = (m_in / m_out if m_in is not None and m_out
                 not in (None, 0.0) and m_out > 0 else None)
        out[i] = {"op": s["best"], "damage_in_mean": m_in,
                  "damage_out_mean": m_out, "in_out_ratio": ratio,
                  "selective": bool(ratio is not None
                                    and ratio >= D2_SELECTIVE_RATIO)}
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_d2(run_dir, label: str) -> dict:
    model, cfg, step = load_checkpoint(run_dir, "final.pt")
    model.eval()
    bundle = data_mod.build_bundle(cfg)

    singles = [td for td in bundle.seen_heldout if td.task.n_tokens == 1]
    x_pool = np.concatenate([td.x[:N_SIG // len(singles)] for td in singles])

    sig = signature(model, x_pool)
    inv = context_invariance(model, bundle, sig)
    forced = {kind: forced_programs(model, bundle,
                                    build_mapping(sig, kind), kind)
              for kind in ("sub", "surf")}
    swaps = atom_swaps(model, bundle, sig)
    abl = selective_ablation(run_dir, sig)

    sharp = {i: s for i, s in sig.items() if s["sharp"]}
    verdict = {
        "n_sharp_atoms": len(sharp),
        "sharp_ops": sorted(set(s["best"] for s in sharp.values())),
        "any_sub_op_signature": any(
            s["best"].startswith("sub:") for s in sharp.values()),
        "all_sharp_invariant": (bool(inv) and all(
            v["invariant"] for v in inv.values())),
        "recomb_L1_surf": forced["surf"]["acc_by_group"].get("L1"),
        "recomb_L3_surf": forced["surf"]["acc_by_group"].get("L3"),
        "recomb_works_L1": bool(
            (forced["surf"]["acc_by_group"].get("L1") or 0)
            >= D2_RECOMB_WORKS
            or (forced["sub"]["acc_by_group"].get("L1") or 0)
            >= D2_RECOMB_WORKS),
        "recomb_works_L3": bool(
            (forced["surf"]["acc_by_group"].get("L3") or 0)
            >= D2_RECOMB_WORKS
            or (forced["sub"]["acc_by_group"].get("L3") or 0)
            >= D2_RECOMB_WORKS),
        "all_selective": (bool(abl) and all(
            v["selective"] for v in abl.values())),
    }
    return {"label": label, "run_dir": str(run_dir), "checkpoint_step": step,
            "arm": cfg.arm, "seed": cfg.seed,
            "thresholds": {"sharp": D2_SHARP, "map": D2_MAP_THRESHOLD,
                           "invariant_min": D2_INVARIANT_MIN,
                           "recomb_works": D2_RECOMB_WORKS,
                           "selective_ratio": D2_SELECTIVE_RATIO},
            "p1_signature": {str(i): s for i, s in sig.items()},
            "p2_invariance": {str(i): v for i, v in inv.items()},
            "p3_forced_programs": forced,
            "p4_swaps": swaps,
            "p5_selective_ablation": {str(i): v for i, v in abl.items()},
            "verdict": verdict}


def main() -> None:
    run_specs = []
    for exp, arm_glob, label in (("e1b", "A6_s*", "A6"),
                                 ("e4", "A14_s*", "A14"),
                                 ("e5", "A16_s*", "A16")):
        for rd in sorted(glob.glob(str(RUNS_DIR / exp / arm_glob))):
            if os.path.exists(os.path.join(rd, "checkpoints", "final.pt")):
                seed = os.path.basename(rd).split("_")[1]
                run_specs.append((rd, f"{label}_{seed}"))

    out = RESULTS_DIR / "d2"
    results = []
    for rd, label in run_specs:
        print(f"=== D2 on {label} ({rd})")
        res = run_d2(rd, label)
        write_json(out / f"{label}.json", res)
        results.append(res)
        v = res["verdict"]
        print(f"    sharp={v['n_sharp_atoms']} {v['sharp_ops']} "
              f"invariant={v['all_sharp_invariant']} "
              f"L1_forced={v['recomb_L1_surf']} "
              f"L3_forced={v['recomb_L3_surf']} "
              f"selective={v['all_selective']}")

    lines = ["# D2 - Atom Factorization Audit (read-only)", "",
             "Registered definition and thresholds in DECISIONS.md "
             "(2026-08-20). Semantics are readout-channel: "
             "decode(A_i(h)) == f(decode(h)) under the model's own frozen "
             "decoder.", ""]

    lines += ["## P-1/P-2: sharp signatures and context invariance", "",
              "| run | sharp atoms (op@rate) | any sub-op? | worst context "
              "group (per atom) |", "|---|---|---|---|"]
    for res in results:
        sig = res["p1_signature"]
        sharp = [f"A{i}={s['best'].split(':')[-1]}@{s['best_rate']:.2f}"
                 for i, s in sorted(sig.items(), key=lambda kv: int(kv[0]))
                 if s["sharp"]]
        inv = res["p2_invariance"]
        worst = [f"A{i}:{v['min_group_rate']:.2f}"
                 for i, v in sorted(inv.items(), key=lambda kv: int(kv[0]))]
        lines.append(f"| {res['label']} | {', '.join(sharp) or '-'} | "
                     f"{res['verdict']['any_sub_op_signature']} | "
                     f"{', '.join(worst) or '-'} |")

    lines += ["", "## P-3: composer-free recomposition (the guillotine)", "",
              "| run | kind | coverage | singletons | trained pairs | L1 | "
              "L2 | L3 |", "|---|---|---|---|---|---|---|---|"]
    for res in results:
        for kind in ("sub", "surf"):
            f = res["p3_forced_programs"][kind]
            g = f["acc_by_group"]
            def cell(name):
                v = g.get(name)
                return "-" if v is None else f"{v:.3f}"
            lines.append(
                f"| {res['label']} | {kind} | {f['coverage']:.2f} | "
                f"{cell('trained_singleton')} | {cell('trained_pair')} | "
                f"{cell('L1')} | {cell('L2')} | {cell('L3')} |")

    lines += ["", "## P-4/P-5: swaps and selectivity", "",
              "| run | duplicate pairs (mean acc drop) | selective ablation "
              "(atom: in/out ratio) |", "|---|---|---|"]
    for res in results:
        sw = res["p4_swaps"]["duplicate_pairs"]
        sw_txt = ", ".join(f"{p['op']} A{p['atoms'][0]}<->A{p['atoms'][1]}: "
                           f"{p['mean_drop']:.3f}" for p in sw) or "none"
        ab = res["p5_selective_ablation"]
        ab_txt = ", ".join(
            f"A{i}({v['op'].split(':')[-1]}): "
            f"{'-' if v['in_out_ratio'] is None else f'{v['in_out_ratio']:.1f}'}"
            for i, v in sorted(ab.items(), key=lambda kv: int(kv[0])))
        lines.append(f"| {res['label']} | {sw_txt} | {ab_txt or '-'} |")

    lines += ["", "Registered verdict rule: FACTORIZED requires sharp "
              "identity + context invariance + selective damage + "
              "composer-free novel recomposition. High routed accuracy with "
              "collapsed forced programs = surface-conditioned routing "
              "programs, not atom factorization. Per-run detail in the "
              "JSON files beside this summary.", ""]
    with open(out / "summary.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
