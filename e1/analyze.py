"""Metric computation for E1. Reads ONLY saved artifacts; never touches the model.

Every number in metrics.json is re-derivable from runs/{run_id}/artifacts without
retraining (spec 6).

Usage:
    python -m e1.analyze runs/A0_0_7245365
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .utils import read_json, write_json

DEAD_ATOM_THRESHOLD = 0.05   # M2: mean_j(d_ij) <= 0.05 -> atom is dead


def load_predictions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def m1_recombination(seen: list[dict], unseen: list[dict]) -> dict:
    acc_seen = float(np.mean([r["correct"] for r in seen]))
    acc_unseen = float(np.mean([r["correct"] for r in unseen]))
    return {
        "M1_acc_seen": acc_seen,
        "M1_acc_unseen": acc_unseen,
        "M1_gap": acc_seen - acc_unseen,
    }


def m2_ablation(d: np.ndarray) -> dict:
    n_atoms = d.shape[0]
    cvs, means, counts = [], [], []
    dead = 0
    for i in range(n_atoms):
        row = d[i][~np.isnan(d[i])]
        counts.append(int(row.size))
        if row.size == 0:
            means.append(float("nan"))
            dead += 1
            continue
        mean = float(row.mean())
        means.append(mean)
        if mean > DEAD_ATOM_THRESHOLD:
            cvs.append(float(row.std(ddof=0) / mean))
        else:
            dead += 1
    # Atoms whose degradation profile rests on a single task cannot show variance;
    # reported separately so a flattering M2_cv is visible for what it is.
    multi = [
        float(d[i][~np.isnan(d[i])].std(ddof=0) / d[i][~np.isnan(d[i])].mean())
        for i in range(n_atoms)
        if counts[i] >= 2 and not np.isnan(means[i]) and means[i] > DEAD_ATOM_THRESHOLD
    ]
    return {
        "M2_cv": float(np.mean(cvs)) if cvs else float("nan"),
        "M2_dead": int(dead),
        "M2_n_atoms_scored": len(cvs),
        "M2_cv_multitask_only": float(np.mean(multi)) if multi else float("nan"),
        "M2_mean_degradation": [None if np.isnan(m) else m for m in means],
        "M2_tasks_per_atom": counts,
    }


def m3_alignment(A: np.ndarray) -> dict:
    srt = np.sort(A, axis=1)
    align = srt[:, -1]
    purity = srt[:, -1] - srt[:, -2]
    return {
        "M3_align": float(align.mean()),
        "M3_purity": float(purity.mean()),
        "M3_align_per_atom": align.tolist(),
        "M3_purity_per_atom": purity.tolist(),
        "M3_argmax_primitive": A.argmax(axis=1).tolist(),
        "M3_n_primitives_covered": int(len(set(A.argmax(axis=1).tolist()))),
    }


def m4_routing(rows: list[dict], A: np.ndarray) -> float:
    """Fraction of routing steps that select the atom aligned to the step's primitive."""
    aligned = A.argmax(axis=1)
    hits = total = 0
    for r in rows:
        for t, chosen in enumerate(r["routing_hard"]):
            total += 1
            if aligned[chosen] == r["instruction"][t]:
                hits += 1
    return hits / max(1, total)


def m5_utilisation(routing_counts: np.ndarray, dead: int) -> dict:
    counts = routing_counts.sum(axis=0).astype(np.float64)
    n = counts.size
    p = counts / max(counts.sum(), 1.0)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(n)) if n > 1 else 0.0
    return {
        "M5_entropy": entropy,
        "M5_dead": int(dead),
        "M5_selection_fraction": (counts / max(counts.sum(), 1.0)).tolist(),
    }


def m7_manifold(drift: np.ndarray, acc_teacher: np.ndarray,
                acc_actual: np.ndarray) -> dict:
    """M7: is the library fine and only the composition operator broken? (D12)"""
    teacher = float(acc_teacher.mean())
    actual = float(acc_actual.mean())
    return {
        "M7_drift_step1": float(drift[:, 0].mean()),
        "M7_drift_final": float(drift[:, -1].mean()),
        "M7_acc_teacher_forced": teacher,
        "M7_recovery": teacher - actual,
    }


def m6_soft_hard(unseen: list[dict]) -> dict:
    hard = float(np.mean([r["correct"] for r in unseen]))
    soft = float(np.mean([r["correct_soft"] for r in unseen]))
    return {"M6_soft_hard_gap": soft - hard, "M6_acc_soft_unseen": soft}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def analyze(run_dir: Path) -> dict:
    art = run_dir / "artifacts"
    seen = load_predictions(art / "predictions_seen_heldout.jsonl")
    unseen = load_predictions(art / "predictions_unseen.jsonl")
    singleton = load_predictions(art / "predictions_singleton.jsonl")

    A = np.load(art / "alignment_matrix.npy")
    A1 = np.load(art / "alignment_matrix_1step.npy")
    d = np.load(art / "ablation_matrix.npy")
    routing_counts = np.load(art / "routing_counts.npy")
    ablate_all = np.load(art / "ablate_all_acc.npy")
    pc = read_json(art / "param_counts.json")
    cfg = read_json(run_dir / "config.json")

    metrics: dict = {"arm": cfg["arm"], "seed": cfg["seed"]}
    if cfg.get("rung"):
        metrics["rung"] = cfg["rung"]
        metrics["code_consistency_weight"] = cfg.get("code_consistency_weight", 0.0)
    metrics.update(m1_recombination(seen, unseen))
    m2 = m2_ablation(d)
    metrics.update(m2)
    metrics.update(m3_alignment(A))
    # Literal one-step probe, retained as a diagnostic (DECISIONS.md D11).
    one = m3_alignment(A1)
    metrics["M3_align_1step"] = one["M3_align"]
    metrics["M3_purity_1step"] = one["M3_purity"]
    metrics["M4_routing_acc_seen"] = m4_routing(seen, A)
    metrics["M4_routing_acc_unseen"] = m4_routing(unseen, A)
    metrics.update(m5_utilisation(routing_counts, m2["M2_dead"]))
    metrics.update(m6_soft_hard(unseen))
    metrics.update(m7_manifold(
        np.load(art / "manifold_drift.npy"),
        np.load(art / "manifold_acc_teacher.npy"),
        np.load(art / "manifold_acc_actual.npy"),
    ))

    # Assumption-free M3 variant (D21): best trailing atom rather than an assumed
    # identity slot. Reported alongside the pre-registered M3, never replacing it.
    t_path = art / "alignment_tensor.npy"
    if t_path.exists():
        T = np.load(t_path)                      # [i, s, p]
        best_over_s = T.max(axis=1)              # [i, p]
        robust = m3_alignment(best_over_s)
        metrics["M3_align_best_s"] = robust["M3_align"]
        metrics["M3_purity_best_s"] = robust["M3_purity"]
        metrics["M3_primitives_covered_best_s"] = robust["M3_n_primitives_covered"]
        # Which trailing atom carried each atom's best reading, and how much work it
        # did -- a large residual there means the reading is inflated, not clean.
        flat = T.reshape(T.shape[0], -1).argmax(axis=1)
        metrics["M3_best_s_per_atom"] = (flat // T.shape[2]).tolist()
        rn_path = art / "atom_residual_norms.npy"
        if rn_path.exists():
            rn = np.load(rn_path)
            metrics["atom_residual_norms"] = rn.tolist()
            metrics["M3_best_s_residual"] = [
                float(rn[s]) for s in metrics["M3_best_s_per_atom"]]

    # Decoder-free closed-map probe (D21): no trailing atom, so no assumption to
    # violate and nothing to inflate it.
    se_path = art / "state_alignment_err.npy"
    if se_path.exists():
        err = np.load(se_path)
        sacc = np.load(art / "state_alignment_acc.npy")
        metrics["M3_closed_map_error"] = float(err.min(axis=1).mean())
        metrics["M3_closed_map_argmin"] = err.argmin(axis=1).tolist()
        # MUST be read with the error: an untrained library scores a NEAR-PERFECT
        # closed-map error, because atoms initialise near zero so h0+atom(h0) ~ h0 =
        # enc(identity(x)) and every atom "implements identity". Coverage is what
        # separates a real closed-map library (a permutation, 8/8) from a dead one
        # (everything collapsed on identity, 1/8). See D24.
        metrics["M3_closed_map_coverage"] = int(len(set(err.argmin(axis=1).tolist())))
        # Error against a one-atom-per-primitive assignment, which a dead library
        # cannot fake: greedy match on the error matrix, no primitive reused.
        e = err.copy()
        used, matched = set(), []
        order = np.argsort(e.min(axis=1))
        for i in order:
            cand = [p for p in range(e.shape[1]) if p not in used]
            best = min(cand, key=lambda p: e[i, p])
            used.add(best)
            matched.append(e[i, best])
        metrics["M3_closed_map_error_matched"] = float(np.mean(matched))
        srt = np.sort(sacc, axis=1)
        metrics["M3_state_align"] = float(srt[:, -1].mean())
        metrics["M3_state_purity"] = float((srt[:, -1] - srt[:, -2]).mean())
        metrics["M3_state_primitives_covered"] = int(len(set(sacc.argmax(1).tolist())))

    # E1b: did the on-manifold constraint bind, and is it being gamed? (D27/D28)
    cr_path = art / "code_residual_unseen.npy"
    if cr_path.exists():
        cr_u = np.load(cr_path)
        cr_s = np.load(art / "code_residual_seen.npy")
        if cr_u.size:
            metrics["code_residual_unseen"] = float(cr_u.mean())
        if cr_s.size:
            metrics["code_residual_seen"] = float(cr_s.mean())
        if cr_u.size and cr_s.size:
            hard = np.concatenate([cr_s, cr_u])
            metrics["code_residual"] = float(hard.mean())
            sp = art / "code_residual_soft_unseen.npy"
            if sp.exists():
                soft = np.concatenate([np.load(art / "code_residual_soft_seen.npy"),
                                       np.load(sp)])
                metrics["code_residual_soft"] = float(soft.mean())
                # The gaming signal: a soft constraint satisfied while the hard one
                # is not means the decode went uninformative rather than h_t
                # becoming a valid code (D27).
                metrics["code_soft_hard_gap"] = float(hard.mean() - soft.mean())
            ep = art / "code_entropy_unseen.npy"
            if ep.exists():
                ent = np.concatenate([np.load(art / "code_entropy_seen.npy"),
                                      np.load(ep)])
                metrics["code_entropy"] = float(ent.mean())
                metrics["code_entropy_frac"] = float(
                    ent.mean() / np.log(cfg["vocab"]))
            dp = art / "code_diversity_unseen.json"
            if dp.exists():
                div = read_json(art / "code_diversity_seen.json") + read_json(dp)
                fr = [d["distinct_frac"] for d in div]
                # R3-specific: a hard projection always looks confident, so entropy
                # cannot see its collapse mode. Diversity can.
                metrics["code_decode_diversity"] = float(np.mean(fr))
                metrics["code_decode_diversity_min"] = float(np.min(fr))

    # Collapse detector (D29): every manifold metric looks excellent on a collapsed
    # code, so spread must be read beside them.
    sp_path = art / "code_spread.json"
    if sp_path.exists():
        metrics.update(read_json(sp_path))

    # A3 only: how much of the library survived phase 2's encoder drift (D20).
    p1_path = art / "alignment_matrix_phase1.npy"
    if p1_path.exists():
        A_p1 = np.load(p1_path)
        one_p1 = m3_alignment(A_p1)
        metrics["M3_align_phase1"] = one_p1["M3_align"]
        metrics["M3_primitives_covered_phase1"] = one_p1["M3_n_primitives_covered"]
        metrics["library_decay"] = one_p1["M3_align"] - metrics["M3_align"]

    metrics["acc_singleton"] = float(np.mean([r["correct"] for r in singleton]))
    metrics["acc_ablate_all"] = float(ablate_all.mean())

    # H5 instrumentation: composer and library are always separate line items.
    metrics["params"] = {
        "composer": pc["composer"],
        "atoms_total": pc["atoms_total"],
        "atoms_each": pc["atoms_each"],
        "keys": pc["keys"],
        "encoder": pc["encoder"],
        "decoder": pc["decoder"],
        "total": pc["total"],
        "composer_over_atoms": pc["composer"] / pc["atoms_total"],
    }
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    metrics = analyze(run_dir)
    write_json(run_dir / "metrics.json", metrics)
    print(json.dumps(
        {k: v for k, v in metrics.items()
         if not isinstance(v, (list, dict))}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
