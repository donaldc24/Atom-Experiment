"""Aggregate completed runs into tables + the mechanical E0 pattern check.

Rules inherited from V1's standing rules:
  - reads metrics.json artifacts only, never a model
  - refuses to mix hostnames (determinism is a within-platform guarantee)
  - refuses to mix smoke and real runs
  - per-seed rows are never discarded; arm summaries are mean +/- std
  - composer and atom library stay separate line items in every table
The E0 verdict (pattern match, not number match) and the E0 instrument audit
are computed here, mechanically, from artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ops
from . import split as split_mod
from . import registered as R
from .utils import RESULTS_DIR, RUNS_DIR, read_json, write_json

HEADLINE = (
    "acc_seen_hard", "acc_seen_soft",
    "acc_unseen_L1_hard", "acc_unseen_L2_hard", "acc_unseen_L3_hard",
    "dissociation_gap_hard", "soft_hard_gap_seen",
    "closed_map_seen", "closed_map_unseen_L1", "closed_map_unseen_L3",
    "closed_map_target_seen", "closed_map_target_unseen_L1",
    "closed_map_target_unseen_L3",
    "census_atoms_in_use", "census_steps_per_token", "census_pass_rate",
    "ablation_cv_median", "standalone_best_acc_mean_in_use",
    "closed_map_atom_matched_error", "closed_map_atom_coverage",
    # C1: renamed - these measure task-identity leakage, not sub-op structure
    "leakage_subop_identity_from_delta", "leakage_subop_identity_h0_floor",
    "decodability_surface_from_delta", "decodability_surface_h0_floor",
    "transfer_task_row_std_mean", "transfer_transplant_row_std_mean",
    # C2: the granularity instrument
    "probe_transfer_subop_mean", "probe_transfer_subop_mean_taskscope",
    # C3: canonical substitution, per level (train = seen_heldout pairs)
    "canon_route_agree_hard_train", "canon_route_agree_hard_L1",
    "canon_route_agree_hard_L2", "canon_route_agree_hard_L3",
    "canon_repair_acc_train", "canon_repair_acc_L1",
    "canon_repair_acc_L2", "canon_repair_acc_L3",
    "canon_repair_delta_train", "canon_repair_delta_L1",
    "canon_repair_delta_L2", "canon_repair_delta_L3",
    "canon_route_kl_train", "canon_route_kl_L1",
    "canon_route_kl_L2", "canon_route_kl_L3",
    # C4: decomposed transplant variance (+ the legacy conflated number)
    "transfer_partner_variance", "transfer_input_variance",
    "transfer_row_std_legacy",
)


def collect(experiment: str, smoke: bool = False) -> list[dict]:
    sub = f"smoke_{experiment}" if smoke else experiment
    rows = []
    hostnames = set()
    source_ids = set()
    seen_run_keys: dict[tuple[str, int], Path] = {}
    for mpath in sorted((RUNS_DIR / sub).rglob("metrics.json")):
        run_dir = mpath.parent
        m = read_json(mpath)
        if m.get("smoke", False) != smoke:
            raise SystemExit(f"{run_dir}: smoke flag mismatch with directory")
        env = read_json(run_dir / "env.json")
        hostnames.add(env.get("hostname", "unknown"))
        revision = m.get("protocol_revision", "pre-revision")
        if revision != R.PROTOCOL_REVISION:
            raise SystemExit(
                f"{run_dir}: protocol revision {revision!r} cannot be pooled "
                f"with current revision {R.PROTOCOL_REVISION!r}")
        # Amendment R10: pooling identity is the harness-source content
        # fingerprint - the code that can change a number. Runs written before
        # the key existed fall back to the old dirty-snapshot/commit id, which
        # also covered incidental files; backfill them (atomv2.backfill) rather
        # than pooling across the two schemes.
        source = env.get("harness_source_sha256")
        if source is None:
            raise SystemExit(
                f"{run_dir}: env.json predates the harness_source_sha256 "
                "provenance key (amendment R10). Run "
                "`python -m atomv2.backfill --rev <commit>` to record the "
                "source identity of these runs before aggregating.")
        source_ids.add(source)
        key = (m["arm"], int(m["seed"]))
        if key in seen_run_keys:
            raise SystemExit(
                f"duplicate run key {key}: {seen_run_keys[key]} and {run_dir}; "
                "never average duplicate seeds")
        seen_run_keys[key] = run_dir
        row = {"run_dir": str(run_dir), "arm": m["arm"], "seed": m["seed"]}
        row["protocol_revision"] = revision
        for k in HEADLINE:
            row[k] = m.get(k)
        row["params_composer"] = m["param_counts"]["composer"]
        row["params_atoms_total"] = m["param_counts"]["atoms_total"]
        rows.append(row)
    if len(hostnames) > 1:
        raise SystemExit(f"runs span multiple hostnames {hostnames}; "
                          "determinism is a within-platform guarantee - "
                          "aggregate one machine's runs at a time.")
    if len(source_ids) > 1:
        raise SystemExit(f"runs span multiple source snapshots {source_ids}; "
                         "aggregate one exact implementation at a time.")
    return rows


def summarise(rows: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    summary = {}
    for arm, rs in sorted(by_arm.items()):
        entry = {"n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs)}
        for k in HEADLINE:
            vals = [r[k] for r in rs if r[k] is not None]
            if vals:
                entry[k + "_mean"] = float(np.mean(vals))
                entry[k + "_std"] = (float(np.std(vals, ddof=1))
                                     if len(vals) > 1 else None)
                entry[k + "_per_seed"] = {str(r["seed"]): r[k] for r in rs}
        summary[arm] = entry
    return summary


# ---------------------------------------------------------------------------
# E0: pattern check + instrument audit
# ---------------------------------------------------------------------------

def e0_verdict(rows: list[dict], summary: dict) -> dict:
    oracle = summary.get("A0-oracle", {})
    free = summary.get("A0-free", {})
    o_l1 = oracle.get("acc_unseen_L1_hard_mean")
    f_l1 = free.get("acc_unseen_L1_hard_mean")
    o_cm = oracle.get("closed_map_target_seen_mean")
    f_cm = free.get("closed_map_target_seen_mean")
    checks = {
        "oracle_L1_high": (o_l1 is not None and o_l1 > R.E0_ORACLE_L1_MIN,
                           {"value": o_l1, "min": R.E0_ORACLE_L1_MIN}),
        "free_L1_floor": (f_l1 is not None and f_l1 < R.E0_FREE_L1_MAX,
                          {"value": f_l1, "max": R.E0_FREE_L1_MAX}),
        "gap_unmistakable": (
            o_l1 is not None and f_l1 is not None
            and (o_l1 - f_l1) >= R.E0_GAP_MIN,
            {"gap": None if o_l1 is None or f_l1 is None else o_l1 - f_l1,
             "min": R.E0_GAP_MIN}),
        "closed_map_target_direction": (
            o_cm is not None and f_cm is not None and o_cm < f_cm,
            {"oracle": o_cm, "free": f_cm,
             "note": "direction only: oracle final distance-to-target < free; "
                     "nearest-prefix headline remains reported but is all-pass gameable"}),
    }
    return {
        "passed": all(ok for ok, _ in checks.values()),
        "protocol_revision": R.PROTOCOL_REVISION,
        "checks": {k: {"ok": ok, **detail} for k, (ok, detail) in checks.items()},
        "n_runs": len(rows),
        "note": "qualitative pattern match, NOT number match - the world "
                "changed (length 6 not 8, 16 slots not 8, micro-steps added)",
    }


def e0_instrument_audit(rows: list[dict]) -> dict:
    """The oracle arm has ground truth, so it audits the instruments.

    The registered 2% census correctly counts only seven atoms: P3 occurs in
    one singleton and no training pair, so its forced share is 1/76 < 2%.
    We separately audit that all eight forced atoms were observed at all.
    """
    audit = {}
    split = split_mod.load_verified()
    seen_ids = list(split["singletons_train"]) + list(split["train_pairs"])
    expected_counts = np.zeros(R.N_SURFACE, dtype=np.int64)
    for tid in seen_ids:
        for p in ops.task_surface_ops(tid):
            expected_counts[ops.SURFACE_NAMES.index(p)] += 1
    expected_share = expected_counts / expected_counts.sum()
    expected_mask = np.zeros(R.N_ATOMS, dtype=bool)
    expected_mask[:R.N_SURFACE] = expected_share > R.CENSUS_EPS
    for r in rows:
        if r["arm"] != "A0-oracle":
            continue
        run_dir = Path(r["run_dir"])
        cen = read_json(run_dir / "panel" / "final" / "census.json")
        actual_share = np.asarray(cen["atom_selection_share"])
        entry = {
            "registered_census_expected": int(expected_mask.sum()),
            "registered_census_matches_expected": (
                cen["in_use_mask"] == expected_mask.tolist()),
            "all_8_forced_atoms_observed": bool(
                (actual_share[:R.N_SURFACE] > 0).all()),
        }
        npz = np.load(run_dir / "panel" / "final" / "ablation.npz",
                      allow_pickle=True)
        usage, task_ids = npz["usage"], [str(t) for t in npz["task_ids"]]
        correct, total = 0, 0
        for i in range(R.N_SURFACE):
            p = ops.SURFACE_NAMES[i]
            for j, tid in enumerate(task_ids):
                should = p in ops.task_surface_ops(tid)
                is_used = usage[i, j] >= R.USAGE_TASK_THRESHOLD
                correct += int(should == is_used)
                total += 1
        entry["usage_matches_ground_truth"] = correct / total
        entry["idle_atoms_stay_idle"] = bool(
            (usage[R.N_SURFACE:] < R.USAGE_TASK_THRESHOLD).all())
        damage = npz["damage"]
        cvs = []
        for i in range(R.N_SURFACE):
            row = damage[i][~np.isnan(damage[i])]
            if len(row) >= 2 and abs(row.mean()) > 1e-9:
                cvs.append(float(row.std(ddof=1) / abs(row.mean())))
        entry["oracle_ablation_cv_median"] = (float(np.median(cvs))
                                              if cvs else None)
        audit[f"{r['arm']}_seed{r['seed']}"] = entry
    return audit


def aggregate(experiment: str, smoke: bool = False) -> Path:
    rows = collect(experiment, smoke=smoke)
    if not rows:
        raise SystemExit(f"no completed runs found for {experiment} (smoke={smoke})")
    summary = summarise(rows)
    sub = f"smoke_{experiment}" if smoke else experiment
    out = RESULTS_DIR / sub
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "per_run.json", rows)
    write_json(out / "summary.json", summary)

    if experiment == "e0":
        verdict = e0_verdict(rows, summary)
        verdict["instrument_audit"] = e0_instrument_audit(rows)
        write_json(out / "e0_verdict.json", verdict)

    # Amendment R9: E1's lambda=0 cell is not re-run; it IS E0's A0-free arm.
    # Carried in as a clearly labelled reference row rather than pooled into
    # the battery, so no table ever implies it was run under E1.
    lambda_zero = None
    if experiment == "e1" and not smoke:
        try:
            ref_rows = collect(R.LAMBDA_ZERO_SOURCE["experiment"], smoke=False)
        except SystemExit:
            ref_rows = []
        ref_rows = [r for r in ref_rows if r["arm"] == R.LAMBDA_ZERO_SOURCE["arm"]]
        if ref_rows:
            lambda_zero = summarise(ref_rows)[R.LAMBDA_ZERO_SOURCE["arm"]]
            lambda_zero["source"] = dict(R.LAMBDA_ZERO_SOURCE)
            lambda_zero["lambda_use"] = R.LAMBDA_GRID["A1"]
            write_json(out / "lambda_zero_reference.json", lambda_zero)

    lines = [f"# {sub} summary", "",
             "| arm | n | " + " | ".join(HEADLINE) + " |",
             "|" + "---|" * (len(HEADLINE) + 2)]
    for arm, e in sorted(summary.items()):
        cells = []
        for k in HEADLINE:
            mean = e.get(k + "_mean")
            std = e.get(k + "_std")
            if mean is None:
                cells.append("-")
            elif std is None:
                cells.append(f"{mean:.4f}")
            else:
                cells.append(f"{mean:.4f}±{std:.4f}")
        lines.append(f"| {arm} | {e['n_seeds']} | " + " | ".join(cells) + " |")
    if lambda_zero is not None:
        cells = []
        for k in HEADLINE:
            mean = lambda_zero.get(k + "_mean")
            std = lambda_zero.get(k + "_std")
            if mean is None:
                cells.append("-")
            elif std is None:
                cells.append(f"{mean:.4f}")
            else:
                cells.append(f"{mean:.4f}±{std:.4f}")
        lines.append(f"| A1=A0-free (ref, from e0) | {lambda_zero['n_seeds']} | "
                     + " | ".join(cells) + " |")
    lines += ["", "Composer and atom library are separate line items by rule; "
              "never sum them into a 'system size'.", ""]
    if lambda_zero is not None:
        lines += ["Amendment R9: the lambda=0 row is E0's A0-free arm, which is "
                  "configurationally identical to A1 (the resolved configs "
                  "differ only in the `arm` and `experiment` strings). It was "
                  "NOT re-run under E1 and is shown as a reference row, not as "
                  "a battery arm.", ""]
    with open(out / "summary.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate completed runs")
    ap.add_argument("--experiment", required=True, choices=["e0", "e1"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = aggregate(a.experiment, smoke=a.smoke)
    print(out)
    if a.experiment == "e0":
        print(json.dumps(read_json(out / "e0_verdict.json")["checks"], indent=2))
