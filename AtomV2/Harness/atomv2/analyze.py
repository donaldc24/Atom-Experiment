"""Derive a run's headline metrics.json from saved artifacts ONLY.

Never touches the model or the training loop; can be re-run (or extended with
new metrics) at any time, and applies retroactively to any completed run.
Per-cell accuracies are carried into metrics.json because per-cell IS the
measurement; level means are summaries (H1Experiments.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import registered as R
from .utils import read_json, write_json


def _final_eval(run_dir: Path) -> dict:
    evals = sorted((run_dir / "evals").glob("step*.json"))
    if not evals:
        raise FileNotFoundError(f"no eval artifacts under {run_dir}/evals")
    return read_json(evals[-1]), [read_json(p) for p in evals]


def analyze(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    cfg = read_json(run_dir / "config.json")
    final_eval, all_evals = _final_eval(run_dir)
    s = final_eval["sets"]

    per_cell = {}
    for set_name, block in s.items():
        for tid, t in block["tasks"].items():
            per_cell[tid] = {"set": set_name, "level": t["level"],
                             "acc_hard": t["acc_hard"], "acc_soft": t["acc_soft"],
                             "closed_map_error": t["closed_map"]["error"],
                             "closed_map_final_dist_to_target":
                                 t["closed_map"]["final_dist_to_target"]}

    curve = [{
        "step": e["step"],
        "seen_hard": e["sets"]["seen_heldout"]["mean_acc_hard"],
        "L1_hard": e["sets"]["unseen_L1"]["mean_acc_hard"],
        "L2_hard": e["sets"]["unseen_L2"]["mean_acc_hard"],
        "L3_hard": e["sets"]["unseen_L3"]["mean_acc_hard"],
        "closed_map_seen": e["sets"]["seen_heldout"]["mean_closed_map_error"],
        "closed_map_unseen_L1": e["sets"]["unseen_L1"]["mean_closed_map_error"],
        "closed_map_target_seen": e["sets"]["seen_heldout"][
            "mean_closed_map_final_dist_to_target"],
        "closed_map_target_unseen_L1": e["sets"]["unseen_L1"][
            "mean_closed_map_final_dist_to_target"],
        "atoms_in_use": e["routing"]["atoms_in_use"],
        "pass_rate": e["routing"]["pass_rate"],
        "steps_per_token": e["routing"]["steps_per_token"],
    } for e in all_evals]

    metrics = {
        "arm": cfg["arm"], "seed": cfg["seed"], "experiment": cfg["experiment"],
        "smoke": cfg.get("smoke", False),
        "protocol_revision": cfg.get("protocol_revision", "pre-revision"),
        "final_step": final_eval["step"],
        # Q1
        "acc_seen_hard": s["seen_heldout"]["mean_acc_hard"],
        "acc_seen_soft": s["seen_heldout"]["mean_acc_soft"],
        "acc_unseen_L1_hard": s["unseen_L1"]["mean_acc_hard"],
        "acc_unseen_L2_hard": s["unseen_L2"]["mean_acc_hard"],
        "acc_unseen_L3_hard": s["unseen_L3"]["mean_acc_hard"],
        "acc_unseen_L1_soft": s["unseen_L1"]["mean_acc_soft"],
        "acc_unseen_L2_soft": s["unseen_L2"]["mean_acc_soft"],
        "acc_unseen_L3_soft": s["unseen_L3"]["mean_acc_soft"],
        "dissociation_gap_hard": final_eval["dissociation_gap_hard"],
        "soft_hard_gap_seen": s["seen_heldout"]["soft_hard_gap"],
        # Q2
        "closed_map_seen": s["seen_heldout"]["mean_closed_map_error"],
        "closed_map_unseen_L1": s["unseen_L1"]["mean_closed_map_error"],
        "closed_map_unseen_L3": s["unseen_L3"]["mean_closed_map_error"],
        "closed_map_target_seen": s["seen_heldout"][
            "mean_closed_map_final_dist_to_target"],
        "closed_map_target_unseen_L1": s["unseen_L1"][
            "mean_closed_map_final_dist_to_target"],
        "closed_map_target_unseen_L3": s["unseen_L3"][
            "mean_closed_map_final_dist_to_target"],
        # per-cell + curve
        "per_cell": per_cell,
        "curve": curve,
    }

    # Q3-Q5 from the final panel (if it has been run).
    panel_dir = run_dir / "panel" / "final"
    if panel_dir.exists():
        cen = read_json(panel_dir / "census.json")
        metrics["census_atoms_in_use"] = cen["atoms_in_use"]
        metrics["census_steps_per_token"] = cen["steps_per_token"]
        metrics["census_pass_rate"] = cen["pass_rate"]

        abl = read_json(panel_dir / "ablation.json")
        cvs = [a["damage_cv"] for a in abl["per_atom"].values()
               if a.get("damage_cv") is not None]
        metrics["ablation_cv_median"] = float(np.median(cvs)) if cvs else None
        metrics["ablation_cv_per_atom"] = {k: a.get("damage_cv")
                                           for k, a in abl["per_atom"].items()}

        sta = read_json(panel_dir / "standalone.json")
        in_use = cen["in_use_mask"]
        best_in_use = [sta[str(i)]["best_acc"] for i in range(R.N_ATOMS)
                       if in_use[i]]
        metrics["standalone_best_acc_mean_in_use"] = (
            float(np.mean(best_in_use)) if best_in_use else None)
        metrics["standalone_best_candidates"] = {
            str(i): sta[str(i)]["best_candidate"] for i in range(R.N_ATOMS)}

        cma = read_json(panel_dir / "closed_map_atom.json")
        metrics["closed_map_atom_matched_error"] = cma["matched_error_mean"]
        metrics["closed_map_atom_coverage"] = cma["coverage"]

        dec = read_json(panel_dir / "decodability.json")
        for key in ("subop_from_delta", "subop_from_state", "subop_h0_floor",
                    "surface_from_delta", "surface_from_state", "surface_h0_floor"):
            metrics[f"decodability_{key}"] = dec.get(key, {}).get("score")
            metrics[f"decodability_{key}_shuffled"] = dec.get(
                key + "_shuffled", {}).get("score")

        tr = read_json(panel_dir / "transfer.json")
        stds = [v for v in tr["task_row_stats"]["row_stds"]
                if not np.isnan(v)]
        metrics["transfer_task_row_std_mean"] = float(np.mean(stds)) if stds else None
        tstds = [v for v in tr["transplant_row_stats"]["row_stds"]
                 if not np.isnan(v)]
        metrics["transfer_transplant_row_std_mean"] = (
            float(np.mean(tstds)) if tstds else None)

    metrics["param_counts"] = read_json(run_dir / "param_counts.json")
    metrics["init_calibration"] = read_json(run_dir / "init_calibration.json")
    write_json(run_dir / "metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute metrics.json from run artifacts")
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    m = analyze(Path(a.run_dir))
    print(json.dumps({k: v for k, v in m.items()
                      if k not in ("per_cell", "curve")}, indent=2, default=str))
