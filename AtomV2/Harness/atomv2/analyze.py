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
        # AMENDMENT C1: the sub-op keys are renamed to say what they measure
        # (task-identity leakage). The legacy decodability_subop_* keys are
        # still emitted as deprecated aliases for one release so older
        # summaries keep parsing; they carry identical values by construction.
        for key in ("leakage_subop_identity_from_delta",
                    "leakage_subop_identity_from_state",
                    "leakage_subop_identity_h0_floor",
                    "surface_from_delta", "surface_from_state",
                    "surface_h0_floor"):
            prefix = "" if key.startswith("leakage_") else "decodability_"
            metrics[f"{prefix}{key}"] = dec.get(key, {}).get("score")
            metrics[f"{prefix}{key}_shuffled"] = dec.get(
                key + "_shuffled", {}).get("score")
        for legacy, current in dec.get("deprecated_aliases", {}).items():
            metrics[legacy] = metrics.get(current)
            metrics[f"{legacy}_shuffled"] = metrics.get(f"{current}_shuffled")
        metrics["deprecated_alias_note"] = (
            "decodability_subop_* are DEPRECATED aliases of "
            "leakage_subop_identity_* (amendment C1): the probe measures "
            "task-identity leakage into deltas, not sub-op localization. "
            "Removed after one release; use leakage_subop_identity_* and, for "
            "granularity, probe_transfer_subop_*.")

        tr = read_json(panel_dir / "transfer.json")
        stds = [v for v in tr["task_row_stats"]["row_stds"]
                if not np.isnan(v)]
        metrics["transfer_task_row_std_mean"] = float(np.mean(stds)) if stds else None
        tstds = [v for v in tr["transplant_row_stats"]["row_stds"]
                 if not np.isnan(v)]
        metrics["transfer_transplant_row_std_mean"] = (
            float(np.mean(tstds)) if tstds else None)
        # AMENDMENT C4: decomposition of the conflated row-std.
        vd = tr.get("variance_decomposition")
        if vd:
            metrics["transfer_partner_variance"] = vd["partner_variance_mean"]
            metrics["transfer_input_variance"] = vd["input_variance_mean"]
            metrics["transfer_row_std_legacy"] = \
                metrics["transfer_transplant_row_std_mean"]

        # AMENDMENT C2: transfer-split sub-op probes (granularity instrument).
        tsp_path = panel_dir / "transfer_split_subop.json"
        if tsp_path.exists():
            tsp = read_json(tsp_path)
            metrics["probe_transfer_subop_mean"] = tsp.get("mean_across_subops")
            metrics["probe_transfer_subop_mean_taskscope"] = tsp.get(
                "mean_across_subops_taskscope")
            for sub, entry in tsp.get("per_subop", {}).items():
                metrics[f"probe_transfer_subop_{sub}_acc"] = \
                    entry.get("mean_balanced_acc")

        # AMENDMENT C3: canonical substitution.
        canon_path = panel_dir / "canonical_substitution.json"
        if canon_path.exists():
            canon = read_json(canon_path)
            metrics["canon_variants_agree"] = canon.get("all_variants_agree")
            for level, block in canon.get("per_level", {}).items():
                for k in ("canon_route_agree_hard", "canon_route_kl",
                          "canon_repair_acc", "canon_repair_delta"):
                    for tag in ("", "_alt"):
                        metrics[f"{k}{tag}_{level}"] = block.get(k + tag)
                metrics[f"canon_baseline_acc_{level}"] = block.get("baseline_acc")

    # E1b liveness artifacts -> validity gates + telemetry headlines
    # (H1-E1bExperiment.md). Evaluated here, from disk, never in the loop.
    lv_dir = run_dir / "liveness"
    if lv_dir.exists():
        metrics.update(_liveness_metrics(lv_dir))

    # E2 noise telemetry + registered robustness sweep (H1-Experiment2.md).
    nt_dir = run_dir / "noise_telemetry"
    if nt_dir.exists():
        metrics.update(_noise_metrics(nt_dir))
    rob_path = run_dir / "noise_robustness.json"
    if rob_path.exists():
        metrics.update(_robustness_metrics(read_json(rob_path)))

    # E3 sandbox telemetry (H1-Experiment3.md).
    st_dir = run_dir / "sandbox_telemetry"
    if st_dir.exists():
        metrics.update(_sandbox_metrics(st_dir))

    # E4 (H1-Experiment4.md): the registered 20k like-for-like checkpoint
    # block (never mixed with 30k finals) + the per-P3-cell dax-crack check.
    if cfg.get("experiment") == "e4":
        metrics.update(_e4_metrics(run_dir, metrics))

    # E5 (H1-Experiment5.md): identical registered measurements (20k
    # like-for-like block, dax-crack check) under the e5_ prefix, plus the
    # two producer telemetry rows (output variance, branch-READ spread).
    if cfg.get("experiment") == "e5":
        metrics.update(_e4_metrics(run_dir, metrics, prefix="e5"))
        pt_dir = run_dir / "producer_telemetry"
        if pt_dir.exists():
            metrics.update(_producer_metrics(pt_dir))

    metrics["param_counts"] = read_json(run_dir / "param_counts.json")
    metrics["init_calibration"] = read_json(run_dir / "init_calibration.json")
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def _e4_metrics(run_dir: Path, metrics: dict, prefix: str = "e4") -> dict:
    """E4 dual-budget reporting + the dax-crack check (H1-Experiment4.md).

    The paired comparison against A6/A9/A12 references happens at the 20k
    checkpoint (like for like); 30k finals stay in the ordinary headline
    keys. Dax crack: any P3 cell's raw hard accuracy at or above the
    registered threshold, evaluated at both budgets, named either way.
    E5 reuses these measurement definitions unchanged under prefix='e5'
    (its references are the E4 arms, compared like for like at 20k).
    """
    out = {}

    def _dax(cells: dict) -> dict:
        l3 = {tid: c["acc_hard"] for tid, c in cells.items()
              if c.get("level") == "L3" or c.get("set") == "unseen_L3"}
        if not l3:
            return {}
        best = max(l3, key=l3.get)
        return {
            "per_cell": l3,
            "n_cells": len(l3),
            "max_cell_acc": l3[best],
            "max_cell": best,
            "cells_at_threshold": sorted(
                t for t, a in l3.items() if a >= R.E4_DAX_CRACK_THRESHOLD),
            "crack": bool(l3[best] >= R.E4_DAX_CRACK_THRESHOLD),
        }

    dax_final = _dax(metrics.get("per_cell", {}))
    if dax_final:
        out[f"{prefix}_dax_final"] = dax_final
        out[f"{prefix}_dax_crack"] = dax_final["crack"]
        out[f"{prefix}_dax_max_cell_acc"] = dax_final["max_cell_acc"]
        out[f"{prefix}_dax_cells_at_threshold"] = len(
            dax_final["cells_at_threshold"])

    ckpt_eval = run_dir / "evals" / "step020000.json"
    if ckpt_eval.exists():
        e = read_json(ckpt_eval)
        s = e["sets"]
        out.update({
            f"{prefix}_acc_seen_hard_20k": s["seen_heldout"]["mean_acc_hard"],
            f"{prefix}_acc_unseen_L1_hard_20k": s["unseen_L1"][
                "mean_acc_hard"],
            f"{prefix}_acc_unseen_L2_hard_20k": s["unseen_L2"][
                "mean_acc_hard"],
            f"{prefix}_acc_unseen_L3_hard_20k": s["unseen_L3"][
                "mean_acc_hard"],
            f"{prefix}_closed_map_seen_20k": s["seen_heldout"][
                "mean_closed_map_error"],
            f"{prefix}_closed_map_target_L3_20k": s["unseen_L3"][
                "mean_closed_map_final_dist_to_target"],
        })
        cells_20k = {}
        for set_name, block in s.items():
            for tid, t in block["tasks"].items():
                cells_20k[tid] = {"set": set_name, "level": t["level"],
                                  "acc_hard": t["acc_hard"]}
        dax_20k = _dax(cells_20k)
        if dax_20k:
            out[f"{prefix}_dax_20k"] = dax_20k
            out[f"{prefix}_dax_crack_20k"] = dax_20k["crack"]
            out[f"{prefix}_dax_max_cell_acc_20k"] = dax_20k["max_cell_acc"]

    canon_20k = run_dir / "panel" / "step020000" / "canonical_substitution.json"
    if canon_20k.exists():
        canon = read_json(canon_20k)
        for level, block in canon.get("per_level", {}).items():
            out[f"{prefix}_canon_repair_acc_{level}_20k"] = block.get(
                "canon_repair_acc")
            out[f"{prefix}_canon_repair_delta_{level}_20k"] = block.get(
                "canon_repair_delta")
            out[f"{prefix}_canon_baseline_acc_{level}_20k"] = block.get(
                "baseline_acc")
    return out


def _producer_metrics(pt_dir: Path) -> dict:
    """E5 producer telemetry -> headline numbers. Descriptive only: the
    producer carries no validity gate of its own (the E1b liveness and E2
    cosine gates still apply to every E5 run); the variance row is the
    registered gaming-audit instrument, the branch-READ spread the
    registered free metric."""
    records = [read_json(p) for p in sorted(pt_dir.glob("step*.json"))]
    if not records:
        return {}
    final = records[-1]
    ov, br = final["output_variance"], final["branch_read"]
    return {
        "e5_lambda_producer": final["lambda_producer"],
        "e5_producer_variance_per_atom_final": ov["per_atom"],
        "e5_producer_variance_min_final": ov["min"],
        "e5_producer_variance_mean_final": ov["mean"],
        "e5_producer_variance_z0_final": ov["z0_reference"],
        "e5_branch_read_mean_final": br["read_mean"],
        "e5_branch_read_spread_mean_final": br["spread_mean"],
        # Collapse trajectory: the variance-min curve, so a mid-training
        # collapse-and-recover cannot hide in the final snapshot.
        "e5_producer_variance_min_curve": [
            {"step": r["step"], "min": r["output_variance"]["min"]}
            for r in records],
    }


def _liveness_metrics(lv_dir: Path) -> dict:
    """Summarize E1b liveness telemetry and evaluate the registered gates.

    Implementation invariant: every eval's max base probability must be at or
    below the arm's p_max + tolerance. Deafness rule: within (2k, 18k), two
    CONSECUTIVE scheduled evals with BOTH median router task-gradient norm
    < 1e-8 AND median router/atom ratio < 1e-3 invalidate the run. Raw norm
    growth beyond 10x the step-1k median for two consecutive evals is a
    non-invalidating audit flag.
    """
    records = [read_json(p) for p in sorted(lv_dir.glob("step*.json"))]
    if not records:
        return {}
    lo, hi = R.E1B_DEAF_WINDOW

    def _deaf(rec) -> bool:
        sig = rec["learning_signal"]
        g = sig["router_total"]["median"]
        ratio = sig["router_atom_ratio"]["median"]
        return (g is not None and g < R.E1B_DEAF_GRAD_NORM
                and ratio is not None and ratio < R.E1B_DEAF_RATIO)

    window = [r for r in records if lo <= r["step"] <= hi]
    deaf_step = None
    for a, b in zip(window, window[1:]):
        if _deaf(a) and _deaf(b):
            deaf_step = b["step"]
            break

    ref = next((r for r in records if r["step"] >= 1000), records[0])
    ref_q = ref["base_geometry"]["raw_query_norm"]["p50"]
    ref_k = ref["base_geometry"]["raw_key_norm"]["p50"]
    growth_step = None
    flags = [(r["step"],
              ref_q > 0 and r["base_geometry"]["raw_query_norm"]["p50"]
              > R.E1B_NORM_GROWTH_FLAG * ref_q
              or ref_k > 0 and r["base_geometry"]["raw_key_norm"]["p50"]
              > R.E1B_NORM_GROWTH_FLAG * ref_k) for r in records]
    for (s1, f1), (s2, f2) in zip(flags, flags[1:]):
        if f1 and f2:
            growth_step = s2
            break

    grads = [r["learning_signal"]["router_total"]["median"] for r in window
             if r["learning_signal"]["router_total"]["median"] is not None]
    ratios = [r["learning_signal"]["router_atom_ratio"]["median"] for r in window
              if r["learning_signal"]["router_atom_ratio"]["median"] is not None]
    final = records[-1]
    prog = final.get("deterministic_programs", {})
    out = {
        "e1b_pmax_observed_max": max(
            r["base_geometry"]["max_base_prob_observed"] for r in records),
        "e1b_pmax_invariant_ok": all(
            r["base_geometry"]["pmax_invariant_ok"] for r in records),
        "e1b_deafness_violated": deaf_step is not None,
        "e1b_deafness_step": deaf_step,
        "e1b_run_valid": deaf_step is None and all(
            r["base_geometry"]["pmax_invariant_ok"] for r in records),
        "e1b_norm_growth_flagged": growth_step is not None,
        "e1b_norm_growth_step": growth_step,
        "e1b_router_grad_median_min_window": (
            float(np.min(grads)) if grads else None),
        "e1b_router_atom_ratio_median_min_window": (
            float(np.min(ratios)) if ratios else None),
        "e1b_frac_base_prob_above_0999_final": final["base_geometry"][
            "frac_above_0999"],
        "e1b_programs_per_task_mean": prog.get("programs_per_task_mean"),
        "e1b_routing_entropy_nats_mean": prog.get("routing_entropy_nats_mean"),
        "e1b_stochastic_unique_seqs_mean": final["stochastic_diversity"][
            "unique_sequences_per_input_mean"],
        "e1b_stochastic_disagreement_rate": final["stochastic_diversity"][
            "route_disagreement_rate"],
    }
    return out


def _noise_metrics(nt_dir: Path) -> dict:
    """E2 noise telemetry -> headline numbers + the cosine implementation
    gate (observed median clean/transmitted cosine within E2_COSINE_TOL of
    the arm's registered target, at every eval)."""
    records = [read_json(p) for p in sorted(nt_dir.glob("step*.json"))]
    if not records:
        return {}
    final = records[-1]
    return {
        "e2_state_noise_sigma": final["state_noise_sigma"],
        "e2_target_cosine": final["target_cosine"],
        "e2_cosine_observed_final": final["handoff"]["cosine"]["p50"],
        "e2_cosine_gate_ok": all(r["handoff"]["cosine_within_tol"]
                                 for r in records),
        "e2_route_flip_rate_final": final["route_flip_rate"],
        "e2_pred_disagreement_final": final["pred_disagreement_mean"],
        "e2_transmitted_pos_mean_abs_final": final["handoff"][
            "transmitted_pos_mean_abs"],
        "e2_transmitted_pos_var_final": final["handoff"][
            "transmitted_pos_var"],
        "e2_closed_map_producer_final": final["closed_map_noisy_forward"][
            "producer_error_mean"],
        "e2_closed_map_transmitted_final": final["closed_map_noisy_forward"][
            "transmitted_error_mean"],
        "e2_closed_map_producer_target_final": final[
            "closed_map_noisy_forward"]["producer_target_dist_mean"],
    }


def _sandbox_metrics(st_dir: Path) -> dict:
    """E3 sandbox telemetry -> headline numbers. Descriptive only: the
    sandbox carries no validity gate of its own (the E1b liveness gates still
    apply to every E3 run); optimization failure under a correctly
    implemented sandbox is a result, not an invalid run."""
    records = [read_json(p) for p in sorted(st_dir.glob("step*.json"))]
    if not records:
        return {}
    final = records[-1]
    return {
        "e3_lambda_sandbox_valid": final["lambda_sandbox_valid"],
        "e3_lambda_sandbox_unique": final["lambda_sandbox_unique"],
        "e3_usage_weighted_atoms_final": final["usage"]["n_weighted"],
        "e3_usage_ema_final": final["usage"]["ema"],
        "e3_standalone_read_final": final["standalone"]["read_mean_weighted"],
        "e3_standalone_cycle_final": final["standalone"][
            "cycle_mean_weighted"],
        "e3_standalone_read_all_final": final["standalone"]["read_mean"],
        "e3_closure_read_final": final["closure"]["read_mean"],
        "e3_closure_cycle_final": final["closure"]["cycle_mean"],
        "e3_unique_pair_dist_mean_final": final["uniqueness"][
            "pair_dist_mean_weighted"],
        "e3_unique_pair_dist_min_final": final["uniqueness"][
            "pair_dist_min_weighted"],
        "e3_nonidentity_dist_min_final": final["uniqueness"][
            "pass_dist_min_weighted"],
        "e3_margin_satisfied_frac_final": final["uniqueness"][
            "margin_satisfied_frac_weighted"],
    }


def _robustness_metrics(rob: dict) -> dict:
    """Flatten the registered sweep: seen + L1 accuracy (plus flip/
    disagreement on seen) at each target cosine."""
    out = {}
    for c in rob.get("target_cosines", []):
        key = f"{c:.3f}"
        tag = f"c{key.replace('.', '')}"
        seen = rob["sets"]["seen_heldout"][key]
        out[f"e2_robust_seen_acc_{tag}"] = seen["mean_acc"]
        out[f"e2_robust_seen_flip_{tag}"] = seen["route_flip_rate"]
        out[f"e2_robust_seen_disagree_{tag}"] = seen["pred_disagreement"]
        out[f"e2_robust_L1_acc_{tag}"] = \
            rob["sets"]["unseen_L1"][key]["mean_acc"]
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute metrics.json from run artifacts")
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    m = analyze(Path(a.run_dir))
    print(json.dumps({k: v for k, v in m.items()
                      if k not in ("per_cell", "curve")}, indent=2, default=str))
