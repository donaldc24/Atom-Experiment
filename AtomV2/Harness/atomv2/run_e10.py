"""Staged E10 driver for the Reynolds Transport Network experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import registered as R
from .e10 import config_for_e10, run_dir_for_e10, train_e10
from .utils import RESULTS_DIR, read_json, write_json


def base_score(metrics: dict) -> dict:
    values = {
        "seen": metrics["acc_seen_hard"],
        "L1": metrics["acc_unseen_L1_hard"],
        "L2": metrics["acc_unseen_L2_hard"],
        "L3": metrics["acc_unseen_L3_hard"],
        "triple": metrics["acc_extrap_triple_hard"],
        "quad": metrics["acc_extrap_quad_hard"],
    }
    checks = {name: values[name] >= threshold
              for name, threshold in R.E10_BASE_GATES.items()}
    return {"values": values, "checks": checks, "passes": all(checks.values())}


def _long_horizon(metrics: dict) -> float:
    return 0.5 * (metrics["acc_extrap_triple_hard"]
                  + metrics["acc_extrap_quad_hard"])


def screen_verdict(by_arm: dict[str, dict]) -> dict:
    required = set(R.E10_ARMS)
    if set(by_arm) != required:
        raise ValueError(f"screen requires {sorted(required)}, got {sorted(by_arm)}")
    base = {arm: base_score(by_arm[arm]) for arm in R.E10_ARMS}
    conservation = {
        arm: by_arm[arm]["mass_conservation_relative_max"]
        <= R.E10_CONSERVATION_MAX for arm in R.E10_ARMS}
    a29_long = _long_horizon(by_arm["A29"])
    a30_long = _long_horizon(by_arm["A30"])
    a31_long = _long_horizon(by_arm["A31"])
    a30_matched = a30_long >= a29_long - R.E10_MATCHED_LONG_HORIZON_TOL
    reynolds_pass = base["A30"]["passes"] and conservation["A30"] \
        and a30_matched
    noise_gain = (by_arm["A31"]["acc_pair_noisy_hard"]
                  - by_arm["A30"]["acc_pair_noisy_hard"])
    viscosity_clean_ok = (
        a31_long >= a30_long - R.E10_VISCOSITY_CLEAN_TOL)
    viscosity_pass = (base["A31"]["passes"] and conservation["A31"]
                      and noise_gain >= R.E10_VISCOSITY_NOISE_MARGIN
                      and viscosity_clean_ok)

    if base["A28"]["passes"]:
        outcome, winner = "LOCAL_REACTION_SUFFICIENT", None
    elif viscosity_pass:
        outcome, winner = "VISCOUS_REYNOLDS_FLOW", "A31"
    elif reynolds_pass:
        outcome, winner = "INVISCID_REYNOLDS_FLOW", "A30"
    elif base["A31"]["passes"] and conservation["A31"]:
        outcome, winner = "STRUCTURED_FLOW_ONLY_WITH_VISCOSITY", "A31"
    elif base["A29"]["passes"]:
        outcome, winner = "DIRECT_TRANSPORT_ONLY", None
    else:
        outcome, winner = "NO_FLOW_SUCCESS", None

    return {
        "outcome": outcome,
        "registered_winner": winner,
        "base": base,
        "conservation": {
            arm: {"value": by_arm[arm]["mass_conservation_relative_max"],
                  "threshold": R.E10_CONSERVATION_MAX,
                  "passes": conservation[arm]} for arm in R.E10_ARMS},
        "reynolds_contrast": {
            "A29_long_horizon_mean": a29_long,
            "A30_long_horizon_mean": a30_long,
            "tolerance": R.E10_MATCHED_LONG_HORIZON_TOL,
            "matched_clean_performance": a30_matched,
            "passes": reynolds_pass,
        },
        "viscosity_contrast": {
            "A30_noisy_pair": by_arm["A30"]["acc_pair_noisy_hard"],
            "A31_noisy_pair": by_arm["A31"]["acc_pair_noisy_hard"],
            "noise_gain": noise_gain,
            "required_gain": R.E10_VISCOSITY_NOISE_MARGIN,
            "A30_long_horizon_mean": a30_long,
            "A31_long_horizon_mean": a31_long,
            "clean_tolerance": R.E10_VISCOSITY_CLEAN_TOL,
            "clean_tolerance_passes": viscosity_clean_ok,
            "passes": viscosity_pass,
        },
        "interpretation_guard": (
            "A positive result concerns a discrete neural transport operator; "
            "it is not a Navier-Stokes solver or theorem transfer."),
    }


def replication_verdict(arm: str, metrics: list[dict]) -> dict:
    if arm not in ("A30", "A31") or len(metrics) != 3:
        raise ValueError("replication requires A30/A31 at exactly three seeds")
    scored = {str(m["seed"]): base_score(m) for m in metrics}
    base_passes = sum(v["passes"] for v in scored.values())
    conservation = {str(m["seed"]):
                    m["mass_conservation_relative_max"]
                    <= R.E10_CONSERVATION_MAX for m in metrics}
    return {
        "arm": arm,
        "seeds": sorted(m["seed"] for m in metrics),
        "base_pass_count": base_passes,
        "base_pass_required": 2,
        "all_conservation_pass": all(conservation.values()),
        "per_seed_base": scored,
        "per_seed_conservation": conservation,
        "claim_holds": base_passes >= 2 and all(conservation.values()),
    }


def _summary_markdown(rows: list[dict]) -> str:
    head = ("| arm | seed | seen | L1 | L2 | L3 | triple | quad | "
            "pair noisy | mass residual | params |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    body = ""
    for m in sorted(rows, key=lambda x: (x["arm"], x["seed"])):
        body += (f"| {m['arm']} | {m['seed']} | {m['acc_seen_hard']:.4f} | "
                 f"{m['acc_unseen_L1_hard']:.4f} | "
                 f"{m['acc_unseen_L2_hard']:.4f} | "
                 f"{m['acc_unseen_L3_hard']:.4f} | "
                 f"{m['acc_extrap_triple_hard']:.4f} | "
                 f"{m['acc_extrap_quad_hard']:.4f} | "
                 f"{m['acc_pair_noisy_hard']:.4f} | "
                 f"{m['mass_conservation_relative_max']:.3e} | "
                 f"{m['param_counts']['total']} |\n")
    return "# E10 Reynolds Transport Network\n\n" + head + body


def _run_one(arm: str, seed: int, smoke: bool,
             allow_dirty: bool) -> tuple[Path, dict]:
    cfg = config_for_e10(arm, seed, smoke=smoke)
    expected = run_dir_for_e10(cfg)
    if (expected / "metrics.json").exists() and \
            (expected / "SHA256SUMS").exists():
        print(f"=== {arm} seed={seed}: complete, reusing {expected}")
        run_dir = expected
    else:
        print(f"=== {arm} seed={seed} transport={cfg.transport} "
              f"nu={cfg.viscosity} steps={cfg.total_steps} -> {expected}")
        run_dir = train_e10(cfg, allow_dirty=allow_dirty)
    metrics = read_json(run_dir / "metrics.json")
    print(f"    seen={metrics['acc_seen_hard']:.4f} "
          f"L1={metrics['acc_unseen_L1_hard']:.4f} "
          f"L2={metrics['acc_unseen_L2_hard']:.4f} "
          f"L3={metrics['acc_unseen_L3_hard']:.4f} "
          f"T3={metrics['acc_extrap_triple_hard']:.4f} "
          f"T4={metrics['acc_extrap_quad_hard']:.4f} "
          f"noise={metrics['acc_pair_noisy_hard']:.4f} "
          f"mass={metrics['mass_conservation_relative_max']:.2e}")
    return run_dir, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run E10 Reynolds transport")
    ap.add_argument("--stage", choices=("smoke", "screen", "replicate"),
                    required=True)
    ap.add_argument("--arms", nargs="*", choices=R.E10_ARMS)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if args.stage == "smoke":
        batch = [(arm, 0) for arm in R.E10_ARMS]
        result_dir = RESULTS_DIR / "smoke_e10"
    elif args.stage == "screen":
        batch = [(arm, R.E10_SCREEN_SEED) for arm in R.E10_ARMS]
        result_dir = RESULTS_DIR / "e10"
    else:
        if not args.arms or len(args.arms) != 1 or \
                args.arms[0] not in ("A30", "A31"):
            raise SystemExit("replicate requires --arms A30 or --arms A31")
        arm = args.arms[0]
        batch = [(arm, seed) for seed in R.E10_REPLICATE_SEEDS]
        result_dir = RESULTS_DIR / "e10"

    result_dir.mkdir(parents=True, exist_ok=True)
    current_rows = []
    for arm, seed in batch:
        _, metrics = _run_one(arm, seed, args.stage == "smoke",
                              args.allow_dirty)
        current_rows.append(metrics)

    if args.stage in ("smoke", "screen"):
        by_arm = {m["arm"]: m for m in current_rows}
        verdict = screen_verdict(by_arm)
        filename = ("e10_smoke_verdict.json" if args.stage == "smoke"
                    else "e10_screen_verdict.json")
        write_json(result_dir / filename, verdict)
        print(f"screen verdict: {verdict['outcome']}"
              + (f"; replicate {verdict['registered_winner']}"
                 if verdict["registered_winner"] else "; stop"))
    else:
        # Include seed 1 from the completed screen in the replication verdict.
        screen_cfg = config_for_e10(arm, R.E10_SCREEN_SEED, smoke=False)
        screen_path = run_dir_for_e10(screen_cfg) / "metrics.json"
        if not screen_path.exists():
            raise SystemExit(f"missing seed-1 screen run {screen_path}")
        all_metrics = [read_json(screen_path)] + current_rows
        verdict = replication_verdict(arm, all_metrics)
        write_json(result_dir / "e10_replication_verdict.json", verdict)
        current_rows = all_metrics
        print("replication verdict: "
              + ("CLAIM HOLDS" if verdict["claim_holds"] else "CLAIM FAILS"))

    # Merge any prior result rows by (arm, seed) without touching run data.
    per_run_path = result_dir / "per_run.json"
    prior = read_json(per_run_path) if per_run_path.exists() else []
    merged = {(m["arm"], m["seed"]): m for m in prior}
    merged.update({(m["arm"], m["seed"]): m for m in current_rows})
    rows = list(merged.values())
    write_json(per_run_path, rows)
    (result_dir / "summary.md").write_text(
        _summary_markdown(rows), encoding="utf-8", newline="\n")
    print(result_dir)


if __name__ == "__main__":
    main()

