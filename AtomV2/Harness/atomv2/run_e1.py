"""Experiment 1 batch driver: A1..A4 (lambda_use grid) x seeds 0,1,2.

REFUSES to start unless the E0 calibration verdict exists and passed - a null
result from an uncalibrated rig is uninterpretable. Override with --force only
alongside --force-reason, which is recorded in the results directory.
"""
from __future__ import annotations

import argparse

from . import registered as R
from .aggregate import aggregate
from .analyze import analyze
from .config import E1_BATTERY_ARMS, config_for_arm, run_dir_for
from .panel import run_all_panels
from .train import train_run
from .utils import RESULTS_DIR, read_json, write_json, write_sha256sums


def _check_e0_gate(force: bool, force_reason: str | None, smoke: bool) -> None:
    verdict_path = RESULTS_DIR / ("smoke_e0" if smoke else "e0") / "e0_verdict.json"
    if verdict_path.exists():
        verdict = read_json(verdict_path)
        if (verdict.get("passed")
                and verdict.get("protocol_revision") == R.PROTOCOL_REVISION):
            return
    if not force:
        raise SystemExit(
            f"E1 refuses to start: {verdict_path} missing, not passed, or for "
            f"a protocol revision other than {R.PROTOCOL_REVISION!r}. "
            "Run the E0 battery first (python -m atomv2.run_e0). Override "
            "only with --force --force-reason '...' (recorded).")
    if not force_reason:
        raise SystemExit("--force requires --force-reason (it gets recorded).")
    write_json(RESULTS_DIR / ("smoke_e1" if smoke else "e1") / "FORCED_START.json",
               {"reason": force_reason})


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E1 battery (lambda_use sweep)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(R.SEEDS))
    ap.add_argument("--arms", nargs="*", default=list(E1_BATTERY_ARMS))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-reason", default=None)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    # Amendment R9: A1 is the same condition as E0's A0-free. Running it here
    # would spend compute reproducing an existing result and would attach two
    # arm names to one condition in the summary tables.
    if "A1" in args.arms:
        raise SystemExit(
            "A1 is not part of the E1 battery (amendment R9): it is "
            "configurationally identical to E0's A0-free arm, which has "
            "already been run. The lambda=0 row is sourced from "
            f"{R.LAMBDA_ZERO_SOURCE['experiment']}/"
            f"{R.LAMBDA_ZERO_SOURCE['arm']} and is labelled as such in the E1 "
            "summary. Re-running it would duplicate a condition under a "
            "second name.")

    batch = [(arm, seed) for arm in args.arms for seed in args.seeds]
    if args.plan:
        for arm, seed in batch:
            print(f"{arm} (lambda_use={R.LAMBDA_GRID[arm]}) seed={seed}")
        return

    _check_e0_gate(args.force, args.force_reason, args.smoke)

    for arm, seed in batch:
        cfg = config_for_arm(arm, seed, smoke=args.smoke)
        run_dir = run_dir_for(cfg)
        if (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists():
            print(f"skip (complete): {run_dir}")
            continue
        print(f"=== {arm} (lambda_use={cfg.lambda_use}) seed={seed} -> {run_dir}")
        if (run_dir / "checkpoints" / "final.pt").exists():
            print("    training already complete; resuming at panel")
        else:
            train_run(cfg, allow_dirty=args.allow_dirty)
        run_all_panels(run_dir)
        metrics = analyze(run_dir)
        write_sha256sums(run_dir)
        print(f"    seen={metrics['acc_seen_hard']:.4f} "
              f"L1={metrics['acc_unseen_L1_hard']:.4f} "
              f"L2={metrics['acc_unseen_L2_hard']:.4f} "
              f"L3={metrics['acc_unseen_L3_hard']:.4f} "
              f"census={metrics.get('census_atoms_in_use')} "
              f"pass_rate={metrics.get('census_pass_rate'):.3f}")

    print(aggregate("e1", smoke=args.smoke))


if __name__ == "__main__":
    main()
