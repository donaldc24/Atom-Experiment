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
    """Amendment R11: E0 no longer BLOCKS E1, but its verdict is recorded here.

    Ungating is not the same as forgetting. E0 still ran, still produced a
    verdict, and that verdict is copied next to the E1 results so no one reads
    an E1 number without the calibration context attached. If E0 was never run
    at all, that IS still a hard stop: the instrument would be unvalidated.
    """
    exp = "smoke_e0" if smoke else "e0"
    verdict_path = RESULTS_DIR / exp / "e0_verdict.json"
    out = RESULTS_DIR / ("smoke_e1" if smoke else "e1")

    if not verdict_path.exists():
        raise SystemExit(
            f"E1 refuses to start: {verdict_path} does not exist. R11 removed "
            "the requirement that E0 PASS, not the requirement that it RAN - "
            "without a calibration verdict the instrument is unvalidated and "
            "an E1 result is uninterpretable. Run python -m atomv2.run_e0.")

    verdict = read_json(verdict_path)
    revision_ok = verdict.get("protocol_revision") == R.PROTOCOL_REVISION
    if not revision_ok and not force:
        raise SystemExit(
            f"E1 refuses to start: the E0 verdict is for protocol revision "
            f"{verdict.get('protocol_revision')!r}, not "
            f"{R.PROTOCOL_REVISION!r}. Revision mismatch is still a hard stop "
            "under R11. Override with --force --force-reason '...'.")

    if R.E1_REQUIRES_E0_PASS and not verdict.get("passed") and not force:
        raise SystemExit(
            "E1 refuses to start: E0 did not pass and E1_REQUIRES_E0_PASS is "
            "set. Override with --force --force-reason '...' (recorded).")

    failed = [k for k, c in verdict.get("checks", {}).items() if not c.get("ok")]
    write_json(out / "e0_context.json", {
        "amendment": "R11: E1 is not gated on the E0 verdict",
        "e0_passed": verdict.get("passed"),
        "e0_protocol_revision": verdict.get("protocol_revision"),
        "e0_failed_checks": failed,
        "e0_checks": verdict.get("checks"),
        "e0_instrument_audit": verdict.get("instrument_audit"),
        "note": "Every E1 number must be reported alongside this context. E0's "
                "instrument audit passed; its failing checks concern the free "
                "arm's absolute L1 level, which is a finding about this "
                "architecture rather than an instrument fault (see R11).",
    })
    if not verdict.get("passed"):
        print(f"NOTE: E0 did not pass (failed: {', '.join(failed)}). "
              "Proceeding under amendment R11; context recorded to "
              f"{out / 'e0_context.json'}")
    if force and force_reason:
        write_json(out / "FORCED_START.json", {"reason": force_reason})


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
