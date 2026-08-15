"""Experiment 0 batch driver: A0-oracle + A0-free x seeds 0,1,2, sequential.

Per run: train -> panel (all registered checkpoints) -> analyze -> SHA256SUMS.
Idempotent skip: a run with metrics.json AND SHA256SUMS is complete.
Circuit-breaker (default ON): if an A0-oracle run finishes with final L1
accuracy below the registered E0_ORACLE_L1_MIN, the batch aborts - if the rig
cannot reproduce the oracle ceiling, nothing else is interpretable; debug the
harness, not the hypothesis.
"""
from __future__ import annotations

import argparse

from . import registered as R
from .aggregate import aggregate
from .analyze import analyze
from .config import E0_ARMS, config_for_arm, run_dir_for
from .panel import run_all_panels
from .train import train_run
from .utils import read_json, write_sha256sums


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E0 calibration battery")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny volumes/steps; pipeline verification only")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(R.SEEDS))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-circuit-breaker", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the batch and exit")
    args = ap.parse_args()

    batch = [(arm, seed) for arm in E0_ARMS for seed in args.seeds]
    if args.plan:
        for arm, seed in batch:
            print(f"{arm} seed={seed}")
        return

    for arm, seed in batch:
        cfg = config_for_arm(arm, seed, smoke=args.smoke)
        run_dir = run_dir_for(cfg)
        if (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists():
            print(f"skip (complete): {run_dir}")
            continue
        print(f"=== {arm} seed={seed} -> {run_dir}")
        if (run_dir / "checkpoints" / "final.pt").exists():
            print("    training already complete; resuming at panel")
        else:
            train_run(cfg, allow_dirty=args.allow_dirty)
        run_all_panels(run_dir)
        metrics = analyze(run_dir)
        write_sha256sums(run_dir)
        print(f"    seen={metrics['acc_seen_hard']:.4f} "
              f"L1={metrics['acc_unseen_L1_hard']:.4f} "
              f"L3={metrics['acc_unseen_L3_hard']:.4f} "
              f"closed_map={metrics['closed_map_seen']:.4f} "
              f"census={metrics.get('census_atoms_in_use')}")

        if (arm == "A0-oracle" and not args.no_circuit_breaker
                and not args.smoke
                and metrics["acc_unseen_L1_hard"] < R.E0_ORACLE_L1_MIN):
            raise SystemExit(
                f"CIRCUIT BREAKER: oracle L1 {metrics['acc_unseen_L1_hard']:.4f} "
                f"< {R.E0_ORACLE_L1_MIN}. The rig is broken somewhere (data "
                "generator, split, model, or metrics); nothing proceeds until "
                "it is found. Debug the harness, not the hypothesis.")

    out = aggregate("e0", smoke=args.smoke)
    verdict = read_json(out / "e0_verdict.json")
    print(f"E0 verdict: {'PASS' if verdict['passed'] else 'FAIL'}")
    for name, c in verdict["checks"].items():
        print(f"  {name}: {'ok' if c['ok'] else 'FAIL'} {c}")


if __name__ == "__main__":
    main()
