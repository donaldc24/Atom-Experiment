"""Sequential batch driver: 5 arms x 5 seeds = 25 runs, one at a time.

Parallel runs contend for the same cores and risk thermal throttling, so this is
strictly sequential (spec 3). Already-completed runs are skipped unless --force.

    python -m e1.run_all                 # everything
    python -m e1.run_all --arms A0       # just the oracle gate
    python -m e1.run_all --seeds 0 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .analyze import analyze
from .config import ARMS, SEEDS
from .train import RUNS_DIR, run_id_for
from .utils import read_json, write_json


def already_done(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-oracle-failure", action="store_true", default=True)
    args = ap.parse_args()

    plan = [(a, s) for a in args.arms for s in args.seeds]
    print(f"planned runs: {len(plan)}")
    t_batch = time.time()

    for idx, (arm, seed) in enumerate(plan, start=1):
        run_dir = RUNS_DIR / run_id_for(arm, seed)
        if already_done(run_dir) and not args.force:
            print(f"[{idx}/{len(plan)}] skip {run_dir.name} (already complete)")
            continue

        cmd = [sys.executable, "-m", "e1.train", "--arm", arm, "--seed", str(seed)]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        t0 = time.time()
        print(f"[{idx}/{len(plan)}] {arm} seed {seed} ...", flush=True)
        # A separate process per run keeps thread pinning and RNG state clean.
        proc = subprocess.run(cmd, cwd=str(RUNS_DIR.parent))
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})")
            return proc.returncode

        metrics = analyze(run_dir)
        write_json(run_dir / "metrics.json", metrics)
        dt = time.time() - t0
        print(f"  done in {dt / 60:.1f} min  "
              f"acc_unseen={metrics['M1_acc_unseen']:.4f}  "
              f"acc_seen={metrics['M1_acc_seen']:.4f}  "
              f"M3_align={metrics['M3_align']:.3f}  "
              f"dead={metrics['M2_dead']}", flush=True)

        # T1 in its substituted operational form (DECISIONS.md D18): the gate asks
        # whether a fully composing solution EXISTS in this architecture, which is
        # measured directly by teacher-forced composition, not by A0's end-to-end
        # accuracy. A0's `acc_unseen` is reported as an observation, not a gate.
        if arm == "A0" and args.stop_on_oracle_failure and \
                metrics["M7_acc_teacher_forced"] < 0.99:
            print("\nT1 GATE FAILED: teacher-forced composition below 0.99 on the oracle.")
            print("No compositional solution is reachable here; every other arm's")
            print("failure would be uninterpretable. Stopping.")
            return 1

    print(f"\nbatch complete in {(time.time() - t_batch) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
