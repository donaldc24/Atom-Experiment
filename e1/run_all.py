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
from .config import ARMS, GENERATIONS, SEEDS, config_for_generation
from .train import RUNS_DIR, run_dir_for
from .utils import read_json, write_json


def already_done(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", default="v1",
                    help="which experiment generation to run (D40)")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--split-seeds", nargs="*", type=int, default=None,
                    help="default: every split seed the generation froze")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-oracle-failure", action="store_true", default=True)
    args = ap.parse_args()

    if args.generation not in GENERATIONS:
        print(f"error: unknown generation {args.generation!r}; "
              f"known: {sorted(GENERATIONS)}", file=sys.stderr)
        return 2
    split_seeds = args.split_seeds or list(GENERATIONS[args.generation]["split_seeds"])

    # Split seed is the OUTER loop: a batch that dies partway then still holds a
    # complete arm x seed block for the splits it finished, rather than a ragged
    # fragment of every split.
    plan = [(ss, a, s) for ss in split_seeds for a in args.arms for s in args.seeds]
    print(f"generation {args.generation}: {len(plan)} runs "
          f"({len(args.arms)} arms x {len(args.seeds)} seeds x "
          f"{len(split_seeds)} splits)")
    t_batch = time.time()

    for idx, (split_seed, arm, seed) in enumerate(plan, start=1):
        cfg = config_for_generation(args.generation, arm, seed, split_seed)
        run_dir = run_dir_for(cfg)
        if already_done(run_dir) and not args.force:
            print(f"[{idx}/{len(plan)}] skip {run_dir.name} (already complete)")
            continue

        cmd = [sys.executable, "-m", "e1.train", "--arm", arm, "--seed", str(seed),
               "--generation", args.generation, "--split-seed", str(split_seed)]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.threads is not None:
            cmd += ["--threads", str(args.threads)]
        t0 = time.time()
        print(f"[{idx}/{len(plan)}] {arm} seed {seed} split {split_seed} ...",
              flush=True)
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
