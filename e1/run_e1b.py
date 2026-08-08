"""E1b manifold ladder: sequential runner.

Does a self-supervised on-manifold constraint substitute for A0's ground truth?
A0's intermediate-state supervision did two things at once -- kept h_t on the encoder
manifold, and said WHICH point on it (i.e. supplied the decomposition). E1b supplies
only the first, using constraints that need no ground truth.

Run order puts the decisive cell early: R0 (harness check) then R2 at the middle
weight is the answer; everything after is confirmation and sweep.

    python -m e1.run_e1b                 # full ladder
    python -m e1.run_e1b --plan          # print the plan and exit
    python -m e1.run_e1b --skip-r3       # drop R3 if R2 clearly recovers
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .analyze import analyze
from .config import config_for_rung
from .train import RUNS_DIR, train
from .utils import git_info, write_json

SEEDS = (0, 1, 2, 3, 4)

# (rung, weight) in execution order. R2 at w=10 is the decision cell and comes second.
# R2 w=1 moved ahead of R1 after w=10 collapsed the code (D29): the weight most
# likely to bind without collapsing should run next. Schedule only -- no cell dropped,
# no threshold moved.
LADDER = [
    # (rung, weight, seeds). Seed counts were reduced for three cells once the
    # result was established -- see D31. Reduced-n cells are NOT headline results
    # and are labelled underpowered wherever they are reported.
    ("R0",  0.0, (0, 1, 2, 3, 4)),   # baseline, full
    ("R2", 10.0, (0, 1, 2, 3, 4)),   # full
    ("R2",  1.0, (0, 1, 2)),         # settled at n=3; seeds 3-4 dropped
    ("R1",  0.0, (0, 1)),            # settled at n=2; seeds 2-4 dropped
    ("R2", 40.0, (0,)),              # slope confirmation only, n=1
    ("R3",  0.0, (0, 1, 2, 3, 4)),   # full -- the only mechanism that can still change the verdict
]


def run_id_for(rung: str, weight: float, seed: int) -> str:
    return f"e1b_{rung}_w{int(weight)}_{seed}_{git_info()['git_sha_short']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--rungs", nargs="*", default=None,
                    help="restrict to these rungs, e.g. --rungs R0 R2")
    ap.add_argument("--skip-r3", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.seeds_override = args.seeds is not None

    ladder = [c for c in LADDER if not (args.skip_r3 and c[0] == "R3")]
    if args.rungs:
        ladder = [c for c in ladder if c[0] in args.rungs]
    # Per-cell seed lists; --seeds overrides them uniformly when given explicitly.
    plan = [(r, w, sd) for r, w, seeds in ladder
            for sd in (args.seeds if args.seeds_override else seeds)]

    print(f"E1b ladder: {len(plan)} runs")
    for rung, weight, seeds in ladder:
        n = len(args.seeds if args.seeds_override else seeds)
        flag = "" if n >= 5 else "   [REDUCED-n, not a headline result]"
        print(f"  {rung:3s} w={weight:<5g} x{n} seeds{flag}")
    if args.plan:
        return 0

    t_batch = time.time()
    for idx, (rung, weight, seed) in enumerate(plan, start=1):
        run_dir = RUNS_DIR / run_id_for(rung, weight, seed)
        if (run_dir / "metrics.json").exists() and not args.force:
            print(f"[{idx}/{len(plan)}] skip {run_dir.name}")
            continue

        cmd = [sys.executable, "-m", "e1.train_e1b",
               "--rung", rung, "--weight", str(weight), "--seed", str(seed)]
        t0 = time.time()
        print(f"[{idx}/{len(plan)}] {rung} w={weight:g} seed {seed} ...", flush=True)
        proc = subprocess.run(cmd, cwd=str(RUNS_DIR.parent))
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})")
            return proc.returncode

        m = analyze(run_dir)
        write_json(run_dir / "metrics.json", m)
        print(f"  done in {(time.time() - t0) / 60:.1f} min  "
              f"closed_err={m['M3_closed_map_error']:.3f}  "
              f"unseen={m['M1_acc_unseen']:.4f}  "
              f"code_resid={m.get('code_residual', float('nan')):.3f}  "
              f"drift={m['M7_drift_step1']:.3f}", flush=True)

    print(f"\nE1b complete in {(time.time() - t_batch) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
