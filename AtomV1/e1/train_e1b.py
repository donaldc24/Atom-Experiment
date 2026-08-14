"""Single-run entry point for an E1b ladder cell. Produces artifacts; no metrics.

    python -m e1.train_e1b --rung R2 --weight 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import config_for_rung
from .train import RUNS_DIR, train
from .utils import git_info


def run_id_for(rung: str, weight: float, seed: int) -> str:
    return f"e1b_{rung}_w{int(weight)}_{seed}_{git_info()['git_sha_short']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    cfg = config_for_rung(args.rung, args.seed, args.weight)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    run_dir = Path(args.out) if args.out else \
        RUNS_DIR / run_id_for(args.rung, args.weight, args.seed)
    print(f"[{args.rung} w={args.weight:g} seed={args.seed}] -> {run_dir}")
    train(cfg, run_dir, allow_dirty=args.allow_dirty)
    print(f"done in {json.load(open(run_dir / 'env.json'))['train_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
