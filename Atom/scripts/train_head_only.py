#!/usr/bin/env python3
"""Run the diagnostic frozen-BERT, head-only baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.experiments import run_head_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-examples", type=int, default=500)
    parser.add_argument("--validation-examples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/development/head_only"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config).with_overrides(
        seed=args.seed,
        tasks=(args.task,),
        train_examples_per_task=args.train_examples,
        validation_examples_per_task=args.validation_examples,
        epochs=args.epochs,
    )
    record = run_head_only(
        config,
        args.task,
        args.output_dir / f"seed_{args.seed}" / args.task,
    )
    print(f"Best primary score: {record['best']['metrics']['primary_score']:.6f}")


if __name__ == "__main__":
    main()
