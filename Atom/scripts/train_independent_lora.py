#!/usr/bin/env python3
"""Train one or more independent LoRA task adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.experiments import run_independent_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--validation-examples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("results/independent_lora"))
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config).with_overrides(seed=args.seed)
    changes = {}
    if args.train_examples is not None:
        changes["train_examples_per_task"] = args.train_examples
    if args.validation_examples is not None:
        changes["validation_examples_per_task"] = args.validation_examples
    if args.epochs is not None:
        changes["epochs"] = args.epochs
    if changes:
        config = config.with_overrides(**changes)
    tasks = tuple(args.tasks or config.tasks)
    result = run_independent_lora(
        config,
        args.output_root,
        tasks=tasks,
        run_kind="development" if args.development else "confirmatory",
        rank=args.rank,
        force=args.force,
    )
    for task, record in result["tasks"].items():
        print(f"{task}: {record['best']['metrics']['primary_score']:.6f}")


if __name__ == "__main__":
    main()
