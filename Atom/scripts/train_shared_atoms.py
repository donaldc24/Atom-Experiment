#!/usr/bin/env python3
"""Train a jointly shared atom dictionary across selected tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.experiments import run_shared_atoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/atoms.yaml"))
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--atom-count", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--validation-examples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sparsity-lambda", type=float)
    parser.add_argument("--freeze-atoms", action="store_true")
    parser.add_argument("--shuffle-training-labels", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("results/shared_atoms"))
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
    result = run_shared_atoms(
        config,
        args.output_root,
        tasks=tuple(args.tasks or config.tasks),
        run_kind="development" if args.development else "confirmatory",
        atom_count=args.atom_count,
        top_k=args.top_k,
        sparsity_lambda=args.sparsity_lambda,
        freeze_atoms=args.freeze_atoms,
        shuffle_training_labels=args.shuffle_training_labels,
        force=args.force,
    )
    for task, record in result["tasks"].items():
        print(
            f"{task}: all={record['all_atoms']['metrics']['primary_score']:.6f}, "
            f"top-k={record['top_k']['metrics']['primary_score']:.6f}"
        )


if __name__ == "__main__":
    main()
