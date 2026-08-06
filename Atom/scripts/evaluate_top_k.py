#!/usr/bin/env python3
"""Reload and evaluate a shared-atom checkpoint under a fixed top-k mask."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.experiments import evaluate_atom_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/atoms.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config).with_overrides(seed=args.seed)
    record = evaluate_atom_checkpoint(config, args.run_dir, top_k=args.k)
    for task, result in record["tasks"].items():
        print(f"{task}: {result['metrics']['primary_score']:.6f}")


if __name__ == "__main__":
    main()
