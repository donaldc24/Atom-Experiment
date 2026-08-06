#!/usr/bin/env python3
"""Run the locked three-seed independent-LoRA/shared-atoms H1 comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.experiments import run_core_seed, validate_expected_core_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--atoms-config", type=Path, default=Path("configs/atoms.yaml"))
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_config(args.baseline_config)
    atoms = load_config(args.atoms_config)
    seeds = tuple(args.seeds or baseline.confirmatory_seeds)
    if seeds != tuple(baseline.confirmatory_seeds):
        print("Warning: a subset of seeds is a preliminary, non-confirmatory run.")
    for seed in seeds:
        print(f"\n=== H1 seed {seed} ===", flush=True)
        independent, shared = run_core_seed(
            baseline.with_overrides(seed=seed),
            atoms.with_overrides(seed=seed),
            args.results_root,
            force=args.force,
        )
        validate_expected_core_counts(independent, shared)
    print("Core H1 runs complete. Generate the decision with scripts/summarize_h1.py.")


if __name__ == "__main__":
    main()
