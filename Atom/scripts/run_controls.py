#!/usr/bin/env python3
"""Run or resume the six locked seed-17 roadmap chunk-25 controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.followups_controls import CONTROL_IDS, run_control_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/baseline.yaml"),
    )
    parser.add_argument(
        "--atoms-config",
        type=Path,
        default=Path("configs/atoms.yaml"),
    )
    parser.add_argument(
        "--independent-root",
        type=Path,
        default=Path("results/independent_lora"),
        help="core independent-LoRA root containing seed_17 compact checkpoints",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/controls"),
    )
    parser.add_argument(
        "--control",
        action="append",
        choices=CONTROL_IDS,
        dest="controls",
        help="run a selected control (repeatable); defaults to all six",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun selected controls even when complete records exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Chunk 25 is intentionally seed-17-only.  There is no CLI seed override.
    baseline = load_config(args.baseline_config).with_overrides(seed=17)
    atoms = load_config(args.atoms_config).with_overrides(seed=17)
    summary, summary_path, report_path = run_control_suite(
        baseline,
        atoms,
        args.output_root,
        args.independent_root,
        controls=args.controls,
        force=args.force,
    )
    print(f"Control suite status: {summary['status']}")
    for control_id, result in summary["controls"].items():
        print(f"{control_id}: mean_primary_score={result['mean_primary_score']:.6f}")
    print(f"JSON: {summary_path}")
    print(f"Markdown: {report_path}")


if __name__ == "__main__":
    main()
