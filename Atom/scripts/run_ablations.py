#!/usr/bin/env python3
"""Run/resume the locked seed-17 H1 atom-count and capacity ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.followups_ablations import (
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    run_ablations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="locked independent-LoRA config (must resolve to seed 17)",
    )
    parser.add_argument(
        "--atoms-config",
        type=Path,
        default=Path("configs/atoms.yaml"),
        help="locked shared-atoms config (must resolve to seed 17)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/followup_ablations"),
        help="variant runs and final reports destination",
    )
    parser.add_argument(
        "--core-independent-root",
        type=Path,
        default=Path("results/independent_lora"),
        help="root containing the reusable core seed_17 rank-4 run",
    )
    parser.add_argument(
        "--core-shared-root",
        type=Path,
        default=Path("results/shared_atoms"),
        help="root containing the reusable core seed_17 8-atom run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun non-core training variants and active-capacity evaluations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_ablations(
        load_config(args.baseline_config),
        load_config(args.atoms_config),
        args.results_root,
        core_independent_root=args.core_independent_root,
        core_shared_root=args.core_shared_root,
        force=args.force,
    )
    atom_answers = summary["atom_count_ablation"]["question_answers"]
    rank_answers = summary["lora_rank_ablation"]["question_answers"]
    active_answers = summary["active_capacity_ablation"]["question_answers"]
    print(
        "Atom quality saturation: "
        f"N={atom_answers['quality_saturation']['atom_count']} "
        "under the documented 0.005 rule."
    )
    print(f"Best observed LoRA rank: r={rank_answers['best_mean_quality_rank']}.")
    print(
        "Smallest active capacity within 0.005 of best: "
        f"k={active_answers['smallest_active_atoms_within_0_005_of_best']}."
    )
    print(f"Strict JSON: {args.results_root / SUMMARY_FILENAME}")
    print(f"Markdown report: {args.results_root / REPORT_FILENAME}")


if __name__ == "__main__":
    main()
