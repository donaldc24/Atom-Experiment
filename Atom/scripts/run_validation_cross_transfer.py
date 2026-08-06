#!/usr/bin/env python3
"""Run/resume the locked 5-target x 3-seed cross-transfer validation grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.validation_cross_transfer import (
    DEFAULT_CORE_RESULTS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    run_validation_cross_transfer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--atoms-config", type=Path, default=Path("configs/atoms.yaml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--core-results-root",
        type=Path,
        default=DEFAULT_CORE_RESULTS_ROOT,
        help="root containing strict results/independent_lora core artifacts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun every cell and sub-run instead of resuming completed artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, summary_path, report_path = run_validation_cross_transfer(
        load_config(args.baseline_config),
        load_config(args.atoms_config),
        args.output_root,
        core_results_root=args.core_results_root,
        force=args.force,
    )
    primary = summary["primary_strong_transfer"]
    print(f"Cross-transfer strong result: {'PASS' if primary['passed'] else 'FAIL'}")
    print(
        "Control-aware reusable-basis support: "
        f"{'PASS' if summary['strong_reusable_basis_support'] else 'FAIL'}"
    )
    print(f"Aggregate retention: {primary['aggregate_retention']:.2%}")
    print(f"JSON: {summary_path}")
    print(f"Markdown: {report_path}")


if __name__ == "__main__":
    main()
