#!/usr/bin/env python3
"""Run/resume Experiment A's locked matched shared-LoRA/atom frontier."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.validation_frontier import (
    DEFAULT_ATOM_REUSE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    run_validation_frontier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--atoms-config", type=Path, default=Path("configs/atoms.yaml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--atom-reuse-root",
        type=Path,
        default=DEFAULT_ATOM_REUSE_ROOT,
        help="root searched for strictly compatible existing shared-atom artifacts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace all 24 complete cells and do not reuse existing atom artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, summary_path, report_path = run_validation_frontier(
        load_config(args.baseline_config),
        load_config(args.atoms_config),
        args.output_root,
        atom_reuse_root=args.atom_reuse_root,
        force=args.force,
    )
    decision = summary["atom_specific_advantage"]
    print(f"Atom-specific advantage: {'PASS' if decision['passed'] else 'FAIL'}")
    print(f"Qualifying capacities: {decision['qualifying_capacities'] or 'none'}")
    print(f"JSON: {summary_path}")
    print(f"Markdown: {report_path}")


if __name__ == "__main__":
    main()
