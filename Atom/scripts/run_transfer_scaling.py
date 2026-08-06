#!/usr/bin/env python3
"""Run locked seed-17 frozen-atom transfer and task-prefix scaling follow-ups."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.followups_transfer import (
    FOLLOWUP_SEED,
    run_frozen_atom_transfer,
    run_scaling_curve,
    run_transfer_and_scaling,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--atoms-config", type=Path, default=Path("configs/atoms.yaml"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--only",
        choices=("transfer", "scaling"),
        help="run only one follow-up (the default runs both)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_config(args.baseline_config).with_overrides(seed=FOLLOWUP_SEED)
    atoms = load_config(args.atoms_config).with_overrides(seed=FOLLOWUP_SEED)
    followups = args.results_root / "followups"
    shared_prefixes = followups / "shared_prefixes"

    if args.only == "transfer":
        transfer = run_frozen_atom_transfer(
            baseline,
            atoms,
            followups / "frozen_atom_transfer",
            core_results_root=args.results_root,
            shared_prefix_root=shared_prefixes,
            force=args.force,
        )
        verdict = "PASS" if transfer["strong_result"]["strong_transfer"] else "FAIL"
        print(f"Frozen-atom transfer: {verdict}")
        return
    if args.only == "scaling":
        scaling = run_scaling_curve(
            baseline,
            atoms,
            followups / "scaling_curve",
            core_results_root=args.results_root,
            shared_prefix_root=shared_prefixes,
            force=args.force,
        )
        print(f"Scaling curve complete: {len(scaling['points'])} task prefixes")
        return

    transfer, scaling = run_transfer_and_scaling(
        baseline,
        atoms,
        args.results_root,
        force=args.force,
    )
    verdict = "PASS" if transfer["strong_result"]["strong_transfer"] else "FAIL"
    print(f"Frozen-atom transfer: {verdict}")
    print(f"Scaling curve complete: {len(scaling['points'])} task prefixes")
    print(f"Reports written below {followups}")


if __name__ == "__main__":
    main()
