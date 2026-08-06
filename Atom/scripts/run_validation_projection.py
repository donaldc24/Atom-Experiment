#!/usr/bin/env python3
"""Run/resume the locked all-target oracle LoRA-to-atom-span projection."""

from __future__ import annotations

import argparse
from pathlib import Path

from cgmoe_h1.config import load_config
from cgmoe_h1.validation_projection import run_projection_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--atoms-config", type=Path, default=Path("configs/atoms.yaml"))
    parser.add_argument(
        "--cross-transfer-root",
        type=Path,
        default=Path("results/atom_validation/cross_transfer"),
    )
    parser.add_argument(
        "--independent-root",
        type=Path,
        default=Path("results/independent_lora"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/atom_validation/oracle_projection"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, summary_path, report_path = run_projection_suite(
        load_config(args.baseline_config),
        load_config(args.atoms_config),
        cross_transfer_root=args.cross_transfer_root,
        independent_root=args.independent_root,
        output_root=args.output_root,
        force=args.force,
    )
    verdict = "PASS" if summary["strong_learned_span_support"] else "FAIL"
    aggregate = summary["aggregate"]
    print(f"Strong learned-span support: {verdict}")
    print(
        "Learned all-eight quality retention: "
        f"{aggregate['learned_all_atoms_quality_retention']:.3%}"
    )
    print(
        "Learned/random relative reconstruction error: "
        f"{aggregate['learned_span_all_atoms_reconstruction']['relative_frobenius_error']:.6f} / "
        f"{aggregate['random_span_all_atoms_reconstruction']['relative_frobenius_error']:.6f}"
    )
    print(f"JSON: {summary_path}")
    print(f"Markdown: {report_path}")


if __name__ == "__main__":
    main()
