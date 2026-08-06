#!/usr/bin/env python3
"""Regenerate the preregistered H1 summary and Markdown report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from cgmoe_h1.config import H1_CONFIRMATORY_SEEDS
from cgmoe_h1.reporting import (
    DEFAULT_INDEPENDENT_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHARED_ROOT,
    generate_h1_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--independent-root",
        type=Path,
        default=DEFAULT_INDEPENDENT_ROOT,
        help="root containing independent_lora seed_<n> directories",
    )
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=DEFAULT_SHARED_ROOT,
        help="root containing shared_atoms seed_<n> directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for h1_summary.json and h1_report.md",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(H1_CONFIRMATORY_SEEDS),
        help="paired seed IDs to summarize (default: 17 29 43)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, summary_path, report_path = generate_h1_report(
        args.independent_root,
        args.shared_root,
        args.output_dir,
        seeds=tuple(args.seeds),
    )
    print(f"H1 decision: {'PASS' if summary['preregistered_pass'] else 'FAIL'} — {summary['decision']}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
