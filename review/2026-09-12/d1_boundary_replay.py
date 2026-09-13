"""Post-hoc independent replay of the D1 self-bottleneck result.

Loads an existing frozen A14 seed-1 checkpoint and the frozen evaluation data.
It performs no training and never writes to AtomV2/runs or AtomV2/results.
The only output file is this review's d1_boundary_replay.json. Pass --no-write
to print the report without writing any output file.

From the repository root:
    .venv/Scripts/python.exe review/2026-09-12/d1_boundary_replay.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "AtomV2" / "Harness"
sys.path.insert(0, str(HARNESS))

import numpy as np
import torch

from atomv2.data import build_bundle
from atomv2.e7_audit import boundary_panel
from atomv2.panel import load_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)

    run_dir = REPO / "AtomV2/runs/e4/A14_s1_e4-stacked-pressures_d0955fd"
    checkpoint = run_dir / "checkpoints/final.pt"
    model, cfg, step = load_checkpoint(run_dir, "final.pt")
    bundle = build_bundle(cfg)
    start = time.perf_counter()
    panel = boundary_panel(model, bundle)

    task_groups = {
        "trained": [td for td in bundle.seen_heldout if td.task.n_tokens == 2],
        **{level: bundle.unseen[level] for level in ("L1", "L2", "L3")},
    }
    groups = {}
    for group, task_data in task_groups.items():
        n_examples = sum(len(td.x) for td in task_data)
        correct = {
            key: sum(
                round(panel["cells"][td.task.task_id][key] * len(td.x))
                for td in task_data
            )
            for key in ("raw_acc", "boundary_decode_acc", "self_bottleneck_acc")
        }
        groups[group] = {
            "n_tasks": len(task_data),
            "n_examples": n_examples,
            "examples_per_task": sorted({len(td.x) for td in task_data}),
            "correct_examples": correct,
            **panel["by_group"][group],
        }

    report = {
        "label": "POST_HOC_INDEPENDENT_REPLAY",
        "scope": "A14 seed 1 final checkpoint, existing frozen evaluation split",
        "mechanism": (
            "Execute token 1, decode its boundary state with the frozen model's "
            "own decoder, re-encode those predicted digits, then execute token 2. "
            "Ground-truth intermediate digits are used only to score the boundary "
            "decode and canonical-distance diagnostic, never as the repaired input."
        ),
        "limitations": [
            "One independently replayed checkpoint on the original task world and split.",
            "This replay is post-hoc, not a new preregistered experiment.",
            "The repair adds encoder/decoder computation and a known digit interface.",
            "Exact-list accuracy; trained group contains held-out inputs of trained pairs only.",
        ],
        "run_dir": run_dir.relative_to(REPO).as_posix(),
        "checkpoint": checkpoint.relative_to(REPO).as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": int(step),
        "seed": int(cfg.seed),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "source_sha256": {
            str(path.relative_to(REPO).as_posix()): sha256(path)
            for path in (
                Path(__file__),
                HARNESS / "atomv2/e7_audit.py",
                HARNESS / "atomv2/model.py",
                HARNESS / "atomv2/data.py",
                HARNESS / "splits/split_v2.json",
            )
        },
        "inference_seconds": time.perf_counter() - start,
        "groups": groups,
        "L3_cells": {
            key: value for key, value in panel["cells"].items()
            if value["group"] == "L3"
        },
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if not args.no_write:
        Path(__file__).with_suffix(".json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
