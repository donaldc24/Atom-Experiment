#!/usr/bin/env python3
"""Smoke-test the local Python and machine-learning environment."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any


DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
ROW_COUNT = 2


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Report the Python, PyTorch, and Transformers environment and load "
            "two rows from the nyu-mll/glue SST-2 training split."
        )
    )


def _import_dependency(name: str, failures: list[str]) -> ModuleType | None:
    """Import a dependency while preserving a useful diagnostic for the report."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # Import-time binary/linker errors are not ImportError.
        failures.append(f"could not import {name}: {type(exc).__name__}: {exc}")
        return None


def _module_version(module: ModuleType | None) -> str:
    if module is None:
        return "UNAVAILABLE"
    return str(getattr(module, "__version__", "unknown"))


def _load_sst2_rows(datasets_module: ModuleType) -> list[dict[str, Any]]:
    load_dataset = getattr(datasets_module, "load_dataset", None)
    if not callable(load_dataset):
        raise RuntimeError("datasets.load_dataset is not available")

    split = f"train[:{ROW_COUNT}]"
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    rows = [dict(row) for row in dataset]
    if len(rows) != ROW_COUNT:
        raise RuntimeError(
            f"requested {ROW_COUNT} rows from {split}, but received {len(rows)}"
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    failures: list[str] = []

    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    torch = _import_dependency("torch", failures)
    transformers = _import_dependency("transformers", failures)

    print(f"PyTorch version: {_module_version(torch)}")
    print(f"Transformers version: {_module_version(transformers)}")

    if torch is None:
        print("Number of CPU threads: UNAVAILABLE")
        print("CUDA availability: UNAVAILABLE")
    else:
        try:
            print(f"Number of CPU threads: {torch.get_num_threads()}")
        except Exception as exc:
            print("Number of CPU threads: UNAVAILABLE")
            failures.append(
                f"could not query PyTorch CPU threads: {type(exc).__name__}: {exc}"
            )

        try:
            print(f"CUDA availability: {torch.cuda.is_available()}")
        except Exception as exc:
            print("CUDA availability: UNAVAILABLE")
            failures.append(
                f"could not query CUDA availability: {type(exc).__name__}: {exc}"
            )

    datasets_module = _import_dependency("datasets", failures)
    if datasets_module is not None:
        try:
            rows = _load_sst2_rows(datasets_module)
        except Exception as exc:
            failures.append(
                f"could not load {DATASET_NAME}/{DATASET_CONFIG}: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            for index, row in enumerate(rows, start=1):
                print(f"SST-2 row {index}: {row}")

    if failures:
        print("Environment smoke test FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "Install the project dependencies and check network access to the "
            "Hugging Face Hub, then rerun this script.",
            file=sys.stderr,
        )
        return 1

    print("Environment smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
