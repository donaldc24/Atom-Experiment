"""Strict, atomic JSON serialization for experiment records."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch


def to_jsonable(value: Any) -> Any:
    """Recursively convert common experiment values to strict JSON values.

    NaN and infinity are rejected because they are not valid JSON and would
    make an H1 run invalid rather than merely inconvenient to read.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize NaN or infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
            converted[key] = to_jsonable(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int = 2,
) -> Path:
    """Atomically write one UTF-8 JSON document and return its path.

    The complete document is encoded before touching the destination.  It is
    then flushed to a temporary file in the same directory and installed with
    ``os.replace``, so readers observe either the old or the complete new file.
    """
    destination = Path(path)
    serializable = to_jsonable(value)
    document = json.dumps(
        serializable,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def read_json(path: str | Path) -> Any:
    """Read one UTF-8 JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Clear aliases for callers that prefer intent-revealing names.
atomic_write_json = write_json
save_json = write_json
