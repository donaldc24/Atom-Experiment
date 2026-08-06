"""Tests for atomic strict-JSON run records."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cgmoe_h1.utils import serialization
from cgmoe_h1.utils.serialization import read_json, to_jsonable, write_json


@dataclasses.dataclass
class TinyRecord:
    task: str
    scores: np.ndarray


def test_json_round_trip_handles_experiment_value_types(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "metrics.json"
    value = {
        "record": TinyRecord("sst2", np.array([0.5, 0.75])),
        "tensor": torch.tensor([[1, 2], [3, 4]]),
        "scalar": np.float32(0.25),
        "checkpoint": Path("results/model.pt"),
        "seeds": (17, 29, 43),
    }

    returned = write_json(path, value)

    assert returned == path
    assert read_json(path) == {
        "checkpoint": "results\\model.pt" if str(Path("results/model.pt")) == "results\\model.pt" else "results/model.pt",
        "record": {"scores": [0.5, 0.75], "task": "sst2"},
        "scalar": 0.25,
        "seeds": [17, 29, 43],
        "tensor": [[1, 2], [3, 4]],
    }
    assert path.read_bytes().endswith(b"\n")


def test_write_atomically_replaces_existing_document(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"state": "old"}\n', encoding="utf-8")

    write_json(path, {"state": "complete", "seed": 17})

    assert json.loads(path.read_text(encoding="utf-8")) == {"state": "complete", "seed": 17}
    assert not list(tmp_path.glob(".result.json.*.tmp"))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_value_does_not_touch_existing_file(tmp_path: Path, bad_value: float) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"state": "old"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="NaN or infinity"):
        write_json(path, {"metric": bad_value})

    assert read_json(path) == {"state": "old"}
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_replace_failure_cleans_temporary_and_preserves_old_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"state": "old"}\n', encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(serialization.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_json(path, {"state": "new"})

    assert read_json(path) == {"state": "old"}
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_json_conversion_rejects_unsupported_types_and_non_string_keys() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        to_jsonable({"bad": object()})
    with pytest.raises(TypeError, match="keys must be strings"):
        to_jsonable({1: "bad key"})
