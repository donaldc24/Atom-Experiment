"""Offline tests for deterministic GLUE preparation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
from datasets import Dataset, DatasetDict

from cgmoe_h1.data import (
    GLUE_DATASET_NAME,
    TASK_SCHEMAS,
    ModelInputCollator,
    get_dataset_provenance,
    load_task_data,
)


class FakeTokenizer:
    """Small tokenizer/padder with the subset of the Transformers protocol used here."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[list[str], ...], dict[str, Any]]] = []

    def __call__(self, *texts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        self.calls.append((texts, kwargs))
        input_ids: list[list[int]] = []
        for row in zip(*texts, strict=True):
            # Variable lengths verify that data is not padded before batching.
            length = min(kwargs["max_length"], 2 + sum(len(value) for value in row) % 5)
            input_ids.append(list(range(1, length + 1)))
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
            "token_type_ids": [[0] * len(ids) for ids in input_ids],
        }

    def pad(
        self,
        features: list[Mapping[str, Any]],
        *,
        padding: str,
        max_length: int | None,
        pad_to_multiple_of: int | None,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding == "longest"
        assert max_length is None
        assert return_tensors == "pt"
        longest = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of:
            longest = ((longest + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        batch: dict[str, torch.Tensor] = {}
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            rows = []
            for feature in features:
                values = list(feature[key])
                rows.append(values + [0] * (longest - len(values)))
            batch[key] = torch.tensor(rows, dtype=torch.long)
        batch["labels"] = torch.tensor([feature["labels"] for feature in features])
        return batch


def make_task_dict(task_name: str, train_size: int = 9, validation_size: int = 3) -> DatasetDict:
    schema = TASK_SCHEMAS[task_name]

    def split(size: int, offset: int) -> Dataset:
        values: dict[str, list[Any]] = {
            field: [f"{field}-{offset + index}" for index in range(size)]
            for field in schema.text_fields
        }
        values["label"] = [index % 2 for index in range(size)]
        values["idx"] = [offset + index for index in range(size)]
        return Dataset.from_dict(values)

    return DatasetDict({"train": split(train_size, 100), "validation": split(validation_size, 900)})


def loader_for(data: DatasetDict, expected_task: str):
    def loader(dataset_name: str, task_name: str) -> DatasetDict:
        assert dataset_name == GLUE_DATASET_NAME
        assert task_name == expected_task
        return data

    return loader


@pytest.mark.parametrize("task_name", tuple(TASK_SCHEMAS))
def test_every_task_uses_declared_fields_and_produces_auditable_data(task_name: str) -> None:
    raw = make_task_dict(task_name)
    tokenizer = FakeTokenizer()
    train, validation = load_task_data(
        task_name,
        tokenizer,  # type: ignore[arg-type]
        train_limit=5,
        validation_limit=500,
        max_length=8,
        seed=17,
        dataset_loader=loader_for(raw, task_name),
    )

    assert len(train) == 5
    assert len(validation) == 3  # never duplicate an undersized validation split
    assert {"input_ids", "attention_mask", "labels", "task_id", "example_id"} <= set(
        train.column_names
    )
    assert set(train["labels"]) <= {0, 1}
    assert set(train["task_id"]) == {task_name}
    assert len(tokenizer.calls[0][0]) == len(TASK_SCHEMAS[task_name].text_fields)
    assert tokenizer.calls[0][1] == {
        "truncation": True,
        "max_length": 8,
        "padding": False,
    }

    expected_ids = tuple(raw["train"].shuffle(seed=17).select(range(5))["idx"])
    provenance = get_dataset_provenance(train)
    assert provenance.selected_row_ids == expected_ids
    assert provenance.source_fingerprint == raw["train"]._fingerprint
    assert provenance.selected_count == 5
    assert provenance.to_dict()["selected_row_ids"] == list(expected_ids)


def test_same_seed_selects_and_orders_identical_examples() -> None:
    raw = make_task_dict("sst2", train_size=30)

    first, _ = load_task_data(
        "sst2", FakeTokenizer(), 10, 3, 16, 29, dataset_loader=loader_for(raw, "sst2")
    )
    second, _ = load_task_data(
        "sst2", FakeTokenizer(), 10, 3, 16, 29, dataset_loader=loader_for(raw, "sst2")
    )
    other, _ = load_task_data(
        "sst2", FakeTokenizer(), 10, 3, 16, 43, dataset_loader=loader_for(raw, "sst2")
    )

    assert first["example_id"] == second["example_id"]
    assert first["input_ids"] == second["input_ids"]
    assert first["example_id"] != other["example_id"]


def test_source_index_is_saved_when_dataset_has_no_idx_column() -> None:
    raw = make_task_dict("mrpc")
    raw = DatasetDict(
        {
            split: dataset.remove_columns("idx")
            for split, dataset in raw.items()
        }
    )

    train, _ = load_task_data(
        "mrpc", FakeTokenizer(), 4, 2, 32, 17, dataset_loader=loader_for(raw, "mrpc")
    )

    expected = tuple(raw["train"].add_column("source", range(9)).shuffle(seed=17)["source"][:4])
    assert get_dataset_provenance(train).selected_row_ids == expected
    assert train["example_id"] == list(expected)


def test_model_collator_dynamically_pads_and_strips_audit_columns() -> None:
    raw = make_task_dict("sst2")
    tokenizer = FakeTokenizer()
    train, _ = load_task_data(
        "sst2", tokenizer, 5, 2, 16, 17, dataset_loader=loader_for(raw, "sst2")
    )

    batch = ModelInputCollator(tokenizer)([train[0], train[1], train[2]])  # type: ignore[arg-type]

    assert set(batch) == {"input_ids", "attention_mask", "token_type_ids", "labels"}
    assert batch["input_ids"].shape[0] == 3
    assert batch["input_ids"].shape[1] == max(len(train[index]["input_ids"]) for index in range(3))
    assert all(isinstance(value, torch.Tensor) for value in batch.values())


@pytest.mark.parametrize(
    "kwargs",
    (
        {"task_name": "cola"},
        {"task_name": "sst2", "train_limit": -1},
        {"task_name": "sst2", "max_length": 0},
        {"task_name": "sst2", "seed": True},
    ),
)
def test_invalid_data_request_is_rejected(kwargs: dict[str, Any]) -> None:
    raw = make_task_dict("sst2")
    arguments = {
        "task_name": "sst2",
        "tokenizer": FakeTokenizer(),
        "train_limit": 3,
        "validation_limit": 2,
        "max_length": 16,
        "seed": 17,
        "dataset_loader": loader_for(raw, "sst2"),
    }
    arguments.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        load_task_data(**arguments)


def test_missing_source_column_is_rejected() -> None:
    raw = make_task_dict("qnli")
    raw["train"] = raw["train"].remove_columns("question")

    with pytest.raises(ValueError, match="missing required column"):
        load_task_data(
            "qnli", FakeTokenizer(), 3, 2, 16, 17, dataset_loader=loader_for(raw, "qnli")
        )
