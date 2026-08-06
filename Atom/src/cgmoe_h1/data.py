"""Deterministic, auditable loading for the five H1 GLUE tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from datasets import Dataset, DatasetDict, load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

GLUE_DATASET_NAME = "nyu-mll/glue"


@dataclass(frozen=True, slots=True)
class TaskSchema:
    """The predeclared input and label schema for one GLUE configuration."""

    name: str
    text_fields: tuple[str, ...]
    num_labels: int = 2
    positive_label: int = 1

    @property
    def is_pair(self) -> bool:
        return len(self.text_fields) == 2


TASK_SCHEMAS: dict[str, TaskSchema] = {
    "sst2": TaskSchema("sst2", ("sentence",)),
    "mrpc": TaskSchema("mrpc", ("sentence1", "sentence2")),
    "rte": TaskSchema("rte", ("sentence1", "sentence2")),
    "qnli": TaskSchema("qnli", ("question", "sentence")),
    "qqp": TaskSchema("qqp", ("question1", "question2")),
}

MODEL_COLUMNS = frozenset({"input_ids", "attention_mask", "token_type_ids", "labels"})
AUDIT_COLUMNS = frozenset({"task_id", "example_id"})
_SOURCE_INDEX_COLUMN = "__cgmoe_source_index__"


@dataclass(frozen=True, slots=True)
class SplitProvenance:
    """Selection evidence attached to each returned tokenized split."""

    dataset_name: str
    task_name: str
    split: str
    source_fingerprint: str | None
    requested_limit: int
    selected_count: int
    seed: int
    selected_row_ids: tuple[int | str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "split": self.split,
            "source_fingerprint": self.source_fingerprint,
            "requested_limit": self.requested_limit,
            "selected_count": self.selected_count,
            "seed": self.seed,
            "selected_row_ids": list(self.selected_row_ids),
        }


class _Tokenizer(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Mapping[str, Sequence[Any]]: ...


class ModelInputCollator:
    """Dynamically pad model inputs while leaving audit columns in the dataset.

    ``task_id`` and ``example_id`` are intentionally not forwarded to BERT.
    Multitask code already knows the task from the loader it selected, and can
    retrieve row IDs directly from the dataset for the run record.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self._collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        model_features = [
            {key: value for key, value in feature.items() if key in MODEL_COLUMNS}
            for feature in features
        ]
        return dict(self._collator(model_features))


def make_data_collator(tokenizer: PreTrainedTokenizerBase) -> ModelInputCollator:
    """Return the standard dynamic-padding collator for H1 data loaders."""
    return ModelInputCollator(tokenizer)


def get_task_schema(task_name: str) -> TaskSchema:
    """Return the locked schema for *task_name*."""
    try:
        return TASK_SCHEMAS[task_name]
    except KeyError as exc:
        choices = ", ".join(TASK_SCHEMAS)
        raise ValueError(f"unknown task {task_name!r}; expected one of: {choices}") from exc


def _validate_limit(name: str, limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise TypeError(f"{name} must be an integer")
    if limit < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")


def _selected_ids(dataset: Dataset, id_column: str) -> tuple[int | str, ...]:
    values = dataset[id_column]
    identifiers: list[int | str] = []
    for value in values:
        # NumPy scalars appear in some Arrow-backed datasets.  ``item`` turns
        # them into values that JSON serialization can audit directly.
        if hasattr(value, "item"):
            value = value.item()
        if not isinstance(value, (int, str)) or isinstance(value, bool):
            value = str(value)
        identifiers.append(value)
    return tuple(identifiers)


def _tokenize_split(
    source: Dataset,
    *,
    split_name: str,
    task_name: str,
    schema: TaskSchema,
    tokenizer: _Tokenizer,
    requested_limit: int,
    max_length: int,
    seed: int,
) -> Dataset:
    source_fingerprint = getattr(source, "_fingerprint", None)
    missing = set(schema.text_fields).union({"label"}).difference(source.column_names)
    if missing:
        raise ValueError(
            f"{task_name}/{split_name} is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    id_column = "idx" if "idx" in source.column_names else _SOURCE_INDEX_COLUMN
    indexed = source
    if id_column == _SOURCE_INDEX_COLUMN:
        indexed = source.add_column(id_column, list(range(len(source))))

    selected_count = min(requested_limit, len(indexed))
    # This order is part of the locked data contract: shuffle first, then take
    # the leading n rows.  No row is repeated when a split is below its limit.
    selected = indexed.shuffle(seed=seed).select(range(selected_count))
    row_ids = _selected_ids(selected, id_column)

    invalid_labels = sorted(set(selected["label"]).difference(range(schema.num_labels)))
    if invalid_labels:
        raise ValueError(
            f"{task_name}/{split_name} contains labels outside "
            f"[0, {schema.num_labels - 1}]: {invalid_labels}"
        )

    def tokenize_batch(batch: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
        text_inputs = [batch[field] for field in schema.text_fields]
        encoded = dict(
            tokenizer(
                *text_inputs,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
        )
        encoded["labels"] = list(batch["label"])
        encoded["task_id"] = [task_name] * len(batch["label"])
        encoded["example_id"] = list(batch[id_column])
        return encoded

    tokenized = selected.map(
        tokenize_batch,
        batched=True,
        remove_columns=selected.column_names,
        desc=f"Tokenizing {task_name}/{split_name}",
    )
    required_outputs = {"input_ids", "attention_mask", "labels", "task_id", "example_id"}
    missing_outputs = required_outputs.difference(tokenized.column_names)
    if missing_outputs:
        raise ValueError(
            "tokenizer output is missing required column(s): "
            + ", ".join(sorted(missing_outputs))
        )

    provenance = SplitProvenance(
        dataset_name=GLUE_DATASET_NAME,
        task_name=task_name,
        split=split_name,
        source_fingerprint=source_fingerprint,
        requested_limit=requested_limit,
        selected_count=selected_count,
        seed=seed,
        selected_row_ids=row_ids,
    )
    # Hugging Face Dataset intentionally permits lightweight Python metadata.
    # Keeping this off the per-example feature table avoids duplicating the
    # source fingerprint and complete ID list for every row.
    setattr(tokenized, "_cgmoe_provenance", provenance)
    return tokenized


def get_dataset_provenance(dataset: Dataset) -> SplitProvenance:
    """Return audit metadata attached by :func:`load_task_data`."""
    provenance = getattr(dataset, "_cgmoe_provenance", None)
    if not isinstance(provenance, SplitProvenance):
        raise ValueError("dataset was not produced by load_task_data or lost its provenance")
    return provenance


def load_task_data(
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    train_limit: int,
    validation_limit: int,
    max_length: int,
    seed: int,
    *,
    dataset_loader: Callable[..., DatasetDict | Mapping[str, Dataset]] | None = None,
) -> tuple[Dataset, Dataset]:
    """Load, deterministically subset, and tokenize one H1 GLUE task.

    Selection happens before tokenization.  The returned datasets retain
    ``task_id`` and ``example_id`` columns plus attached source fingerprints;
    use :func:`make_data_collator` to dynamically pad only model-facing keys.
    """
    schema = get_task_schema(task_name)
    _validate_limit("train_limit", train_limit)
    _validate_limit("validation_limit", validation_limit)
    _validate_seed(seed)
    if not isinstance(max_length, int) or isinstance(max_length, bool):
        raise TypeError("max_length must be an integer")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    loader = load_dataset if dataset_loader is None else dataset_loader
    loaded = loader(GLUE_DATASET_NAME, task_name)
    missing_splits = {"train", "validation"}.difference(loaded)
    if missing_splits:
        raise ValueError(
            f"{task_name} dataset is missing split(s): {', '.join(sorted(missing_splits))}"
        )

    train = _tokenize_split(
        loaded["train"],
        split_name="train",
        task_name=task_name,
        schema=schema,
        tokenizer=tokenizer,
        requested_limit=train_limit,
        max_length=max_length,
        seed=seed,
    )
    validation = _tokenize_split(
        loaded["validation"],
        split_name="validation",
        task_name=task_name,
        schema=schema,
        tokenizer=tokenizer,
        requested_limit=validation_limit,
        max_length=max_length,
        seed=seed,
    )
    return train, validation
