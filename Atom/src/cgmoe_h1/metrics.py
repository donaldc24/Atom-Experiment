"""Predeclared task metrics for the H1 comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

F1_TASKS = frozenset({"mrpc", "qqp"})
H1_TASKS = frozenset({"sst2", "mrpc", "rte", "qnli", "qqp"})
PRIMARY_SCORE_NAMES: dict[str, str] = {
    "sst2": "accuracy",
    "mrpc": "accuracy_f1_mean",
    "rte": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy_f1_mean",
}


def _as_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _prediction_labels(predictions: Any) -> np.ndarray:
    array = _as_numpy(predictions)
    if array.ndim == 2:
        if array.shape[1] != 2:
            raise ValueError(f"expected binary logits with shape [N, 2], got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("predictions contain NaN or infinity")
        return array.argmax(axis=1).astype(np.int64, copy=False)
    if array.ndim != 1:
        raise ValueError(f"predictions must be labels [N] or logits [N, 2], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("predictions contain NaN or infinity")
    if not np.equal(array, np.floor(array)).all():
        raise ValueError("one-dimensional predictions must contain integer class labels")
    return array.astype(np.int64, copy=False)


def _reference_labels(labels: Any) -> np.ndarray:
    array = _as_numpy(labels)
    if array.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {array.shape}")
    if array.size == 0:
        raise ValueError("metrics require at least one example")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("labels must be finite numbers")
    if not np.equal(array, np.floor(array)).all():
        raise ValueError("labels must contain integer class labels")
    return array.astype(np.int64, copy=False)


def _validated_label_pair(predictions: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    predicted = _prediction_labels(predictions)
    references = _reference_labels(labels)
    if predicted.shape[0] != references.shape[0]:
        raise ValueError(
            f"prediction/label length mismatch: {predicted.shape[0]} != {references.shape[0]}"
        )
    allowed = np.array([0, 1])
    if not np.isin(predicted, allowed).all():
        raise ValueError("predictions must contain only binary labels 0 and 1")
    if not np.isin(references, allowed).all():
        raise ValueError("labels must contain only binary labels 0 and 1")
    return predicted, references


def accuracy(predictions: Any, labels: Any) -> float:
    """Compute binary classification accuracy as a fraction in ``[0, 1]``."""
    predicted, references = _validated_label_pair(predictions, labels)
    return float(np.mean(predicted == references))


def binary_f1(predictions: Any, labels: Any, *, positive_label: int = 1) -> float:
    """Compute binary F1, returning zero when its denominator is zero."""
    if positive_label != 1:
        raise ValueError("the locked H1 positive label is 1")
    predicted, references = _validated_label_pair(predictions, labels)
    true_positive = int(np.sum((predicted == 1) & (references == 1)))
    false_positive = int(np.sum((predicted == 1) & (references == 0)))
    false_negative = int(np.sum((predicted == 0) & (references == 1)))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2.0 * true_positive) / denominator


def primary_score(task_name: str, metrics: Mapping[str, float]) -> float:
    """Compute the task's preregistered scalar from already computed metrics."""
    if task_name not in H1_TASKS:
        raise ValueError(f"unknown H1 task: {task_name!r}")
    try:
        task_accuracy = float(metrics["accuracy"])
    except KeyError as exc:
        raise ValueError("metrics are missing accuracy") from exc
    if not math.isfinite(task_accuracy) or not 0.0 <= task_accuracy <= 1.0:
        raise ValueError("accuracy must be a finite fraction in [0, 1]")
    if task_name not in F1_TASKS:
        return task_accuracy
    try:
        task_f1 = float(metrics["f1"])
    except KeyError as exc:
        raise ValueError(f"metrics for {task_name} are missing f1") from exc
    if not math.isfinite(task_f1) or not 0.0 <= task_f1 <= 1.0:
        raise ValueError("f1 must be a finite fraction in [0, 1]")
    return (task_accuracy + task_f1) / 2.0


def compute_task_metrics(
    task_name: str,
    predictions: Any,
    labels: Any,
    *,
    validation_loss: float | None = None,
) -> dict[str, float]:
    """Return accuracy, required F1, primary score, and optional mean loss."""
    if task_name not in H1_TASKS:
        raise ValueError(f"unknown H1 task: {task_name!r}")
    predicted, references = _validated_label_pair(predictions, labels)
    result = {"accuracy": float(np.mean(predicted == references))}
    if task_name in F1_TASKS:
        result["f1"] = binary_f1(predicted, references)
    result["primary_score"] = primary_score(task_name, result)
    if validation_loss is not None:
        loss = float(validation_loss)
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError("validation_loss must be finite and non-negative")
        result["validation_loss"] = loss
    return result


def compute_metrics(
    predictions: Any,
    labels: Any,
    task_name: str,
    *,
    validation_loss: float | None = None,
) -> dict[str, float]:
    """Conventional argument-order alias for :func:`compute_task_metrics`."""
    return compute_task_metrics(
        task_name,
        predictions,
        labels,
        validation_loss=validation_loss,
    )


def example_weighted_mean_loss(
    batch_mean_losses: Sequence[float],
    batch_example_counts: Sequence[int],
) -> float:
    """Combine batch-mean losses without overweighting a short final batch."""
    if len(batch_mean_losses) != len(batch_example_counts):
        raise ValueError("loss and example-count sequences must have equal length")
    if not batch_mean_losses:
        raise ValueError("at least one batch is required")
    weighted_sum = 0.0
    total_examples = 0
    for loss, count in zip(batch_mean_losses, batch_example_counts, strict=True):
        numeric_loss = float(loss)
        if not math.isfinite(numeric_loss) or numeric_loss < 0.0:
            raise ValueError("batch losses must be finite and non-negative")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("batch example counts must be non-negative integers")
        weighted_sum += numeric_loss * count
        total_examples += count
    if total_examples == 0:
        raise ValueError("at least one example is required")
    return weighted_sum / total_examples
