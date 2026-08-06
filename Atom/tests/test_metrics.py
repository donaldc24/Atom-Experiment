"""Tests for preregistered task metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cgmoe_h1.metrics import (
    accuracy,
    binary_f1,
    compute_metrics,
    compute_task_metrics,
    example_weighted_mean_loss,
    primary_score,
)


def test_accuracy_accepts_logits_and_torch_tensors() -> None:
    logits = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]])
    labels = torch.tensor([1, 0, 0, 1])

    assert accuracy(logits, labels) == 0.75


def test_binary_f1_uses_label_one_as_positive() -> None:
    # tp=2, fp=1, fn=1 -> 4 / 6
    assert binary_f1([1, 1, 1, 0, 0], [1, 1, 0, 1, 0]) == pytest.approx(2 / 3)


def test_binary_f1_zero_denominator_is_zero() -> None:
    assert binary_f1([0, 0, 0], [0, 0, 0]) == 0.0


@pytest.mark.parametrize("task", ("sst2", "rte", "qnli"))
def test_accuracy_only_tasks_have_predeclared_primary(task: str) -> None:
    result = compute_task_metrics(task, [0, 1, 0, 0], [0, 1, 1, 0], validation_loss=0.4)

    assert result == {"accuracy": 0.75, "primary_score": 0.75, "validation_loss": 0.4}


@pytest.mark.parametrize("task", ("mrpc", "qqp"))
def test_pair_tasks_average_accuracy_and_f1(task: str) -> None:
    result = compute_task_metrics(task, [1, 1, 0, 0], [1, 0, 1, 0])

    assert result["accuracy"] == 0.5
    assert result["f1"] == 0.5
    assert result["primary_score"] == 0.5
    assert primary_score(task, result) == 0.5


def test_compute_metrics_conventional_alias() -> None:
    expected = compute_task_metrics("sst2", [1, 0], [1, 1])

    assert compute_metrics([1, 0], [1, 1], "sst2") == expected


def test_example_weighted_loss_handles_short_final_batch() -> None:
    # A naive mean would produce 2.0; 8 examples at 1 and 2 at 3 is 1.4.
    assert example_weighted_mean_loss([1.0, 3.0], [8, 2]) == pytest.approx(1.4)


@pytest.mark.parametrize(
    ("predictions", "labels", "message"),
    (
        ([0, 1], [0], "length mismatch"),
        ([0, 2], [0, 1], "binary labels"),
        ([0.2, 1.0], [0, 1], "integer class labels"),
        ([[0.1, 0.2, 0.7]], [1], "binary logits"),
        ([0], [], "at least one example"),
        ([math.nan], [0], "NaN"),
    ),
)
def test_invalid_metric_inputs_are_rejected(
    predictions: object,
    labels: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        accuracy(predictions, labels)


def test_metric_rejects_nonfinite_or_negative_loss() -> None:
    with pytest.raises(ValueError, match="validation_loss"):
        compute_task_metrics("sst2", np.array([0]), np.array([0]), validation_loss=np.nan)
    with pytest.raises(ValueError, match="batch losses"):
        example_weighted_mean_loss([-0.1], [1])
