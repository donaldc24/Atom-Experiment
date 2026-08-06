from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from cgmoe_h1.training.trainer import (
    assert_nonzero_gradient,
    assert_zero_or_no_gradient,
    capture_model_state,
    evaluate,
    gradient_l2_norm,
    restore_model_state,
    train_single_task,
    train_step,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def _loader(*, shuffle: bool = False) -> DataLoader:
    features = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    rows = [
        {
            "features": feature,
            "labels": label,
            "example_id": f"row-{index}",
            "provenance": {"split": "synthetic"},
        }
        for index, (feature, label) in enumerate(zip(features, labels, strict=True))
    ]
    return DataLoader(rows, batch_size=2, shuffle=shuffle)


def test_evaluate_filters_metadata_and_reports_exact_accounting() -> None:
    model = TinyClassifier()
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
        model.classifier.bias.zero_()

    result = evaluate(model, _loader(), "cpu")

    assert result.metrics == {"accuracy": 1.0}
    assert result.examples == 6
    assert result.batches == 3
    assert result.loss > 0
    assert result.predictions == (0, 0, 0, 1, 1, 1)


def test_single_task_training_decreases_loss_and_captures_best_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    checkpoint = tmp_path / "best.pt"

    result = train_single_task(
        model,
        _loader(),
        _loader(),
        optimizer,
        epochs=3,
        checkpoint_path=checkpoint,
    )

    assert len(result.history) == 3
    assert result.history[-1].training.loss < result.history[0].training.loss
    assert result.best_epoch == 1  # Accuracy ties; the contract keeps the earlier epoch.
    assert result.best_score == 1.0
    assert result.final_validation.metrics["accuracy"] == 1.0
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["epoch"] == 1
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), result.best_state_dict[name])


def test_state_snapshot_is_not_aliased() -> None:
    model = TinyClassifier()
    state = capture_model_state(model)
    original = state["classifier.weight"].clone()
    with torch.no_grad():
        model.classifier.weight.add_(100)
    assert torch.equal(state["classifier.weight"], original)

    restore_model_state(model, state)
    assert torch.equal(model.classifier.weight, original)


def test_train_step_regularization_and_gradient_debug_helpers() -> None:
    torch.manual_seed(9)
    model = TinyClassifier()
    frozen = nn.Parameter(torch.ones(1), requires_grad=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = next(iter(_loader()))

    def check_gradients(current: nn.Module, task_id: str | None) -> None:
        assert task_id is None
        assert_nonzero_gradient(current.parameters(), "classifier")
        assert_zero_or_no_gradient([frozen], "frozen")
        assert gradient_l2_norm(current.parameters()) > 0

    step = train_step(
        model,
        batch,
        optimizer,
        "cpu",
        regularization_fn=lambda current, task: sum(
            parameter.square().mean() for parameter in current.parameters()
        )
        * 1e-3,
        gradient_callback=check_gradients,
    )

    assert step.regularization_loss > 0
    assert math.isclose(
        step.loss,
        step.classification_loss + step.regularization_loss,
        rel_tol=1e-6,
    )


def test_empty_validation_loader_fails_clearly() -> None:
    with pytest.raises(ValueError, match="produced no examples"):
        evaluate(TinyClassifier(), [], "cpu")
