"""Reusable single-task training and evaluation loops.

The functions in this module deliberately know very little about BERT.  A model may
return a tensor, a mapping containing ``logits``, or an object with a ``logits``
attribute.  Multitask models can either expose ``set_task``/``set_active_task`` or
accept ``task_id`` as a keyword argument to ``forward``.
"""

from __future__ import annotations

import copy
import inspect
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Batch: TypeAlias = Mapping[str, Any]
MetricValue: TypeAlias = float | int | Tensor
MetricOutput: TypeAlias = Mapping[str, MetricValue] | MetricValue
MetricFunction: TypeAlias = Callable[[Tensor, Tensor], MetricOutput]
RegularizationFunction: TypeAlias = Callable[[nn.Module, str | None], Tensor | float]
GradientCallback: TypeAlias = Callable[[nn.Module, str | None], None]
StateCaptureFunction: TypeAlias = Callable[[nn.Module], Mapping[str, Any]]
StateRestoreFunction: TypeAlias = Callable[[nn.Module, Mapping[str, Any]], None]

_STANDARD_MODEL_INPUT_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "head_mask",
        "inputs_embeds",
    }
)


@dataclass(frozen=True)
class EvaluationResult:
    """Loss, metrics, and accounting from one evaluation pass."""

    loss: float
    metrics: dict[str, float]
    examples: int
    batches: int
    predictions: tuple[int, ...] = field(repr=False)
    labels: tuple[int, ...] = field(repr=False)

    def metric(self, name: str) -> float:
        """Return ``name`` with an informative error for a misspelled metric."""

        if name not in self.metrics:
            available = ", ".join(sorted(self.metrics)) or "<none>"
            raise KeyError(f"metric {name!r} was not computed; available: {available}")
        return self.metrics[name]

    def to_dict(self, *, include_outputs: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "loss": self.loss,
            "metrics": dict(self.metrics),
            "examples": self.examples,
            "batches": self.batches,
        }
        if include_outputs:
            result["predictions"] = list(self.predictions)
            result["labels"] = list(self.labels)
        return result


@dataclass(frozen=True)
class StepResult:
    """Scalar losses and example count for one optimizer update."""

    loss: float
    classification_loss: float
    regularization_loss: float
    examples: int


@dataclass(frozen=True)
class TrainEpochResult:
    """Example-weighted losses and accounting from one training epoch."""

    loss: float
    classification_loss: float
    regularization_loss: float
    examples: int
    batches: int
    optimizer_steps: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "classification_loss": self.classification_loss,
            "regularization_loss": self.regularization_loss,
            "examples": self.examples,
            "batches": self.batches,
            "optimizer_steps": self.optimizer_steps,
        }


@dataclass(frozen=True)
class EpochResult:
    """Training and validation record for a completed epoch."""

    epoch: int
    training: TrainEpochResult
    validation: EvaluationResult
    selection_score: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "training": self.training.to_dict(),
            "validation": self.validation.to_dict(),
            "selection_score": self.selection_score,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class TrainingResult:
    """Complete single-task history plus an immutable best-checkpoint snapshot."""

    history: tuple[EpochResult, ...]
    best_epoch: int
    best_score: float
    best_state_dict: dict[str, Any] = field(repr=False)

    @property
    def best_validation(self) -> EvaluationResult:
        return self.history[self.best_epoch - 1].validation

    @property
    def final_validation(self) -> EvaluationResult:
        return self.history[-1].validation

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": [epoch.to_dict() for epoch in self.history],
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_validation": self.best_validation.to_dict(),
            "final_validation": self.final_validation.to_dict(),
        }


def move_batch_to_device(batch: Batch, device: torch.device | str) -> dict[str, Any]:
    """Move tensor values in a batch to ``device`` while preserving metadata."""

    destination = torch.device(device)
    return {
        key: value.to(destination) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def split_labels(batch: Batch) -> tuple[dict[str, Any], Tensor]:
    """Separate labels from model inputs, accepting common singular/plural keys."""

    label_key = "labels" if "labels" in batch else "label" if "label" in batch else None
    if label_key is None:
        raise KeyError("each batch must contain a 'labels' (or 'label') tensor")
    labels = batch[label_key]
    if not isinstance(labels, Tensor):
        raise TypeError(f"batch {label_key!r} must be a torch.Tensor")
    inputs = {key: value for key, value in batch.items() if key != label_key}
    return inputs, labels.long()


def set_model_task(model: nn.Module, task_id: str | None) -> bool:
    """Set a model-owned task context when one is available.

    Returns whether a task setter was called.  Forward dispatch remains independent:
    a wrapper is allowed to both own a context and explicitly accept ``task_id``.
    """

    if task_id is None:
        return False
    for method_name in ("set_task", "set_active_task"):
        method = getattr(model, method_name, None)
        if callable(method):
            method(task_id)
            return True
    return False


def _forward_accepts_task_id(model: nn.Module) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "task_id" in signature.parameters


def _filter_model_inputs(model: nn.Module, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop provenance/collation metadata before dispatching to the classifier.

    Explicitly named forward arguments are retained, which keeps small synthetic and
    custom classifiers generic.  For wrappers accepting ``**kwargs``, only canonical
    Transformer tensor inputs are added; fields such as ``example_id`` and
    ``provenance`` must never leak into BERT.
    """

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        parameters = {}
    allowed = {
        name
        for name, parameter in parameters.items()
        if name not in {"self", "task_id"}
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        allowed.update(_STANDARD_MODEL_INPUT_KEYS)
    if not parameters:
        allowed.update(_STANDARD_MODEL_INPUT_KEYS)
    return {
        key: value
        for key, value in inputs.items()
        if key in allowed and isinstance(value, Tensor)
    }


def extract_logits(output: Any) -> Tensor:
    """Extract logits from common PyTorch and Transformers return conventions."""

    logits: Any
    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping) and "logits" in output:
        logits = output["logits"]
    elif hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, Sequence) and output:
        logits = output[0]
    else:
        raise TypeError(
            "model output must be a Tensor, contain 'logits', or expose a logits attribute"
        )
    if not isinstance(logits, Tensor):
        raise TypeError("extracted logits are not a torch.Tensor")
    if logits.ndim < 2:
        raise ValueError(
            f"classification logits must have at least 2 dimensions, got {logits.shape}"
        )
    return logits


def forward_logits(
    model: nn.Module,
    inputs: Mapping[str, Any],
    task_id: str | None = None,
) -> Tensor:
    """Run a duck-typed classifier and return its logits tensor."""

    set_model_task(model, task_id)
    kwargs = _filter_model_inputs(model, inputs)
    if task_id is not None and _forward_accepts_task_id(model):
        kwargs["task_id"] = task_id
    return extract_logits(model(**kwargs))


def _as_regularization_loss(
    value: Tensor | float,
    reference: Tensor,
) -> Tensor:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("regularization_fn must return a scalar")
        return value.to(device=reference.device, dtype=reference.dtype)
    return reference.new_tensor(float(value))


def train_step(
    model: nn.Module,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    task_id: str | None = None,
    regularization_fn: RegularizationFunction | None = None,
    gradient_callback: GradientCallback | None = None,
) -> StepResult:
    """Perform one explicit mean-cross-entropy optimizer update."""

    moved = move_batch_to_device(batch, device)
    inputs, labels = split_labels(moved)
    if labels.numel() == 0:
        raise ValueError("empty batches are not supported")

    optimizer.zero_grad(set_to_none=True)
    logits = forward_logits(model, inputs, task_id)
    classification_loss = F.cross_entropy(logits, labels, reduction="mean")
    regularization_loss = classification_loss.new_zeros(())
    if regularization_fn is not None:
        regularization_loss = _as_regularization_loss(
            regularization_fn(model, task_id), classification_loss
        )
    total_loss = classification_loss + regularization_loss
    if not bool(torch.isfinite(total_loss).item()):
        raise FloatingPointError(
            f"non-finite loss for task {task_id!r}: {float(total_loss.detach().cpu())}"
        )
    total_loss.backward()
    if gradient_callback is not None:
        gradient_callback(model, task_id)
    optimizer.step()

    return StepResult(
        loss=float(total_loss.detach().cpu()),
        classification_loss=float(classification_loss.detach().cpu()),
        regularization_loss=float(regularization_loss.detach().cpu()),
        examples=int(labels.numel()),
    )


def _finish_epoch(
    *,
    weighted_loss: float,
    weighted_classification_loss: float,
    weighted_regularization_loss: float,
    examples: int,
    batches: int,
) -> TrainEpochResult:
    if batches == 0 or examples == 0:
        raise ValueError("the training loader produced no examples")
    return TrainEpochResult(
        loss=weighted_loss / examples,
        classification_loss=weighted_classification_loss / examples,
        regularization_loss=weighted_regularization_loss / examples,
        examples=examples,
        batches=batches,
        optimizer_steps=batches,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    task_id: str | None = None,
    regularization_fn: RegularizationFunction | None = None,
    gradient_callback: GradientCallback | None = None,
) -> TrainEpochResult:
    """Train for one complete pass through ``dataloader``."""

    model.train()
    weighted_loss = 0.0
    weighted_classification = 0.0
    weighted_regularization = 0.0
    examples = 0
    batches = 0
    for batch in dataloader:
        result = train_step(
            model,
            batch,
            optimizer,
            device,
            task_id=task_id,
            regularization_fn=regularization_fn,
            gradient_callback=gradient_callback,
        )
        weighted_loss += result.loss * result.examples
        weighted_classification += result.classification_loss * result.examples
        weighted_regularization += result.regularization_loss * result.examples
        examples += result.examples
        batches += 1
    return _finish_epoch(
        weighted_loss=weighted_loss,
        weighted_classification_loss=weighted_classification,
        weighted_regularization_loss=weighted_regularization,
        examples=examples,
        batches=batches,
    )


def _normalise_metrics(output: MetricOutput, *, scalar_name: str = "score") -> dict[str, float]:
    raw_metrics = output if isinstance(output, Mapping) else {scalar_name: output}
    metrics: dict[str, float] = {}
    for name, value in raw_metrics.items():
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(f"metric {name!r} is not scalar")
            numeric = float(value.detach().cpu())
        else:
            numeric = float(value)
        if not math.isfinite(numeric):
            raise FloatingPointError(f"metric {name!r} is non-finite: {numeric}")
        metrics[str(name)] = numeric
    return metrics


def _accuracy(predictions: Tensor, labels: Tensor) -> dict[str, float]:
    return {"accuracy": float((predictions == labels).float().mean())}


def evaluate(
    model: nn.Module,
    dataloader: Iterable[Batch],
    device: torch.device | str,
    *,
    task_id: str | None = None,
    metric_fn: MetricFunction | None = None,
    scalar_metric_name: str = "score",
) -> EvaluationResult:
    """Evaluate a classifier with example-weighted cross-entropy loss."""

    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    examples = 0
    batches = 0
    all_predictions: list[Tensor] = []
    all_labels: list[Tensor] = []
    try:
        with torch.no_grad():
            for batch in dataloader:
                moved = move_batch_to_device(batch, device)
                inputs, labels = split_labels(moved)
                if labels.numel() == 0:
                    continue
                logits = forward_logits(model, inputs, task_id)
                loss = F.cross_entropy(logits, labels, reduction="mean")
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f"non-finite validation loss for task {task_id!r}")
                predictions = logits.argmax(dim=-1)
                weighted_loss += float(loss.detach().cpu()) * int(labels.numel())
                examples += int(labels.numel())
                batches += 1
                all_predictions.append(predictions.detach().cpu())
                all_labels.append(labels.detach().cpu())
    finally:
        model.train(was_training)

    if batches == 0 or examples == 0:
        raise ValueError("the validation loader produced no examples")
    predictions_tensor = torch.cat(all_predictions)
    labels_tensor = torch.cat(all_labels)
    metrics = _normalise_metrics(
        (metric_fn or _accuracy)(predictions_tensor, labels_tensor),
        scalar_name=scalar_metric_name,
    )
    return EvaluationResult(
        loss=weighted_loss / examples,
        metrics=metrics,
        examples=examples,
        batches=batches,
        predictions=tuple(int(value) for value in predictions_tensor.tolist()),
        labels=tuple(int(value) for value in labels_tensor.tolist()),
    )


def capture_model_state(model: nn.Module) -> dict[str, Any]:
    """Clone a state dict onto CPU so later updates cannot mutate the snapshot."""

    snapshot: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        snapshot[name] = (
            value.detach().cpu().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
        )
    return snapshot


def _capture_with(
    model: nn.Module,
    capture_fn: StateCaptureFunction | None,
) -> dict[str, Any]:
    if capture_fn is None:
        return capture_model_state(model)
    captured = capture_fn(model)
    if not isinstance(captured, Mapping):
        raise TypeError("state_capture_fn must return a mapping")
    return {
        name: value.detach().cpu().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
        for name, value in captured.items()
    }


def restore_model_state(model: nn.Module, state_dict: Mapping[str, Any]) -> None:
    """Restore a snapshot produced by :func:`capture_model_state`."""

    model.load_state_dict(dict(state_dict), strict=True)


def _write_checkpoint(
    path: Path,
    *,
    epoch: int,
    score: float,
    state_dict: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(
        {"epoch": epoch, "score": score, "model_state_dict": dict(state_dict)},
        temporary_path,
    )
    temporary_path.replace(path)


def train_single_task(
    model: nn.Module,
    train_loader: Iterable[Batch],
    validation_loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    device: torch.device | str = "cpu",
    task_id: str | None = None,
    metric_fn: MetricFunction | None = None,
    primary_metric: str = "accuracy",
    score_fn: Callable[[EvaluationResult], float] | None = None,
    regularization_fn: RegularizationFunction | None = None,
    gradient_callback: GradientCallback | None = None,
    checkpoint_path: str | Path | None = None,
    restore_best: bool = True,
    state_capture_fn: StateCaptureFunction | None = None,
    state_restore_fn: StateRestoreFunction | None = None,
    epoch_callback: Callable[[EpochResult], None] | None = None,
) -> TrainingResult:
    """Train for a fixed budget, evaluating and retaining the best epoch.

    Ties deliberately keep the earlier epoch.  Training always runs for all requested
    epochs; restoring the best state happens only after the final-epoch evaluation.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if (state_capture_fn is None) != (state_restore_fn is None):
        raise ValueError("state_capture_fn and state_restore_fn must be supplied together")
    destination = torch.device(device)
    model.to(destination)
    history: list[EpochResult] = []
    best_epoch = 0
    best_score = -math.inf
    best_state: dict[str, Any] | None = None
    path = Path(checkpoint_path) if checkpoint_path is not None else None

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training = train_one_epoch(
            model,
            train_loader,
            optimizer,
            destination,
            task_id=task_id,
            regularization_fn=regularization_fn,
            gradient_callback=gradient_callback,
        )
        validation = evaluate(
            model,
            validation_loader,
            destination,
            task_id=task_id,
            metric_fn=metric_fn,
            scalar_metric_name=primary_metric,
        )
        selection_score = float(
            score_fn(validation) if score_fn else validation.metric(primary_metric)
        )
        if not math.isfinite(selection_score):
            raise FloatingPointError(f"selection score is non-finite: {selection_score}")
        record = EpochResult(
            epoch=epoch,
            training=training,
            validation=validation,
            selection_score=selection_score,
            elapsed_seconds=time.perf_counter() - started,
        )
        history.append(record)
        if selection_score > best_score:
            best_epoch = epoch
            best_score = selection_score
            best_state = _capture_with(model, state_capture_fn)
            if path is not None:
                _write_checkpoint(path, epoch=epoch, score=selection_score, state_dict=best_state)
        if epoch_callback is not None:
            epoch_callback(record)

    assert best_state is not None  # epochs > 0 and finite scores guarantee this.
    result = TrainingResult(tuple(history), best_epoch, best_score, best_state)
    if restore_best:
        (state_restore_fn or restore_model_state)(model, best_state)
    return result


def gradient_l2_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Compute a group gradient norm without modifying gradients."""

    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(squared)


def parameter_l2_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Compute a parameter-group L2 norm."""

    squared = sum(
        float(parameter.detach().float().pow(2).sum().cpu()) for parameter in parameters
    )
    return math.sqrt(squared)


def assert_nonzero_gradient(parameters: Iterable[nn.Parameter], name: str) -> None:
    """Assert that at least one tensor in a parameter group has a nonzero gradient."""

    parameters = tuple(parameters)
    if not parameters:
        raise AssertionError(f"gradient group {name!r} is empty")
    if not any(
        parameter.grad is not None and bool(torch.any(parameter.grad.detach() != 0).item())
        for parameter in parameters
    ):
        raise AssertionError(f"gradient group {name!r} has no nonzero gradient")


def assert_zero_or_no_gradient(parameters: Iterable[nn.Parameter], name: str) -> None:
    """Assert that every present gradient in a parameter group is exactly zero."""

    for parameter in parameters:
        if parameter.grad is not None and bool(torch.any(parameter.grad.detach() != 0).item()):
            raise AssertionError(f"gradient group {name!r} unexpectedly has a nonzero gradient")


def create_adamw_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    epsilon: float = 1e-8,
) -> torch.optim.AdamW:
    """Create the contract optimizer over exactly the trainable parameters."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=epsilon,
    )
