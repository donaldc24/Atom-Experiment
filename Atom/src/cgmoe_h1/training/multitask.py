"""Deterministic batch scheduling and shared-model multitask training."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .trainer import (
    Batch,
    EvaluationResult,
    GradientCallback,
    MetricFunction,
    RegularizationFunction,
    StateCaptureFunction,
    StateRestoreFunction,
    TrainEpochResult,
    _finish_epoch,
    _write_checkpoint,
    _capture_with,
    assert_nonzero_gradient,
    assert_zero_or_no_gradient,
    evaluate,
    gradient_l2_norm,
    parameter_l2_norm,
    restore_model_state,
    train_step,
)

ScheduleMode = Literal["complete_pass", "uniform"]


@dataclass(frozen=True)
class TaskBatch:
    """A batch paired with the task context required to interpret it."""

    task_id: str
    batch: Batch


def _loader_length(loader: Iterable[Batch], task_id: str) -> int:
    try:
        length = len(loader)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(
            f"task loader {task_id!r} has no length; complete_pass scheduling "
            "requires sized loaders"
        ) from error
    if length <= 0:
        raise ValueError(f"task loader {task_id!r} is empty")
    return int(length)


class UniformTaskBatchIterator:
    """Yield repeatably scheduled batches from multiple task loaders.

    ``complete_pass`` is the confirmatory H1 mode: every batch from every loader is
    consumed exactly once in a seeded shuffled schedule.  ``uniform`` is the roadmap's
    development sampler: select a task uniformly for a fixed number of steps and reset
    its iterator after exhaustion.
    """

    def __init__(
        self,
        task_loaders: Mapping[str, Iterable[Batch]],
        *,
        seed: int = 0,
        epoch: int = 0,
        mode: ScheduleMode = "complete_pass",
        steps_per_epoch: int | None = None,
    ) -> None:
        if not task_loaders:
            raise ValueError("at least one task loader is required")
        if any(not task_id for task_id in task_loaders):
            raise ValueError("task IDs must be non-empty strings")
        if mode not in ("complete_pass", "uniform"):
            raise ValueError(f"unsupported schedule mode: {mode!r}")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if mode == "complete_pass" and steps_per_epoch is not None:
            raise ValueError("steps_per_epoch is only valid for uniform mode")

        self.task_loaders = dict(task_loaders)
        self.task_ids = tuple(self.task_loaders)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.mode = mode
        if mode == "complete_pass":
            self._lengths = {
                task_id: _loader_length(loader, task_id)
                for task_id, loader in self.task_loaders.items()
            }
            self.steps_per_epoch = sum(self._lengths.values())
        else:
            self._lengths = {}
            if steps_per_epoch is None or steps_per_epoch <= 0:
                raise ValueError("uniform mode requires a positive steps_per_epoch")
            self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _random(self) -> random.Random:
        # Do not use hash(): its process-level salt would defeat reproducibility.
        return random.Random(self.seed + self.epoch * 1_000_003)

    def task_schedule(self) -> tuple[str, ...]:
        """Return the task-ID schedule without consuming any loader."""

        rng = self._random()
        if self.mode == "complete_pass":
            schedule = [
                task_id
                for task_id in self.task_ids
                for _ in range(self._lengths[task_id])
            ]
            rng.shuffle(schedule)
            return tuple(schedule)
        return tuple(rng.choice(self.task_ids) for _ in range(self.steps_per_epoch))

    def expected_batch_counts(self) -> dict[str, int]:
        """Count selections in this epoch's deterministic schedule."""

        counts = Counter(self.task_schedule())
        return {task_id: counts[task_id] for task_id in self.task_ids}

    def for_epoch(self, epoch: int) -> UniformTaskBatchIterator:
        """Create the same scheduler configuration for a different epoch."""

        return type(self)(
            self.task_loaders,
            seed=self.seed,
            epoch=epoch,
            mode=self.mode,
            steps_per_epoch=self.steps_per_epoch if self.mode == "uniform" else None,
        )

    def __iter__(self) -> Iterator[TaskBatch]:
        iterators = {task_id: iter(loader) for task_id, loader in self.task_loaders.items()}
        for task_id in self.task_schedule():
            iterator = iterators[task_id]
            try:
                batch = next(iterator)
            except StopIteration:
                if self.mode == "complete_pass":
                    raise RuntimeError(
                        f"task loader {task_id!r} ended before its reported length"
                    ) from None
                iterator = iter(self.task_loaders[task_id])
                iterators[task_id] = iterator
                try:
                    batch = next(iterator)
                except StopIteration:
                    raise ValueError(f"task loader {task_id!r} is empty") from None
            yield TaskBatch(task_id, batch)


# The explicit alias makes the contract-safe intent discoverable while retaining the
# roadmap's suggested UniformTaskBatchIterator name.
BalancedTaskBatchIterator = UniformTaskBatchIterator


def _iter_atom_like_layers(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Find atom layers by their public tensor/task interface, without a hard import."""

    for name, module in model.named_modules():
        if all(hasattr(module, attribute) for attribute in ("atom_u", "atom_v", "coefficients")):
            coefficients = getattr(module, "coefficients")
            task_ids = getattr(module, "task_ids", None)
            if isinstance(coefficients, torch.Tensor) and task_ids is not None:
                yield name or "<root>", module


def assert_shared_atom_gradient_contract(model: nn.Module, task_id: str | None) -> None:
    """Assert the roadmap's shared-atom gradient invariants after ``backward``.

    This function has the same signature as ``gradient_callback`` and is intended
    for development runs.  It checks shared atom flow, active/inactive coefficient
    rows, and every parameter explicitly marked frozen.
    """

    if task_id is None:
        raise AssertionError("shared-atom gradient checks require an active task ID")
    layers = tuple(_iter_atom_like_layers(model))
    if not layers:
        raise AssertionError("model contains no atom-like layers")

    atom_parameters = [
        parameter
        for _, layer in layers
        for parameter in (getattr(layer, "atom_u"), getattr(layer, "atom_v"))
    ]
    assert_nonzero_gradient(atom_parameters, "shared atoms")

    active_rows: list[torch.Tensor] = []
    inactive_rows: list[torch.Tensor] = []
    for layer_name, layer in layers:
        task_ids = tuple(getattr(layer, "task_ids"))
        if task_id not in task_ids:
            raise AssertionError(f"atom layer {layer_name!r} does not support task {task_id!r}")
        gradient = getattr(layer, "coefficients").grad
        active_index = task_ids.index(task_id)
        if gradient is None:
            continue
        active_rows.append(gradient[active_index])
        inactive_rows.extend(
            gradient[index] for index in range(len(task_ids)) if index != active_index
        )
    # Rows are tensors rather than Parameters, but the generic helpers only rely
    # on their ``grad`` attribute.  Check row values directly because slicing a
    # gradient tensor does not itself expose ``.grad``.
    if not any(bool(torch.any(row != 0).item()) for row in active_rows):
        raise AssertionError("active task coefficient rows have no nonzero gradient")
    if any(bool(torch.any(row != 0).item()) for row in inactive_rows):
        raise AssertionError("inactive task coefficient rows unexpectedly have gradients")

    frozen_parameters = [
        parameter for parameter in model.parameters() if not parameter.requires_grad
    ]
    assert_zero_or_no_gradient(frozen_parameters, "frozen parameters")


def collect_shared_atom_diagnostics(
    model: nn.Module,
    *,
    near_zero_threshold: float = 1e-6,
    top_n: int = 4,
) -> dict[str, Any]:
    """Collect JSON-friendly coefficient and atom diagnostics for every task."""

    if near_zero_threshold < 0:
        raise ValueError("near_zero_threshold must be non-negative")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    layers = tuple(_iter_atom_like_layers(model))
    if not layers:
        raise ValueError("model contains no atom-like layers")

    atom_parameters = [
        parameter
        for _, layer in layers
        for parameter in (getattr(layer, "atom_u"), getattr(layer, "atom_v"))
    ]
    task_ids = tuple(getattr(layers[0][1], "task_ids"))
    task_diagnostics: dict[str, Any] = {}
    for task_id in task_ids:
        rows: list[torch.Tensor] = []
        top_by_layer: dict[str, list[dict[str, float | int]]] = {}
        for layer_name, layer in layers:
            layer_tasks = tuple(getattr(layer, "task_ids"))
            if layer_tasks != task_ids:
                raise ValueError("all atom layers must use the same task order")
            row = getattr(layer, "coefficients")[layer_tasks.index(task_id)].detach()
            rows.append(row.reshape(-1))
            magnitudes = row.abs().cpu().tolist()
            selected = sorted(
                range(len(magnitudes)),
                key=lambda index: (-magnitudes[index], index),
            )[: min(top_n, len(magnitudes))]
            top_by_layer[layer_name] = [
                {"atom_index": index, "absolute_coefficient": float(magnitudes[index])}
                for index in selected
            ]
        coefficients = torch.cat(rows)
        task_diagnostics[task_id] = {
            "coefficient_l2_norm": float(torch.linalg.vector_norm(coefficients).cpu()),
            "coefficient_near_zero_count": int(
                torch.count_nonzero(coefficients.abs() <= near_zero_threshold).cpu()
            ),
            "coefficient_count": int(coefficients.numel()),
            "top_used_atoms_by_layer": top_by_layer,
        }

    return {
        "atom_gradient_l2_norm_last_step": gradient_l2_norm(atom_parameters),
        "atom_parameter_l2_norm": parameter_l2_norm(atom_parameters),
        "tasks": task_diagnostics,
    }


@dataclass(frozen=True)
class MultitaskEpochResult:
    """Per-task and global records for one shared-model epoch."""

    epoch: int
    training: dict[str, TrainEpochResult]
    global_training: TrainEpochResult
    validation: dict[str, EvaluationResult]
    selection_score: float
    elapsed_seconds: float
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "training": {task: result.to_dict() for task, result in self.training.items()},
            "global_training": self.global_training.to_dict(),
            "validation": {task: result.to_dict() for task, result in self.validation.items()},
            "selection_score": self.selection_score,
            "elapsed_seconds": self.elapsed_seconds,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class MultitaskTrainingResult:
    """Shared-model training history and best checkpoint."""

    history: tuple[MultitaskEpochResult, ...]
    best_epoch: int
    best_score: float
    best_state_dict: dict[str, Any] = field(repr=False)

    @property
    def best_validation(self) -> dict[str, EvaluationResult]:
        return self.history[self.best_epoch - 1].validation

    @property
    def final_validation(self) -> dict[str, EvaluationResult]:
        return self.history[-1].validation

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": [epoch.to_dict() for epoch in self.history],
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_validation": {
                task: result.to_dict() for task, result in self.best_validation.items()
            },
            "final_validation": {
                task: result.to_dict() for task, result in self.final_validation.items()
            },
        }


@dataclass
class _Accumulator:
    weighted_loss: float = 0.0
    weighted_classification: float = 0.0
    weighted_regularization: float = 0.0
    examples: int = 0
    batches: int = 0

    def add(self, loss: float, classification: float, regularization: float, examples: int) -> None:
        self.weighted_loss += loss * examples
        self.weighted_classification += classification * examples
        self.weighted_regularization += regularization * examples
        self.examples += examples
        self.batches += 1

    def finish(self) -> TrainEpochResult:
        return _finish_epoch(
            weighted_loss=self.weighted_loss,
            weighted_classification_loss=self.weighted_classification,
            weighted_regularization_loss=self.weighted_regularization,
            examples=self.examples,
            batches=self.batches,
        )


def train_multitask_epoch(
    model: nn.Module,
    task_loaders: Mapping[str, Iterable[Batch]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    seed: int,
    epoch: int,
    regularization_fn: RegularizationFunction | None = None,
    gradient_callback: GradientCallback | None = None,
    schedule_mode: ScheduleMode = "complete_pass",
    steps_per_epoch: int | None = None,
) -> tuple[dict[str, TrainEpochResult], TrainEpochResult]:
    """Train one shared-model epoch and return exact per-task accounting."""

    model.train()
    scheduler = UniformTaskBatchIterator(
        task_loaders,
        seed=seed,
        epoch=epoch,
        mode=schedule_mode,
        steps_per_epoch=steps_per_epoch,
    )
    accumulators = {task_id: _Accumulator() for task_id in task_loaders}
    global_accumulator = _Accumulator()
    for task_batch in scheduler:
        result = train_step(
            model,
            task_batch.batch,
            optimizer,
            device,
            task_id=task_batch.task_id,
            regularization_fn=regularization_fn,
            gradient_callback=gradient_callback,
        )
        accumulators[task_batch.task_id].add(
            result.loss,
            result.classification_loss,
            result.regularization_loss,
            result.examples,
        )
        global_accumulator.add(
            result.loss,
            result.classification_loss,
            result.regularization_loss,
            result.examples,
        )
    return (
        {task_id: accumulator.finish() for task_id, accumulator in accumulators.items()},
        global_accumulator.finish(),
    )


def train_multitask(
    model: nn.Module,
    train_loaders: Mapping[str, Iterable[Batch]],
    validation_loaders: Mapping[str, Iterable[Batch]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    seed: int,
    device: torch.device | str = "cpu",
    metric_fns: Mapping[str, MetricFunction] | None = None,
    primary_metrics: Mapping[str, str] | None = None,
    regularization_fn: RegularizationFunction | None = None,
    gradient_callback: GradientCallback | None = None,
    diagnostics_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    schedule_mode: ScheduleMode = "complete_pass",
    steps_per_epoch: int | None = None,
    restore_best: bool = True,
    checkpoint_path: str | Path | None = None,
    state_capture_fn: StateCaptureFunction | None = None,
    state_restore_fn: StateRestoreFunction | None = None,
    epoch_callback: Callable[[MultitaskEpochResult], None] | None = None,
) -> MultitaskTrainingResult:
    """Train shared parameters and heads with unweighted task checkpoint selection."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if (state_capture_fn is None) != (state_restore_fn is None):
        raise ValueError("state_capture_fn and state_restore_fn must be supplied together")
    if set(train_loaders) != set(validation_loaders):
        raise ValueError("train and validation loaders must have identical task IDs")
    if not train_loaders:
        raise ValueError("at least one task is required")
    metric_fns = dict(metric_fns or {})
    primary_metrics = dict(primary_metrics or {})
    unknown_metric_tasks = set(metric_fns) - set(train_loaders)
    unknown_primary_tasks = set(primary_metrics) - set(train_loaders)
    if unknown_metric_tasks or unknown_primary_tasks:
        raise ValueError("metric configuration contains unknown task IDs")

    destination = torch.device(device)
    model.to(destination)
    history: list[MultitaskEpochResult] = []
    best_epoch = 0
    best_score = -math.inf
    best_state: dict[str, Any] | None = None
    path = Path(checkpoint_path) if checkpoint_path is not None else None

    for epoch_number in range(1, epochs + 1):
        started = time.perf_counter()
        training, global_training = train_multitask_epoch(
            model,
            train_loaders,
            optimizer,
            destination,
            seed=seed,
            epoch=epoch_number - 1,
            regularization_fn=regularization_fn,
            gradient_callback=gradient_callback,
            schedule_mode=schedule_mode,
            steps_per_epoch=steps_per_epoch,
        )
        validation = {
            task_id: evaluate(
                model,
                validation_loaders[task_id],
                destination,
                task_id=task_id,
                metric_fn=metric_fns.get(task_id),
                scalar_metric_name=primary_metrics.get(task_id, "accuracy"),
            )
            for task_id in train_loaders
        }
        scores = [
            result.metric(primary_metrics.get(task_id, "accuracy"))
            for task_id, result in validation.items()
        ]
        selection_score = sum(scores) / len(scores)
        if not math.isfinite(selection_score):
            raise FloatingPointError(f"multitask selection score is non-finite: {selection_score}")
        if diagnostics_fn is not None:
            diagnostics = dict(diagnostics_fn(model))
        elif any(_iter_atom_like_layers(model)):
            diagnostics = collect_shared_atom_diagnostics(model)
        else:
            diagnostics = {}
        record = MultitaskEpochResult(
            epoch=epoch_number,
            training=training,
            global_training=global_training,
            validation=validation,
            selection_score=selection_score,
            elapsed_seconds=time.perf_counter() - started,
            diagnostics=diagnostics,
        )
        history.append(record)
        # Strict comparison implements the contract's earlier-epoch tie break.
        if selection_score > best_score:
            best_epoch = epoch_number
            best_score = selection_score
            best_state = _capture_with(model, state_capture_fn)
            if path is not None:
                _write_checkpoint(
                    path,
                    epoch=epoch_number,
                    score=selection_score,
                    state_dict=best_state,
                )
        if epoch_callback is not None:
            epoch_callback(record)

    assert best_state is not None
    result = MultitaskTrainingResult(tuple(history), best_epoch, best_score, best_state)
    if restore_best:
        (state_restore_fn or restore_model_state)(model, best_state)
    return result
