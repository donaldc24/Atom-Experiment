"""Experiment A: matched shared-LoRA/shared-atom validation frontier.

The protocol in ``docs/atom_validation_spec.md`` is deliberately encoded here
rather than inferred from existing result filenames.  A completed cell is the
smallest resumable unit; a cell is reusable only after its raw evaluations,
data provenance, model identity, accounting, and compact checkpoint have all
been validated.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypeAlias

import torch

from cgmoe_h1.config import H1_CONFIRMATORY_SEEDS, H1_TASKS, ExperimentConfig
from cgmoe_h1.experiments import (
    PreparedData,
    _run_metadata,
    build_atom_model,
    build_loaders,
    build_lora_model,
    prepare_data,
    save_compact_checkpoint,
)
from cgmoe_h1.metrics import compute_task_metrics
from cgmoe_h1.models.atoms import coefficient_l1_regularization
from cgmoe_h1.models.injection import extract_adapter_state_dict, load_adapter_state_dict
from cgmoe_h1.training.multitask import train_multitask
from cgmoe_h1.training.trainer import create_adamw_optimizer, evaluate
from cgmoe_h1.utils.parameters import active_adapter_operations, categorized_parameter_counts
from cgmoe_h1.utils.reproducibility import set_seed
from cgmoe_h1.utils.runtime import RuntimeMonitor
from cgmoe_h1.utils.serialization import read_json, write_json


SCHEMA_VERSION = 1
CAPACITIES = (1, 2, 4, 8)
SYSTEMS = ("shared_lora", "shared_atoms")
SEEDS = H1_CONFIRMATORY_SEEDS
TASKS = H1_TASKS
UPDATES = 3_750
MEAN_TOLERANCE = 0.005
WORST_TASK_TOLERANCE = 0.01
STORAGE_RATIO_LIMIT = 1.02
TARGET_COUNT = 4
TARGET_DIMENSION = (128, 128)
DEFAULT_OUTPUT_ROOT = Path("results/atom_validation/frontier")
DEFAULT_ATOM_REUSE_ROOT = Path("results")
PROTOCOL_FILENAME = "frontier_protocol.json"
SUMMARY_FILENAME = "frontier_summary.json"
REPORT_FILENAME = "frontier_report.md"

CellRunner: TypeAlias = Callable[
    [ExperimentConfig, ExperimentConfig, str, int, Path, PreparedData | None, bool],
    dict[str, Any],
]


def validate_locked_configs(baseline: ExperimentConfig, atoms: ExperimentConfig) -> None:
    baseline.validate_h1_contract()
    atoms.validate_h1_contract()
    if baseline.experiment_name != "independent_lora":
        raise ValueError("baseline config must select independent_lora")
    if atoms.experiment_name != "shared_atoms":
        raise ValueError("atom config must select shared_atoms")
    if baseline.tasks != TASKS or atoms.tasks != TASKS:
        raise ValueError(f"task order must be exactly {TASKS!r}")
    if baseline.confirmatory_seeds != SEEDS or atoms.confirmatory_seeds != SEEDS:
        raise ValueError(f"seeds must be exactly {SEEDS!r}")


def capacity_config(config: ExperimentConfig, system: str, capacity: int) -> ExperimentConfig:
    if system not in SYSTEMS:
        raise ValueError(f"unknown frontier system: {system}")
    if capacity not in CAPACITIES:
        raise ValueError(f"capacity must be one of {CAPACITIES!r}")
    if system == "shared_lora":
        return config.with_overrides(lora_rank=capacity, lora_alpha=float(capacity))
    return config.with_overrides(
        atom_count=capacity,
        active_atoms_during_training=capacity,
        active_atoms_for_primary_evaluation=capacity,
    )


def cell_directory(root: str | Path, system: str, capacity: int, seed: int) -> Path:
    return Path(root) / system / f"capacity_{capacity}" / f"seed_{seed}"


def cell_path(root: str | Path, system: str, capacity: int, seed: int) -> Path:
    return cell_directory(root, system, capacity, seed) / "cell.json"


def _mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("scores must be a non-empty finite sequence")
    return math.fsum(values) / len(values)


def _validate_provenance(
    value: Any, config: ExperimentConfig, *, name: str = "dataset provenance"
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(TASKS):
        raise ValueError(f"{name} must contain exactly the five locked tasks")
    for task in TASKS:
        task_value = value[task]
        if not isinstance(task_value, Mapping) or set(task_value) != {"train", "validation"}:
            raise ValueError(f"{name} for {task} must contain train and validation")
        for split, limit in (
            ("train", config.train_examples_per_task),
            ("validation", config.validation_examples_per_task),
        ):
            record = task_value[split]
            required = {
                "dataset_name", "task_name", "split", "source_fingerprint",
                "requested_limit", "selected_count", "seed", "selected_row_ids",
            }
            if not isinstance(record, Mapping) or set(record) != required:
                raise ValueError(f"{name} {task}/{split} has incomplete fields")
            ids = record["selected_row_ids"]
            selected_count = record["selected_count"]
            count_is_valid = (
                selected_count == limit
                if split == "train"
                else isinstance(selected_count, int) and 0 < selected_count <= limit
            )
            if (
                record["task_name"] != task
                or record["split"] != split
                or record["requested_limit"] != limit
                or not count_is_valid
                or record["seed"] != config.seed
                or not isinstance(record["source_fingerprint"], str)
                or not record["source_fingerprint"]
                or not isinstance(ids, list)
                or len(ids) != selected_count
                or len({str(item) for item in ids}) != selected_count
            ):
                raise ValueError(f"{name} {task}/{split} violates the locked rows")


def _validate_evaluation(
    evaluation: Any, task: str, *, name: str, expected_examples: int
) -> None:
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{name} is not an object")
    predictions = evaluation.get("predictions")
    labels = evaluation.get("labels")
    examples = evaluation.get("examples")
    if (
        examples != expected_examples
        or not isinstance(predictions, list)
        or not isinstance(labels, list)
        or len(predictions) != examples
        or len(labels) != examples
    ):
        raise ValueError(
            f"{name} lacks all {expected_examples} raw predictions and labels"
        )
    expected = compute_task_metrics(task, predictions, labels)
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{name} lacks task metrics")
    for metric, value in expected.items():
        try:
            observed = float(metrics[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} lacks metric {metric}") from error
        if not math.isclose(observed, value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name} metric {metric} does not match raw outputs")


def _component_shapes(system: str, capacity: int) -> dict[str, list[tuple[int, ...]]]:
    heads = [(2, 128)] * 5 + [(2,)] * 5
    if system == "shared_lora":
        return {"adapter": [(capacity, 128)] * 4 + [(128, capacity)] * 4, "heads": heads}
    return {
        "atoms": [(capacity, 128)] * 8,
        "coefficients": [(5, capacity)] * 4,
        "heads": heads,
    }


def _tensor_shapes(state: Mapping[str, Any]) -> list[tuple[int, ...]]:
    return sorted(
        (tuple(value.shape) for value in state.values() if isinstance(value, torch.Tensor)),
        key=lambda shape: (len(shape), shape),
    )


def _validate_checkpoint(
    checkpoint: Any,
    directory: Path,
    *,
    system: str,
    capacity: int,
    seed: int,
) -> None:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("cell lacks compact checkpoint metadata")
    expected = _component_shapes(system, capacity)
    paths = checkpoint.get("paths")
    byte_counts = checkpoint.get("bytes_by_component")
    if not isinstance(paths, Mapping) or set(paths) != set(expected):
        raise ValueError("compact checkpoint has wrong components")
    if not isinstance(byte_counts, Mapping) or set(byte_counts) != set(expected):
        raise ValueError("compact checkpoint byte accounting is incomplete")
    total = 0
    for component, wanted_shapes in expected.items():
        path = Path(paths[component])
        if path.resolve() != (directory / f"{component}.pt").resolve() or not path.is_file():
            raise ValueError(f"invalid {component} checkpoint path")
        size = path.stat().st_size
        if byte_counts[component] != size or size <= 0:
            raise ValueError(f"invalid {component} checkpoint bytes")
        total += size
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("component") != component
            or not isinstance(metadata, Mapping)
            or metadata.get("experiment") != "validation_frontier"
            or metadata.get("system") != system
            or metadata.get("capacity") != capacity
            or metadata.get("seed") != seed
            or metadata.get("tasks") != list(TASKS)
        ):
            raise ValueError(f"invalid {component} component metadata")
        state = payload.get("state_dict")
        if not isinstance(state, Mapping) or _tensor_shapes(state) != sorted(
            wanted_shapes, key=lambda shape: (len(shape), shape)
        ):
            raise ValueError(f"invalid {component} tensor shapes")
    if checkpoint.get("total_bytes") != total or checkpoint.get("format") != "torch.save":
        raise ValueError("invalid compact checkpoint total or format")


def _expected_counts(system: str, capacity: int) -> dict[str, int]:
    heads = 1_290
    if system == "shared_lora":
        return {
            "lora_adapter_parameters": 1_024 * capacity,
            "atom_parameters": 0,
            "coefficient_parameters": 0,
            "head_parameters": heads,
            "persistent_adaptation_parameters": 1_024 * capacity + heads,
        }
    return {
        "lora_adapter_parameters": 0,
        "atom_parameters": 1_024 * capacity,
        "coefficient_parameters": 20 * capacity,
        "head_parameters": heads,
        "persistent_adaptation_parameters": 1_044 * capacity + heads,
    }


def validate_frontier_cell(
    record: Mapping[str, Any],
    *,
    directory: str | Path | None = None,
    expected_system: str | None = None,
    expected_capacity: int | None = None,
    expected_seed: int | None = None,
) -> None:
    system = record.get("system")
    capacity = record.get("capacity")
    seed = record.get("seed")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("experiment") != "validation_frontier"
        or record.get("run_kind") != "validation_frontier"
        or record.get("status") != "complete"
        or system not in SYSTEMS
        or capacity not in CAPACITIES
        or seed not in SEEDS
    ):
        raise ValueError("frontier cell identity is invalid")
    if expected_system is not None and system != expected_system:
        raise ValueError("frontier cell system does not match its grid position")
    if expected_capacity is not None and capacity != expected_capacity:
        raise ValueError("frontier cell capacity does not match its grid position")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("frontier cell seed does not match its grid position")
    try:
        config = ExperimentConfig.from_mapping(record["resolved_config"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frontier cell resolved config is invalid") from error
    expected_config = capacity_config(config, str(system), int(capacity))
    if expected_config.to_dict() != config.to_dict() or config.seed != seed:
        raise ValueError("frontier cell resolved config does not match capacity and seed")
    canonical = (
        config.with_overrides(lora_rank=4, lora_alpha=4.0)
        if system == "shared_lora"
        else config.with_overrides(
            atom_count=8,
            active_atoms_during_training=8,
            active_atoms_for_primary_evaluation=4,
        )
    )
    canonical.validate_h1_contract()
    expected_experiment_name = (
        "independent_lora" if system == "shared_lora" else "shared_atoms"
    )
    if config.experiment_name != expected_experiment_name:
        raise ValueError("frontier cell selected the wrong system configuration")
    if config.tasks != TASKS or config.confirmatory_seeds != SEEDS:
        raise ValueError("frontier cell resolved config changed the locked tasks or seeds")
    if record.get("model") != "prajjwal1/bert-tiny" or not isinstance(
        record.get("model_revision"), str
    ) or not record["model_revision"]:
        raise ValueError("frontier cell lacks exact model revision")
    targets = record.get("target_modules")
    dimensions = record.get("target_dimensions")
    if (
        not isinstance(targets, list)
        or len(targets) != TARGET_COUNT
        or len(set(targets)) != TARGET_COUNT
        or not isinstance(dimensions, Mapping)
        or set(dimensions) != set(targets)
        or any(tuple(dimensions[name]) != TARGET_DIMENSION for name in targets)
        or sum(str(name).endswith(".query") for name in targets) != 2
        or sum(str(name).endswith(".value") for name in targets) != 2
    ):
        raise ValueError("frontier cell target components or dimensions are invalid")
    environment = record.get("environment")
    if not isinstance(environment, Mapping) or not {
        "python", "platform", "cpu_threads", "cuda_available", "packages"
    }.issubset(environment) or not isinstance(environment["packages"], Mapping):
        raise ValueError("frontier cell environment provenance is incomplete")
    _validate_provenance(record.get("dataset_provenance"), config)
    tasks = record.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASKS):
        raise ValueError("frontier cell task evaluations are incomplete")
    for task in TASKS:
        _validate_evaluation(
            tasks[task], task, name=f"{system}/{capacity}/{seed}/{task}",
            expected_examples=record["dataset_provenance"][task]["validation"]["selected_count"],
        )
    history = record.get("history")
    if (
        not isinstance(history, Mapping)
        or not isinstance(history.get("epochs"), list)
        or len(history["epochs"]) != 3
        or history.get("best_epoch") not in (1, 2, 3)
    ):
        raise ValueError("frontier cell lacks the complete three-epoch history")
    for epoch in history["epochs"]:
        global_training = epoch.get("global_training") if isinstance(epoch, Mapping) else None
        if (
            not isinstance(global_training, Mapping)
            or global_training.get("batches") != 1_250
            or global_training.get("optimizer_steps") != 1_250
            or global_training.get("examples") != 10_000
        ):
            raise ValueError("frontier cell history does not prove 3,750 optimizer updates")
    counts = record.get("parameter_counts")
    expected_counts = _expected_counts(str(system), int(capacity))
    if not isinstance(counts, Mapping) or counts.get("base_model_parameters") != 4_385_920:
        raise ValueError("frontier cell base parameter accounting is invalid")
    if counts.get("base_trainable_parameters") != 0 or counts.get(
        "uncategorized_trainable_parameters"
    ) != 0:
        raise ValueError("frontier cell has unexpected trainable parameters")
    if any(counts.get(field) != value for field, value in expected_counts.items()):
        raise ValueError("frontier cell persistent parameter accounting is invalid")
    storage = record.get("storage")
    if (
        not isinstance(storage, Mapping)
        or storage.get("persistent_parameters") != expected_counts["persistent_adaptation_parameters"]
        or storage.get("persistent_tensor_bytes") != 4 * expected_counts["persistent_adaptation_parameters"]
        or storage.get("common_frozen_base_parameters") != 4_385_920
        or storage.get("common_frozen_base_tensor_bytes") != 17_543_680
    ):
        raise ValueError("frontier cell storage accounting is invalid")
    if record.get("active_adapter_operations_per_token") != 1_024 * capacity:
        raise ValueError("frontier cell active operation accounting is invalid")
    budget = record.get("locked_budget")
    if not isinstance(budget, Mapping) or budget.get("optimizer_updates") != UPDATES:
        raise ValueError("frontier cell optimizer-update budget is invalid")
    runtime = record.get("runtime")
    if not isinstance(runtime, Mapping) or not {
        "elapsed_seconds", "training_elapsed_seconds", "evaluation_elapsed_seconds",
        "peak_rss_bytes", "inference_seconds_by_task"
    }.issubset(runtime):
        raise ValueError("frontier cell runtime accounting is incomplete")
    inference_times = runtime.get("inference_seconds_by_task")
    numeric_runtime = (
        runtime.get("elapsed_seconds"), runtime.get("training_elapsed_seconds"),
        runtime.get("evaluation_elapsed_seconds"), runtime.get("peak_rss_bytes"),
    )
    if (
        not isinstance(inference_times, Mapping)
        or set(inference_times) != set(TASKS)
        or any(not isinstance(value, (int, float)) or value < 0 for value in inference_times.values())
        or any(not isinstance(value, (int, float)) or value < 0 for value in numeric_runtime)
        or not math.isclose(
            float(runtime["elapsed_seconds"]),
            float(runtime["training_elapsed_seconds"]) + float(runtime["evaluation_elapsed_seconds"]),
            rel_tol=0.0, abs_tol=1e-9,
        )
        or not math.isclose(
            float(runtime["evaluation_elapsed_seconds"]),
            math.fsum(float(value) for value in inference_times.values()),
            rel_tol=0.0, abs_tol=1e-9,
        )
    ):
        raise ValueError("frontier cell runtime values are invalid or inconsistent")
    target_directory = Path(directory) if directory is not None else Path(record["artifact_directory"])
    _validate_checkpoint(
        record.get("checkpoint"), target_directory,
        system=str(system), capacity=int(capacity), seed=int(seed),
    )
    if storage.get("checkpoint_bytes") != record["checkpoint"]["total_bytes"]:
        raise ValueError("frontier cell checkpoint byte accounting is inconsistent")


def _optimizer(model: torch.nn.Module, config: ExperimentConfig) -> torch.optim.AdamW:
    return create_adamw_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        epsilon=config.adam_epsilon,
    )


def _metric(task: str):
    return partial(compute_task_metrics, task)


def _evaluate_tasks(model, loaders, config: ExperimentConfig) -> tuple[dict[str, Any], dict[str, float]]:
    records: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    for task in TASKS:
        started = time.perf_counter()
        result = evaluate(
            model, loaders[task], config.device, task_id=task,
            metric_fn=_metric(task), scalar_metric_name="primary_score",
        )
        seconds[task] = time.perf_counter() - started
        records[task] = result.to_dict(include_outputs=True)
    return records, seconds


def _checkpoint_metadata(system: str, capacity: int, seed: int) -> dict[str, Any]:
    return {
        "experiment": "validation_frontier",
        "system": system,
        "capacity": capacity,
        "seed": seed,
        "tasks": list(TASKS),
    }


def _record(
    *,
    config: ExperimentConfig,
    system: str,
    capacity: int,
    directory: Path,
    metadata: Mapping[str, Any],
    tasks: Mapping[str, Any],
    history: Mapping[str, Any],
    counts: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    runtime: Mapping[str, Any],
    reused_from: str | None = None,
) -> dict[str, Any]:
    return {
        **dict(metadata),
        "experiment": "validation_frontier",
        "run_kind": "validation_frontier",
        "status": "complete",
        "system": system,
        "capacity": capacity,
        "tasks": dict(tasks),
        "history": dict(history),
        "parameter_counts": dict(counts),
        "storage": {
            "persistent_parameters": _expected_counts(system, capacity)["persistent_adaptation_parameters"],
            "persistent_tensor_bytes": 4 * _expected_counts(system, capacity)["persistent_adaptation_parameters"],
            "checkpoint_bytes": checkpoint["total_bytes"],
            "common_frozen_base_parameters": 4_385_920,
            "common_frozen_base_tensor_bytes": 17_543_680,
        },
        "checkpoint": dict(checkpoint),
        "runtime": dict(runtime),
        "active_adapter_operations_per_token": 1_024 * capacity,
        "locked_budget": {
            "train_examples_per_task": 2_000,
            "validation_examples_per_task": 500,
            "batch_size": 8,
            "max_length": 128,
            "epochs": 3,
            "optimizer_updates": UPDATES,
            "schedule": "seeded balanced complete pass",
        },
        "artifact_directory": str(directory),
        "reused_atom_artifact": reused_from,
    }


def run_frontier_cell(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    system: str,
    capacity: int,
    directory: Path,
    prepared: PreparedData | None,
    force: bool,
) -> dict[str, Any]:
    """Run one new cell.  Existing completed cells are handled by the orchestrator."""

    del force
    base = baseline_config if system == "shared_lora" else atom_config
    config = capacity_config(base, system, capacity)
    set_seed(config.seed)
    prepared = prepared or prepare_data(config, tasks=TASKS)
    train_loaders, validation_loaders = build_loaders(prepared, config, tasks=TASKS)
    if sum(len(loader) for loader in train_loaders.values()) * config.epochs != UPDATES:
        raise ValueError("locked data loaders do not yield exactly 3,750 optimizer updates")
    if system == "shared_lora":
        model, target_names = build_lora_model(config, TASKS, rank=capacity)
        regularizer = None
    else:
        model, target_names = build_atom_model(config, TASKS, atom_count=capacity)
        regularizer = lambda value, task: coefficient_l1_regularization(
            value, task, config.sparsity_lambda
        )
    optimizer = _optimizer(model, config)
    metric_fns = {task: _metric(task) for task in TASKS}
    with RuntimeMonitor() as monitor:
        result = train_multitask(
            model,
            train_loaders,
            validation_loaders,
            optimizer,
            epochs=config.epochs,
            seed=config.seed,
            device=config.device,
            metric_fns=metric_fns,
            primary_metrics={task: "primary_score" for task in TASKS},
            regularization_fn=regularizer,
            schedule_mode="complete_pass",
            state_capture_fn=lambda value: extract_adapter_state_dict(value, include_heads=True),
            state_restore_fn=lambda value, state: load_adapter_state_dict(
                value, state, include_heads=True
            ),
        )
    runtime_result = monitor.result()
    task_records, inference_seconds = _evaluate_tasks(model, validation_loaders, config)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = save_compact_checkpoint(
        model, directory, metadata=_checkpoint_metadata(system, capacity, config.seed)
    )
    counts = categorized_parameter_counts(model)
    metadata = _run_metadata(config, model, target_names, prepared, "validation_frontier")
    record = _record(
        config=config,
        system=system,
        capacity=capacity,
        directory=directory,
        metadata=metadata,
        tasks=task_records,
        history=result.to_dict(),
        counts=counts,
        checkpoint=checkpoint,
        runtime={
            "elapsed_seconds": runtime_result.elapsed_seconds + math.fsum(inference_seconds.values()),
            "training_elapsed_seconds": runtime_result.elapsed_seconds,
            "evaluation_elapsed_seconds": math.fsum(inference_seconds.values()),
            "peak_rss_bytes": runtime_result.peak_rss_bytes,
            "inference_seconds_by_task": inference_seconds,
        },
    )
    validate_frontier_cell(record, directory=directory)
    return record


def _validate_source_atom_record(
    source: Mapping[str, Any], source_directory: Path, config: ExperimentConfig, capacity: int
) -> None:
    expected_config = capacity_config(config, "shared_atoms", capacity)
    if (
        source.get("schema_version") != SCHEMA_VERSION
        or source.get("system") != "shared_atoms"
        or source.get("seed") != config.seed
        or source.get("task_ids") != list(TASKS)
        or source.get("atom_count") != capacity
        or source.get("active_atoms_during_training") != capacity
        or source.get("atoms_frozen") is not False
        or source.get("training_labels_shuffled") is not False
        or source.get("sparsity_lambda") != config.sparsity_lambda
    ):
        raise ValueError("candidate atom artifact identity is incompatible")
    stored = ExperimentConfig.from_mapping(source.get("resolved_config", {}))
    if stored.to_dict() != expected_config.to_dict():
        raise ValueError("candidate atom artifact resolved config is incompatible")
    if not isinstance(source.get("model_revision"), str) or not source["model_revision"]:
        raise ValueError("candidate atom artifact lacks model revision")
    _validate_provenance(source.get("dataset_provenance"), expected_config, name="candidate provenance")
    targets = source.get("target_modules")
    dimensions = source.get("target_dimensions")
    if (
        not isinstance(targets, list) or len(targets) != 4 or len(set(targets)) != 4
        or not isinstance(dimensions, Mapping) or set(dimensions) != set(targets)
        or any(tuple(dimensions[name]) != TARGET_DIMENSION for name in targets)
    ):
        raise ValueError("candidate atom target metadata is incompatible")
    tasks = source.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASKS):
        raise ValueError("candidate atom evaluations are incomplete")
    for task in TASKS:
        task_value = tasks[task]
        if not isinstance(task_value, Mapping) or "all_atoms" not in task_value:
            raise ValueError("candidate atom artifact lacks all-active evaluation")
        _validate_evaluation(
            task_value["all_atoms"], task, name=f"candidate/{task}",
            expected_examples=source["dataset_provenance"][task]["validation"]["selected_count"],
        )
    history = source.get("history")
    if not isinstance(history, Mapping) or len(history.get("epochs", [])) != 3:
        raise ValueError("candidate atom history is incomplete")
    if any(
        not isinstance(epoch, Mapping)
        or not isinstance(epoch.get("global_training"), Mapping)
        or epoch["global_training"].get("batches") != 1_250
        or epoch["global_training"].get("optimizer_steps") != 1_250
        or epoch["global_training"].get("examples") != 10_000
        for epoch in history["epochs"]
    ):
        raise ValueError("candidate atom history does not prove the locked update budget")
    expected = _expected_counts("shared_atoms", capacity)
    counts = source.get("parameter_counts")
    mapped_persistent = counts.get("total_persistent_task_parameters") if isinstance(counts, Mapping) else None
    if (
        not isinstance(counts, Mapping)
        or counts.get("base_model_parameters") != 4_385_920
        or counts.get("atom_parameters") != expected["atom_parameters"]
        or counts.get("coefficient_parameters") != expected["coefficient_parameters"]
        or counts.get("head_parameters") != expected["head_parameters"]
        or counts.get("base_trainable_parameters") != 0
        or counts.get("uncategorized_trainable_parameters") != 0
        or mapped_persistent != expected["persistent_adaptation_parameters"]
    ):
        raise ValueError("candidate atom parameter accounting is incompatible")
    checkpoint = source.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint.get("paths", {})) != {
        "atoms", "coefficients", "heads"
    }:
        raise ValueError("candidate atom checkpoint components are incomplete")
    actual_total = 0
    for component, shapes in _component_shapes("shared_atoms", capacity).items():
        path = Path(checkpoint["paths"][component])
        if not path.is_file() or path.parent.resolve() != source_directory.resolve():
            raise ValueError("candidate atom checkpoint path is invalid")
        if checkpoint.get("bytes_by_component", {}).get(component) != path.stat().st_size:
            raise ValueError("candidate atom checkpoint bytes are invalid")
        actual_total += path.stat().st_size
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("component") != component
            or not isinstance(payload.get("state_dict"), Mapping)
            or not isinstance(metadata, Mapping)
            or metadata.get("system") != "shared_atoms"
            or metadata.get("seed") != config.seed
            or metadata.get("tasks") != list(TASKS)
            or metadata.get("atom_count") != capacity
            or _tensor_shapes(payload.get("state_dict", {}))
            != sorted(shapes, key=lambda shape: (len(shape), shape))
        ):
            raise ValueError("candidate atom component metadata or shapes are invalid")
    if checkpoint.get("total_bytes") != actual_total or checkpoint.get("format") != "torch.save":
        raise ValueError("candidate atom checkpoint total or format is invalid")
    operations = source.get("active_adapter_operations_per_token")
    if not isinstance(operations, Mapping) or operations.get("all_atoms") != 1_024 * capacity:
        raise ValueError("candidate atom active-operation accounting is invalid")


def _copy_reused_atom_cell(
    source: Mapping[str, Any], source_directory: Path, destination: Path,
    config: ExperimentConfig, capacity: int,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    byte_counts: dict[str, int] = {}
    metadata = _checkpoint_metadata("shared_atoms", capacity, config.seed)
    for component in ("atoms", "coefficients", "heads"):
        payload = torch.load(
            Path(source["checkpoint"]["paths"][component]), map_location="cpu", weights_only=True
        )
        target = destination / f"{component}.pt"
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "component": component,
                    "metadata": metadata,
                    "state_dict": payload["state_dict"],
                },
                temporary,
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        paths[component] = str(target)
        byte_counts[component] = target.stat().st_size
    checkpoint = {
        "paths": paths,
        "bytes_by_component": byte_counts,
        "total_bytes": sum(byte_counts.values()),
        "format": "torch.save",
        "dtype": source["checkpoint"].get("dtype", "torch.float32"),
    }
    counts = dict(source["parameter_counts"])
    counts["lora_adapter_parameters"] = 0
    counts["persistent_adaptation_parameters"] = counts["total_persistent_task_parameters"]
    source_inference = {
        task: float(source["runtime"]["inference_seconds_by_task"][task]["all_atoms"])
        for task in TASKS
    }
    training_seconds = float(source["runtime"]["elapsed_seconds"])
    evaluation_seconds = math.fsum(source_inference.values())
    record = _record(
        config=config,
        system="shared_atoms",
        capacity=capacity,
        directory=destination,
        metadata={
            key: source[key]
            for key in (
                "schema_version", "run_kind", "model", "model_revision", "seed",
                "resolved_config", "target_modules", "target_dimensions",
                "dataset_provenance", "environment",
            )
        },
        tasks={task: source["tasks"][task]["all_atoms"] for task in TASKS},
        history=source["history"],
        counts=counts,
        checkpoint=checkpoint,
        runtime={
            "elapsed_seconds": training_seconds + evaluation_seconds,
            "training_elapsed_seconds": training_seconds,
            "evaluation_elapsed_seconds": evaluation_seconds,
            "peak_rss_bytes": source["runtime"]["peak_rss_bytes"],
            "inference_seconds_by_task": source_inference,
        },
        reused_from=str(source_directory),
    )
    record["reused_atom_source_run_kind"] = source.get("run_kind")
    validate_frontier_cell(record, directory=destination)
    return record


def atom_reuse_candidates(root: str | Path, capacity: int, seed: int) -> tuple[Path, ...]:
    base = Path(root)
    candidates = [
        base / "followup_ablations" / "atom_count" / f"atoms_{capacity}" / f"seed_{seed}",
    ]
    if capacity == 8:
        candidates.append(base / "shared_atoms" / f"seed_{seed}")
    return tuple(candidates)


def _try_reuse_atom(
    reuse_root: Path, destination: Path, config: ExperimentConfig, capacity: int
) -> dict[str, Any] | None:
    for directory in atom_reuse_candidates(reuse_root, capacity, config.seed):
        path = directory / "metrics_by_task.json"
        if not path.is_file():
            continue
        source = read_json(path)
        try:
            _validate_source_atom_record(source, directory, config, capacity)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            print(f"Rejecting incompatible atom reuse candidate {path}: {error}", flush=True)
            continue
        print(f"Reusing fully validated atom artifact {path}", flush=True)
        return _copy_reused_atom_cell(source, directory, destination, config, capacity)
    return None


def pareto_frontier(
    points: Sequence[Mapping[str, Any]], *, quality_tolerance: float = 0.0
) -> list[str]:
    """Return point IDs not dominated on mean, worst, storage, and operations."""

    if quality_tolerance < 0:
        raise ValueError("quality_tolerance must be non-negative")

    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        no_worse = (
            float(left["mean_primary_score"]) >= float(right["mean_primary_score"]) - quality_tolerance
            and float(left["worst_task_score"]) >= float(right["worst_task_score"]) - quality_tolerance
            and int(left["persistent_parameters"]) <= int(right["persistent_parameters"])
            and int(left["active_operations"]) <= int(right["active_operations"])
        )
        materially_better = (
            float(left["mean_primary_score"]) > float(right["mean_primary_score"]) + quality_tolerance
            or float(left["worst_task_score"]) > float(right["worst_task_score"]) + quality_tolerance
            or int(left["persistent_parameters"]) < int(right["persistent_parameters"])
            or int(left["active_operations"]) < int(right["active_operations"])
        )
        return no_worse and materially_better

    result = []
    for point in points:
        if not any(other is not point and dominates(other, point) for other in points):
            result.append(str(point["id"]))
    return result


def build_frontier_summary(cells: Sequence[Mapping[str, Any]], *, output_root: str | Path) -> dict[str, Any]:
    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for cell in cells:
        validate_frontier_cell(cell)
        key = (str(cell["system"]), int(cell["capacity"]), int(cell["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate frontier cell: {key}")
        indexed[key] = cell
    expected = {(system, capacity, seed) for system in SYSTEMS for capacity in CAPACITIES for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"frontier grid must contain all 24 locked cells; missing={sorted(expected-set(indexed))}, unexpected={sorted(set(indexed)-expected)}")

    results: dict[str, Any] = {system: {} for system in SYSTEMS}
    points: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for capacity in CAPACITIES:
            by_task: dict[str, Any] = {}
            for task in TASKS:
                scores = [
                    float(indexed[(system, capacity, seed)]["tasks"][task]["metrics"]["primary_score"])
                    for seed in SEEDS
                ]
                by_task[task] = {
                    "mean": _mean(scores),
                    "std_population": statistics.pstdev(scores),
                    "seed_scores": {str(seed): score for seed, score in zip(SEEDS, scores, strict=True)},
                }
            first = indexed[(system, capacity, SEEDS[0])]
            storage_values = {
                int(indexed[(system, capacity, seed)]["storage"]["persistent_parameters"])
                for seed in SEEDS
            }
            operation_values = {
                int(indexed[(system, capacity, seed)]["active_adapter_operations_per_token"])
                for seed in SEEDS
            }
            if len(storage_values) != 1 or len(operation_values) != 1:
                raise ValueError("accounting changed across seeds")
            task_means = [by_task[task]["mean"] for task in TASKS]
            worst_score = min(task_means)
            result = {
                "by_task": by_task,
                "mean_primary_score": _mean(task_means),
                "worst_task_score": worst_score,
                "worst_tasks": [
                    task for task in TASKS
                    if math.isclose(by_task[task]["mean"], worst_score, rel_tol=0.0, abs_tol=1e-15)
                ],
                "persistent_parameters": next(iter(storage_values)),
                "persistent_tensor_bytes": int(first["storage"]["persistent_tensor_bytes"]),
                "checkpoint_bytes_by_seed": {
                    str(seed): int(indexed[(system, capacity, seed)]["storage"]["checkpoint_bytes"])
                    for seed in SEEDS
                },
                "active_operations": next(iter(operation_values)),
                "runtime_by_seed": {
                    str(seed): dict(indexed[(system, capacity, seed)]["runtime"]) for seed in SEEDS
                },
            }
            results[system][str(capacity)] = result
            points.append({"id": f"{system}:c{capacity}", **{k: result[k] for k in ("mean_primary_score", "worst_task_score", "persistent_parameters", "active_operations")}})

    paired: dict[str, Any] = {}
    qualifying: list[int] = []
    for capacity in CAPACITIES:
        lora = results["shared_lora"][str(capacity)]
        atoms = results["shared_atoms"][str(capacity)]
        mean_delta = atoms["mean_primary_score"] - lora["mean_primary_score"]
        worst_delta = atoms["worst_task_score"] - lora["worst_task_score"]
        storage_ratio = atoms["persistent_parameters"] / lora["persistent_parameters"]
        operations_equal = atoms["active_operations"] == lora["active_operations"]
        criteria = {
            "atom_mean_minus_lora_at_least_0_005": mean_delta >= MEAN_TOLERANCE,
            "atom_worst_no_more_than_0_01_below_lora": worst_delta >= -WORST_TASK_TOLERANCE,
            "atom_storage_no_more_than_1_02_lora": storage_ratio <= STORAGE_RATIO_LIMIT,
            "active_operations_identical": operations_equal,
        }
        if all(criteria.values()):
            qualifying.append(capacity)
        paired[str(capacity)] = {
            "mean_delta_atom_minus_lora": mean_delta,
            "worst_task_delta_atom_minus_lora": worst_delta,
            "storage_ratio_atom_over_lora": storage_ratio,
            "active_operations_equal": operations_equal,
            "criteria": criteria,
            "qualifies": all(criteria.values()),
            "by_task_mean_delta": {
                task: atoms["by_task"][task]["mean"] - lora["by_task"][task]["mean"]
                for task in TASKS
            },
            "paired_seed_task_deltas": {
                task: {
                    str(seed): atoms["by_task"][task]["seed_scores"][str(seed)]
                    - lora["by_task"][task]["seed_scores"][str(seed)]
                    for seed in SEEDS
                }
                for task in TASKS
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "validation_frontier",
        "status": "complete",
        "predeclared_before_live_training": True,
        "systems": list(SYSTEMS),
        "capacities": list(CAPACITIES),
        "seeds": list(SEEDS),
        "tasks": list(TASKS),
        "cell_count": 24,
        "aggregation_rule": "mean each task over three seeds, then unweighted mean over tasks",
        "standard_deviation": "population standard deviation across the three seeds",
        "results": results,
        "matched_capacity_deltas": paired,
        "pareto_points": points,
        "pareto_frontiers": {
            "exact": pareto_frontier(points),
            "quality_tolerance_0_005": pareto_frontier(points, quality_tolerance=MEAN_TOLERANCE),
        },
        "atom_specific_advantage": {
            "passed": bool(qualifying),
            "qualifying_capacities": qualifying,
            "rule": {
                "minimum_mean_delta": MEAN_TOLERANCE,
                "minimum_worst_task_delta": -WORST_TASK_TOLERANCE,
                "maximum_storage_ratio": STORAGE_RATIO_LIMIT,
                "active_operations": "identical",
            },
        },
        "common_frozen_base": {"parameters": 4_385_920, "tensor_bytes": 17_543_680},
        "protocol": str(Path(output_root) / PROTOCOL_FILENAME),
        "output_root": str(Path(output_root)),
        "cell_artifacts": {
            system: {
                str(capacity): {
                    str(seed): str(cell_path(output_root, system, capacity, seed)) for seed in SEEDS
                } for capacity in CAPACITIES
            } for system in SYSTEMS
        },
    }


def build_protocol_record(
    baseline: ExperimentConfig, atoms: ExperimentConfig, output_root: str | Path,
    atom_reuse_root: str | Path,
) -> dict[str, Any]:
    validate_locked_configs(baseline, atoms)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "validation_frontier",
        "status": "locked_before_live_training",
        "systems": {
            "shared_lora": "one rank-c bank shared by all five tasks; alpha=c",
            "shared_atoms": "N=c rank-1 atoms per target layer; all N active",
        },
        "capacities": list(CAPACITIES),
        "seeds": list(SEEDS),
        "tasks": list(TASKS),
        "schedule": "same seeded balanced complete-pass multitask schedule",
        "optimizer_updates_per_cell": UPDATES,
        "aggregation_rule": "mean each task over three seeds, then unweighted mean over tasks",
        "mean_quality_tolerance": MEAN_TOLERANCE,
        "pareto_dimensions": {
            "maximize": ["mean_primary_score", "worst_task_score"],
            "minimize": ["persistent_parameters", "active_adapter_operations_per_token"],
        },
        "atom_specific_advantage_rule": {
            "exists_matched_capacity": True,
            "atom_mean_minus_lora": {"operator": ">=", "threshold": MEAN_TOLERANCE},
            "atom_worst_minus_lora": {"operator": ">=", "threshold": -WORST_TASK_TOLERANCE},
            "atom_storage_over_lora": {"operator": "<=", "threshold": STORAGE_RATIO_LIMIT},
            "active_operations": "identical",
        },
        "resumability": "completed atomic cell records; --force replaces complete cells",
        "atom_reuse_policy": "reuse only after strict config, revision, rows, fingerprints, shapes, metadata, outputs, and accounting validation",
        "shared_lora_reuse_policy": "never reuse legacy results; train every cell in this experiment",
        "output_root": str(Path(output_root)),
        "atom_reuse_root": str(Path(atom_reuse_root)),
        "baseline_config": baseline.to_dict(),
        "atom_config": atoms.to_dict(),
    }


def render_frontier_markdown(summary: Mapping[str, Any]) -> str:
    decision = summary["atom_specific_advantage"]
    lines = [
        "# Experiment A: Matched Shared-LoRA/Atom Frontier", "",
        "Status: **COMPLETE**", "",
        f"Atom-specific advantage: **{'PASS' if decision['passed'] else 'FAIL'}**. "
        f"Qualifying matched capacities: {decision['qualifying_capacities'] or 'none'}.", "",
        "Scores first average each task across seeds 17, 29, and 43, then average the five task means. Standard deviations below are population standard deviations.", "",
        "## Frontier summary", "",
        "| System | c | Mean | Worst task (score) | Parameters | Tensor bytes | Active ops/token | Checkpoint bytes (17/29/43) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for system in SYSTEMS:
        for capacity in CAPACITIES:
            row = summary["results"][system][str(capacity)]
            byte_text = "/".join(str(row["checkpoint_bytes_by_seed"][str(seed)]) for seed in SEEDS)
            lines.append(
                f"| {system} | {capacity} | {row['mean_primary_score']:.6f} | {','.join(row['worst_tasks'])} ({row['worst_task_score']:.6f}) | "
                f"{row['persistent_parameters']:,} | {row['persistent_tensor_bytes']:,} | {row['active_operations']:,} | {byte_text} |"
            )
    lines += ["", "## Every task and seed score", ""]
    for capacity in CAPACITIES:
        lines += [f"### Capacity {capacity}", "", "| System | Task | Seed 17 | Seed 29 | Seed 43 | Mean | Std |", "|---|---|---:|---:|---:|---:|---:|"]
        for system in SYSTEMS:
            for task in TASKS:
                row = summary["results"][system][str(capacity)]["by_task"][task]
                scores = row["seed_scores"]
                lines.append(
                    f"| {system} | {task} | {scores['17']:.6f} | {scores['29']:.6f} | {scores['43']:.6f} | {row['mean']:.6f} | {row['std_population']:.6f} |"
                )
        lines.append("")
    lines += ["## Matched-capacity decisions", "", "| c | Mean delta | Worst delta | Storage ratio | Ops equal | Qualifies |", "|---:|---:|---:|---:|---:|---:|"]
    for capacity in CAPACITIES:
        row = summary["matched_capacity_deltas"][str(capacity)]
        lines.append(
            f"| {capacity} | {row['mean_delta_atom_minus_lora']:+.6f} | {row['worst_task_delta_atom_minus_lora']:+.6f} | {row['storage_ratio_atom_over_lora']:.6f} | "
            f"{'yes' if row['active_operations_equal'] else 'no'} | {'PASS' if row['qualifies'] else 'FAIL'} |"
        )
    lines += [
        "", "## Pareto frontiers", "",
        f"- Exact: {', '.join(summary['pareto_frontiers']['exact']) or 'none'}",
        f"- Quality tolerance 0.005: {', '.join(summary['pareto_frontiers']['quality_tolerance_0_005']) or 'none'}",
        "", "## Runtime by cell", "",
        "| System | c | Seed | Total seconds | Training seconds | Evaluation seconds | Peak RSS bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        for capacity in CAPACITIES:
            for seed in SEEDS:
                runtime = summary["results"][system][str(capacity)]["runtime_by_seed"][str(seed)]
                lines.append(
                    f"| {system} | {capacity} | {seed} | {float(runtime['elapsed_seconds']):.6f} | "
                    f"{float(runtime['training_elapsed_seconds']):.6f} | {float(runtime['evaluation_elapsed_seconds']):.6f} | "
                    f"{int(runtime['peak_rss_bytes']):,} |"
                )
    lines += [
        "", "The common frozen base is reported separately: 4,385,920 parameters (17,543,680 raw tensor bytes). Raw predictions, labels, histories, provenance, and compact component paths are retained in each cell JSON.", "",
    ]
    return "\n".join(lines)


def _write_report(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def run_validation_frontier(
    baseline: ExperimentConfig,
    atoms: ExperimentConfig,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    atom_reuse_root: str | Path = DEFAULT_ATOM_REUSE_ROOT,
    force: bool = False,
    cell_runner: CellRunner | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Run/resume the locked 24-cell frontier and write final JSON/Markdown."""

    validate_locked_configs(baseline, atoms)
    destination = Path(output_root)
    reuse_root = Path(atom_reuse_root)
    protocol = build_protocol_record(baseline, atoms, destination, reuse_root)
    protocol_path = destination / PROTOCOL_FILENAME
    if protocol_path.is_file():
        if read_json(protocol_path) != protocol:
            raise ValueError(f"existing frontier protocol differs from locked design: {protocol_path}")
    else:
        write_json(protocol_path, protocol)

    runner = cell_runner or run_frontier_cell
    prepared_by_seed: dict[int, PreparedData] = {}
    cells: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_baseline = baseline.with_overrides(seed=seed)
        seed_atoms = atoms.with_overrides(seed=seed)
        for capacity in CAPACITIES:
            for system in SYSTEMS:
                directory = cell_directory(destination, system, capacity, seed)
                path = directory / "cell.json"
                if path.is_file() and not force:
                    cell = read_json(path)
                    validate_frontier_cell(
                        cell, directory=directory, expected_system=system,
                        expected_capacity=capacity, expected_seed=seed,
                    )
                    print(f"Skipping complete frontier {system}, c={capacity}, seed={seed}.", flush=True)
                else:
                    cell = None
                    if system == "shared_atoms" and cell_runner is None and not force:
                        cell = _try_reuse_atom(reuse_root, directory, seed_atoms, capacity)
                    if cell is None:
                        if cell_runner is None and seed not in prepared_by_seed:
                            prepared_by_seed[seed] = prepare_data(seed_baseline, tasks=TASKS)
                        print(f"Running frontier {system}, c={capacity}, seed={seed}.", flush=True)
                        cell = runner(
                            seed_baseline, seed_atoms, system, capacity, directory,
                            prepared_by_seed.get(seed), force,
                        )
                    validate_frontier_cell(
                        cell, directory=directory, expected_system=system,
                        expected_capacity=capacity, expected_seed=seed,
                    )
                    write_json(path, cell)
                cells.append(cell)

    # Paired identity and row checks are deliberately global, not implied by a seed.
    for seed in SEEDS:
        revisions = {cell["model_revision"] for cell in cells if cell["seed"] == seed}
        provenances = [
            cell["dataset_provenance"] for cell in cells if cell["seed"] == seed
        ]
        if (
            len(revisions) != 1
            or not provenances
            or any(value != provenances[0] for value in provenances[1:])
        ):
            raise ValueError(f"paired cells for seed {seed} do not share model revision and selected rows")
    summary = build_frontier_summary(cells, output_root=destination)
    summary_path = write_json(destination / SUMMARY_FILENAME, summary)
    report_path = _write_report(destination / REPORT_FILENAME, render_frontier_markdown(summary))
    return summary, summary_path, report_path


__all__ = [
    "CAPACITIES", "DEFAULT_ATOM_REUSE_ROOT", "DEFAULT_OUTPUT_ROOT", "MEAN_TOLERANCE",
    "PROTOCOL_FILENAME", "REPORT_FILENAME", "SEEDS", "SUMMARY_FILENAME", "SYSTEMS",
    "atom_reuse_candidates", "build_frontier_summary", "build_protocol_record",
    "capacity_config", "cell_directory", "cell_path", "pareto_frontier",
    "render_frontier_markdown", "run_frontier_cell", "run_validation_frontier",
    "validate_frontier_cell", "validate_locked_configs",
]
