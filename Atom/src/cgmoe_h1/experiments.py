"""Executable H1 experiment assembly, training, evaluation, and persistence."""

from __future__ import annotations

import gc
import importlib.metadata
import math
import os
import platform
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from cgmoe_h1.config import ExperimentConfig
from cgmoe_h1.data import (
    get_dataset_provenance,
    load_task_data,
    make_data_collator,
)
from cgmoe_h1.metrics import compute_task_metrics
from cgmoe_h1.models.atoms import (
    AtomLinear,
    coefficient_l1_regularization,
    iter_atom_layers,
)
from cgmoe_h1.models.classifier import BertTaskClassifier
from cgmoe_h1.models.injection import (
    extract_adapter_state_dict,
    inject_atoms,
    inject_lora,
    load_adapter_state_dict,
)
from cgmoe_h1.models.lora import iter_lora_layers
from cgmoe_h1.training.multitask import (
    collect_shared_atom_diagnostics,
    train_multitask,
)
from cgmoe_h1.training.trainer import (
    EvaluationResult,
    create_adamw_optimizer,
    evaluate,
    train_single_task,
)
from cgmoe_h1.utils.parameters import (
    active_adapter_operations,
    assert_frozen_base,
    categorized_parameter_counts,
    checkpoint_bytes,
    shared_atom_parameter_totals,
)
from cgmoe_h1.utils.reproducibility import make_torch_generator, set_seed
from cgmoe_h1.utils.runtime import RuntimeMonitor
from cgmoe_h1.utils.serialization import read_json, write_json


TASK_LABEL_COUNTS = {task: 2 for task in ("sst2", "mrpc", "rte", "qnli", "qqp")}
EXPECTED_BERT_TINY_TARGETS = 4
SCHEMA_VERSION = 1


@dataclass(slots=True)
class PreparedData:
    tokenizer: PreTrainedTokenizerBase
    train: dict[str, Dataset]
    validation: dict[str, Dataset]
    provenance: dict[str, dict[str, Any]]


def environment_record() -> dict[str, Any]:
    packages = (
        "datasets",
        "evaluate",
        "numpy",
        "psutil",
        "scikit-learn",
        "torch",
        "transformers",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "packages": versions,
    }


def prepare_data(
    config: ExperimentConfig,
    *,
    tasks: Sequence[str] | None = None,
) -> PreparedData:
    selected_tasks = tuple(tasks or config.tasks)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    train: dict[str, Dataset] = {}
    validation: dict[str, Dataset] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        print(f"Loading/tokenizing {task} for seed {config.seed}...", flush=True)
        train[task], validation[task] = load_task_data(
            task,
            tokenizer,
            config.train_examples_per_task,
            config.validation_examples_per_task,
            config.max_length,
            config.seed,
        )
        provenance[task] = {
            "train": get_dataset_provenance(train[task]).to_dict(),
            "validation": get_dataset_provenance(validation[task]).to_dict(),
        }
    return PreparedData(tokenizer, train, validation, provenance)


def build_loaders(
    prepared: PreparedData,
    config: ExperimentConfig,
    *,
    tasks: Sequence[str] | None = None,
    shuffle_training_labels: bool = False,
) -> tuple[dict[str, DataLoader[Any]], dict[str, DataLoader[Any]]]:
    selected_tasks = tuple(tasks or prepared.train)
    collator = make_data_collator(prepared.tokenizer)
    train_loaders: dict[str, DataLoader[Any]] = {}
    validation_loaders: dict[str, DataLoader[Any]] = {}
    for task_index, task in enumerate(selected_tasks):
        train_dataset = prepared.train[task]
        if shuffle_training_labels:
            labels = list(train_dataset["labels"])
            generator = make_torch_generator(config.seed + 50_000 + task_index)
            permutation = torch.randperm(len(labels), generator=generator).tolist()
            train_dataset = train_dataset.remove_columns("labels").add_column(
                "labels", [labels[index] for index in permutation]
            )
        train_loaders[task] = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collator,
            generator=make_torch_generator(config.seed + 1_000 + task_index),
        )
        validation_loaders[task] = DataLoader(
            prepared.validation[task],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
    return train_loaders, validation_loaders


def _metric_fn(task: str) -> Callable[[Tensor, Tensor], dict[str, float]]:
    return partial(compute_task_metrics, task)


def _new_classifier(config: ExperimentConfig, tasks: Sequence[str]) -> BertTaskClassifier:
    encoder = AutoModel.from_pretrained(config.base_model)
    model = BertTaskClassifier(
        encoder,
        task_num_labels={task: TASK_LABEL_COUNTS[task] for task in tasks},
    )
    return model


def build_head_only_model(config: ExperimentConfig, task: str) -> BertTaskClassifier:
    model = _new_classifier(config, (task,))
    assert_frozen_base(model)
    return model


def build_lora_model(
    config: ExperimentConfig,
    tasks: Sequence[str],
    *,
    rank: int | None = None,
) -> tuple[BertTaskClassifier, list[str]]:
    model = _new_classifier(config, tasks)
    resolved_rank = rank or config.lora_rank
    names = inject_lora(
        model.encoder,
        config.target_modules,
        rank=resolved_rank,
        alpha=float(resolved_rank),
        dropout=config.lora_dropout,
        expected_count=EXPECTED_BERT_TINY_TARGETS,
    )
    model.freeze_base_encoder()
    assert_frozen_base(model)
    return model, names


def build_atom_model(
    config: ExperimentConfig,
    tasks: Sequence[str],
    *,
    atom_count: int | None = None,
    freeze_atoms: bool = False,
) -> tuple[BertTaskClassifier, list[str]]:
    model = _new_classifier(config, tasks)
    names = inject_atoms(
        model.encoder,
        tasks,
        config.target_modules,
        atom_count=atom_count or config.atom_count,
        scaling=config.atom_scaling,
        expected_count=EXPECTED_BERT_TINY_TARGETS,
    )
    model.freeze_base_encoder()
    if freeze_atoms:
        for layer in iter_atom_layers(model):
            layer.atom_u.requires_grad_(False)
            layer.atom_v.requires_grad_(False)
    assert_frozen_base(model)
    return model, names


def _optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.AdamW:
    return create_adamw_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        epsilon=config.adam_epsilon,
    )


def _evaluation_dict(result: EvaluationResult, *, include_outputs: bool = False) -> dict[str, Any]:
    return result.to_dict(include_outputs=include_outputs)


def _atomic_torch_save(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _split_adapter_state(
    state: Mapping[str, Any],
) -> dict[str, OrderedDict[str, Any]]:
    components = {
        "adapter": OrderedDict(),
        "atoms": OrderedDict(),
        "coefficients": OrderedDict(),
        "heads": OrderedDict(),
    }
    for key, value in state.items():
        if key.startswith("heads."):
            components["heads"][key] = value
        elif key.endswith(".coefficients"):
            components["coefficients"][key] = value
        elif ".lora_a." in key or ".lora_b." in key:
            components["adapter"][key] = value
        elif key.endswith((".atom_u", ".atom_v", "._extra_state")):
            components["atoms"][key] = value
        else:
            raise ValueError(f"unrecognized compact state key: {key}")
    return components


def save_compact_checkpoint(
    model: nn.Module,
    directory: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(directory)
    state = extract_adapter_state_dict(model, include_heads=True)
    components = _split_adapter_state(state)
    paths: dict[str, str] = {}
    byte_counts: dict[str, int] = {}
    for name, component_state in components.items():
        if not component_state:
            continue
        path = destination / f"{name}.pt"
        _atomic_torch_save(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "component": name,
                "metadata": dict(metadata),
                "state_dict": component_state,
            },
        )
        paths[name] = str(path)
        byte_counts[name] = checkpoint_bytes(path)
    return {
        "paths": paths,
        "bytes_by_component": byte_counts,
        "total_bytes": sum(byte_counts.values()),
        "format": "torch.save",
        "dtype": str(next(model.parameters()).dtype),
    }


def load_compact_checkpoint(model: nn.Module, directory: str | Path) -> None:
    source = Path(directory)
    combined: OrderedDict[str, Any] = OrderedDict()
    for name in ("adapter", "atoms", "coefficients", "heads"):
        path = source / f"{name}.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        combined.update(payload["state_dict"])
    load_adapter_state_dict(model, combined, include_heads=True, strict=True)


def _run_metadata(
    config: ExperimentConfig,
    model: BertTaskClassifier,
    target_names: Sequence[str],
    prepared: PreparedData,
    run_kind: str,
) -> dict[str, Any]:
    encoder_config = model.encoder.config
    return {
        "schema_version": SCHEMA_VERSION,
        "run_kind": run_kind,
        "model": config.base_model,
        "model_revision": getattr(encoder_config, "_commit_hash", None),
        "seed": config.seed,
        "resolved_config": config.to_dict(),
        "target_modules": list(target_names),
        "target_dimensions": {
            name: [module.out_features, module.in_features]
            for name, module in (
                (name, model.encoder.get_submodule(name)) for name in target_names
            )
        },
        "dataset_provenance": prepared.provenance,
        "environment": environment_record(),
    }


def _print_single_epoch(task: str) -> Callable[[Any], None]:
    def callback(record: Any) -> None:
        score = record.validation.metrics.get("primary_score", record.selection_score)
        print(
            f"  {task} epoch {record.epoch}: train_loss={record.training.loss:.5f}, "
            f"validation={score:.5f}, seconds={record.elapsed_seconds:.1f}",
            flush=True,
        )

    return callback


def _print_multitask_epoch(record: Any) -> None:
    task_scores = ", ".join(
        f"{task}={evaluation.metrics['primary_score']:.4f}"
        for task, evaluation in record.validation.items()
    )
    print(
        f"  shared epoch {record.epoch}: mean={record.selection_score:.5f}, "
        f"{task_scores}, seconds={record.elapsed_seconds:.1f}",
        flush=True,
    )


def run_head_only(
    config: ExperimentConfig,
    task: str,
    output_dir: str | Path,
    *,
    prepared: PreparedData | None = None,
    run_kind: str = "development",
) -> dict[str, Any]:
    set_seed(config.seed)
    prepared = prepared or prepare_data(config, tasks=(task,))
    train_loaders, validation_loaders = build_loaders(prepared, config, tasks=(task,))
    model = build_head_only_model(config, task)
    optimizer = _optimizer(model, config)
    with RuntimeMonitor() as monitor:
        result = train_single_task(
            model,
            train_loaders[task],
            validation_loaders[task],
            optimizer,
            epochs=config.epochs,
            device=config.device,
            task_id=task,
            metric_fn=_metric_fn(task),
            primary_metric="primary_score",
            state_capture_fn=lambda value: extract_adapter_state_dict(value, include_heads=True),
            state_restore_fn=lambda value, state: load_adapter_state_dict(
                value, state, include_heads=True
            ),
            epoch_callback=_print_single_epoch(task),
        )
    runtime = monitor.result()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = save_compact_checkpoint(
        model,
        destination,
        metadata={"system": "head_only", "seed": config.seed, "task": task},
    )
    counts = categorized_parameter_counts(model)
    record = {
        "schema_version": SCHEMA_VERSION,
        "system": "head_only",
        "run_kind": run_kind,
        "seed": config.seed,
        "task": task,
        "history": result.to_dict(),
        "best": _evaluation_dict(result.best_validation),
        "final": _evaluation_dict(result.final_validation),
        "parameter_counts": counts,
        "checkpoint": checkpoint,
        "runtime": {
            "elapsed_seconds": runtime.elapsed_seconds,
            "peak_rss_bytes": runtime.peak_rss_bytes,
        },
        "resolved_config": config.to_dict(),
        "dataset_provenance": prepared.provenance[task],
    }
    write_json(destination / "metrics.json", record)
    return record


def _aggregate_independent_counts(task_records: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    entries = [record["parameter_counts"] for record in task_records.values()]
    base_sizes = {int(entry["base_model_parameters"]) for entry in entries}
    if len(base_sizes) != 1:
        raise ValueError("independent tasks do not share the same base parameter count")
    return {
        "base_model_parameters": next(iter(base_sizes)),
        "base_trainable_parameters": sum(
            int(entry["base_trainable_parameters"]) for entry in entries
        ),
        "adapter_parameters": sum(
            int(entry["lora_adapter_parameters"]) for entry in entries
        ),
        "head_parameters": sum(int(entry["head_parameters"]) for entry in entries),
        "uncategorized_trainable_parameters": sum(
            int(entry["uncategorized_trainable_parameters"]) for entry in entries
        ),
        "total_persistent_task_parameters": sum(
            int(entry["persistent_adaptation_parameters"]) for entry in entries
        ),
    }


def run_independent_lora(
    config: ExperimentConfig,
    output_root: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    prepared: PreparedData | None = None,
    run_kind: str = "confirmatory",
    rank: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    selected_tasks = tuple(tasks or config.tasks)
    prepared = prepared or prepare_data(config, tasks=selected_tasks)
    seed_directory = Path(output_root) / f"seed_{config.seed}"
    seed_directory.mkdir(parents=True, exist_ok=True)
    task_records: dict[str, Any] = {}
    for task in selected_tasks:
        task_directory = seed_directory / task
        metrics_path = task_directory / "metrics.json"
        if not force and metrics_path.exists():
            print(f"Skipping completed independent {task}, seed {config.seed}.", flush=True)
            task_records[task] = read_json(metrics_path)
            continue
        set_seed(config.seed)
        train_loaders, validation_loaders = build_loaders(prepared, config, tasks=(task,))
        model, target_names = build_lora_model(config, (task,), rank=rank)
        optimizer = _optimizer(model, config)
        print(
            f"Training independent LoRA: task={task}, seed={config.seed}, "
            f"rank={rank or config.lora_rank}",
            flush=True,
        )
        with RuntimeMonitor() as monitor:
            result = train_single_task(
                model,
                train_loaders[task],
                validation_loaders[task],
                optimizer,
                epochs=config.epochs,
                device=config.device,
                task_id=task,
                metric_fn=_metric_fn(task),
                primary_metric="primary_score",
                state_capture_fn=lambda value: extract_adapter_state_dict(
                    value, include_heads=True
                ),
                state_restore_fn=lambda value, state: load_adapter_state_dict(
                    value, state, include_heads=True
                ),
                epoch_callback=_print_single_epoch(task),
            )
        runtime = monitor.result()
        metadata = _run_metadata(config, model, target_names, prepared, run_kind)
        checkpoint = save_compact_checkpoint(
            model,
            task_directory,
            metadata={"system": "independent_lora", "seed": config.seed, "task": task},
        )
        counts = categorized_parameter_counts(model)
        record = {
            **metadata,
            "system": "independent_lora",
            "task": task,
            "rank": rank or config.lora_rank,
            "history": result.to_dict(),
            "best": _evaluation_dict(result.best_validation, include_outputs=True),
            "final": _evaluation_dict(result.final_validation, include_outputs=True),
            "parameter_counts": counts,
            "checkpoint": checkpoint,
            "runtime": {
                "elapsed_seconds": runtime.elapsed_seconds,
                "peak_rss_bytes": runtime.peak_rss_bytes,
            },
            "active_adapter_operations_per_token": active_adapter_operations(model),
        }
        write_json(metrics_path, record)
        task_records[task] = record
        del optimizer, model
        gc.collect()

    parameter_counts = _aggregate_independent_counts(task_records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "system": "independent_lora",
        "run_kind": run_kind,
        "seed": config.seed,
        "rank": rank or config.lora_rank,
        "tasks": task_records,
        "parameter_counts": parameter_counts,
        "checkpoint_bytes": sum(
            int(record["checkpoint"]["total_bytes"]) for record in task_records.values()
        ),
        "resolved_config": config.to_dict(),
    }
    write_json(seed_directory / "metrics_by_task.json", summary)
    write_json(seed_directory / "parameter_counts.json", parameter_counts)
    return summary


def _top_k_masks(model: nn.Module, tasks: Sequence[str], top_k: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    named_layers = [
        (name, layer) for name, layer in model.named_modules() if isinstance(layer, AtomLinear)
    ]
    for task in tasks:
        result[task] = {
            name: [int(index) for index in layer.topk_mask(task, top_k).nonzero().flatten()]
            for name, layer in named_layers
        }
    return result


def coefficient_analysis(
    model: nn.Module,
    tasks: Sequence[str],
    *,
    top_k: int,
    dead_threshold: float = 1e-6,
) -> dict[str, Any]:
    layers = tuple(iter_atom_layers(model))
    if not layers:
        raise ValueError("coefficient analysis requires atom layers")
    atom_count = layers[0].atom_count
    usage_by_task: dict[str, list[float]] = {}
    top_atoms: dict[str, list[int]] = {}
    masks = _top_k_masks(model, tasks, top_k)
    utilization = [0] * atom_count
    for task in tasks:
        rows = torch.stack([layer.coefficient_row(task).detach().cpu() for layer in layers])
        mean_abs = rows.abs().mean(dim=0)
        usage_by_task[task] = [float(value) for value in mean_abs]
        selected = sorted(range(atom_count), key=lambda index: (-float(mean_abs[index]), index))[
            :top_k
        ]
        top_atoms[task] = selected
        for index in set(selected):
            utilization[index] += 1

    similarities: dict[str, float] = {}
    for left_index, left in enumerate(tasks):
        left_vector = torch.tensor(usage_by_task[left])
        for right in tasks[left_index + 1 :]:
            right_vector = torch.tensor(usage_by_task[right])
            denominator = float(left_vector.norm() * right_vector.norm())
            similarity = 0.0 if denominator == 0 else float(torch.dot(left_vector, right_vector)) / denominator
            similarities[f"{left}:{right}"] = similarity

    max_usage = [max(usage_by_task[task][index] for task in tasks) for index in range(atom_count)]
    return {
        "usage_by_task": usage_by_task,
        "top_atoms_by_task": top_atoms,
        "top_k_masks_by_layer": masks,
        "pairwise_cosine_similarity": similarities,
        "atom_utilization_count": utilization,
        "dead_atoms": [index for index, value in enumerate(max_usage) if value <= dead_threshold],
        "task_exclusive_atoms": [index for index, count in enumerate(utilization) if count == 1],
        "reused_atoms": [index for index, count in enumerate(utilization) if count >= 2],
    }


def run_shared_atoms(
    config: ExperimentConfig,
    output_root: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    prepared: PreparedData | None = None,
    run_kind: str = "confirmatory",
    atom_count: int | None = None,
    top_k: int | None = None,
    sparsity_lambda: float | None = None,
    freeze_atoms: bool = False,
    shuffle_training_labels: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    selected_tasks = tuple(tasks or config.tasks)
    resolved_atom_count = atom_count or config.atom_count
    resolved_top_k = top_k or min(config.active_atoms_for_primary_evaluation, resolved_atom_count)
    if resolved_top_k > resolved_atom_count:
        raise ValueError("top_k cannot exceed atom_count")
    seed_directory = Path(output_root) / f"seed_{config.seed}"
    metrics_path = seed_directory / "metrics_by_task.json"
    if not force and metrics_path.exists():
        print(f"Skipping completed shared atoms, seed {config.seed}.", flush=True)
        return read_json(metrics_path)

    set_seed(config.seed)
    prepared = prepared or prepare_data(config, tasks=selected_tasks)
    train_loaders, validation_loaders = build_loaders(
        prepared,
        config,
        tasks=selected_tasks,
        shuffle_training_labels=shuffle_training_labels,
    )
    model, target_names = build_atom_model(
        config,
        selected_tasks,
        atom_count=resolved_atom_count,
        freeze_atoms=freeze_atoms,
    )
    optimizer = _optimizer(model, config)
    metric_fns = {task: _metric_fn(task) for task in selected_tasks}
    primary_metrics = {task: "primary_score" for task in selected_tasks}
    resolved_lambda = config.sparsity_lambda if sparsity_lambda is None else sparsity_lambda
    regularizer = lambda value, task: coefficient_l1_regularization(
        value, task, resolved_lambda
    )
    print(
        f"Training shared atoms: tasks={','.join(selected_tasks)}, seed={config.seed}, "
        f"atoms={resolved_atom_count}, lambda={resolved_lambda:g}, "
        f"freeze_atoms={freeze_atoms}",
        flush=True,
    )
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
            primary_metrics=primary_metrics,
            regularization_fn=regularizer,
            diagnostics_fn=lambda value: collect_shared_atom_diagnostics(value),
            schedule_mode="complete_pass",
            state_capture_fn=lambda value: extract_adapter_state_dict(
                value, include_heads=True
            ),
            state_restore_fn=lambda value, state: load_adapter_state_dict(
                value, state, include_heads=True
            ),
            epoch_callback=_print_multitask_epoch,
        )
    runtime = monitor.result()

    task_records: dict[str, Any] = {}
    inference_seconds: dict[str, dict[str, float]] = {}
    for task in selected_tasks:
        started = time.perf_counter()
        all_atoms = evaluate(
            model,
            validation_loaders[task],
            config.device,
            task_id=task,
            metric_fn=metric_fns[task],
            scalar_metric_name="primary_score",
        )
        all_seconds = time.perf_counter() - started

        model.set_active_task(task)
        model.set_atom_top_k(resolved_top_k)
        started = time.perf_counter()
        top_result = evaluate(
            model,
            validation_loaders[task],
            config.device,
            task_id=None,
            metric_fn=metric_fns[task],
            scalar_metric_name="primary_score",
        )
        top_seconds = time.perf_counter() - started
        model.clear_atom_top_k()
        task_records[task] = {
            "all_atoms": _evaluation_dict(all_atoms, include_outputs=True),
            "top_k": _evaluation_dict(top_result, include_outputs=True),
            "top_k_value": resolved_top_k,
        }
        inference_seconds[task] = {
            "all_atoms": all_seconds,
            "top_k": top_seconds,
        }

    analysis = coefficient_analysis(model, selected_tasks, top_k=resolved_top_k)
    metadata = _run_metadata(config, model, target_names, prepared, run_kind)
    seed_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = save_compact_checkpoint(
        model,
        seed_directory,
        metadata={
            "system": "shared_atoms",
            "seed": config.seed,
            "tasks": list(selected_tasks),
            "atom_count": resolved_atom_count,
        },
    )
    counts = shared_atom_parameter_totals(model)
    summary = {
        **metadata,
        "system": "shared_atoms",
        "seed": config.seed,
        "tasks": task_records,
        "task_ids": list(selected_tasks),
        "atom_count": resolved_atom_count,
        "active_atoms_during_training": resolved_atom_count,
        "active_atoms_for_evaluation": resolved_top_k,
        "sparsity_lambda": resolved_lambda,
        "atoms_frozen": freeze_atoms,
        "training_labels_shuffled": shuffle_training_labels,
        "history": result.to_dict(),
        "parameter_counts": counts,
        "checkpoint": checkpoint,
        "runtime": {
            "elapsed_seconds": runtime.elapsed_seconds,
            "peak_rss_bytes": runtime.peak_rss_bytes,
            "inference_seconds_by_task": inference_seconds,
        },
        "active_adapter_operations_per_token": {
            "all_atoms": active_adapter_operations(model, active_atoms=resolved_atom_count),
            "top_k": active_adapter_operations(model, active_atoms=resolved_top_k),
        },
        "coefficient_analysis": analysis,
    }
    write_json(metrics_path, summary)
    write_json(seed_directory / "parameter_counts.json", counts)
    write_json(seed_directory / "training_history.json", result.to_dict())
    write_json(seed_directory / "top_k_masks.json", analysis["top_k_masks_by_layer"])
    return summary


def evaluate_atom_checkpoint(
    config: ExperimentConfig,
    run_directory: str | Path,
    *,
    top_k: int,
    tasks: Sequence[str] | None = None,
    prepared: PreparedData | None = None,
) -> dict[str, Any]:
    """Reload a compact atom checkpoint and evaluate a fixed top-k mask."""

    source = Path(run_directory)
    existing = read_json(source / "metrics_by_task.json")
    stored_config = ExperimentConfig.from_mapping(existing["resolved_config"])
    if config.seed != stored_config.seed:
        raise ValueError(
            f"requested seed {config.seed} does not match checkpoint seed {stored_config.seed}"
        )
    config = stored_config
    selected_tasks = tuple(
        tasks
        or existing.get("task_ids")
        or [task for task in config.tasks if task in existing["tasks"]]
    )
    atom_count = int(existing["atom_count"])
    if not 1 <= top_k <= atom_count:
        raise ValueError(f"top_k must be in [1, {atom_count}], got {top_k}")
    set_seed(config.seed)
    prepared = prepared or prepare_data(config, tasks=selected_tasks)
    _, validation_loaders = build_loaders(prepared, config, tasks=selected_tasks)
    model, _ = build_atom_model(
        config,
        selected_tasks,
        atom_count=atom_count,
        freeze_atoms=bool(existing.get("atoms_frozen", False)),
    )
    load_compact_checkpoint(model, source)
    results: dict[str, Any] = {}
    for task in selected_tasks:
        model.set_active_task(task)
        model.set_atom_top_k(top_k)
        evaluation = evaluate(
            model,
            validation_loaders[task],
            config.device,
            task_id=None,
            metric_fn=_metric_fn(task),
            scalar_metric_name="primary_score",
        )
        model.clear_atom_top_k()
        results[task] = _evaluation_dict(evaluation, include_outputs=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "system": "shared_atoms",
        "seed": config.seed,
        "atom_count": atom_count,
        "top_k": top_k,
        "tasks": results,
        "masks": _top_k_masks(model, selected_tasks, top_k),
    }
    write_json(source / f"top_k_{top_k}_evaluation.json", record)
    return record


def run_core_seed(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    results_root: str | Path = "results",
    *,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if baseline_config.seed != atom_config.seed:
        raise ValueError("paired core systems must use the same seed")
    if baseline_config.experiment_name != "independent_lora":
        raise ValueError("baseline config must select independent_lora")
    if atom_config.experiment_name != "shared_atoms":
        raise ValueError("atom config must select shared_atoms")
    baseline_config.validate_h1_contract()
    atom_config.validate_h1_contract()
    prepared = prepare_data(baseline_config)
    root = Path(results_root)
    independent = run_independent_lora(
        baseline_config,
        root / "independent_lora",
        prepared=prepared,
        force=force,
    )
    shared = run_shared_atoms(
        atom_config,
        root / "shared_atoms",
        prepared=prepared,
        force=force,
    )
    return independent, shared


def validate_expected_core_counts(
    independent: Mapping[str, Any],
    shared: Mapping[str, Any],
) -> None:
    independent_count = independent["parameter_counts"]["total_persistent_task_parameters"]
    shared_count = shared["parameter_counts"]["total_persistent_task_parameters"]
    if int(independent_count) != 21_770:
        raise AssertionError(f"expected 21,770 independent parameters, got {independent_count}")
    if int(shared_count) != 9_642:
        raise AssertionError(f"expected 9,642 shared parameters, got {shared_count}")
    ratio = int(shared_count) / int(independent_count)
    if not math.isclose(ratio, 0.4429030776, rel_tol=1e-9):
        raise AssertionError(f"unexpected storage ratio: {ratio}")


__all__ = [
    "PreparedData",
    "build_atom_model",
    "build_head_only_model",
    "build_loaders",
    "build_lora_model",
    "coefficient_analysis",
    "environment_record",
    "evaluate_atom_checkpoint",
    "load_compact_checkpoint",
    "prepare_data",
    "run_core_seed",
    "run_head_only",
    "run_independent_lora",
    "run_shared_atoms",
    "save_compact_checkpoint",
    "validate_expected_core_counts",
]
