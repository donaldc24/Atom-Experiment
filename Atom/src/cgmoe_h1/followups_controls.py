"""Roadmap chunk 25: random, averaging, retrieval, and multitask controls.

The confirmatory H1 decision is intentionally kept separate from these controls.
This module runs the six predeclared diagnostics at the locked development seed
17, reusing the exact core data selection and compact independent-LoRA
checkpoints.  Every expensive control has its own completion record, so an
interrupted suite resumes without repeating finished work.
"""

from __future__ import annotations

import gc
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch import Tensor, nn

from cgmoe_h1.config import ExperimentConfig, H1_TASKS
from cgmoe_h1.experiments import (
    PreparedData,
    build_loaders,
    build_lora_model,
    prepare_data,
    run_shared_atoms,
    save_compact_checkpoint,
)
from cgmoe_h1.metrics import compute_task_metrics
from cgmoe_h1.models.injection import (
    extract_adapter_state_dict,
    load_adapter_state_dict,
)
from cgmoe_h1.training.multitask import train_multitask
from cgmoe_h1.training.trainer import create_adamw_optimizer, evaluate
from cgmoe_h1.utils.parameters import (
    active_adapter_operations,
    categorized_parameter_counts,
)
from cgmoe_h1.utils.reproducibility import make_torch_generator, set_seed
from cgmoe_h1.utils.runtime import RuntimeMonitor
from cgmoe_h1.utils.serialization import read_json, write_json


CONTROL_SEED = 17
CONTROL_SCHEMA_VERSION = 1
CONTROL_RESULTS_FILENAME = "control_results.json"
CONTROL_REPORT_FILENAME = "control_report.md"


@dataclass(frozen=True, slots=True)
class ControlSpec:
    """Stable identity and interpretation for one predeclared control."""

    control_id: str
    title: str
    rules_out: str


CONTROL_SPECS = (
    ControlSpec(
        "random_frozen_atoms",
        "Random frozen atoms + trained coefficients/heads",
        "random projection capacity",
    ),
    ControlSpec(
        "average_independent_loras",
        "Average independent LoRA effective updates",
        "simple adapter averaging",
    ),
    ControlSpec(
        "nearest_other_task_lora",
        "Nearest other-task LoRA retrieval",
        "memorized task lookup or nearest-adapter reuse",
    ),
    ControlSpec(
        "shared_multitask_lora",
        "One shared balanced multitask LoRA",
        "ordinary multitask sharing without atom composition",
    ),
    ControlSpec(
        "shared_atoms_shuffled_labels",
        "Shared atoms with deterministically shuffled labels",
        "data-independent capacity or leakage",
    ),
    ControlSpec(
        "shared_atoms_no_sparsity",
        "Shared atoms without coefficient sparsity",
        "an overly strong sparsity penalty",
    ),
)
CONTROL_IDS = tuple(spec.control_id for spec in CONTROL_SPECS)
CONTROL_BY_ID = {spec.control_id: spec for spec in CONTROL_SPECS}


CompactState: TypeAlias = OrderedDict[str, Any]


@dataclass(slots=True)
class ControlRunContext:
    """Shared immutable inputs and paths supplied to control executors."""

    baseline_config: ExperimentConfig
    atom_config: ExperimentConfig
    output_root: Path
    independent_root: Path
    prepared: PreparedData | None
    force: bool

    def directory(self, control_id: str) -> Path:
        return self.output_root / control_id / f"seed_{CONTROL_SEED}"


ControlExecutor: TypeAlias = Callable[[ControlRunContext], dict[str, Any]]


def validate_control_configs(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
) -> None:
    """Require the complete locked H1 contract at the predeclared seed 17."""

    baseline_config.validate_h1_contract()
    atom_config.validate_h1_contract()
    if baseline_config.experiment_name != "independent_lora":
        raise ValueError("baseline control config must select independent_lora")
    if atom_config.experiment_name != "shared_atoms":
        raise ValueError("atom control config must select shared_atoms")
    if baseline_config.seed != CONTROL_SEED or atom_config.seed != CONTROL_SEED:
        raise ValueError(f"chunk-25 controls are locked to seed {CONTROL_SEED}")
    comparable_fields = (
        "base_model",
        "tasks",
        "train_examples_per_task",
        "validation_examples_per_task",
        "max_length",
        "batch_size",
        "learning_rate",
        "epochs",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "target_modules",
    )
    mismatches = [
        name
        for name in comparable_fields
        if getattr(baseline_config, name) != getattr(atom_config, name)
    ]
    if mismatches:
        raise ValueError(f"paired control configs differ in: {', '.join(mismatches)}")


def deterministic_label_permutation(length: int, seed: int, task_index: int) -> list[int]:
    """Return the exact label permutation used by core ``build_loaders``."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if task_index < 0:
        raise ValueError("task_index must be non-negative")
    generator = make_torch_generator(seed + 50_000 + task_index)
    return torch.randperm(length, generator=generator).tolist()


def deterministically_shuffle_labels(
    labels: Sequence[int],
    *,
    seed: int,
    task_index: int,
) -> list[int]:
    """Shuffle labels without altering their multiset or the underlying rows."""

    values = list(labels)
    return [values[index] for index in deterministic_label_permutation(len(values), seed, task_index)]


def _require_tensor(state: Mapping[str, Any], key: str) -> Tensor:
    value = state.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"compact LoRA state is missing tensor {key!r}")
    return value


def _lora_a_keys(state: Mapping[str, Any]) -> tuple[str, ...]:
    keys = tuple(sorted(key for key in state if key.endswith(".lora_a.weight")))
    if not keys:
        raise ValueError("compact state contains no LoRA A tensors")
    allowed = {
        key
        for a_key in keys
        for key in (a_key, a_key.removesuffix("lora_a.weight") + "lora_b.weight")
    }
    unexpected = sorted(set(state) - allowed)
    if unexpected:
        raise ValueError(f"LoRA adapter state contains non-adapter keys: {unexpected}")
    return keys


def effective_lora_updates(
    state: Mapping[str, Any],
    *,
    scaling: float = 1.0,
) -> OrderedDict[str, Tensor]:
    """Materialize each effective ``scale * B @ A`` update in stable path order."""

    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("scaling must be finite and positive")
    updates: OrderedDict[str, Tensor] = OrderedDict()
    for a_key in _lora_a_keys(state):
        prefix = a_key.removesuffix("lora_a.weight")
        b_key = prefix + "lora_b.weight"
        a = _require_tensor(state, a_key)
        b = _require_tensor(state, b_key)
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(
                f"incompatible LoRA factors at {prefix!r}: A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        updates[prefix.removesuffix(".")] = (b.double() @ a.double()) * scaling
    return updates


def effective_lora_vector(state: Mapping[str, Any], *, scaling: float = 1.0) -> Tensor:
    """Concatenate all effective updates for cosine comparison."""

    return torch.cat(
        [update.reshape(-1) for update in effective_lora_updates(state, scaling=scaling).values()]
    )


def cosine_effective_lora_similarity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    scaling: float = 1.0,
) -> float:
    """Cosine similarity of path-aligned effective LoRA update matrices."""

    left_updates = effective_lora_updates(left, scaling=scaling)
    right_updates = effective_lora_updates(right, scaling=scaling)
    if tuple(left_updates) != tuple(right_updates):
        raise ValueError("LoRA states do not contain the same target module paths")
    for name in left_updates:
        if left_updates[name].shape != right_updates[name].shape:
            raise ValueError(f"effective update shape mismatch at {name!r}")
    left_vector = torch.cat([value.reshape(-1) for value in left_updates.values()])
    right_vector = torch.cat([value.reshape(-1) for value in right_updates.values()])
    denominator = float(left_vector.norm() * right_vector.norm())
    if denominator == 0.0:
        return 0.0
    return float(torch.dot(left_vector, right_vector) / denominator)


def nearest_other_task_adapters(
    adapter_states: Mapping[str, Mapping[str, Any]],
    tasks: Sequence[str] = H1_TASKS,
) -> dict[str, dict[str, Any]]:
    """Choose the closest different task, breaking exact ties by task order."""

    task_ids = tuple(tasks)
    if len(task_ids) < 2:
        raise ValueError("nearest-other-task retrieval requires at least two tasks")
    missing = [task for task in task_ids if task not in adapter_states]
    if missing:
        raise ValueError(f"missing adapter states for: {', '.join(missing)}")
    result: dict[str, dict[str, Any]] = {}
    for target in task_ids:
        candidates: list[tuple[str, float]] = []
        for source in task_ids:
            if source == target:
                continue
            similarity = cosine_effective_lora_similarity(
                adapter_states[target], adapter_states[source]
            )
            candidates.append((source, similarity))
        selected, _ = min(
            candidates,
            key=lambda item: (-item[1], task_ids.index(item[0])),
        )
        result[target] = {
            "source_task": selected,
            "similarities": {source: value for source, value in candidates},
        }
    return result


def average_effective_lora_state(
    adapter_states: Mapping[str, Mapping[str, Any]],
    *,
    rank: int,
) -> CompactState:
    """Average effective updates and deterministically refactor to locked rank.

    Directly averaging independently initialized A/B factors is basis-dependent
    and introduces cross terms.  Instead this control averages ``B @ A`` and
    uses a truncated SVD to obtain the best rank-``rank`` LoRA representation.
    """

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if not adapter_states:
        raise ValueError("at least one adapter state is required")
    task_states = list(adapter_states.values())
    template = task_states[0]
    template_keys = tuple(template)
    if any(tuple(state) != template_keys for state in task_states[1:]):
        raise ValueError("adapter states must have identical ordered keys")

    updates_by_task = [effective_lora_updates(state) for state in task_states]
    averaged: CompactState = OrderedDict()
    factors: dict[str, tuple[Tensor, Tensor]] = {}
    for prefix in updates_by_task[0]:
        matrices = [updates[prefix] for updates in updates_by_task]
        if any(matrix.shape != matrices[0].shape for matrix in matrices[1:]):
            raise ValueError(f"effective update shape mismatch at {prefix!r}")
        mean_update = torch.stack(matrices).mean(dim=0)
        u, singular_values, vh = torch.linalg.svd(mean_update, full_matrices=False)
        retained = min(rank, singular_values.numel())
        sqrt_s = singular_values[:retained].clamp_min(0).sqrt()
        b_factor = torch.zeros(mean_update.shape[0], rank, dtype=mean_update.dtype)
        a_factor = torch.zeros(rank, mean_update.shape[1], dtype=mean_update.dtype)
        b_factor[:, :retained] = u[:, :retained] * sqrt_s.unsqueeze(0)
        a_factor[:retained] = sqrt_s.unsqueeze(1) * vh[:retained]
        factors[prefix] = (a_factor, b_factor)

    for key, value in template.items():
        if not isinstance(value, Tensor):
            raise ValueError(f"LoRA adapter entry {key!r} must be a tensor")
        if key.endswith(".lora_a.weight"):
            prefix = key.removesuffix(".lora_a.weight")
            factor = factors[prefix][0]
        elif key.endswith(".lora_b.weight"):
            prefix = key.removesuffix(".lora_b.weight")
            factor = factors[prefix][1]
        else:  # _lora_a_keys above normally makes this unreachable.
            raise ValueError(f"unexpected LoRA key {key!r}")
        averaged[key] = factor.to(dtype=value.dtype, device="cpu")
    return averaged


def _load_compact_component(directory: Path, component: str) -> CompactState:
    path = directory / f"{component}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing compact checkpoint component: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("component") != component:
        raise ValueError(f"invalid compact {component} checkpoint: {path}")
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError(f"compact checkpoint has no state_dict mapping: {path}")
    state: CompactState = OrderedDict(raw_state)
    if component == "adapter":
        _lora_a_keys(state)
    elif component == "heads":
        unexpected = sorted(key for key in state if not key.startswith("heads."))
        if unexpected:
            raise ValueError(f"head checkpoint contains non-head keys: {unexpected}")
    return state


def load_independent_compact_states(
    independent_root: str | Path,
    *,
    tasks: Sequence[str] = H1_TASKS,
    seed: int = CONTROL_SEED,
) -> tuple[dict[str, CompactState], dict[str, CompactState]]:
    """Load only adapter/head components from the selected core checkpoints."""

    source = Path(independent_root) / f"seed_{seed}"
    adapter_states: dict[str, CompactState] = {}
    head_states: dict[str, CompactState] = {}
    for task in tasks:
        task_directory = source / task
        adapter_states[task] = _load_compact_component(task_directory, "adapter")
        head_states[task] = _load_compact_component(task_directory, "heads")
        expected_head_prefix = f"heads.{task}."
        if not head_states[task] or any(
            not key.startswith(expected_head_prefix) for key in head_states[task]
        ):
            raise ValueError(f"{task} head checkpoint does not contain only its target head")
    reference_keys = tuple(adapter_states[tasks[0]])
    for task in tasks[1:]:
        if tuple(adapter_states[task]) != reference_keys:
            raise ValueError(f"independent adapter paths differ for task {task!r}")
    return adapter_states, head_states


def validate_independent_core_checkpoints(
    config: ExperimentConfig,
    independent_root: str | Path,
) -> dict[str, Any]:
    """Audit the seed-17 independent run before any control training starts."""

    seed_directory = Path(independent_root) / f"seed_{CONTROL_SEED}"
    summary_path = seed_directory / "metrics_by_task.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"chunk-25 controls require the core seed-17 independent run: {summary_path}"
        )
    summary = read_json(summary_path)
    if int(summary.get("seed", -1)) != CONTROL_SEED:
        raise ValueError("independent checkpoint summary is not seed 17")
    if int(summary.get("rank", -1)) != config.lora_rank:
        raise ValueError("independent checkpoint rank does not match the locked config")
    if set(summary.get("tasks", {})) != set(config.tasks):
        raise ValueError("independent checkpoint set does not contain all five locked tasks")
    stored = ExperimentConfig.from_mapping(summary["resolved_config"])
    stored.validate_h1_contract()
    if stored.seed != CONTROL_SEED:
        raise ValueError("stored independent config is not seed 17")
    load_independent_compact_states(
        independent_root,
        tasks=config.tasks,
        seed=CONTROL_SEED,
    )
    return summary


def _require_prepared(context: ControlRunContext) -> PreparedData:
    if context.prepared is None:
        raise RuntimeError("real control execution requires prepared locked data")
    return context.prepared


def _metric_fn(task: str) -> Callable[[Tensor, Tensor], dict[str, float]]:
    return partial(compute_task_metrics, task)


def _optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.AdamW:
    return create_adamw_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        epsilon=config.adam_epsilon,
    )


def _evaluate_lora_tasks(
    model: nn.Module,
    validation_loaders: Mapping[str, Any],
    config: ExperimentConfig,
    tasks: Sequence[str],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        started = time.perf_counter()
        evaluation = evaluate(
            model,
            validation_loaders[task],
            config.device,
            task_id=task,
            metric_fn=_metric_fn(task),
            scalar_metric_name="primary_score",
        )
        results[task] = {
            "primary": evaluation.to_dict(include_outputs=True),
            "inference_seconds": time.perf_counter() - started,
        }
    return results


def _base_record(
    control_id: str,
    config: ExperimentConfig,
    *,
    design: Mapping[str, Any],
    tasks: Mapping[str, Any],
) -> dict[str, Any]:
    spec = CONTROL_BY_ID[control_id]
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "control_id": control_id,
        "title": spec.title,
        "rules_out": spec.rules_out,
        "status": "complete",
        "seed": CONTROL_SEED,
        "run_kind": "followup_control",
        "locked_budget": {
            "train_examples_per_task": config.train_examples_per_task,
            "validation_examples_per_task": config.validation_examples_per_task,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "tasks": list(config.tasks),
        },
        "design": dict(design),
        "tasks": dict(tasks),
        "resolved_config": config.to_dict(),
        "compact_state_only": True,
    }


def _standardize_atom_control(
    context: ControlRunContext,
    control_id: str,
    *,
    freeze_atoms: bool = False,
    shuffle_training_labels: bool = False,
    sparsity_lambda: float | None = None,
) -> dict[str, Any]:
    config = context.atom_config
    core_record = run_shared_atoms(
        config,
        context.output_root / control_id,
        prepared=_require_prepared(context),
        run_kind="followup_control",
        atom_count=config.atom_count,
        top_k=config.active_atoms_for_primary_evaluation,
        sparsity_lambda=sparsity_lambda,
        freeze_atoms=freeze_atoms,
        shuffle_training_labels=shuffle_training_labels,
        force=context.force,
    )
    task_records = {
        task: {
            "primary": core_record["tasks"][task]["top_k"],
            "unpruned": core_record["tasks"][task]["all_atoms"],
        }
        for task in config.tasks
    }
    resolved_lambda = (
        config.sparsity_lambda if sparsity_lambda is None else float(sparsity_lambda)
    )
    design: dict[str, Any] = {
        "model": "shared_atom_dictionary",
        "atom_count": config.atom_count,
        "active_atoms_for_evaluation": config.active_atoms_for_primary_evaluation,
        "atoms_frozen": freeze_atoms,
        "coefficients_trainable": True,
        "heads_trainable": True,
        "sparsity_lambda": resolved_lambda,
        "training_labels_shuffled": shuffle_training_labels,
        "multitask_schedule": "seeded balanced complete pass",
    }
    if shuffle_training_labels:
        design["label_shuffle_seed_by_task"] = {
            task: CONTROL_SEED + 50_000 + index
            for index, task in enumerate(config.tasks)
        }
    record = _base_record(
        control_id,
        config,
        design=design,
        tasks=task_records,
    )
    record.update(
        {
            "history": core_record["history"],
            "parameter_counts": core_record["parameter_counts"],
            "checkpoint": core_record["checkpoint"],
            "runtime": core_record["runtime"],
            "dataset_provenance": core_record["dataset_provenance"],
            "coefficient_analysis": core_record["coefficient_analysis"],
            "source_metrics": str(
                context.output_root
                / control_id
                / f"seed_{CONTROL_SEED}"
                / "metrics_by_task.json"
            ),
        }
    )
    return record


def run_random_frozen_atoms(context: ControlRunContext) -> dict[str, Any]:
    """Train only task coefficients/heads over a fixed random atom dictionary."""

    return _standardize_atom_control(
        context,
        "random_frozen_atoms",
        freeze_atoms=True,
    )


def run_shared_atoms_shuffled_labels(context: ControlRunContext) -> dict[str, Any]:
    """Train shared atoms after a seeded within-task permutation of labels."""

    return _standardize_atom_control(
        context,
        "shared_atoms_shuffled_labels",
        shuffle_training_labels=True,
    )


def run_shared_atoms_no_sparsity(context: ControlRunContext) -> dict[str, Any]:
    """Train shared atoms with the coefficient L1 weight set exactly to zero."""

    return _standardize_atom_control(
        context,
        "shared_atoms_no_sparsity",
        sparsity_lambda=0.0,
    )


def run_average_independent_loras(context: ControlRunContext) -> dict[str, Any]:
    """Evaluate a rank-4 refactorization of the mean effective LoRA update."""

    config = context.baseline_config
    prepared = _require_prepared(context)
    adapter_states, head_states = load_independent_compact_states(
        context.independent_root,
        tasks=config.tasks,
    )
    averaged_adapter = average_effective_lora_state(
        adapter_states,
        rank=config.lora_rank,
    )
    set_seed(CONTROL_SEED)
    _, validation_loaders = build_loaders(prepared, config, tasks=config.tasks)
    model, target_names = build_lora_model(config, config.tasks)
    combined: CompactState = OrderedDict(averaged_adapter)
    for task in config.tasks:
        combined.update(head_states[task])
    load_adapter_state_dict(model, combined, include_heads=True, strict=True)
    task_records = _evaluate_lora_tasks(model, validation_loaders, config, config.tasks)
    checkpoint = save_compact_checkpoint(
        model,
        context.directory("average_independent_loras"),
        metadata={
            "control_id": "average_independent_loras",
            "seed": CONTROL_SEED,
            "source_tasks": list(config.tasks),
            "factorization": "truncated_svd_of_mean_effective_update",
        },
    )
    record = _base_record(
        "average_independent_loras",
        config,
        design={
            "model": "one_shared_rank4_lora",
            "average_space": "effective_delta_weight",
            "formula": "mean_t(B_t @ A_t)",
            "rank_projection": "deterministic truncated SVD",
            "rank": config.lora_rank,
            "classification_heads": "each target task's own selected core head",
            "additional_training_updates": 0,
        },
        tasks=task_records,
    )
    record.update(
        {
            "target_modules": target_names,
            "parameter_counts": categorized_parameter_counts(model),
            "checkpoint": checkpoint,
            "source_checkpoints": {
                task: str(
                    context.independent_root / f"seed_{CONTROL_SEED}" / task / "adapter.pt"
                )
                for task in config.tasks
            },
            "dataset_provenance": prepared.provenance,
            "active_adapter_operations_per_token": active_adapter_operations(model),
        }
    )
    return record


def run_nearest_other_task_lora(context: ControlRunContext) -> dict[str, Any]:
    """Retrieve a different task's closest effective update for every target."""

    config = context.baseline_config
    prepared = _require_prepared(context)
    adapter_states, head_states = load_independent_compact_states(
        context.independent_root,
        tasks=config.tasks,
    )
    retrieval = nearest_other_task_adapters(adapter_states, config.tasks)
    set_seed(CONTROL_SEED)
    _, validation_loaders = build_loaders(prepared, config, tasks=config.tasks)
    task_records: dict[str, Any] = {}
    checkpoint_records: dict[str, Any] = {}
    parameter_counts: dict[str, Any] = {}
    for target in config.tasks:
        source = retrieval[target]["source_task"]
        model, target_names = build_lora_model(config, (target,))
        combined: CompactState = OrderedDict(adapter_states[source])
        combined.update(head_states[target])
        load_adapter_state_dict(model, combined, include_heads=True, strict=True)
        evaluation = evaluate(
            model,
            validation_loaders[target],
            config.device,
            task_id=target,
            metric_fn=_metric_fn(target),
            scalar_metric_name="primary_score",
        )
        task_records[target] = {
            "primary": evaluation.to_dict(include_outputs=True),
            "retrieved_source_task": source,
            "cosine_similarity": retrieval[target]["similarities"][source],
            "similarities_to_other_tasks": retrieval[target]["similarities"],
        }
        checkpoint_records[target] = save_compact_checkpoint(
            model,
            context.directory("nearest_other_task_lora") / target,
            metadata={
                "control_id": "nearest_other_task_lora",
                "seed": CONTROL_SEED,
                "target_task": target,
                "retrieved_source_task": source,
            },
        )
        parameter_counts[target] = categorized_parameter_counts(model)
        del model
        gc.collect()

    record = _base_record(
        "nearest_other_task_lora",
        config,
        design={
            "model": "one_retrieved_independent_rank4_lora_per_target",
            "retrieval_query": "target task's selected independent LoRA",
            "similarity": "cosine of concatenated effective B @ A updates",
            "candidate_pool": "all four other tasks; target adapter excluded",
            "tie_break": "locked task order",
            "classification_heads": "target task's own selected core head",
            "additional_training_updates": 0,
        },
        tasks=task_records,
    )
    record.update(
        {
            "retrieval": retrieval,
            "parameter_counts_by_target": parameter_counts,
            "checkpoints_by_target": checkpoint_records,
            "source_checkpoint_root": str(
                context.independent_root / f"seed_{CONTROL_SEED}"
            ),
            "dataset_provenance": prepared.provenance,
            "target_modules": target_names,
        }
    )
    return record


def run_shared_multitask_lora(context: ControlRunContext) -> dict[str, Any]:
    """Train one rank-4 LoRA with five heads on the balanced complete-pass schedule."""

    config = context.baseline_config
    prepared = _require_prepared(context)
    set_seed(CONTROL_SEED)
    train_loaders, validation_loaders = build_loaders(
        prepared,
        config,
        tasks=config.tasks,
    )
    model, target_names = build_lora_model(config, config.tasks)
    optimizer = _optimizer(model, config)
    metric_fns = {task: _metric_fn(task) for task in config.tasks}
    primary_metrics = {task: "primary_score" for task in config.tasks}
    with RuntimeMonitor() as monitor:
        training_result = train_multitask(
            model,
            train_loaders,
            validation_loaders,
            optimizer,
            epochs=config.epochs,
            seed=CONTROL_SEED,
            device=config.device,
            metric_fns=metric_fns,
            primary_metrics=primary_metrics,
            schedule_mode="complete_pass",
            state_capture_fn=lambda value: extract_adapter_state_dict(
                value, include_heads=True
            ),
            state_restore_fn=lambda value, state: load_adapter_state_dict(
                value, state, include_heads=True
            ),
        )
    runtime = monitor.result()
    task_records = _evaluate_lora_tasks(model, validation_loaders, config, config.tasks)
    checkpoint = save_compact_checkpoint(
        model,
        context.directory("shared_multitask_lora"),
        metadata={
            "control_id": "shared_multitask_lora",
            "seed": CONTROL_SEED,
            "tasks": list(config.tasks),
            "rank": config.lora_rank,
        },
    )
    counts = categorized_parameter_counts(model)
    record = _base_record(
        "shared_multitask_lora",
        config,
        design={
            "model": "one_shared_lora_bank_plus_five_task_heads",
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "multitask_schedule": "seeded balanced complete pass",
            "updates": sum(len(loader) for loader in train_loaders.values()) * config.epochs,
            "checkpoint_selection": "earliest epoch with highest unweighted mean task score",
        },
        tasks=task_records,
    )
    record.update(
        {
            "history": training_result.to_dict(),
            "target_modules": target_names,
            "parameter_counts": counts,
            "checkpoint": checkpoint,
            "runtime": {
                "elapsed_seconds": runtime.elapsed_seconds,
                "peak_rss_bytes": runtime.peak_rss_bytes,
            },
            "dataset_provenance": prepared.provenance,
            "active_adapter_operations_per_token": active_adapter_operations(model),
        }
    )
    del optimizer, model
    gc.collect()
    return record


DEFAULT_EXECUTORS: Mapping[str, ControlExecutor] = {
    "random_frozen_atoms": run_random_frozen_atoms,
    "average_independent_loras": run_average_independent_loras,
    "nearest_other_task_lora": run_nearest_other_task_lora,
    "shared_multitask_lora": run_shared_multitask_lora,
    "shared_atoms_shuffled_labels": run_shared_atoms_shuffled_labels,
    "shared_atoms_no_sparsity": run_shared_atoms_no_sparsity,
}


def control_result_path(output_root: str | Path, control_id: str) -> Path:
    if control_id not in CONTROL_BY_ID:
        raise ValueError(f"unknown control ID {control_id!r}")
    return Path(output_root) / control_id / f"seed_{CONTROL_SEED}" / "control_result.json"


def _primary_score(task_record: Mapping[str, Any]) -> float:
    primary = task_record.get("primary")
    if not isinstance(primary, Mapping):
        raise ValueError("control task record is missing primary evaluation")
    metrics = primary.get("metrics")
    if not isinstance(metrics, Mapping) or "primary_score" not in metrics:
        raise ValueError("control task evaluation is missing metrics.primary_score")
    score = float(metrics["primary_score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"invalid control primary score: {score}")
    return score


def validate_control_record(
    record: Mapping[str, Any],
    control_id: str,
    *,
    tasks: Sequence[str] = H1_TASKS,
) -> None:
    """Reject stale, partial, or malformed records before treating them as resumable."""

    if record.get("control_id") != control_id:
        raise ValueError(f"control record ID mismatch for {control_id!r}")
    if record.get("status") != "complete":
        raise ValueError(f"control record {control_id!r} is not complete")
    if int(record.get("seed", -1)) != CONTROL_SEED:
        raise ValueError(f"control record {control_id!r} is not seed {CONTROL_SEED}")
    task_records = record.get("tasks")
    if not isinstance(task_records, Mapping) or set(task_records) != set(tasks):
        raise ValueError(f"control record {control_id!r} does not contain all locked tasks")
    for task in tasks:
        _primary_score(task_records[task])
    if record.get("compact_state_only") is not True:
        raise ValueError(f"control record {control_id!r} does not attest compact-only state")


def _control_digest(record: Mapping[str, Any], result_path: Path) -> dict[str, Any]:
    task_scores = {
        task: _primary_score(record["tasks"][task])
        for task in H1_TASKS
    }
    digest: dict[str, Any] = {
        "control_id": record["control_id"],
        "title": record["title"],
        "rules_out": record["rules_out"],
        "status": record["status"],
        "seed": CONTROL_SEED,
        "task_primary_scores": task_scores,
        "mean_primary_score": sum(task_scores.values()) / len(task_scores),
        "design": record["design"],
        "result_path": str(result_path),
        "compact_state_only": True,
    }
    if "checkpoint" in record:
        digest["checkpoint"] = record["checkpoint"]
    if "checkpoints_by_target" in record:
        digest["checkpoints_by_target"] = record["checkpoints_by_target"]
    if "retrieval" in record:
        digest["retrieval"] = record["retrieval"]
    return digest


def build_control_summary(
    records: Mapping[str, Mapping[str, Any]],
    output_root: str | Path,
    *,
    requested_controls: Sequence[str] = CONTROL_IDS,
    last_error: str | None = None,
) -> dict[str, Any]:
    """Build a small aggregate that points to full per-control audit records."""

    ordered_records = {
        control_id: records[control_id]
        for control_id in CONTROL_IDS
        if control_id in records
    }
    completed = tuple(ordered_records)
    missing = [control_id for control_id in CONTROL_IDS if control_id not in ordered_records]
    summary: dict[str, Any] = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "followup": "roadmap_chunk_25_controls",
        "seed": CONTROL_SEED,
        "status": "complete" if not missing else "partial",
        "controls_requested": list(requested_controls),
        "controls_completed": list(completed),
        "controls_missing": missing,
        "locked_budget": {
            "tasks": list(H1_TASKS),
            "train_examples_per_task": 2000,
            "validation_examples_per_task": 500,
            "epochs": 3,
            "batch_size": 8,
        },
        "controls": {
            control_id: _control_digest(
                record,
                control_result_path(output_root, control_id),
            )
            for control_id, record in ordered_records.items()
        },
        "interpretation": (
            "These are diagnostic controls and do not replace or alter the preregistered H1 decision."
        ),
    }
    if last_error is not None:
        summary["last_error"] = last_error
    return summary


def render_control_report(summary: Mapping[str, Any]) -> str:
    """Render a concise audit-oriented Markdown report for the six controls."""

    status = str(summary.get("status", "partial")).upper()
    lines = [
        "# H1 Follow-up Controls (Roadmap Chunk 25)",
        "",
        f"Status: **{status}**",
        "",
        (
            "All controls use the locked seed 17 data selection, 2,000 training rows per task "
            "(or the upstream split minimum), up to 500 validation rows, batch size 8, and three epochs."
        ),
        "",
        "These diagnostics do not change the preregistered H1 decision.",
        "",
        "## Primary scores",
        "",
        "| Control | SST-2 | MRPC | RTE | QNLI | QQP | Mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    controls = summary.get("controls", {})
    if isinstance(controls, Mapping):
        for control_id in CONTROL_IDS:
            digest = controls.get(control_id)
            if not isinstance(digest, Mapping):
                continue
            scores = digest["task_primary_scores"]
            lines.append(
                "| "
                + str(digest["title"])
                + " | "
                + " | ".join(f"{float(scores[task]):.4f}" for task in H1_TASKS)
                + f" | {float(digest['mean_primary_score']):.4f} |"
            )

    lines.extend(["", "## Control definitions", ""])
    for spec in CONTROL_SPECS:
        digest = controls.get(spec.control_id) if isinstance(controls, Mapping) else None
        state = "complete" if isinstance(digest, Mapping) else "not yet run"
        lines.extend(
            [
                f"### {spec.title}",
                "",
                f"State: **{state}**. Intended to rule out {spec.rules_out}.",
                "",
            ]
        )
        if isinstance(digest, Mapping):
            design = digest.get("design", {})
            if isinstance(design, Mapping):
                for key, value in design.items():
                    if isinstance(value, (str, int, float, bool)):
                        lines.append(f"- {key.replace('_', ' ')}: `{value}`")
                lines.append("")
            retrieval = digest.get("retrieval")
            if isinstance(retrieval, Mapping):
                lines.extend(
                    [
                        "| Target task | Retrieved other task | Cosine similarity |",
                        "|---|---|---:|",
                    ]
                )
                for target in H1_TASKS:
                    selection = retrieval[target]
                    source = selection["source_task"]
                    similarity = selection["similarities"][source]
                    lines.append(f"| {target} | {source} | {float(similarity):.4f} |")
                lines.append("")

    missing = summary.get("controls_missing", [])
    if missing:
        lines.extend(
            [
                "## Resume status",
                "",
                "Completed controls are persisted individually. Re-run the same command to execute: "
                + ", ".join(str(value) for value in missing)
                + ".",
                "",
            ]
        )
    if summary.get("last_error"):
        lines.extend(["Last error:", "", f"`{summary['last_error']}`", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_control_outputs(
    summary: Mapping[str, Any],
    output_root: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_root) / f"seed_{CONTROL_SEED}"
    summary_path = write_json(directory / CONTROL_RESULTS_FILENAME, summary)
    report_path = directory / CONTROL_REPORT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    try:
        temporary.write_text(render_control_report(summary), encoding="utf-8", newline="\n")
        temporary.replace(report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary_path, report_path


def _selected_control_ids(controls: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(controls or CONTROL_IDS)
    if not selected:
        raise ValueError("at least one control must be selected")
    unknown = [control_id for control_id in selected if control_id not in CONTROL_BY_ID]
    if unknown:
        raise ValueError(f"unknown control ID(s): {', '.join(unknown)}")
    if len(set(selected)) != len(selected):
        raise ValueError("selected controls must not contain duplicates")
    return selected


def run_control_suite(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    output_root: str | Path = "results/controls",
    independent_root: str | Path = "results/independent_lora",
    *,
    controls: Sequence[str] | None = None,
    force: bool = False,
    prepared: PreparedData | None = None,
    executors: Mapping[str, ControlExecutor] | None = None,
    validate_core: bool = True,
) -> tuple[dict[str, Any], Path, Path]:
    """Run/resume selected controls and always persist aggregate progress.

    ``executors`` is an explicit test seam: offline tests can exercise
    orchestration and persistence with deterministic mock compute.  Any control
    absent from that mapping uses its real executor.
    """

    validate_control_configs(baseline_config, atom_config)
    selected = _selected_control_ids(controls)
    destination = Path(output_root)
    source = Path(independent_root)
    executor_map = dict(DEFAULT_EXECUTORS)
    if executors is not None:
        executor_map.update(executors)

    real_controls = [
        control_id
        for control_id in selected
        if executors is None or control_id not in executors
    ]
    if validate_core and real_controls:
        validate_independent_core_checkpoints(baseline_config, source)
    if prepared is None and real_controls:
        prepared = prepare_data(baseline_config, tasks=baseline_config.tasks)

    context = ControlRunContext(
        baseline_config=baseline_config,
        atom_config=atom_config,
        output_root=destination,
        independent_root=source,
        prepared=prepared,
        force=force,
    )
    records: dict[str, Mapping[str, Any]] = {}
    for control_id in CONTROL_IDS:
        path = control_result_path(destination, control_id)
        if path.is_file() and not force:
            existing = read_json(path)
            validate_control_record(existing, control_id)
            records[control_id] = existing

    for control_id in selected:
        if control_id in records and not force:
            print(f"Skipping completed control {control_id}.", flush=True)
            continue
        print(f"Running control {control_id} (seed {CONTROL_SEED})...", flush=True)
        try:
            record = executor_map[control_id](context)
            validate_control_record(record, control_id)
            path = control_result_path(destination, control_id)
            write_json(path, record)
            records[control_id] = record
        except Exception as error:
            summary = build_control_summary(
                records,
                destination,
                requested_controls=selected,
                last_error=f"{type(error).__name__}: {error}",
            )
            write_control_outputs(summary, destination)
            raise
        summary = build_control_summary(
            records,
            destination,
            requested_controls=selected,
        )
        write_control_outputs(summary, destination)

    final_summary = build_control_summary(
        records,
        destination,
        requested_controls=selected,
    )
    summary_path, report_path = write_control_outputs(final_summary, destination)
    return final_summary, summary_path, report_path


__all__ = [
    "CONTROL_IDS",
    "CONTROL_REPORT_FILENAME",
    "CONTROL_RESULTS_FILENAME",
    "CONTROL_SEED",
    "CONTROL_SPECS",
    "ControlRunContext",
    "ControlSpec",
    "average_effective_lora_state",
    "build_control_summary",
    "control_result_path",
    "cosine_effective_lora_similarity",
    "deterministic_label_permutation",
    "deterministically_shuffle_labels",
    "effective_lora_updates",
    "effective_lora_vector",
    "load_independent_compact_states",
    "nearest_other_task_adapters",
    "render_control_report",
    "run_average_independent_loras",
    "run_control_suite",
    "run_nearest_other_task_lora",
    "run_random_frozen_atoms",
    "run_shared_atoms_no_sparsity",
    "run_shared_atoms_shuffled_labels",
    "run_shared_multitask_lora",
    "validate_control_configs",
    "validate_control_record",
    "validate_independent_core_checkpoints",
    "write_control_outputs",
]
