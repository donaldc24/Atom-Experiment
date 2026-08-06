"""Frozen-dictionary transfer and task-count scaling follow-ups for H1.

The two experiments in this module deliberately use the locked seed-17 H1
budget.  Expensive sub-runs write their own metrics before the aggregate report
is assembled, so an interrupted invocation can continue without repeating
completed training.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from cgmoe_h1.config import H1_TASKS, ExperimentConfig
from cgmoe_h1.experiments import (
    PreparedData,
    build_atom_model,
    build_loaders,
    prepare_data,
    run_head_only,
    run_independent_lora,
    run_shared_atoms,
    save_compact_checkpoint,
)
from cgmoe_h1.metrics import compute_task_metrics
from cgmoe_h1.models.atoms import coefficient_l1_regularization, iter_atom_layers
from cgmoe_h1.models.injection import (
    extract_adapter_state_dict,
    load_adapter_state_dict,
)
from cgmoe_h1.training.trainer import create_adamw_optimizer, evaluate, train_single_task
from cgmoe_h1.utils.parameters import (
    active_adapter_operations,
    categorized_parameter_counts,
)
from cgmoe_h1.utils.reproducibility import set_seed
from cgmoe_h1.utils.runtime import RuntimeMonitor
from cgmoe_h1.utils.serialization import read_json, write_json


FOLLOWUP_SEED = 17
TRANSFER_SOURCE_TASKS = H1_TASKS[:4]
TRANSFER_TARGET_TASK = H1_TASKS[4]
TRANSFER_QUALITY_THRESHOLD = 0.95
SCHEMA_VERSION = 1


def _validate_followup_configs(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
) -> None:
    """Require the registered H1 budget and the single follow-up seed."""

    baseline_config.validate_h1_contract()
    atom_config.validate_h1_contract()
    if baseline_config.experiment_name != "independent_lora":
        raise ValueError("baseline config must select independent_lora")
    if atom_config.experiment_name != "shared_atoms":
        raise ValueError("atom config must select shared_atoms")
    if baseline_config.seed != FOLLOWUP_SEED or atom_config.seed != FOLLOWUP_SEED:
        raise ValueError(f"chunks 21-22 are locked to seed {FOLLOWUP_SEED}")
    if baseline_config.tasks != H1_TASKS or atom_config.tasks != H1_TASKS:
        raise ValueError(f"follow-up task order must be {H1_TASKS!r}")


def _primary_score(record: Mapping[str, Any], field: str = "best") -> float:
    try:
        value = float(record[field]["metrics"]["primary_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"record has no {field} primary score") from error
    if not math.isfinite(value):
        raise ValueError("primary score must be finite")
    return value


def _shared_scores(record: Mapping[str, Any], tasks: Sequence[str]) -> list[float]:
    scores: list[float] = []
    for task in tasks:
        try:
            value = float(record["tasks"][task]["top_k"]["metrics"]["primary_score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"shared record has no top-k primary score for {task}") from error
        if not math.isfinite(value):
            raise ValueError(f"shared primary score for {task} must be finite")
        scores.append(value)
    return scores


def _transfer_primary_score(record: Mapping[str, Any]) -> float:
    """Read the registered top-k transfer score, accepting old/mock records."""

    if "top_k" in record:
        return _primary_score(record, "top_k")
    return _primary_score(record)


def _dictionary_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    found = False
    for name, module in model.named_modules():
        if not hasattr(module, "atom_u") or not hasattr(module, "atom_v"):
            continue
        found = True
        for tensor_name in ("atom_u", "atom_v"):
            tensor = getattr(module, tensor_name).detach().cpu().contiguous()
            digest.update(f"{name}.{tensor_name}:{tuple(tensor.shape)}".encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    if not found:
        raise ValueError("model contains no atom dictionary")
    return digest.hexdigest()


def copy_frozen_atom_dictionary(
    model: nn.Module,
    checkpoint_directory: str | Path,
) -> dict[str, Any]:
    """Copy only ``atom_u``/``atom_v`` tensors and freeze them.

    The source checkpoint's coefficient rows, task metadata, and heads are
    intentionally ignored.  This is what makes the target task genuinely new.
    """

    checkpoint_path = Path(checkpoint_directory) / "atoms.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError(f"invalid atom checkpoint: {checkpoint_path}")

    parameters = dict(model.named_parameters())
    expected = {
        name
        for name in parameters
        if name.endswith((".atom_u", ".atom_v"))
    }
    supplied = {
        name
        for name, value in state.items()
        if name.endswith((".atom_u", ".atom_v")) and isinstance(value, Tensor)
    }
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    if missing or unexpected:
        raise RuntimeError(
            "atom dictionary checkpoint does not match target architecture: "
            f"missing={missing}, unexpected={unexpected}"
        )

    copied_parameters = 0
    with torch.no_grad():
        for name in sorted(expected):
            destination = parameters[name]
            source = state[name]
            if destination.shape != source.shape:
                raise RuntimeError(
                    f"shape mismatch for {name}: {tuple(source.shape)} != "
                    f"{tuple(destination.shape)}"
                )
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
            destination.requires_grad_(False)
            copied_parameters += destination.numel()

    # Check every atom layer rather than relying only on key suffixes.
    for layer in iter_atom_layers(model):
        layer.atom_u.requires_grad_(False)
        layer.atom_v.requires_grad_(False)
    return {
        "checkpoint": str(checkpoint_path),
        "tensor_count": len(expected),
        "parameter_count": copied_parameters,
        "sha256": _dictionary_digest(model),
    }


def assert_transfer_trainable_contract(model: nn.Module, task: str) -> list[str]:
    """Assert that only the new coefficient tensors and target head can train."""

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [
        name
        for name in trainable
        if not name.endswith(".coefficients") and not name.startswith(f"heads.{task}.")
    ]
    if invalid:
        raise AssertionError(f"unexpected trainable transfer parameters: {invalid}")
    if not any(name.endswith(".coefficients") for name in trainable):
        raise AssertionError("transfer model has no trainable coefficients")
    if not any(name.startswith(f"heads.{task}.") for name in trainable):
        raise AssertionError(f"transfer model has no trainable {task} head")
    for layer in iter_atom_layers(model):
        if layer.atom_u.requires_grad or layer.atom_v.requires_grad:
            raise AssertionError("transfer atom dictionary must be frozen")
    return trainable


def _optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.AdamW:
    return create_adamw_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        epsilon=config.adam_epsilon,
    )


def run_frozen_atom_target(
    config: ExperimentConfig,
    output_directory: str | Path,
    *,
    prepared: PreparedData | None = None,
    dictionary_checkpoint: str | Path | None,
    force: bool = False,
) -> dict[str, Any]:
    """Train QQP coefficients/head over learned or seeded-random frozen atoms."""

    task = TRANSFER_TARGET_TASK
    destination = Path(output_directory)
    metrics_path = destination / "metrics.json"
    if metrics_path.is_file() and not force:
        return read_json(metrics_path)

    set_seed(config.seed)
    prepared = prepared or prepare_data(config, tasks=(task,))
    train_loaders, validation_loaders = build_loaders(prepared, config, tasks=(task,))
    model, target_names = build_atom_model(
        config,
        (task,),
        atom_count=config.atom_count,
        freeze_atoms=True,
    )
    if dictionary_checkpoint is None:
        dictionary = {
            "checkpoint": None,
            "tensor_count": sum(2 for _ in iter_atom_layers(model)),
            "parameter_count": sum(
                layer.atom_u.numel() + layer.atom_v.numel()
                for layer in iter_atom_layers(model)
            ),
            "sha256": _dictionary_digest(model),
        }
        system = "random_frozen_atoms"
    else:
        dictionary = copy_frozen_atom_dictionary(model, dictionary_checkpoint)
        system = "transferred_frozen_atoms"
    trainable_names = assert_transfer_trainable_contract(model, task)
    optimizer = _optimizer(model, config)

    def epoch_callback(record: Any) -> None:
        print(
            f"  {system} epoch {record.epoch}: "
            f"validation={record.selection_score:.5f}, seconds={record.elapsed_seconds:.1f}",
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
            metric_fn=partial(compute_task_metrics, task),
            primary_metric="primary_score",
            regularization_fn=lambda value, task_id: coefficient_l1_regularization(
                value, task_id, config.sparsity_lambda
            ),
            state_capture_fn=lambda value: extract_adapter_state_dict(
                value, include_heads=True
            ),
            state_restore_fn=lambda value, state: load_adapter_state_dict(
                value, state, include_heads=True
            ),
            epoch_callback=epoch_callback,
        )
    runtime = monitor.result()
    model.set_active_task(task)
    model.set_atom_top_k(config.active_atoms_for_primary_evaluation)
    try:
        top_k_evaluation = evaluate(
            model,
            validation_loaders[task],
            config.device,
            task_id=None,
            metric_fn=partial(compute_task_metrics, task),
            scalar_metric_name="primary_score",
        )
    finally:
        model.clear_atom_top_k()
    counts = categorized_parameter_counts(model)
    new_parameters = counts["coefficient_parameters"] + counts["head_parameters"]
    if new_parameters != counts["model_trainable_parameters"]:
        raise AssertionError(
            "transfer trainable count does not equal coefficients plus head: "
            f"{counts['model_trainable_parameters']} != {new_parameters}"
        )
    checkpoint = save_compact_checkpoint(
        model,
        destination,
        metadata={"system": system, "seed": config.seed, "task": task},
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "system": system,
        "run_kind": "followup",
        "seed": config.seed,
        "task": task,
        "source_tasks": list(TRANSFER_SOURCE_TASKS) if dictionary_checkpoint else [],
        "dictionary": dictionary,
        "dictionary_frozen": True,
        "trainable_parameter_names": trainable_names,
        "new_task_parameters": new_parameters,
        "reused_dictionary_parameters": counts["atom_parameters"],
        "parameter_counts": counts,
        "primary_evaluation": "top_k",
        "active_atoms_for_primary_evaluation": config.active_atoms_for_primary_evaluation,
        "best": result.best_validation.to_dict(include_outputs=True),
        "all_atoms": result.best_validation.to_dict(include_outputs=True),
        "top_k": top_k_evaluation.to_dict(include_outputs=True),
        "final": result.final_validation.to_dict(include_outputs=True),
        "history": result.to_dict(),
        "target_modules": target_names,
        "active_adapter_operations_per_token": {
            "all_atoms": active_adapter_operations(model, active_atoms=config.atom_count),
            "top_k": active_adapter_operations(
                model, active_atoms=config.active_atoms_for_primary_evaluation
            ),
        },
        "checkpoint": checkpoint,
        "runtime": {
            "elapsed_seconds": runtime.elapsed_seconds,
            "peak_rss_bytes": runtime.peak_rss_bytes,
        },
        "resolved_config": config.to_dict(),
        "dataset_provenance": prepared.provenance[task],
    }
    write_json(metrics_path, record)
    return record


def evaluate_strong_transfer(
    transferred_score: float,
    fresh_lora_score: float,
    transferred_new_parameters: int,
    fresh_lora_new_parameters: int,
) -> dict[str, Any]:
    """Apply chunk 21's inclusive quality and strict parameter criteria."""

    values = (float(transferred_score), float(fresh_lora_score))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("transfer scores must be finite")
    if fresh_lora_score <= 0:
        raise ValueError("fresh LoRA score must be positive")
    if transferred_new_parameters < 0 or fresh_lora_new_parameters <= 0:
        raise ValueError("parameter counts must be non-negative and baseline-positive")
    quality_ratio = transferred_score / fresh_lora_score
    quality_passed = transferred_score >= TRANSFER_QUALITY_THRESHOLD * fresh_lora_score
    parameter_passed = transferred_new_parameters < fresh_lora_new_parameters
    return {
        "quality_threshold": TRANSFER_QUALITY_THRESHOLD,
        "quality_ratio": quality_ratio,
        "quality_passed": quality_passed,
        "fewer_new_parameters": parameter_passed,
        "new_parameter_ratio": transferred_new_parameters / fresh_lora_new_parameters,
        "strong_transfer": quality_passed and parameter_passed,
    }


def _fresh_lora_new_parameters(record: Mapping[str, Any]) -> int:
    try:
        counts = record["parameter_counts"]
        return int(counts["lora_adapter_parameters"]) + int(counts["head_parameters"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fresh LoRA record lacks per-task parameter counts") from error


def _head_new_parameters(record: Mapping[str, Any]) -> int:
    try:
        return int(record["parameter_counts"]["head_parameters"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("head-only record lacks parameter counts") from error


def _load_or_run_core_independent(
    config: ExperimentConfig,
    core_results_root: Path,
) -> dict[str, Any]:
    path = core_results_root / "independent_lora" / f"seed_{config.seed}" / "metrics_by_task.json"
    if path.is_file():
        return read_json(path)
    print("Core independent seed-17 result missing; generating it first.", flush=True)
    return run_independent_lora(
        config,
        core_results_root / "independent_lora",
        tasks=H1_TASKS,
        run_kind="confirmatory",
    )


def _load_or_run_core_shared(
    config: ExperimentConfig,
    core_results_root: Path,
) -> dict[str, Any]:
    path = core_results_root / "shared_atoms" / f"seed_{config.seed}" / "metrics_by_task.json"
    if path.is_file():
        return read_json(path)
    print("Core shared seed-17 result missing; generating it first.", flush=True)
    return run_shared_atoms(
        config,
        core_results_root / "shared_atoms",
        tasks=H1_TASKS,
        run_kind="confirmatory",
    )


def _load_or_run_head_only(
    config: ExperimentConfig,
    output_directory: Path,
    prepared: PreparedData,
    *,
    force: bool,
    reuse_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    path = output_directory / "metrics.json"
    if path.is_file() and not force:
        return read_json(path)
    if not force:
        for candidate in reuse_paths:
            if candidate.is_file():
                print(f"Reusing completed locked QQP head-only run at {candidate}.", flush=True)
                return read_json(candidate)
    return run_head_only(
        config,
        TRANSFER_TARGET_TASK,
        output_directory,
        prepared=prepared,
        run_kind="followup",
    )


def render_transfer_markdown(summary: Mapping[str, Any]) -> str:
    rows = summary["systems"]
    fresh_score = float(rows["fresh_lora"]["score"])
    lines = [
        "# H1 Chunk 21: Frozen-Atom Transfer",
        "",
        f"Seed: {summary['seed']}; source tasks: {', '.join(summary['source_tasks'])}; "
        f"target task: {summary['target_task']}.",
        "",
        "| System | QQP primary score | Fresh-LoRA quality | New parameters |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("fresh_lora", "Fresh rank-4 LoRA"),
        ("transferred_frozen_atoms", "Frozen learned atoms + new coefficients/head"),
        ("head_only", "Head only"),
        ("random_frozen_atoms", "Frozen random atoms + new coefficients/head"),
    ):
        row = rows[key]
        ratio = float(row["score"]) / fresh_score
        lines.append(
            f"| {label} | {float(row['score']):.4f} | {ratio:.3f} | "
            f"{int(row['new_parameters']):,} |"
        )
    verdict = summary["strong_result"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Strong frozen-atom transfer: **{'PASS' if verdict['strong_transfer'] else 'FAIL'}**.",
            "",
            f"The transferred model retained {float(verdict['quality_ratio']):.1%} of "
            f"fresh-LoRA quality and used {float(verdict['new_parameter_ratio']):.1%} "
            "as many new parameters.",
            "",
        ]
    )
    return "\n".join(lines)


def run_frozen_atom_transfer(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    output_directory: str | Path,
    *,
    core_results_root: str | Path = "results",
    shared_prefix_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run and report roadmap chunk 21."""

    _validate_followup_configs(baseline_config, atom_config)
    destination = Path(output_directory)
    summary_path = destination / "frozen_atom_transfer.json"
    report_path = destination / "frozen_atom_transfer.md"
    if summary_path.is_file() and report_path.is_file() and not force:
        return read_json(summary_path)

    prefix_root = (
        Path(shared_prefix_root)
        if shared_prefix_root is not None
        else destination.parent / "shared_prefixes"
    )
    source_root = prefix_root / "prefix_4"
    source_record = run_shared_atoms(
        atom_config,
        source_root,
        tasks=TRANSFER_SOURCE_TASKS,
        run_kind="followup",
        force=force,
    )
    source_checkpoint = source_root / f"seed_{atom_config.seed}"
    if not (source_checkpoint / "atoms.pt").is_file():
        # Mock runners and relocated records may publish the exact component path.
        try:
            source_checkpoint = Path(source_record["checkpoint"]["paths"]["atoms"]).parent
        except (KeyError, TypeError) as error:
            raise FileNotFoundError(source_checkpoint / "atoms.pt") from error

    prepared = prepare_data(atom_config, tasks=(TRANSFER_TARGET_TASK,))
    learned = run_frozen_atom_target(
        atom_config,
        destination / "transferred_frozen_atoms",
        prepared=prepared,
        dictionary_checkpoint=source_checkpoint,
        force=force,
    )
    random_control = run_frozen_atom_target(
        atom_config,
        destination / "random_frozen_atoms",
        prepared=prepared,
        dictionary_checkpoint=None,
        force=force,
    )
    reusable_head_path = (
        destination.parent
        / "transfer"
        / "head_only"
        / f"seed_{FOLLOWUP_SEED}"
        / TRANSFER_TARGET_TASK
        / "metrics.json"
    )
    head_only = _load_or_run_head_only(
        baseline_config,
        destination / "head_only",
        prepared,
        force=force,
        reuse_paths=(reusable_head_path,),
    )
    local_head_path = destination / "head_only" / "metrics.json"
    head_result_path = (
        reusable_head_path
        if reusable_head_path.is_file() and not local_head_path.is_file()
        else local_head_path
    )
    independent = _load_or_run_core_independent(
        baseline_config, Path(core_results_root)
    )
    try:
        fresh_lora = independent["tasks"][TRANSFER_TARGET_TASK]
    except (KeyError, TypeError) as error:
        raise ValueError("core independent result has no QQP task") from error

    learned_score = _transfer_primary_score(learned)
    fresh_score = _primary_score(fresh_lora)
    learned_new = int(learned["new_task_parameters"])
    fresh_new = _fresh_lora_new_parameters(fresh_lora)
    verdict = evaluate_strong_transfer(learned_score, fresh_score, learned_new, fresh_new)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "frozen_atom_transfer",
        "seed": FOLLOWUP_SEED,
        "source_tasks": list(TRANSFER_SOURCE_TASKS),
        "target_task": TRANSFER_TARGET_TASK,
        "budget": {
            "train_examples": atom_config.train_examples_per_task,
            "validation_examples": atom_config.validation_examples_per_task,
            "epochs": atom_config.epochs,
        },
        "systems": {
            "fresh_lora": {
                "score": fresh_score,
                "new_parameters": fresh_new,
                "result": str(
                    Path(core_results_root)
                    / "independent_lora"
                    / f"seed_{FOLLOWUP_SEED}"
                    / TRANSFER_TARGET_TASK
                    / "metrics.json"
                ),
            },
            "transferred_frozen_atoms": {
                "score": learned_score,
                "new_parameters": learned_new,
                "reused_dictionary_parameters": int(learned["reused_dictionary_parameters"]),
                "result": str(destination / "transferred_frozen_atoms" / "metrics.json"),
            },
            "head_only": {
                "score": _primary_score(head_only),
                "new_parameters": _head_new_parameters(head_only),
                "result": str(head_result_path),
            },
            "random_frozen_atoms": {
                "score": _transfer_primary_score(random_control),
                "new_parameters": int(random_control["new_task_parameters"]),
                "result": str(destination / "random_frozen_atoms" / "metrics.json"),
            },
        },
        "strong_result": verdict,
        "source_four_task_result": str(
            source_root / f"seed_{FOLLOWUP_SEED}" / "metrics_by_task.json"
        ),
    }
    write_json(summary_path, summary)
    destination.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_transfer_markdown(summary), encoding="utf-8", newline="\n")
    return summary


def _independent_prefix_statistics(
    independent: Mapping[str, Any],
    tasks: Sequence[str],
) -> tuple[list[float], int]:
    scores: list[float] = []
    storage = 0
    for task in tasks:
        try:
            record = independent["tasks"][task]
            storage += int(record["parameter_counts"]["persistent_adaptation_parameters"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"core independent result is incomplete for {task}") from error
        scores.append(_primary_score(record))
    return scores, storage


def build_scaling_point(
    task_count: int,
    tasks: Sequence[str],
    independent: Mapping[str, Any],
    shared: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one JSON-friendly task-prefix comparison row."""

    selected = tuple(tasks)
    if task_count != len(selected) or task_count < 1:
        raise ValueError("task_count must equal a non-empty task prefix length")
    independent_scores, independent_storage = _independent_prefix_statistics(
        independent, selected
    )
    shared_scores = _shared_scores(shared, selected)
    try:
        shared_storage = int(shared["parameter_counts"]["total_persistent_task_parameters"])
        active_atoms = int(shared["active_atoms_for_evaluation"])
        operations = int(shared["active_adapter_operations_per_token"]["top_k"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("shared result lacks storage or active-capacity accounting") from error
    if independent_storage <= 0 or shared_storage <= 0:
        raise ValueError("persistent storage counts must be positive")
    return {
        "task_count": task_count,
        "tasks": list(selected),
        "mean_quality": sum(shared_scores) / task_count,
        "worst_task_score": min(shared_scores),
        "independent_mean_quality": sum(independent_scores) / task_count,
        "independent_worst_task_score": min(independent_scores),
        "shared_mean_quality": sum(shared_scores) / task_count,
        "shared_worst_task_score": min(shared_scores),
        "quality_retention": (sum(shared_scores) / sum(independent_scores))
        if sum(independent_scores) > 0
        else None,
        "independent_storage_parameters": independent_storage,
        "shared_storage_parameters": shared_storage,
        "relative_storage": shared_storage / independent_storage,
        "active_capacity": active_atoms,
        "active_capacity_atoms": active_atoms,
        "active_capacity_operations_per_token": operations,
    }


def render_scaling_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# H1 Chunk 22: Task-Count Scaling Curve",
        "",
        f"Locked seed {summary['seed']}; task prefixes follow the registered order.",
        "",
        "| Tasks | Shared mean | Shared worst | Independent mean | Independent storage | "
        "Shared storage | Relative storage | Active atoms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in summary["points"]:
        lines.append(
            f"| {int(point['task_count'])} | {float(point['shared_mean_quality']):.4f} | "
            f"{float(point['shared_worst_task_score']):.4f} | "
            f"{float(point['independent_mean_quality']):.4f} | "
            f"{int(point['independent_storage_parameters']):,} | "
            f"{int(point['shared_storage_parameters']):,} | "
            f"{float(point['relative_storage']):.3f} | "
            f"{int(point['active_capacity_atoms'])} |"
        )
    lines.extend(
        [
            "",
            "Independent storage is computed by summing the exact per-task core LoRA "
            "counts. Shared storage is the exact persistent dictionary, coefficient, "
            "and head count for each prefix model.",
            "",
        ]
    )
    return "\n".join(lines)


def run_scaling_curve(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    output_directory: str | Path,
    *,
    core_results_root: str | Path = "results",
    shared_prefix_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run and report roadmap chunk 22 for registered task prefixes 1..5."""

    _validate_followup_configs(baseline_config, atom_config)
    destination = Path(output_directory)
    summary_path = destination / "scaling_curve.json"
    report_path = destination / "scaling_curve.md"
    if summary_path.is_file() and report_path.is_file() and not force:
        return read_json(summary_path)

    core_root = Path(core_results_root)
    prefix_root = (
        Path(shared_prefix_root)
        if shared_prefix_root is not None
        else destination.parent / "shared_prefixes"
    )
    independent = _load_or_run_core_independent(baseline_config, core_root)
    core_shared = _load_or_run_core_shared(atom_config, core_root)
    points: list[dict[str, Any]] = []
    result_paths: dict[str, str] = {}
    for task_count in range(1, len(H1_TASKS) + 1):
        tasks = H1_TASKS[:task_count]
        if task_count == len(H1_TASKS):
            shared = core_shared
            result_path = (
                core_root
                / "shared_atoms"
                / f"seed_{FOLLOWUP_SEED}"
                / "metrics_by_task.json"
            )
        else:
            run_root = prefix_root / f"prefix_{task_count}"
            shared = run_shared_atoms(
                atom_config,
                run_root,
                tasks=tasks,
                run_kind="followup",
                force=force,
            )
            result_path = run_root / f"seed_{FOLLOWUP_SEED}" / "metrics_by_task.json"
        points.append(build_scaling_point(task_count, tasks, independent, shared))
        result_paths[str(task_count)] = str(result_path)

    independent_storage = [point["independent_storage_parameters"] for point in points]
    shared_storage = [point["shared_storage_parameters"] for point in points]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "task_count_scaling_curve",
        "seed": FOLLOWUP_SEED,
        "task_order": list(H1_TASKS),
        "budget": {
            "train_examples_per_task": atom_config.train_examples_per_task,
            "validation_examples_per_task": atom_config.validation_examples_per_task,
            "epochs": atom_config.epochs,
        },
        "points": points,
        "result_paths": result_paths,
        "diagnostics": {
            "independent_storage_strictly_increases": all(
                right > left
                for left, right in zip(independent_storage, independent_storage[1:])
            ),
            "shared_storage_strictly_increases": all(
                right > left for left, right in zip(shared_storage, shared_storage[1:])
            ),
            "final_relative_storage": points[-1]["relative_storage"],
            "shared_quality_range": max(point["mean_quality"] for point in points)
            - min(point["mean_quality"] for point in points),
        },
    }
    write_json(summary_path, summary)
    destination.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_scaling_markdown(summary), encoding="utf-8", newline="\n")
    return summary


def run_transfer_and_scaling(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    results_root: str | Path = "results",
    *,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run chunks 21 and 22 while sharing the four-task prefix checkpoint."""

    root = Path(results_root)
    followups = root / "followups"
    prefix_root = followups / "shared_prefixes"
    transfer = run_frozen_atom_transfer(
        baseline_config,
        atom_config,
        followups / "frozen_atom_transfer",
        core_results_root=root,
        shared_prefix_root=prefix_root,
        force=force,
    )
    scaling = run_scaling_curve(
        baseline_config,
        atom_config,
        followups / "scaling_curve",
        core_results_root=root,
        shared_prefix_root=prefix_root,
        force=force,
    )
    return transfer, scaling


__all__ = [
    "FOLLOWUP_SEED",
    "TRANSFER_QUALITY_THRESHOLD",
    "TRANSFER_SOURCE_TASKS",
    "TRANSFER_TARGET_TASK",
    "assert_transfer_trainable_contract",
    "build_scaling_point",
    "copy_frozen_atom_dictionary",
    "evaluate_strong_transfer",
    "render_scaling_markdown",
    "render_transfer_markdown",
    "run_frozen_atom_target",
    "run_frozen_atom_transfer",
    "run_scaling_curve",
    "run_transfer_and_scaling",
]
