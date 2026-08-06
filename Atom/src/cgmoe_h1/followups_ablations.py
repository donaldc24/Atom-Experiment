"""Seed-17 atom-count, LoRA-rank, and active-capacity H1 ablations.

The roadmap deliberately leaves these tests as stronger, exploratory evidence.
This module keeps the original data sample and training budget locked, varies
only the requested capacity, and records enough provenance to distinguish a
new training run from a reused core checkpoint.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cgmoe_h1.config import ExperimentConfig
from cgmoe_h1.experiments import (
    evaluate_atom_checkpoint,
    prepare_data,
    run_independent_lora,
    run_shared_atoms,
)
from cgmoe_h1.utils.serialization import read_json, write_json

SEED = 17
ATOM_COUNTS = (2, 4, 6, 8, 12, 16)
LORA_RANKS = (1, 2, 4, 8)
ACTIVE_ATOM_COUNTS = (1, 2, 4, 8)
SATURATION_TOLERANCE = 0.005
DEAD_ATOM_THRESHOLD = 1e-6
SUMMARY_FILENAME = "h1_followup_ablations.json"
REPORT_FILENAME = "h1_followup_ablations.md"


class AblationError(ValueError):
    """A configuration or saved record cannot support the ablation report."""


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AblationError(f"{path} must be an object")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AblationError(f"{path} must be an array")
    return value


def _finite_number(value: object, path: str, *, unit_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AblationError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise AblationError(f"{path} must be finite")
    if unit_interval and not 0.0 <= result <= 1.0:
        raise AblationError(f"{path} must be in [0, 1], got {result}")
    return result


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AblationError(f"{path} must be an integer")
    if value < minimum:
        raise AblationError(f"{path} must be >= {minimum}, got {value}")
    return value


def _validate_locked_config(config: ExperimentConfig, expected_system: str) -> None:
    if config.experiment_name != expected_system:
        raise AblationError(
            f"expected {expected_system!r} config, got {config.experiment_name!r}"
        )
    if config.seed != SEED:
        raise AblationError(f"follow-up ablations are locked to seed {SEED}, got {config.seed}")
    try:
        config.validate_h1_contract()
    except (TypeError, ValueError) as error:
        raise AblationError(str(error)) from error


def _validate_core_record(
    record: Mapping[str, Any],
    *,
    config: ExperimentConfig,
    system: str,
    capacity_name: str,
    capacity: int,
) -> None:
    if record.get("system") != system:
        raise AblationError(f"core record system must be {system!r}")
    if record.get("seed") != SEED:
        raise AblationError(f"core {system} record must use seed {SEED}")
    if record.get(capacity_name) != capacity:
        raise AblationError(
            f"core {system} record {capacity_name} must be {capacity}, "
            f"got {record.get(capacity_name)!r}"
        )
    task_records = _mapping(record.get("tasks"), f"core {system}.tasks")
    if set(task_records) != set(config.tasks):
        raise AblationError(f"core {system} record must contain exactly {list(config.tasks)}")

    stored_values = _mapping(
        record.get("resolved_config"), f"core {system}.resolved_config"
    )
    try:
        stored = ExperimentConfig.from_mapping(stored_values)
        stored.validate_h1_contract()
    except (TypeError, ValueError) as error:
        raise AblationError(f"invalid core {system} resolved config: {error}") from error
    if stored.to_dict() != config.to_dict():
        raise AblationError(f"core {system} config does not match the requested locked config")


def _normalize_variant_config(
    record: Mapping[str, Any],
    *,
    expected: ExperimentConfig,
    allowed_legacy_fields: set[str],
    path: str,
) -> dict[str, Any]:
    """Upgrade cached variant metadata without concealing unrelated config drift.

    Early ablation records received capacity through a runner argument, so their
    nested ``resolved_config`` retained the core capacity.  Accept that specific
    legacy drift, reject every other mismatch, and persist the actual variant.
    """

    expected_values = expected.to_dict()

    def normalize_one(value: object, item_path: str) -> dict[str, Any]:
        stored_values = _mapping(value, item_path)
        try:
            stored = ExperimentConfig.from_mapping(stored_values).to_dict()
        except (TypeError, ValueError) as error:
            raise AblationError(f"invalid {item_path}: {error}") from error
        mismatches = [
            name
            for name, expected_value in expected_values.items()
            if name not in allowed_legacy_fields and stored[name] != expected_value
        ]
        if mismatches:
            raise AblationError(
                f"{item_path} differs outside the capacity override: "
                f"{', '.join(sorted(mismatches))}"
            )
        return expected_values

    normalized = dict(record)
    normalized["resolved_config"] = normalize_one(
        record.get("resolved_config"), f"{path}.resolved_config"
    )

    # Independent summaries embed the complete per-task records. Normalize
    # their provenance too when present; shared task entries are evaluations.
    tasks_value = record.get("tasks")
    if isinstance(tasks_value, Mapping):
        normalized_tasks: dict[str, Any] = {}
        for task, task_value in tasks_value.items():
            if isinstance(task_value, Mapping) and "resolved_config" in task_value:
                normalized_task = dict(task_value)
                normalized_task["resolved_config"] = normalize_one(
                    task_value["resolved_config"],
                    f"{path}.tasks.{task}.resolved_config",
                )
                normalized_tasks[str(task)] = normalized_task
            else:
                normalized_tasks[str(task)] = task_value
        normalized["tasks"] = normalized_tasks
    return normalized


def _task_scores(
    record: Mapping[str, Any],
    tasks: Sequence[str],
    *,
    variant: str | None,
    path: str,
) -> dict[str, float]:
    records = _mapping(record.get("tasks"), f"{path}.tasks")
    if set(records) != set(tasks):
        raise AblationError(f"{path}.tasks must contain exactly {list(tasks)}")
    scores: dict[str, float] = {}
    for task in tasks:
        task_record = _mapping(records[task], f"{path}.tasks.{task}")
        evaluation = (
            _mapping(task_record.get(variant), f"{path}.tasks.{task}.{variant}")
            if variant is not None
            else task_record
        )
        metrics = _mapping(evaluation.get("metrics"), f"{path}.tasks.{task}.metrics")
        scores[task] = _finite_number(
            metrics.get("primary_score"),
            f"{path}.tasks.{task}.metrics.primary_score",
            unit_interval=True,
        )
    return scores


def _persistent_parameters(record: Mapping[str, Any], path: str) -> int:
    counts = _mapping(record.get("parameter_counts"), f"{path}.parameter_counts")
    return _integer(
        counts.get("total_persistent_task_parameters"),
        f"{path}.parameter_counts.total_persistent_task_parameters",
        minimum=1,
    )


def _independent_checkpoint_bytes(record: Mapping[str, Any], path: str) -> int:
    return _integer(record.get("checkpoint_bytes"), f"{path}.checkpoint_bytes", minimum=1)


def _shared_checkpoint_bytes(record: Mapping[str, Any], path: str) -> int:
    checkpoint = _mapping(record.get("checkpoint"), f"{path}.checkpoint")
    return _integer(checkpoint.get("total_bytes"), f"{path}.checkpoint.total_bytes", minimum=1)


def _independent_operations(
    record: Mapping[str, Any], tasks: Sequence[str], path: str
) -> int:
    task_records = _mapping(record.get("tasks"), f"{path}.tasks")
    values = {
        _integer(
            _mapping(task_records[task], f"{path}.tasks.{task}").get(
                "active_adapter_operations_per_token"
            ),
            f"{path}.tasks.{task}.active_adapter_operations_per_token",
            minimum=1,
        )
        for task in tasks
    }
    if len(values) != 1:
        raise AblationError(f"{path} tasks have inconsistent active operation counts")
    return next(iter(values))


def _shared_operations(record: Mapping[str, Any], path: str, variant: str) -> int:
    operations = _mapping(
        record.get("active_adapter_operations_per_token"),
        f"{path}.active_adapter_operations_per_token",
    )
    return _integer(operations.get(variant), f"{path}.operations.{variant}", minimum=1)


def _score_summary(scores: Mapping[str, float]) -> tuple[float, float]:
    values = list(scores.values())
    return statistics.fmean(values), min(values)


def _validated_indices(
    value: object, path: str, *, atom_count: int
) -> list[int]:
    result = [_integer(item, f"{path}[]") for item in _sequence(value, path)]
    if len(set(result)) != len(result):
        raise AblationError(f"{path} must not contain duplicate atom indices")
    if any(index >= atom_count for index in result):
        raise AblationError(f"{path} contains an index outside [0, {atom_count - 1}]")
    return result


def _coefficient_counts(
    record: Mapping[str, Any], *, atom_count: int, task_count: int, path: str
) -> dict[str, Any]:
    analysis = _mapping(record.get("coefficient_analysis"), f"{path}.coefficient_analysis")
    reused = _validated_indices(
        analysis.get("reused_atoms"),
        f"{path}.coefficient_analysis.reused_atoms",
        atom_count=atom_count,
    )
    private = _validated_indices(
        analysis.get("task_exclusive_atoms"),
        f"{path}.coefficient_analysis.task_exclusive_atoms",
        atom_count=atom_count,
    )
    dead = _validated_indices(
        analysis.get("dead_atoms"),
        f"{path}.coefficient_analysis.dead_atoms",
        atom_count=atom_count,
    )
    utilization = [
        _integer(item, f"{path}.coefficient_analysis.atom_utilization_count[]")
        for item in _sequence(
            analysis.get("atom_utilization_count"),
            f"{path}.coefficient_analysis.atom_utilization_count",
        )
    ]
    if len(utilization) != atom_count:
        raise AblationError(
            f"{path}.coefficient_analysis.atom_utilization_count must have "
            f"{atom_count} entries"
        )
    if any(count > task_count for count in utilization):
        raise AblationError(f"{path} atom utilization cannot exceed the task count")
    return {
        "reused_atoms": reused,
        "reused_atom_count": len(reused),
        "reuse_fraction": len(reused) / atom_count,
        "task_private_atoms": private,
        "task_private_atom_count": len(private),
        "task_private_fraction": len(private) / atom_count,
        "dead_atoms": dead,
        "dead_atom_count": len(dead),
        "dead_atom_fraction": len(dead) / atom_count,
        "top_k_unused_atom_count": sum(count == 0 for count in utilization),
        "atom_utilization_count": utilization,
    }


def _validate_shared_variant(
    record: Mapping[str, Any], *, atom_count: int, top_k: int, tasks: Sequence[str]
) -> None:
    if record.get("system") != "shared_atoms" or record.get("seed") != SEED:
        raise AblationError(f"atom-count {atom_count} returned the wrong system or seed")
    if record.get("atom_count") != atom_count:
        raise AblationError(
            f"atom-count variant expected {atom_count}, "
            f"got {record.get('atom_count')!r}"
        )
    records = _mapping(record.get("tasks"), f"atom-count {atom_count}.tasks")
    if set(records) != set(tasks):
        raise AblationError(f"atom-count {atom_count} must contain all five locked tasks")
    for task in tasks:
        task_record = _mapping(records[task], f"atom-count {atom_count}.tasks.{task}")
        if task_record.get("top_k_value") != top_k:
            raise AblationError(
                f"atom-count {atom_count} task {task} expected top-k {top_k}"
            )


def _validate_rank_variant(
    record: Mapping[str, Any], *, rank: int, tasks: Sequence[str]
) -> None:
    if record.get("system") != "independent_lora" or record.get("seed") != SEED:
        raise AblationError(f"LoRA rank {rank} returned the wrong system or seed")
    if record.get("rank") != rank:
        raise AblationError(f"LoRA rank variant expected {rank}, got {record.get('rank')!r}")
    records = _mapping(record.get("tasks"), f"LoRA rank {rank}.tasks")
    if set(records) != set(tasks):
        raise AblationError(f"LoRA rank {rank} must contain all five locked tasks")


def _validate_active_evaluation(
    record: Mapping[str, Any], *, top_k: int, tasks: Sequence[str]
) -> None:
    if record.get("system") != "shared_atoms" or record.get("seed") != SEED:
        raise AblationError(f"active top-k {top_k} returned the wrong system or seed")
    if record.get("atom_count") != 8 or record.get("top_k") != top_k:
        raise AblationError(f"active top-k {top_k} did not evaluate the core 8-atom model")
    records = _mapping(record.get("tasks"), f"active top-k {top_k}.tasks")
    if set(records) != set(tasks):
        raise AblationError(f"active top-k {top_k} must contain all five locked tasks")


def _curve_pattern(values: Sequence[float]) -> str:
    changes = [right - left for left, right in zip(values, values[1:], strict=False)]
    positive = [change > 1e-12 for change in changes]
    negative = [change < -1e-12 for change in changes]
    if not any(positive) and not any(negative):
        return "constant"
    if any(positive) and any(negative):
        return "non_monotonic"
    if all(positive):
        return "strictly_increasing"
    if all(negative):
        return "strictly_decreasing"
    if any(positive):
        return "nondecreasing"
    return "nonincreasing"


def _atom_question_answers(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best_quality = max(float(row["mean_quality"]) for row in rows)
    near_best_counts = [
        int(row["atom_count"])
        for row in rows
        if float(row["mean_quality"]) >= best_quality - SATURATION_TOLERANCE
    ]
    operational_count = min(near_best_counts)
    operational_index = next(
        index
        for index, row in enumerate(rows)
        if int(row["atom_count"]) == operational_count
    )
    persistent_near_best = all(
        float(row["mean_quality"]) >= best_quality - SATURATION_TOLERANCE
        for row in rows[operational_index:]
    )
    conventional_plateau = operational_index > 0 and persistent_near_best
    first, last = rows[0], rows[-1]
    core = next(row for row in rows if row["atom_count"] == 8)
    larger = [row for row in rows if int(row["atom_count"]) > 8]
    reused_counts = [int(row["reused_atom_count"]) for row in rows]
    reuse_fractions = [float(row["reuse_fraction"]) for row in rows]
    atom_counts = [int(row["atom_count"]) for row in rows]
    peak_reused = max(reused_counts)
    return {
        "quality_saturation": {
            "definition": (
                "operational near-best rule: smallest atom count whose mean top-k "
                "quality is within 0.005 absolute score of the best observed mean"
            ),
            "absolute_tolerance": SATURATION_TOLERANCE,
            "best_observed_mean_quality": best_quality,
            # Retain atom_count for schema compatibility while naming its
            # operational meaning explicitly in the adjacent field.
            "atom_count": operational_count,
            "operational_near_best_atom_count": operational_count,
            "atom_counts_within_tolerance": near_best_counts,
            "quality_curve_pattern": _curve_pattern(
                [float(row["mean_quality"]) for row in rows]
            ),
            "conventional_rise_then_plateau_observed": conventional_plateau,
            "interpretation": (
                "The operational count is a compact near-best selection, not by itself "
                "evidence that quality rose and then saturated."
            ),
        },
        "storage_below_independent_baseline": {
            "answer": all(bool(row["storage_below_independent_baseline"]) for row in rows),
            "largest_relative_storage": max(
                float(row["relative_storage_to_independent_rank4"]) for row in rows
            ),
            "variants_not_below": [
                int(row["atom_count"])
                for row in rows
                if not bool(row["storage_below_independent_baseline"])
            ],
        },
        "reuse_with_dictionary_size": {
            "absolute_count_pattern": _curve_pattern(reused_counts),
            "reuse_fraction_pattern": _curve_pattern(reuse_fractions),
            "absolute_count_trend": _curve_pattern(reused_counts),
            "reuse_fraction_trend": _curve_pattern(reuse_fractions),
            "reused_atom_count_by_atom_count": {
                str(atom_count): count
                for atom_count, count in zip(atom_counts, reused_counts, strict=True)
            },
            "reuse_fraction_by_atom_count": {
                str(atom_count): fraction
                for atom_count, fraction in zip(
                    atom_counts, reuse_fractions, strict=True
                )
            },
            "peak_reused_atom_count": peak_reused,
            "peak_reused_atom_counts": [
                atom_count
                for atom_count, count in zip(atom_counts, reused_counts, strict=True)
                if count == peak_reused
            ],
            "interpretation": (
                "Report the complete curve: endpoint comparisons alone can hide a "
                "mid-curve peak or reversal."
            ),
            "smallest_dictionary_reused_atoms": int(first["reused_atom_count"]),
            "largest_dictionary_reused_atoms": int(last["reused_atom_count"]),
            "smallest_dictionary_reuse_fraction": float(first["reuse_fraction"]),
            "largest_dictionary_reuse_fraction": float(last["reuse_fraction"]),
        },
        "extra_atoms_become_task_private": {
            "answer": any(
                int(row["task_private_atom_count"])
                > int(core["task_private_atom_count"])
                for row in larger
            ),
            "core_atom_count": 8,
            "core_task_private_atom_count": int(core["task_private_atom_count"]),
            "larger_dictionary_private_counts": {
                str(row["atom_count"]): int(row["task_private_atom_count"])
                for row in larger
            },
            "maximum_additional_private_atoms": max(
                0,
                max(
                    int(row["task_private_atom_count"])
                    - int(core["task_private_atom_count"])
                    for row in larger
                ),
            ),
        },
    }


def _rank_question_answers(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best = max(rows, key=lambda row: (float(row["mean_quality"]), -int(row["rank"])))
    qualities = [float(row["mean_quality"]) for row in rows]
    core = next(row for row in rows if row["rank"] == 4)
    return {
        "best_mean_quality_rank": int(best["rank"]),
        "best_mean_quality": float(best["mean_quality"]),
        "mean_quality_range_across_ranks": max(qualities) - min(qualities),
        "rank4_gap_from_best": float(best["mean_quality"]) - float(core["mean_quality"]),
        "interpretation": (
            "This is a single-seed sensitivity curve; the observed range is reported "
            "without imposing an unregistered robustness threshold."
        ),
    }


def _active_question_answers(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best_quality = max(float(row["mean_quality"]) for row in rows)
    smallest_near_best = min(
        int(row["active_atoms"])
        for row in rows
        if float(row["mean_quality"]) >= best_quality - SATURATION_TOLERANCE
    )
    best = max(
        rows,
        key=lambda row: (float(row["mean_quality"]), -int(row["active_atoms"])),
    )
    per_atom_operations = {
        int(row["active_adapter_operations_per_token"]) / int(row["active_atoms"])
        for row in rows
    }
    return {
        "best_mean_quality_active_atoms": int(best["active_atoms"]),
        "best_mean_quality": float(best["mean_quality"]),
        "smallest_active_atoms_within_0_005_of_best": smallest_near_best,
        "active_compute_is_linear_in_k": len(per_atom_operations) == 1,
        "adapter_operations_per_token_per_active_atom": min(per_atom_operations),
        "persistent_storage_changes_with_k": len(
            {int(row["persistent_adaptation_parameters"]) for row in rows}
        )
        > 1,
    }


def _complete_locked_settings(
    baseline_config: ExperimentConfig, atom_config: ExperimentConfig
) -> dict[str, Any]:
    baseline = baseline_config.to_dict()
    atoms = atom_config.to_dict()
    baseline_name = str(baseline.pop("experiment_name"))
    atom_name = str(atoms.pop("experiment_name"))
    if baseline != atoms:
        differing = sorted(
            name for name in baseline if baseline[name] != atoms.get(name)
        )
        raise AblationError(
            "locked baseline and atom settings differ unexpectedly: "
            + ", ".join(differing)
        )
    return {
        "experiment_names": {
            "independent_lora": baseline_name,
            "shared_atoms": atom_name,
        },
        **baseline,
    }


def summarize_ablations(
    *,
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    core_independent: Mapping[str, Any],
    core_shared: Mapping[str, Any],
    atom_records: Mapping[int, Mapping[str, Any]],
    rank_records: Mapping[int, Mapping[str, Any]],
    active_records: Mapping[int, Mapping[str, Any]],
    variant_directories: Mapping[str, Mapping[int, Path]],
    core_independent_directory: Path,
    core_shared_directory: Path,
) -> dict[str, Any]:
    """Create the strict, machine-readable answer from saved/live records."""

    tasks = tuple(baseline_config.tasks)
    independent_parameters = _persistent_parameters(core_independent, "core independent")
    shared_parameters = _persistent_parameters(core_shared, "core shared")
    independent_checkpoint = _independent_checkpoint_bytes(core_independent, "core independent")
    shared_checkpoint = _shared_checkpoint_bytes(core_shared, "core shared")
    independent_rank4_operations = _independent_operations(
        core_independent, tasks, "core independent"
    )
    shared_all_operations = _shared_operations(core_shared, "core shared", "all_atoms")
    if shared_all_operations % 8:
        raise AblationError("core 8-atom operation count is not divisible by eight")
    operations_per_atom = shared_all_operations // 8

    atom_rows: list[dict[str, Any]] = []
    for atom_count in ATOM_COUNTS:
        record = atom_records[atom_count]
        top_k = min(4, atom_count)
        _validate_shared_variant(record, atom_count=atom_count, top_k=top_k, tasks=tasks)
        scores = _task_scores(
            record,
            tasks,
            variant="top_k",
            path=f"atom-count {atom_count}",
        )
        mean_quality, worst_task_score = _score_summary(scores)
        parameters = _persistent_parameters(record, f"atom-count {atom_count}")
        row = {
            "atom_count": atom_count,
            "evaluation_top_k": top_k,
            "task_scores": scores,
            "mean_quality": mean_quality,
            "worst_task_score": worst_task_score,
            "persistent_adaptation_parameters": parameters,
            "relative_storage_to_independent_rank4": parameters / independent_parameters,
            "storage_below_independent_baseline": parameters < independent_parameters,
            "checkpoint_bytes": _shared_checkpoint_bytes(
                record, f"atom-count {atom_count}"
            ),
            "active_adapter_operations_per_token": _shared_operations(
                record, f"atom-count {atom_count}", "top_k"
            ),
            **_coefficient_counts(
                record,
                atom_count=atom_count,
                task_count=len(tasks),
                path=f"atom-count {atom_count}",
            ),
            "variant_directory": str(variant_directories["atom_count"][atom_count]),
            "source_directory": str(
                core_shared_directory
                if atom_count == 8
                else variant_directories["atom_count"][atom_count]
            ),
            "reused_core_run": atom_count == 8,
        }
        atom_rows.append(row)

    rank_rows: list[dict[str, Any]] = []
    for rank in LORA_RANKS:
        record = rank_records[rank]
        _validate_rank_variant(record, rank=rank, tasks=tasks)
        scores = _task_scores(record, tasks, variant="best", path=f"LoRA rank {rank}")
        mean_quality, worst_task_score = _score_summary(scores)
        parameters = _persistent_parameters(record, f"LoRA rank {rank}")
        rank_rows.append(
            {
                "rank": rank,
                "task_scores": scores,
                "mean_quality": mean_quality,
                "worst_task_score": worst_task_score,
                "persistent_adaptation_parameters": parameters,
                "relative_storage_to_independent_rank4": parameters
                / independent_parameters,
                "relative_storage_to_core_shared_atoms": parameters / shared_parameters,
                "checkpoint_bytes": _independent_checkpoint_bytes(
                    record, f"LoRA rank {rank}"
                ),
                "active_adapter_operations_per_token": _independent_operations(
                    record, tasks, f"LoRA rank {rank}"
                ),
                "variant_directory": str(variant_directories["lora_rank"][rank]),
                "source_directory": str(
                    core_independent_directory
                    if rank == 4
                    else variant_directories["lora_rank"][rank]
                ),
                "reused_core_run": rank == 4,
            }
        )

    core_rank4_mean = next(
        float(row["mean_quality"]) for row in rank_rows if row["rank"] == 4
    )
    active_rows: list[dict[str, Any]] = []
    for top_k in ACTIVE_ATOM_COUNTS:
        record = active_records[top_k]
        _validate_active_evaluation(record, top_k=top_k, tasks=tasks)
        scores = _task_scores(record, tasks, variant=None, path=f"active top-k {top_k}")
        mean_quality, worst_task_score = _score_summary(scores)
        operations = operations_per_atom * top_k
        active_rows.append(
            {
                "active_atoms": top_k,
                "task_scores": scores,
                "mean_quality": mean_quality,
                "worst_task_score": worst_task_score,
                "quality_retention_vs_independent_rank4": (
                    mean_quality / core_rank4_mean if core_rank4_mean else None
                ),
                "persistent_adaptation_parameters": shared_parameters,
                "relative_storage_to_independent_rank4": shared_parameters
                / independent_parameters,
                "checkpoint_bytes": shared_checkpoint,
                "active_adapter_operations_per_token": operations,
                "relative_active_compute_to_independent_rank4": operations
                / independent_rank4_operations,
                "relative_active_compute_to_all_8_atoms": operations
                / shared_all_operations,
                "variant_directory": str(variant_directories["active_atoms"][top_k]),
                "checkpoint_source_directory": str(core_shared_directory),
                "retrained": False,
            }
        )
    active_k4_mean = next(
        float(row["mean_quality"]) for row in active_rows if row["active_atoms"] == 4
    )
    for row in active_rows:
        row["mean_quality_delta_from_k4"] = float(row["mean_quality"]) - active_k4_mean

    return {
        "schema_version": 1,
        "experiment": "h1_followup_ablations",
        "run_kind": "exploratory_single_seed_followup",
        "seed": SEED,
        "tasks": list(tasks),
        "locked_budget": _complete_locked_settings(baseline_config, atom_config),
        "core_sources": {
            "independent_rank4": str(core_independent_directory),
            "shared_atoms8": str(core_shared_directory),
        },
        "atom_count_ablation": {
            "varied_field": "atom_count",
            "values": list(ATOM_COUNTS),
            "evaluation_rule": "top min(4, atom_count) atoms",
            "coefficient_usage_definitions": {
                "dead_atom_threshold": DEAD_ATOM_THRESHOLD,
                "dead_atom": (
                    "maximum across tasks of the per-task mean absolute coefficient "
                    "across adapted layers is <= dead_atom_threshold"
                ),
                "top_k_unused_atom": (
                    "selected by zero tasks under the per-task mean-absolute-"
                    "coefficient top-k usage analysis"
                ),
                "distinction": (
                    "a non-dead atom may still be top-k unused; dead and top-k unused "
                    "are not interchangeable"
                ),
            },
            "rows": atom_rows,
            "question_answers": _atom_question_answers(atom_rows),
        },
        "lora_rank_ablation": {
            "varied_field": "lora_rank",
            "values": list(LORA_RANKS),
            "rows": rank_rows,
            "question_answers": _rank_question_answers(rank_rows),
        },
        "active_capacity_ablation": {
            "varied_field": "active_atoms_at_evaluation",
            "values": list(ACTIVE_ATOM_COUNTS),
            "checkpoint_atom_count": 8,
            "training_runs": 0,
            "rows": active_rows,
            "question_answers": _active_question_answers(active_rows),
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _score(value: float) -> str:
    return f"{value:.4f}"


def _bytes(value: int) -> str:
    return f"{value:,}"


def _task_score_table(
    rows: Sequence[Mapping[str, Any]], key: str, tasks: Sequence[str]
) -> list[str]:
    lines = [
        "| " + key + " | " + " | ".join(tasks) + " |",
        "|---:" + "|---:" * len(tasks) + "|",
    ]
    for row in rows:
        scores = _mapping(row["task_scores"], f"{key}.task_scores")
        lines.append(
            f"| {row[key]} | "
            + " | ".join(_score(float(scores[task])) for task in tasks)
            + " |"
        )
    return lines


def render_ablations_report(summary: Mapping[str, Any]) -> str:
    """Render the ablation summary as readable Markdown tables and answers."""

    tasks = [str(task) for task in _sequence(summary.get("tasks"), "summary.tasks")]
    atom_section = _mapping(summary.get("atom_count_ablation"), "atom_count_ablation")
    atom_rows = list(_sequence(atom_section.get("rows"), "atom_count_ablation.rows"))
    atom_answers = _mapping(atom_section.get("question_answers"), "atom answers")
    usage_definitions = _mapping(
        atom_section.get("coefficient_usage_definitions"),
        "coefficient usage definitions",
    )
    saturation = _mapping(atom_answers["quality_saturation"], "quality saturation")
    storage = _mapping(atom_answers["storage_below_independent_baseline"], "storage")
    reuse = _mapping(atom_answers["reuse_with_dictionary_size"], "reuse")
    private = _mapping(atom_answers["extra_atoms_become_task_private"], "private")

    rank_section = _mapping(summary.get("lora_rank_ablation"), "lora_rank_ablation")
    rank_rows = list(_sequence(rank_section.get("rows"), "lora_rank_ablation.rows"))
    rank_answers = _mapping(rank_section.get("question_answers"), "rank answers")

    active_section = _mapping(
        summary.get("active_capacity_ablation"), "active_capacity_ablation"
    )
    active_rows = list(_sequence(active_section.get("rows"), "active rows"))
    active_answers = _mapping(active_section.get("question_answers"), "active answers")

    lines = [
        "# H1 Follow-up Ablations - Seed 17",
        "",
        (
            "These are exploratory single-seed follow-ups. All five tasks, sampled examples, "
            "optimizer settings, and the locked three-epoch budget are unchanged from H1."
        ),
        "",
        "## Atom-count ablation",
        "",
        (
            "Operational near-best selection (an explicitly exploratory "
            f"+/-{float(saturation['absolute_tolerance']):.3f} rule): "
            f"**{saturation['operational_near_best_atom_count']} atoms**; counts "
            f"within tolerance {saturation['atom_counts_within_tolerance']}; best "
            f"observed mean {_score(float(saturation['best_observed_mean_quality']))}."
        ),
        (
            "This rule does "
            "**"
            f"{'show' if saturation['conventional_rise_then_plateau_observed'] else 'not show'}"
            "** "
            "a conventional rise-then-plateau pattern; it identifies the smallest "
            "near-best dictionary."
        ),
        (
            f"Storage below the independent rank-4 baseline: "
            f"**{'yes' if storage['answer'] else 'no'}**; largest relative storage "
            f"{_percent(float(storage['largest_relative_storage']))}."
        ),
        (
            "Reuse is **"
            f"{str(reuse['absolute_count_pattern']).replace('_', '-')}** in absolute "
            "count and **"
            f"{str(reuse['reuse_fraction_pattern']).replace('_', '-')}** by fraction. "
            "Reused counts by N: "
            + " -> ".join(
                str(_mapping(value, "atom row")["reused_atom_count"])
                for value in atom_rows
            )
            + "."
        ),
        (
            f"Do extra atoms become task-private above N=8? "
            f"**{'yes' if private['answer'] else 'no'}**; maximum additional private atoms "
            f"{private['maximum_additional_private_atoms']}."
        ),
        (
            f"Dead atom means the learned coefficient criterion is <= "
            f"{float(usage_definitions['dead_atom_threshold']):g}; top-k unused means "
            "selected by zero tasks. A non-dead atom can therefore be top-k unused."
        ),
        "",
        "| Atoms | Eval top-k | Mean | Worst task | Persistent params | "
        "vs rank-4 LoRA | Checkpoint bytes | Reused | Reuse | Private | Dead | "
        "Top-k unused | Ops/token |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in atom_rows:
        row = _mapping(value, "atom row")
        lines.append(
            f"| {row['atom_count']} | {row['evaluation_top_k']} | "
            f"{_score(float(row['mean_quality']))} | "
            f"{_score(float(row['worst_task_score']))} | "
            f"{int(row['persistent_adaptation_parameters']):,} | "
            f"{_percent(float(row['relative_storage_to_independent_rank4']))} | "
            f"{_bytes(int(row['checkpoint_bytes']))} | {row['reused_atom_count']} | "
            f"{_percent(float(row['reuse_fraction']))} | {row['task_private_atom_count']} | "
            f"{row['dead_atom_count']} | {row['top_k_unused_atom_count']} | "
            f"{int(row['active_adapter_operations_per_token']):,} |"
        )
    lines.extend(["", "### Atom-count task scores", ""])
    lines.extend(_task_score_table(atom_rows, "atom_count", tasks))

    lines.extend(
        [
            "",
            "## Independent-LoRA rank ablation",
            "",
            (
                "Best observed mean quality used rank "
                f"**{rank_answers['best_mean_quality_rank']}**. "
                f"The single-seed quality range was "
                f"{_score(float(rank_answers['mean_quality_range_across_ranks']))}; no "
                "unregistered pass/fail robustness threshold is applied."
            ),
            "",
            "| Rank | Mean | Worst task | Persistent params | vs rank 4 | "
            "vs shared N=8 | Checkpoint bytes | Ops/token |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for value in rank_rows:
        row = _mapping(value, "rank row")
        lines.append(
            f"| {row['rank']} | {_score(float(row['mean_quality']))} | "
            f"{_score(float(row['worst_task_score']))} | "
            f"{int(row['persistent_adaptation_parameters']):,} | "
            f"{_percent(float(row['relative_storage_to_independent_rank4']))} | "
            f"{_percent(float(row['relative_storage_to_core_shared_atoms']))} | "
            f"{_bytes(int(row['checkpoint_bytes']))} | "
            f"{int(row['active_adapter_operations_per_token']):,} |"
        )
    lines.extend(["", "### Rank task scores", ""])
    lines.extend(_task_score_table(rank_rows, "rank", tasks))

    lines.extend(
        [
            "",
            "## Active-atom capacity ablation",
            "",
            (
                "All rows reload the same core 8-atom checkpoint; no retraining occurs. "
                "Best observed mean used "
                f"k=**{active_answers['best_mean_quality_active_atoms']}**; smallest k "
                "within 0.005 was "
                f"**{active_answers['smallest_active_atoms_within_0_005_of_best']}**."
            ),
            (
                f"Estimated active adapter compute is "
                "**"
                f"{'linear' if active_answers['active_compute_is_linear_in_k'] else 'not linear'}"
                "** in k. Persistent storage is unchanged because every row deploys the "
                "full dictionary."
            ),
            "",
            "| Active atoms | Mean | Worst task | delta mean vs k=4 | Persistent params | "
            "vs rank-4 LoRA storage | Checkpoint bytes | Ops/token | "
            "vs rank-4 LoRA compute |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for value in active_rows:
        row = _mapping(value, "active row")
        lines.append(
            f"| {row['active_atoms']} | {_score(float(row['mean_quality']))} | "
            f"{_score(float(row['worst_task_score']))} | "
            f"{float(row['mean_quality_delta_from_k4']):+.4f} | "
            f"{int(row['persistent_adaptation_parameters']):,} | "
            f"{_percent(float(row['relative_storage_to_independent_rank4']))} | "
            f"{_bytes(int(row['checkpoint_bytes']))} | "
            f"{int(row['active_adapter_operations_per_token']):,} | "
            f"{_percent(float(row['relative_active_compute_to_independent_rank4']))} |"
        )
    lines.extend(["", "### Active-capacity task scores", ""])
    lines.extend(_task_score_table(active_rows, "active_atoms", tasks))
    return "\n".join(lines) + "\n"


def write_ablations_outputs(
    summary: Mapping[str, Any], output_directory: str | Path
) -> tuple[Path, Path]:
    """Write strict JSON plus its human-readable Markdown companion."""

    destination = Path(output_directory)
    summary_path = write_json(destination / SUMMARY_FILENAME, summary)
    report_path = destination / REPORT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_ablations_report(summary), encoding="utf-8", newline="\n")
    return summary_path, report_path


def _read_record(path: Path, description: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    return _mapping(read_json(path), str(path))


def run_ablations(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    results_root: str | Path = "results/followup_ablations",
    *,
    core_independent_root: str | Path = "results/independent_lora",
    core_shared_root: str | Path = "results/shared_atoms",
    force: bool = False,
) -> dict[str, Any]:
    """Run/resume chunks 23–24 and persist their complete answer.

    Rank 4 and atom count 8 are references to the locked core seed-17 runs.
    Active-capacity rows reload that same 8-atom checkpoint and never train.
    """

    _validate_locked_config(baseline_config, "independent_lora")
    _validate_locked_config(atom_config, "shared_atoms")
    if baseline_config.tasks != atom_config.tasks:
        raise AblationError("baseline and atom configs must use identical locked tasks")

    destination = Path(results_root)
    core_independent_directory = Path(core_independent_root) / f"seed_{SEED}"
    core_shared_directory = Path(core_shared_root) / f"seed_{SEED}"
    core_independent = _read_record(
        core_independent_directory / "metrics_by_task.json", "core independent rank-4 run"
    )
    core_shared = _read_record(
        core_shared_directory / "metrics_by_task.json", "core shared 8-atom run"
    )
    _validate_core_record(
        core_independent,
        config=baseline_config,
        system="independent_lora",
        capacity_name="rank",
        capacity=4,
    )
    _validate_core_record(
        core_shared,
        config=atom_config,
        system="shared_atoms",
        capacity_name="atom_count",
        capacity=8,
    )

    variant_directories: dict[str, dict[int, Path]] = {
        "atom_count": {
            count: destination / "atom_count" / f"atoms_{count}" / f"seed_{SEED}"
            for count in ATOM_COUNTS
        },
        "lora_rank": {
            rank: destination / "lora_rank" / f"rank_{rank}" / f"seed_{SEED}"
            for rank in LORA_RANKS
        },
        "active_atoms": {
            top_k: destination / "active_atoms" / f"top_k_{top_k}" / f"seed_{SEED}"
            for top_k in ACTIVE_ATOM_COUNTS
        },
    }

    prepared: Any | None = None

    def locked_data() -> Any:
        nonlocal prepared
        if prepared is None:
            prepared = prepare_data(baseline_config, tasks=baseline_config.tasks)
        return prepared

    atom_records: dict[int, Mapping[str, Any]] = {}
    for atom_count in ATOM_COUNTS:
        variant_directory = variant_directories["atom_count"][atom_count]
        top_k = min(4, atom_count)
        variant_config = atom_config.with_overrides(
            atom_count=atom_count,
            active_atoms_during_training=atom_count,
            active_atoms_for_primary_evaluation=top_k,
        )
        if atom_count == 8:
            record = core_shared
            write_json(
                variant_directory / "reused_core.json",
                {
                    "system": "shared_atoms",
                    "seed": SEED,
                    "atom_count": 8,
                    "source_directory": core_shared_directory,
                    "retrained": False,
                },
            )
        else:
            metrics_path = variant_directory / "metrics_by_task.json"
            if not force and metrics_path.is_file():
                record = _mapping(read_json(metrics_path), str(metrics_path))
            else:
                output_root = variant_directory.parent
                record = _mapping(
                    run_shared_atoms(
                        variant_config,
                        output_root,
                        tasks=variant_config.tasks,
                        prepared=locked_data(),
                        run_kind="followup_ablation_seed17_locked",
                        atom_count=atom_count,
                        top_k=top_k,
                        force=force,
                    ),
                    f"run_shared_atoms(atom_count={atom_count})",
                )
            record = _normalize_variant_config(
                record,
                expected=variant_config,
                allowed_legacy_fields={
                    "atom_count",
                    "active_atoms_during_training",
                    "active_atoms_for_primary_evaluation",
                },
                path=f"atom-count {atom_count}",
            )
            # This is also the lightweight migration path for already trained
            # variants whose original aggregate metadata retained N=8.
            write_json(metrics_path, record)
        _validate_shared_variant(
            record,
            atom_count=atom_count,
            top_k=top_k,
            tasks=atom_config.tasks,
        )
        atom_records[atom_count] = record

    rank_records: dict[int, Mapping[str, Any]] = {}
    for rank in LORA_RANKS:
        variant_directory = variant_directories["lora_rank"][rank]
        variant_config = baseline_config.with_overrides(lora_rank=rank)
        if rank == 4:
            record = core_independent
            write_json(
                variant_directory / "reused_core.json",
                {
                    "system": "independent_lora",
                    "seed": SEED,
                    "rank": 4,
                    "source_directory": core_independent_directory,
                    "retrained": False,
                },
            )
        else:
            metrics_path = variant_directory / "metrics_by_task.json"
            if not force and metrics_path.is_file():
                record = _mapping(read_json(metrics_path), str(metrics_path))
            else:
                output_root = variant_directory.parent
                record = _mapping(
                    run_independent_lora(
                        variant_config,
                        output_root,
                        tasks=variant_config.tasks,
                        prepared=locked_data(),
                        run_kind="followup_ablation_seed17_locked",
                        rank=rank,
                        force=force,
                    ),
                    f"run_independent_lora(rank={rank})",
                )
            record = _normalize_variant_config(
                record,
                expected=variant_config,
                allowed_legacy_fields={"lora_rank"},
                path=f"LoRA rank {rank}",
            )
            # Rewrite only aggregate metadata/results; checkpoints are never
            # touched and cached task scores remain byte-for-byte values.
            write_json(metrics_path, record)
        _validate_rank_variant(record, rank=rank, tasks=baseline_config.tasks)
        rank_records[rank] = record

    active_records: dict[int, Mapping[str, Any]] = {}
    for top_k in ACTIVE_ATOM_COUNTS:
        variant_directory = variant_directories["active_atoms"][top_k]
        evaluation_path = variant_directory / "evaluation.json"
        if not force and evaluation_path.is_file():
            record = _mapping(read_json(evaluation_path), str(evaluation_path))
        else:
            record = _mapping(
                evaluate_atom_checkpoint(
                    atom_config,
                    core_shared_directory,
                    top_k=top_k,
                    tasks=atom_config.tasks,
                    prepared=locked_data(),
                ),
                f"evaluate_atom_checkpoint(top_k={top_k})",
            )
            write_json(evaluation_path, record)
        _validate_active_evaluation(record, top_k=top_k, tasks=atom_config.tasks)
        active_records[top_k] = record

    summary = summarize_ablations(
        baseline_config=baseline_config,
        atom_config=atom_config,
        core_independent=core_independent,
        core_shared=core_shared,
        atom_records=atom_records,
        rank_records=rank_records,
        active_records=active_records,
        variant_directories=variant_directories,
        core_independent_directory=core_independent_directory,
        core_shared_directory=core_shared_directory,
    )
    write_ablations_outputs(summary, destination)
    return summary


# Intent-revealing alias for callers using the roadmap terminology.
run_followup_ablations = run_ablations


__all__ = [
    "ACTIVE_ATOM_COUNTS",
    "ATOM_COUNTS",
    "AblationError",
    "LORA_RANKS",
    "REPORT_FILENAME",
    "SATURATION_TOLERANCE",
    "SEED",
    "SUMMARY_FILENAME",
    "render_ablations_report",
    "run_ablations",
    "run_followup_ablations",
    "summarize_ablations",
    "write_ablations_outputs",
]
