"""Reproducible, seed-first aggregation and reporting for the H1 experiment.

The functions in this module intentionally operate on saved run records rather
than live models.  A report can therefore be regenerated on a fresh machine
without loading BERT or rerunning either training system.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from cgmoe_h1.config import H1_CONFIRMATORY_SEEDS, H1_TASKS
from cgmoe_h1.utils.serialization import read_json, to_jsonable, write_json

INDEPENDENT_SYSTEM = "independent_lora"
SHARED_SYSTEM = "shared_atoms"
DEFAULT_RESULT_FILENAME = "metrics_by_task.json"
DEFAULT_INDEPENDENT_ROOT = Path("results/independent_lora")
DEFAULT_SHARED_ROOT = Path("results/shared_atoms")
DEFAULT_OUTPUT_DIR = Path("results/h1_report")

QUALITY_RETENTION_THRESHOLD = 0.97
WORST_TASK_GAP_THRESHOLD = 0.03
RELATIVE_STORAGE_THRESHOLD = 0.50


class ReportingError(ValueError):
    """A saved result cannot support a valid H1 summary."""


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportingError(f"{path} must be an object")
    return value


def _finite_number(
    value: object,
    path: str,
    *,
    unit_interval: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportingError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ReportingError(f"{path} must be finite")
    if unit_interval and not 0.0 <= result <= 1.0:
        raise ReportingError(f"{path} must be in [0, 1], got {result}")
    return result


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportingError(f"{path} must be an integer")
    if value <= 0:
        raise ReportingError(f"{path} must be positive, got {value}")
    return value


def _primary_score(record: Mapping[str, Any], path: str) -> float:
    metrics = _mapping(record.get("metrics"), f"{path}.metrics")
    return _finite_number(
        metrics.get("primary_score"),
        f"{path}.metrics.primary_score",
        unit_interval=True,
    )


def load_seed_records(
    root: str | Path,
    *,
    system: str,
    seeds: Sequence[int] = H1_CONFIRMATORY_SEEDS,
    filename: str = DEFAULT_RESULT_FILENAME,
) -> list[dict[str, Any]]:
    """Load one aggregate run record from each ``seed_<n>`` directory."""

    result_root = Path(root)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        path = result_root / f"seed_{seed}" / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing {system} result for seed {seed}: {path}")
        value = read_json(path)
        record = dict(_mapping(value, str(path)))
        records.append(record)
    return records


def _record_sequence(records: object, name: str) -> list[Mapping[str, Any]]:
    if isinstance(records, Mapping):
        if "system" in records:
            values: Iterable[object] = (records,)
        else:
            values = records.values()
    elif isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
        values = records
    else:
        raise ReportingError(f"{name} records must be an iterable or seed mapping")

    result: list[Mapping[str, Any]] = []
    for index, value in enumerate(values):
        result.append(_mapping(value, f"{name} records[{index}]"))
    if not result:
        raise ReportingError(f"{name} records must not be empty")
    return result


def _records_by_seed(
    records: object,
    *,
    name: str,
    expected_system: str,
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for index, record in enumerate(_record_sequence(records, name)):
        path = f"{name} records[{index}]"
        system = record.get("system")
        if system != expected_system:
            raise ReportingError(
                f"{path}.system must be {expected_system!r}, got {system!r}"
            )
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ReportingError(f"{path}.seed must be an integer")
        if seed in result:
            raise ReportingError(f"duplicate {name} record for seed {seed}")
        result[seed] = record
    return result


def _validate_seed_sets(
    independent: Mapping[int, Mapping[str, Any]],
    shared: Mapping[int, Mapping[str, Any]],
    expected_seeds: Sequence[int] | None,
) -> tuple[int, ...]:
    independent_seeds = set(independent)
    shared_seeds = set(shared)
    if independent_seeds != shared_seeds:
        missing_shared = sorted(independent_seeds - shared_seeds)
        missing_independent = sorted(shared_seeds - independent_seeds)
        details: list[str] = []
        if missing_shared:
            details.append(f"shared records missing seeds {missing_shared}")
        if missing_independent:
            details.append(f"independent records missing seeds {missing_independent}")
        raise ReportingError("paired seed sets differ: " + "; ".join(details))

    if expected_seeds is None:
        seeds = tuple(sorted(independent_seeds))
    else:
        seeds = tuple(expected_seeds)
        if len(set(seeds)) != len(seeds):
            raise ReportingError("expected seeds must be unique")
        if independent_seeds != set(seeds):
            missing = sorted(set(seeds) - independent_seeds)
            unexpected = sorted(independent_seeds - set(seeds))
            details = []
            if missing:
                details.append(f"missing seeds {missing}")
            if unexpected:
                details.append(f"unexpected seeds {unexpected}")
            raise ReportingError("invalid confirmatory seed set: " + "; ".join(details))
    if not seeds:
        raise ReportingError("at least one paired seed is required")
    return seeds


def _extract_scores(
    records: Mapping[int, Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    tasks: Sequence[str],
    variant: str,
) -> dict[int, dict[str, float]]:
    scores: dict[int, dict[str, float]] = {}
    for seed in seeds:
        record = records[seed]
        task_records = _mapping(record.get("tasks"), f"seed {seed}.tasks")
        missing = [task for task in tasks if task not in task_records]
        if missing:
            raise ReportingError(f"seed {seed}.tasks is missing required tasks: {missing}")
        scores[seed] = {}
        for task in tasks:
            task_record = _mapping(task_records[task], f"seed {seed}.tasks.{task}")
            variant_record = _mapping(
                task_record.get(variant),
                f"seed {seed}.tasks.{task}.{variant}",
            )
            scores[seed][task] = _primary_score(
                variant_record,
                f"seed {seed}.tasks.{task}.{variant}",
            )
    return scores


def _persistent_parameter_count(
    records: Mapping[int, Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    name: str,
) -> int:
    by_seed: dict[int, int] = {}
    for seed in seeds:
        counts = _mapping(records[seed].get("parameter_counts"), f"{name} seed {seed}.parameter_counts")
        by_seed[seed] = _positive_integer(
            counts.get("total_persistent_task_parameters"),
            f"{name} seed {seed}.parameter_counts.total_persistent_task_parameters",
        )
    unique = set(by_seed.values())
    if len(unique) != 1:
        rendered = ", ".join(f"seed {seed}: {count}" for seed, count in by_seed.items())
        raise ReportingError(
            f"{name} persistent parameter count differs across seeds ({rendered})"
        )
    return next(iter(unique))


def _aggregate_model(
    seed_scores: Mapping[int, Mapping[str, float]],
    *,
    seeds: Sequence[int],
    tasks: Sequence[str],
    persistent_parameters: int | None = None,
) -> dict[str, Any]:
    task_scores: dict[str, float] = {}
    task_standard_deviations: dict[str, float] = {}
    scores_by_task_and_seed: dict[str, dict[str, float]] = {}
    for task in tasks:
        values = [seed_scores[seed][task] for seed in seeds]
        task_scores[task] = statistics.fmean(values)
        task_standard_deviations[task] = statistics.pstdev(values)
        scores_by_task_and_seed[task] = {
            str(seed): seed_scores[seed][task] for seed in seeds
        }
    result: dict[str, Any] = {
        "seed_scores": {
            str(seed): {task: seed_scores[seed][task] for task in tasks}
            for seed in seeds
        },
        "task_scores": task_scores,
        "task_standard_deviations": task_standard_deviations,
        "scores_by_task_and_seed": scores_by_task_and_seed,
        "mean_score": statistics.fmean(task_scores[task] for task in tasks),
    }
    if persistent_parameters is not None:
        result["persistent_adaptation_parameters"] = persistent_parameters
    return result


def _threshold(value: float, operator: str, limit: float) -> dict[str, Any]:
    if operator == ">=":
        passed = value >= limit
    elif operator == "<=":
        passed = value <= limit
    else:  # pragma: no cover - internal invariant
        raise AssertionError(f"unsupported threshold operator: {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": limit,
        "passed": passed,
    }


def summarize_records(
    independent_records: object,
    shared_records: object,
    *,
    seeds: Sequence[int] | None = H1_CONFIRMATORY_SEEDS,
    tasks: Sequence[str] = H1_TASKS,
    coefficient_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the pure JSON-friendly H1 summary from paired saved records.

    Aggregation is seed-first: each task is averaged over seeds, then the five
    task means are averaged with equal task weight.  The independent system's
    retained ``best`` checkpoint is the preregistered baseline.  Its ``final``
    epoch is retained as a diagnostic, while shared ``top_k`` is the primary
    shared result and ``all_atoms`` is the required unpruned diagnostic.
    """

    normalized_tasks = tuple(tasks)
    if not normalized_tasks or len(set(normalized_tasks)) != len(normalized_tasks):
        raise ReportingError("tasks must be a non-empty sequence of unique names")
    if any(not isinstance(task, str) or not task for task in normalized_tasks):
        raise ReportingError("every task name must be a non-empty string")

    independent = _records_by_seed(
        independent_records,
        name="independent",
        expected_system=INDEPENDENT_SYSTEM,
    )
    shared = _records_by_seed(
        shared_records,
        name="shared",
        expected_system=SHARED_SYSTEM,
    )
    normalized_seeds = _validate_seed_sets(independent, shared, seeds)

    independent_best_scores = _extract_scores(
        independent,
        seeds=normalized_seeds,
        tasks=normalized_tasks,
        variant="best",
    )
    independent_final_scores = _extract_scores(
        independent,
        seeds=normalized_seeds,
        tasks=normalized_tasks,
        variant="final",
    )
    shared_all_scores = _extract_scores(
        shared,
        seeds=normalized_seeds,
        tasks=normalized_tasks,
        variant="all_atoms",
    )
    shared_top_scores = _extract_scores(
        shared,
        seeds=normalized_seeds,
        tasks=normalized_tasks,
        variant="top_k",
    )

    independent_parameters = _persistent_parameter_count(
        independent,
        seeds=normalized_seeds,
        name="independent",
    )
    shared_parameters = _persistent_parameter_count(
        shared,
        seeds=normalized_seeds,
        name="shared",
    )

    independent_summary = _aggregate_model(
        independent_best_scores,
        seeds=normalized_seeds,
        tasks=normalized_tasks,
        persistent_parameters=independent_parameters,
    )
    independent_final_summary = _aggregate_model(
        independent_final_scores,
        seeds=normalized_seeds,
        tasks=normalized_tasks,
    )
    shared_all_summary = _aggregate_model(
        shared_all_scores,
        seeds=normalized_seeds,
        tasks=normalized_tasks,
        persistent_parameters=shared_parameters,
    )
    shared_top_summary = _aggregate_model(
        shared_top_scores,
        seeds=normalized_seeds,
        tasks=normalized_tasks,
        persistent_parameters=shared_parameters,
    )

    independent_mean = independent_summary["mean_score"]
    if independent_mean == 0.0:
        raise ReportingError("independent LoRA mean score is zero; retention is undefined")
    quality_retention = shared_top_summary["mean_score"] / independent_mean
    task_gaps = {
        task: independent_summary["task_scores"][task]
        - shared_top_summary["task_scores"][task]
        for task in normalized_tasks
    }
    all_atoms_task_gaps = {
        task: independent_summary["task_scores"][task]
        - shared_all_summary["task_scores"][task]
        for task in normalized_tasks
    }
    worst_task_gap = max(0.0, max(task_gaps.values()))
    all_atoms_worst_task_gap = max(0.0, max(all_atoms_task_gaps.values()))
    relative_storage = shared_parameters / independent_parameters
    per_task_retention = {
        task: (
            shared_top_summary["task_scores"][task]
            / independent_summary["task_scores"][task]
            if independent_summary["task_scores"][task] != 0.0
            else None
        )
        for task in normalized_tasks
    }

    thresholds = {
        "quality_retention": _threshold(
            quality_retention,
            ">=",
            QUALITY_RETENTION_THRESHOLD,
        ),
        "worst_task_gap": _threshold(
            worst_task_gap,
            "<=",
            WORST_TASK_GAP_THRESHOLD,
        ),
        "relative_storage": _threshold(
            relative_storage,
            "<=",
            RELATIVE_STORAGE_THRESHOLD,
        ),
    }
    passed = all(item["passed"] for item in thresholds.values())

    models = {
        "independent_lora": independent_summary,
        "independent_lora_final": independent_final_summary,
        "shared_atoms_all": shared_all_summary,
        "shared_atoms_top_k": shared_top_summary,
    }
    comparison = {
        "quality_retention": quality_retention,
        "task_gaps": task_gaps,
        "all_atoms_task_gaps": all_atoms_task_gaps,
        "per_task_retention": per_task_retention,
        "worst_task_gap": worst_task_gap,
        "all_atoms_worst_task_gap": all_atoms_worst_task_gap,
        "relative_storage": relative_storage,
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "aggregation": "mean over seeds per task, then unweighted mean over tasks",
        "standard_deviation": "population",
        "seeds": list(normalized_seeds),
        "tasks": list(normalized_tasks),
        "models": models,
        "comparison": comparison,
        "thresholds": thresholds,
        "preregistered_pass": passed,
        "passed": passed,
        "decision": "Supported" if passed else "Not supported",
        # Compact mirrors make the machine-readable result convenient without
        # requiring report consumers to know the richer model layout.
        "mean_scores": {
            "independent_lora": independent_summary["mean_score"],
            "shared_atoms_all": shared_all_summary["mean_score"],
            "shared_atoms_top_k": shared_top_summary["mean_score"],
        },
        "quality_retention": quality_retention,
        "worst_task_gap": worst_task_gap,
        "relative_storage": relative_storage,
        "parameter_counts": {
            "independent_lora": independent_parameters,
            "shared_atoms": shared_parameters,
        },
    }
    if coefficient_analysis is not None:
        summary["coefficient_analysis"] = to_jsonable(coefficient_analysis)
    return summary


def _coefficient_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif isinstance(value, (list, tuple)):
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return None
    else:
        return None
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float64)
    else:
        tensor = tensor.to(torch.float64)
    if not torch.isfinite(tensor).all():
        return None
    return tensor


def _payload_task_ids(payload: Mapping[str, Any], fallback: Sequence[str]) -> tuple[str, ...]:
    direct = payload.get("task_ids")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        if all(isinstance(item, str) for item in direct):
            return tuple(direct)
    for key, value in payload.items():
        if str(key).endswith("_extra_state") and isinstance(value, Mapping):
            nested = value.get("task_ids")
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                if all(isinstance(item, str) for item in nested):
                    return tuple(nested)
    return tuple(fallback)


def _extract_coefficient_matrices(
    payload: object,
    tasks: Sequence[str],
) -> tuple[list[str], torch.Tensor] | None:
    """Normalize common checkpoint payloads to ``[layer, task, atom]``."""

    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]

    if isinstance(payload, Mapping):
        task_ids = _payload_task_ids(payload, tasks)
        if set(task_ids) != set(tasks):
            return None
        task_permutation = [task_ids.index(task) for task in tasks]

        coefficient_value = payload.get("coefficients")
        if isinstance(coefficient_value, Mapping):
            candidates = list(coefficient_value.items())
        elif coefficient_value is not None:
            tensor = _coefficient_tensor(coefficient_value)
            if tensor is None:
                candidates = []
            elif tensor.ndim == 2:
                candidates = [("layer_0", tensor)]
            elif tensor.ndim == 3:
                if tensor.shape[1] == len(task_ids):
                    candidates = [(f"layer_{index}", matrix) for index, matrix in enumerate(tensor)]
                elif tensor.shape[0] == len(task_ids):
                    tensor = tensor.permute(1, 0, 2)
                    candidates = [(f"layer_{index}", matrix) for index, matrix in enumerate(tensor)]
                else:
                    candidates = []
            else:
                candidates = []
        else:
            candidates = [
                (str(key), value)
                for key, value in payload.items()
                if str(key).split(".")[-1] == "coefficients"
            ]
            if not candidates:
                # A dedicated coefficients.pt often contains only layer-name
                # to matrix entries, without the repeated field suffix.
                candidates = [
                    (str(key), value)
                    for key, value in payload.items()
                    if _coefficient_tensor(value) is not None
                    and _coefficient_tensor(value).ndim == 2
                    and _coefficient_tensor(value).shape[0] == len(task_ids)
                ]

        names: list[str] = []
        matrices: list[torch.Tensor] = []
        atom_count: int | None = None
        for name, value in candidates:
            matrix = _coefficient_tensor(value)
            if matrix is None or matrix.ndim != 2 or matrix.shape[0] != len(task_ids):
                continue
            matrix = matrix[task_permutation]
            if atom_count is None:
                atom_count = int(matrix.shape[1])
            if matrix.shape[1] != atom_count:
                return None
            names.append(name.removesuffix(".coefficients"))
            matrices.append(matrix)
        if matrices:
            return names, torch.stack(matrices)

        # JSON may store a direct task -> [layer, atom] mapping.
        if all(task in payload for task in tasks):
            rows = [_coefficient_tensor(payload[task]) for task in tasks]
            if all(row is not None and row.ndim in (1, 2) for row in rows):
                normalized = [row.unsqueeze(0) if row.ndim == 1 else row for row in rows]
                shapes = {tuple(row.shape) for row in normalized}
                if len(shapes) == 1:
                    tensor = torch.stack(normalized).permute(1, 0, 2)
                    return [f"layer_{index}" for index in range(tensor.shape[0])], tensor

    tensor = _coefficient_tensor(payload)
    if tensor is not None:
        if tensor.ndim == 2 and tensor.shape[0] == len(tasks):
            return ["layer_0"], tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[1] == len(tasks):
            return [f"layer_{index}" for index in range(tensor.shape[0])], tensor
    return None


def analyze_coefficient_payload(
    payload: object,
    *,
    tasks: Sequence[str] = H1_TASKS,
    top_k: int = 4,
) -> dict[str, Any]:
    """Compute reuse diagnostics from a tensor/list checkpoint payload."""

    extracted = _extract_coefficient_matrices(payload, tasks)
    if extracted is None:
        raise ReportingError("coefficient payload contains no recognizable coefficient matrices")
    layer_names, matrices = extracted
    layer_count, task_count, atom_count = matrices.shape
    if top_k <= 0 or top_k > atom_count:
        raise ReportingError(f"top_k must be in [1, {atom_count}], got {top_k}")

    selected = torch.zeros_like(matrices, dtype=torch.bool)
    for layer_index in range(layer_count):
        for task_index in range(task_count):
            magnitudes = matrices[layer_index, task_index].abs().tolist()
            indices = sorted(range(atom_count), key=lambda index: (-magnitudes[index], index))[:top_k]
            selected[layer_index, task_index, indices] = True

    task_by_atom_usage = {
        task: selected[:, task_index, :].sum(dim=0).tolist()
        for task_index, task in enumerate(tasks)
    }
    mean_magnitudes = matrices.abs().mean(dim=0)
    top_atoms_per_task: dict[str, list[dict[str, float | int]]] = {}
    for task_index, task in enumerate(tasks):
        values = mean_magnitudes[task_index].tolist()
        indices = sorted(range(atom_count), key=lambda index: (-values[index], index))[:top_k]
        top_atoms_per_task[task] = [
            {"atom_index": index, "mean_absolute_coefficient": values[index]}
            for index in indices
        ]

    pairwise_similarity: dict[str, float | None] = {}
    flattened = matrices.permute(1, 0, 2).reshape(task_count, -1)
    for left in range(task_count):
        for right in range(left + 1, task_count):
            denominator = torch.linalg.vector_norm(flattened[left]) * torch.linalg.vector_norm(
                flattened[right]
            )
            key = f"{tasks[left]}|{tasks[right]}"
            if float(denominator) == 0.0:
                pairwise_similarity[key] = None
            else:
                pairwise_similarity[key] = float(
                    torch.dot(flattened[left], flattened[right]) / denominator
                )

    utilization_by_layer: dict[str, list[int]] = {}
    utilization_values: list[int] = []
    for layer_index, layer_name in enumerate(layer_names):
        counts = selected[layer_index].sum(dim=0).tolist()
        utilization_by_layer[layer_name] = counts
        utilization_values.extend(counts)
    reuse = {
        "atom_slots": layer_count * atom_count,
        "dead_atoms": sum(count == 0 for count in utilization_values),
        "task_exclusive_atoms": sum(count == 1 for count in utilization_values),
        "reused_by_two_or_more_tasks": sum(count >= 2 for count in utilization_values),
    }
    return {
        "layer_names": layer_names,
        "layer_count": layer_count,
        "atom_count": atom_count,
        "top_k": top_k,
        "task_by_atom_usage": task_by_atom_usage,
        "top_atoms_per_task": top_atoms_per_task,
        "pairwise_coefficient_similarity": pairwise_similarity,
        "pairwise_similarity_measure": "signed_flattened_coefficients",
        "utilization_by_layer": utilization_by_layer,
        "reuse_unit": "layer_atom_slot",
        "reuse": reuse,
    }


def _analysis_from_training_diagnostics(
    payload: Mapping[str, Any],
    *,
    tasks: Sequence[str],
) -> dict[str, Any] | None:
    task_payloads = payload.get("tasks")
    if not isinstance(task_payloads, Mapping):
        return None
    if not all(task in task_payloads for task in tasks):
        return None

    selections: dict[str, dict[str, list[tuple[int, float]]]] = {}
    max_atom = -1
    layer_names: set[str] = set()
    for task in tasks:
        task_data = task_payloads[task]
        if not isinstance(task_data, Mapping):
            return None
        by_layer = task_data.get("top_used_atoms_by_layer")
        if not isinstance(by_layer, Mapping):
            return None
        selections[task] = {}
        for layer, entries in by_layer.items():
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                return None
            parsed: list[tuple[int, float]] = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    return None
                index = entry.get("atom_index")
                magnitude = entry.get("absolute_coefficient", 0.0)
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    return None
                try:
                    numeric_magnitude = _finite_number(magnitude, "absolute_coefficient")
                except ReportingError:
                    return None
                parsed.append((index, numeric_magnitude))
                max_atom = max(max_atom, index)
            layer_name = str(layer)
            layer_names.add(layer_name)
            selections[task][layer_name] = parsed
    if max_atom < 0:
        return None

    atom_count = max_atom + 1
    usage = {task: [0] * atom_count for task in tasks}
    magnitude_sums = {task: [0.0] * atom_count for task in tasks}
    magnitude_counts = {task: [0] * atom_count for task in tasks}
    utilization_by_layer: dict[str, list[int]] = {}
    for layer in sorted(layer_names):
        counts = [0] * atom_count
        for task in tasks:
            for index, magnitude in selections[task].get(layer, []):
                usage[task][index] += 1
                counts[index] += 1
                magnitude_sums[task][index] += magnitude
                magnitude_counts[task][index] += 1
        utilization_by_layer[layer] = counts

    top_atoms: dict[str, list[dict[str, float | int]]] = {}
    for task in tasks:
        averages = [
            magnitude_sums[task][index] / magnitude_counts[task][index]
            if magnitude_counts[task][index]
            else 0.0
            for index in range(atom_count)
        ]
        chosen = sorted(
            range(atom_count),
            key=lambda index: (-averages[index], -usage[task][index], index),
        )[: max((len(items) for items in selections[task].values()), default=0)]
        top_atoms[task] = [
            {"atom_index": index, "mean_absolute_coefficient": averages[index]}
            for index in chosen
        ]

    flat_counts = [count for counts in utilization_by_layer.values() for count in counts]
    return {
        "layer_names": sorted(layer_names),
        "layer_count": len(layer_names),
        "atom_count": atom_count,
        "top_k": max((len(items) for task in selections.values() for items in task.values()), default=0),
        "task_by_atom_usage": usage,
        "top_atoms_per_task": top_atoms,
        "pairwise_coefficient_similarity": {},
        "pairwise_similarity_measure": "unavailable",
        "utilization_by_layer": utilization_by_layer,
        "reuse_unit": "layer_atom_slot",
        "reuse": {
            "atom_slots": len(flat_counts),
            "dead_atoms": sum(count == 0 for count in flat_counts),
            "task_exclusive_atoms": sum(count == 1 for count in flat_counts),
            "reused_by_two_or_more_tasks": sum(count >= 2 for count in flat_counts),
        },
    }


def _analysis_from_experiment_record(
    payload: Mapping[str, Any],
    *,
    tasks: Sequence[str],
) -> dict[str, Any] | None:
    """Normalize the coefficient analysis emitted by ``experiments.py``."""

    usage = payload.get("usage_by_task")
    top_atoms = payload.get("top_atoms_by_task")
    utilization = payload.get("atom_utilization_count")
    if not isinstance(usage, Mapping) or not isinstance(top_atoms, Mapping):
        return None
    if not isinstance(utilization, Sequence) or isinstance(utilization, (str, bytes)):
        return None
    if not all(task in usage and task in top_atoms for task in tasks):
        return None

    atom_count = len(utilization)
    normalized_usage: dict[str, list[float]] = {}
    normalized_top: dict[str, list[dict[str, float | int]]] = {}
    for task in tasks:
        values = usage[task]
        indices = top_atoms[task]
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != atom_count
            or not isinstance(indices, Sequence)
            or isinstance(indices, (str, bytes))
        ):
            return None
        try:
            numeric_values = [
                _finite_number(value, f"usage_by_task.{task}[{index}]")
                for index, value in enumerate(values)
            ]
        except ReportingError:
            return None
        normalized_indices: list[int] = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < atom_count:
                return None
            normalized_indices.append(index)
        normalized_usage[task] = numeric_values
        normalized_top[task] = [
            {
                "atom_index": index,
                "mean_absolute_coefficient": numeric_values[index],
            }
            for index in normalized_indices
        ]

    normalized_utilization: list[int] = []
    for value in utilization:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        normalized_utilization.append(value)

    similarities: dict[str, float | None] = {}
    raw_similarities = payload.get("pairwise_cosine_similarity")
    if isinstance(raw_similarities, Mapping):
        for pair, value in raw_similarities.items():
            try:
                numeric = _finite_number(value, f"pairwise_cosine_similarity.{pair}")
            except ReportingError:
                continue
            similarities[str(pair).replace(":", "|")] = numeric

    masks = payload.get("top_k_masks_by_layer")
    layer_names: set[str] = set()
    if isinstance(masks, Mapping):
        for task in tasks:
            task_masks = masks.get(task)
            if isinstance(task_masks, Mapping):
                layer_names.update(str(layer) for layer in task_masks)

    dead = payload.get("dead_atoms", ())
    exclusive = payload.get("task_exclusive_atoms", ())
    reused = payload.get("reused_atoms", ())
    if not all(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        for value in (dead, exclusive, reused)
    ):
        return None
    return {
        "layer_names": sorted(layer_names),
        "layer_count": len(layer_names),
        "atom_count": atom_count,
        "top_k": max((len(normalized_top[task]) for task in tasks), default=0),
        "usage_measure": "mean_absolute_coefficient",
        "task_by_atom_usage": normalized_usage,
        "top_atoms_per_task": normalized_top,
        "pairwise_coefficient_similarity": similarities,
        "pairwise_similarity_measure": "cosine_of_mean_absolute_coefficients",
        "atom_utilization_count": normalized_utilization,
        "top_k_masks_by_layer": to_jsonable(masks) if isinstance(masks, Mapping) else {},
        "reuse_unit": "atom_index",
        "reuse": {
            "atom_slots": atom_count,
            "dead_atoms": len(dead),
            "task_exclusive_atoms": len(exclusive),
            "reused_by_two_or_more_tasks": len(reused),
        },
    }


def _normalize_json_analysis(
    payload: Mapping[str, Any],
    *,
    tasks: Sequence[str],
    top_k: int,
) -> dict[str, Any]:
    try:
        analyzed = analyze_coefficient_payload(payload, tasks=tasks, top_k=top_k)
        analyzed["source"] = "metrics_by_task.json coefficient_analysis"
        return analyzed
    except ReportingError:
        pass
    experiment_analysis = _analysis_from_experiment_record(payload, tasks=tasks)
    if experiment_analysis is not None:
        experiment_analysis["source"] = "metrics_by_task.json coefficient_analysis"
        return experiment_analysis
    diagnostics = _analysis_from_training_diagnostics(payload, tasks=tasks)
    if diagnostics is not None:
        diagnostics["source"] = "metrics_by_task.json coefficient_analysis"
        return diagnostics
    if "reuse" in payload or "task_by_atom_usage" in payload:
        result = dict(to_jsonable(payload))
        result.setdefault("source", "metrics_by_task.json coefficient_analysis")
        return result
    return {
        "source": "metrics_by_task.json coefficient_analysis",
        "provided": to_jsonable(payload),
    }


def _aggregate_coefficient_analyses(
    per_seed: Mapping[int, Mapping[str, Any]],
    *,
    tasks: Sequence[str],
) -> dict[str, Any] | None:
    reusable = {seed: analysis for seed, analysis in per_seed.items() if isinstance(analysis.get("reuse"), Mapping)}
    if not reusable:
        return None

    reuse_keys = ("atom_slots", "dead_atoms", "task_exclusive_atoms", "reused_by_two_or_more_tasks")
    aggregate: dict[str, Any] = {
        "seeds_analyzed": sorted(reusable),
        "reuse_mean_across_seeds": {},
    }
    usage_measures = {
        str(analysis["usage_measure"])
        for analysis in reusable.values()
        if analysis.get("usage_measure") is not None
    }
    if len(usage_measures) == 1:
        aggregate["task_by_atom_usage_measure"] = next(iter(usage_measures))
    reuse_units = {
        str(analysis["reuse_unit"])
        for analysis in reusable.values()
        if analysis.get("reuse_unit") is not None
    }
    if len(reuse_units) == 1:
        aggregate["reuse_unit"] = next(iter(reuse_units))
    similarity_measures = {
        str(analysis["pairwise_similarity_measure"])
        for analysis in reusable.values()
        if analysis.get("pairwise_similarity_measure") is not None
    }
    if len(similarity_measures) == 1:
        aggregate["pairwise_similarity_measure"] = next(iter(similarity_measures))
    for key in reuse_keys:
        values: list[float] = []
        for analysis in reusable.values():
            reuse = analysis["reuse"]
            value = reuse.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if values:
            aggregate["reuse_mean_across_seeds"][key] = statistics.fmean(values)

    max_atoms = max(
        (
            len(values)
            for analysis in reusable.values()
            for values in (
                analysis.get("task_by_atom_usage", {}).values()
                if isinstance(analysis.get("task_by_atom_usage"), Mapping)
                else []
            )
        ),
        default=0,
    )
    usage_mean: dict[str, list[float]] = {}
    for task in tasks:
        rows: list[Sequence[Any]] = []
        for analysis in reusable.values():
            usage = analysis.get("task_by_atom_usage")
            if isinstance(usage, Mapping) and isinstance(usage.get(task), Sequence):
                rows.append(usage[task])
        if rows:
            usage_mean[task] = [
                statistics.fmean(
                    float(row[index]) if index < len(row) else 0.0 for row in rows
                )
                for index in range(max_atoms)
            ]
    if usage_mean:
        aggregate["task_by_atom_usage_mean"] = usage_mean

    pair_values: dict[str, list[float]] = {}
    for analysis in reusable.values():
        pairs = analysis.get("pairwise_coefficient_similarity")
        if not isinstance(pairs, Mapping):
            continue
        for pair, value in pairs.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                pair_values.setdefault(str(pair), []).append(float(value))
    if pair_values:
        aggregate["pairwise_coefficient_similarity_mean"] = {
            pair: statistics.fmean(values) for pair, values in sorted(pair_values.items())
        }
    return aggregate


def collect_coefficient_analysis(
    shared_root: str | Path,
    shared_records: object,
    *,
    seeds: Sequence[int],
    tasks: Sequence[str] = H1_TASKS,
    top_k: int = 4,
) -> dict[str, Any] | None:
    """Collect optional JSON or checkpoint coefficient diagnostics per seed.

    Optional diagnostic corruption is reported as a warning and never changes
    the quality/storage threshold decision.
    """

    by_seed = _records_by_seed(
        shared_records,
        name="shared",
        expected_system=SHARED_SYSTEM,
    )
    root = Path(shared_root)
    analyses: dict[int, Mapping[str, Any]] = {}
    warnings: list[str] = []
    for seed in seeds:
        record = by_seed.get(seed)
        if record is None:
            continue
        embedded = record.get("coefficient_analysis")
        if isinstance(embedded, Mapping):
            analyses[seed] = _normalize_json_analysis(
                embedded,
                tasks=tasks,
                top_k=top_k,
            )
            continue

        path = root / f"seed_{seed}" / "coefficients.pt"
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            analysis = analyze_coefficient_payload(payload, tasks=tasks, top_k=top_k)
            analysis["source"] = str(path)
            analyses[seed] = analysis
        except Exception as exc:  # Optional diagnostics must not block H1 scores.
            warnings.append(f"seed {seed}: could not analyze {path}: {type(exc).__name__}: {exc}")

    if not analyses and not warnings:
        return None
    result: dict[str, Any] = {
        "available": bool(analyses),
        "per_seed": {str(seed): to_jsonable(analysis) for seed, analysis in sorted(analyses.items())},
    }
    aggregate = _aggregate_coefficient_analyses(analyses, tasks=tasks)
    if aggregate is not None:
        result["aggregate"] = aggregate
    if warnings:
        result["warnings"] = warnings
    return result


def summarize_h1(
    independent_root: str | Path = DEFAULT_INDEPENDENT_ROOT,
    shared_root: str | Path = DEFAULT_SHARED_ROOT,
    *,
    seeds: Sequence[int] = H1_CONFIRMATORY_SEEDS,
    tasks: Sequence[str] = H1_TASKS,
    filename: str = DEFAULT_RESULT_FILENAME,
) -> dict[str, Any]:
    """Load both result trees and return a complete H1 summary."""

    independent_records = load_seed_records(
        independent_root,
        system=INDEPENDENT_SYSTEM,
        seeds=seeds,
        filename=filename,
    )
    shared_records = load_seed_records(
        shared_root,
        system=SHARED_SYSTEM,
        seeds=seeds,
        filename=filename,
    )
    coefficient_analysis = collect_coefficient_analysis(
        shared_root,
        shared_records,
        seeds=seeds,
        tasks=tasks,
    )
    return summarize_records(
        independent_records,
        shared_records,
        seeds=seeds,
        tasks=tasks,
        coefficient_analysis=coefficient_analysis,
    )


def _score(value: float) -> str:
    return f"{value:.4f}"


def _percentage(value: float) -> str:
    return f"{value:.2%}"


def _task_label(task: str) -> str:
    labels = {"sst2": "SST-2", "mrpc": "MRPC", "rte": "RTE", "qnli": "QNLI", "qqp": "QQP"}
    return labels.get(task, task)


def render_h1_report(summary: Mapping[str, Any]) -> str:
    """Render the machine-readable summary as the required Markdown report."""

    models = _mapping(summary.get("models"), "summary.models")
    independent = _mapping(models.get("independent_lora"), "summary.models.independent_lora")
    independent_final = _mapping(
        models.get("independent_lora_final"),
        "summary.models.independent_lora_final",
    )
    shared_all = _mapping(models.get("shared_atoms_all"), "summary.models.shared_atoms_all")
    shared_top = _mapping(models.get("shared_atoms_top_k"), "summary.models.shared_atoms_top_k")
    comparison = _mapping(summary.get("comparison"), "summary.comparison")
    thresholds = _mapping(summary.get("thresholds"), "summary.thresholds")
    tasks = tuple(summary.get("tasks", ()))
    seeds = tuple(summary.get("seeds", ()))
    passed = bool(summary.get("preregistered_pass"))
    decision = str(summary.get("decision", "Supported" if passed else "Not supported"))

    independent_parameters = int(independent["persistent_adaptation_parameters"])
    shared_parameters = int(shared_top["persistent_adaptation_parameters"])
    relative_storage = float(comparison["relative_storage"])

    lines = [
        "# H1 Experiment Report",
        "",
        f"Decision: **{decision}**",
        "",
        f"Preregistered result: **{'PASS' if passed else 'FAIL'}**. A pass requires all three locked thresholds; no narrative interpretation overrides this result.",
        "",
        "## Comparison",
        "",
        "| Model | Mean score | Worst task gap | Persistent adaptation params | Relative storage | Active rank/atoms |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Independent LoRA | {_score(float(independent['mean_score']))} | 0.0000 | {independent_parameters:,} | 100.00% | rank 4 |",
        f"| Shared atoms, all 8 | {_score(float(shared_all['mean_score']))} | {_score(float(comparison['all_atoms_worst_task_gap']))} | {shared_parameters:,} | {_percentage(relative_storage)} | 8 |",
        f"| Shared atoms, top 4 | {_score(float(shared_top['mean_score']))} | {_score(float(comparison['worst_task_gap']))} | {shared_parameters:,} | {_percentage(relative_storage)} | 4 |",
        "",
        "Top-4 evaluation changes active capacity, not stored parameters; all eight learned atoms and every coefficient remain in the shared deployment count.",
        "",
        "## Per-task primary comparison",
        "",
        "| Task | Independent LoRA | Shared atoms top-4 | Absolute gap | Relative retention |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in tasks:
        retention = comparison["per_task_retention"][task]
        retention_text = "N/A" if retention is None else _percentage(float(retention))
        lines.append(
            f"| {_task_label(task)} | {_score(float(independent['task_scores'][task]))} | "
            f"{_score(float(shared_top['task_scores'][task]))} | "
            f"{_score(float(comparison['task_gaps'][task]))} | {retention_text} |"
        )
    lines.extend(
        [
            "",
            "The gap is LoRA minus shared top-4 in absolute score units; a negative value means the shared system scored higher.",
            "",
            "## Locked threshold decision",
            "",
            "| Criterion | Observed | Required | Result |",
            "|---|---:|---:|---:|",
        ]
    )
    threshold_labels = {
        "quality_retention": "Quality retention",
        "worst_task_gap": "Worst task gap",
        "relative_storage": "Relative storage",
    }
    for key in ("quality_retention", "worst_task_gap", "relative_storage"):
        item = thresholds[key]
        value = float(item["value"])
        threshold_value = float(item["threshold"])
        display_value = _percentage(value) if key != "worst_task_gap" else _score(value)
        display_threshold = (
            _percentage(threshold_value) if key != "worst_task_gap" else _score(threshold_value)
        )
        lines.append(
            f"| {threshold_labels[key]} | {display_value} | {item['operator']} {display_threshold} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Per-seed primary scores",
            "",
            "| Model | Seed | " + " | ".join(_task_label(task) for task in tasks) + " | Seed mean |",
            "|---|---:|" + "---:|" * (len(tasks) + 1),
        ]
    )
    seed_models = (
        ("Independent LoRA", independent),
        ("Shared atoms, all 8", shared_all),
        ("Shared atoms, top 4", shared_top),
    )
    for label, model in seed_models:
        for seed in seeds:
            scores = model["seed_scores"][str(seed)]
            seed_mean = statistics.fmean(float(scores[task]) for task in tasks)
            lines.append(
                f"| {label} | {seed} | "
                + " | ".join(_score(float(scores[task])) for task in tasks)
                + f" | {_score(seed_mean)} |"
            )

    lines.extend(
        [
            "",
            "## Across-seed population standard deviation",
            "",
            "| Model | " + " | ".join(_task_label(task) for task in tasks) + " |",
            "|---|" + "---:|" * len(tasks),
        ]
    )
    for label, model in seed_models:
        lines.append(
            f"| {label} | "
            + " | ".join(
                _score(float(model["task_standard_deviations"][task])) for task in tasks
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Independent final-epoch diagnostic",
            "",
            f"The independent final-epoch mean was {_score(float(independent_final['mean_score']))}; the retained best checkpoints, not final epochs, define the baseline above.",
            "",
            "## Coefficient reuse",
            "",
        ]
    )
    coefficient = summary.get("coefficient_analysis")
    if not isinstance(coefficient, Mapping) or not coefficient.get("available"):
        lines.append(
            "Coefficient reuse artifacts were unavailable. This diagnostic absence does not alter the quality/storage threshold arithmetic."
        )
        if isinstance(coefficient, Mapping) and coefficient.get("warnings"):
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in coefficient["warnings"])
    else:
        aggregate = coefficient.get("aggregate")
        if isinstance(aggregate, Mapping):
            reuse = aggregate.get("reuse_mean_across_seeds")
            if isinstance(reuse, Mapping):
                if aggregate.get("reuse_unit") == "atom_index":
                    reuse_description = "Mean atom-index counts across analyzed seeds:"
                    reuse_heading = "Atom indices"
                else:
                    reuse_description = (
                        "Mean counts across analyzed seeds (an atom slot is one atom in one target layer):"
                    )
                    reuse_heading = "Atom slots"
                lines.extend(
                    [
                        reuse_description,
                        "",
                        f"| {reuse_heading} | Dead atoms | Task-exclusive atoms | Reused by 2+ tasks |",
                        "|---:|---:|---:|---:|",
                        "| "
                        + " | ".join(
                            f"{float(reuse.get(key, 0.0)):.2f}"
                            for key in (
                                "atom_slots",
                                "dead_atoms",
                                "task_exclusive_atoms",
                                "reused_by_two_or_more_tasks",
                            )
                        )
                        + " |",
                    ]
                )
            usage = aggregate.get("task_by_atom_usage_mean")
            if isinstance(usage, Mapping) and usage:
                atom_count = max(len(row) for row in usage.values())
                measure = aggregate.get("task_by_atom_usage_measure")
                if measure == "mean_absolute_coefficient":
                    usage_description = "Mean task-by-atom absolute coefficient magnitude:"
                else:
                    usage_description = (
                        "Mean task-by-atom top-k usage (number of target layers selecting each atom index):"
                    )
                lines.extend(
                    [
                        "",
                        usage_description,
                        "",
                        "| Task | " + " | ".join(f"Atom {index}" for index in range(atom_count)) + " |",
                        "|---|" + "---:|" * atom_count,
                    ]
                )
                for task in tasks:
                    row = usage.get(task)
                    if isinstance(row, Sequence):
                        lines.append(
                            f"| {_task_label(task)} | "
                            + " | ".join(f"{float(value):.2f}" for value in row)
                            + " |"
                        )
            pairs = aggregate.get("pairwise_coefficient_similarity_mean")
            if isinstance(pairs, Mapping) and pairs:
                if aggregate.get("pairwise_similarity_measure") == "cosine_of_mean_absolute_coefficients":
                    similarity_description = (
                        "Mean pairwise cosine similarity of per-atom absolute coefficient magnitudes:"
                    )
                else:
                    similarity_description = (
                        "Mean pairwise signed cosine similarity of coefficient vectors:"
                    )
                lines.extend(
                    [
                        "",
                        similarity_description,
                        "",
                        "| Task pair | Similarity |",
                        "|---|---:|",
                    ]
                )
                lines.extend(f"| {pair.replace('|', ' / ')} | {float(value):.4f} |" for pair, value in pairs.items())

        per_seed = coefficient.get("per_seed")
        if isinstance(per_seed, Mapping):
            top_rows: list[tuple[str, str, str]] = []
            for seed, analysis in per_seed.items():
                if not isinstance(analysis, Mapping):
                    continue
                top_by_task = analysis.get("top_atoms_per_task")
                if not isinstance(top_by_task, Mapping):
                    continue
                for task in tasks:
                    entries = top_by_task.get(task)
                    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                        continue
                    indices = [
                        str(entry["atom_index"])
                        for entry in entries
                        if isinstance(entry, Mapping) and "atom_index" in entry
                    ]
                    top_rows.append((str(seed), task, ", ".join(indices)))
            if top_rows:
                lines.extend(
                    [
                        "",
                        "Top atoms per task and seed:",
                        "",
                        "| Seed | Task | Top atom indices |",
                        "|---:|---|---|",
                    ]
                )
                lines.extend(
                    f"| {seed} | {_task_label(task)} | {indices} |"
                    for seed, task, indices in top_rows
                )
        if coefficient.get("warnings"):
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in coefficient["warnings"])

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "The shared top-4 system meets every preregistered quality, worst-task, and storage threshold, so H1 is supported under the locked comparison."
                if passed
                else "The shared top-4 system misses at least one preregistered threshold, so H1 is not supported by this locked comparison."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_h1_outputs(
    summary: Mapping[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write ``h1_summary.json`` and ``h1_report.md`` and return both paths."""

    destination = Path(output_dir)
    summary_path = write_json(destination / "h1_summary.json", summary)
    report_path = destination / "h1_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_h1_report(summary), encoding="utf-8", newline="\n")
    return summary_path, report_path


def generate_h1_report(
    independent_root: str | Path = DEFAULT_INDEPENDENT_ROOT,
    shared_root: str | Path = DEFAULT_SHARED_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    seeds: Sequence[int] = H1_CONFIRMATORY_SEEDS,
    tasks: Sequence[str] = H1_TASKS,
    filename: str = DEFAULT_RESULT_FILENAME,
) -> tuple[dict[str, Any], Path, Path]:
    """Load, aggregate, render, and persist the complete H1 result."""

    summary = summarize_h1(
        independent_root,
        shared_root,
        seeds=seeds,
        tasks=tasks,
        filename=filename,
    )
    summary_path, report_path = write_h1_outputs(summary, output_dir)
    return summary, summary_path, report_path


# Readable aliases for callers that use analysis-oriented terminology.
aggregate_h1_results = summarize_records
build_h1_summary = summarize_records


__all__ = [
    "DEFAULT_INDEPENDENT_ROOT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RESULT_FILENAME",
    "DEFAULT_SHARED_ROOT",
    "INDEPENDENT_SYSTEM",
    "QUALITY_RETENTION_THRESHOLD",
    "RELATIVE_STORAGE_THRESHOLD",
    "ReportingError",
    "SHARED_SYSTEM",
    "WORST_TASK_GAP_THRESHOLD",
    "aggregate_h1_results",
    "analyze_coefficient_payload",
    "build_h1_summary",
    "collect_coefficient_analysis",
    "generate_h1_report",
    "load_seed_records",
    "render_h1_report",
    "summarize_h1",
    "summarize_records",
    "write_h1_outputs",
]
