"""Oracle projection of held-out LoRA updates into frozen atom spans.

The numerical helpers in this module are deliberately independent of model
training.  They answer a narrow question: given a target update matrix and a
fixed bank of rank-one atom matrices, what is the best Frobenius-norm
reconstruction available in that span?
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cgmoe_h1.config import H1_CONFIRMATORY_SEEDS, H1_TASKS, ExperimentConfig
from cgmoe_h1.experiments import (
    PreparedData,
    build_atom_model,
    build_loaders,
    environment_record,
    prepare_data,
    save_compact_checkpoint,
)
from cgmoe_h1.followups_controls import effective_lora_updates
from cgmoe_h1.followups_transfer import copy_frozen_atom_dictionary
from cgmoe_h1.metrics import compute_task_metrics
from cgmoe_h1.models.atoms import AtomLinear
from cgmoe_h1.models.injection import load_adapter_state_dict
from cgmoe_h1.training.trainer import evaluate
from cgmoe_h1.utils.parameters import active_adapter_operations, categorized_parameter_counts
from cgmoe_h1.utils.reproducibility import set_seed
from cgmoe_h1.utils.serialization import read_json, write_json
from cgmoe_h1.validation_cross_transfer import (
    PROTOCOL_FILENAME as CROSS_TRANSFER_PROTOCOL_FILENAME,
    _validate_compact_checkpoint,
    _validate_environment,
    _validate_frozen_atom_record,
    _validate_head_only_record,
    _validate_source_record,
    _selected_validation_examples,
    build_cross_transfer_summary,
    build_protocol_record as build_cross_transfer_protocol,
    validate_cross_transfer_cell,
)


SCHEMA_VERSION = 1
DEFAULT_CAPACITY = 8
DEFAULT_TOP_K = 4
QUALITY_RETENTION_THRESHOLD = 0.95
PER_TARGET_RETENTION_THRESHOLD = 0.90
RANDOM_QUALITY_MARGIN = 0.005
PROTOCOL_FILENAME = "oracle_projection_protocol.json"
EXPERIMENT_NAME = "oracle_lora_update_projection"
MODEL_NAME = "prajjwal1/bert-tiny"
EXPECTED_TARGET_MODULES = (
    "encoder.layer.0.attention.self.query",
    "encoder.layer.0.attention.self.value",
    "encoder.layer.1.attention.self.query",
    "encoder.layer.1.attention.self.value",
)
EXPECTED_STATE_PREFIXES = tuple(f"encoder.{name}" for name in EXPECTED_TARGET_MODULES)
EXPECTED_TARGET_DIMENSIONS = {name: [128, 128] for name in EXPECTED_TARGET_MODULES}
EXPECTED_FRESH_PARAMETER_COUNTS = {
    "base_trainable_parameters": 0,
    "lora_adapter_parameters": 4096,
    "atom_parameters": 0,
    "coefficient_parameters": 0,
    "head_parameters": 258,
    "uncategorized_trainable_parameters": 0,
    "model_trainable_parameters": 4354,
    "persistent_adaptation_parameters": 4354,
}
EXPECTED_PROJECTED_PARAMETER_COUNTS = {
    "base_trainable_parameters": 0,
    "lora_adapter_parameters": 0,
    "atom_parameters": 8192,
    "coefficient_parameters": 32,
    "head_parameters": 258,
    "uncategorized_trainable_parameters": 0,
    "model_trainable_parameters": 290,
    "persistent_adaptation_parameters": 8482,
}


def _require_finite_matrix(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-two tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def atom_design_matrix(
    atom_u: Tensor,
    atom_v: Tensor,
    *,
    scaling: float = 1.0,
) -> Tensor:
    """Return flattened effective atom matrices as ``[elements, atoms]``.

    ``atom_u`` is shaped ``[N, d_out]`` and ``atom_v`` is shaped
    ``[N, d_in]``.  Column ``k`` is
    ``scaling * outer(atom_u[k], atom_v[k]).flatten()``.
    """

    atom_u = _require_finite_matrix("atom_u", atom_u)
    atom_v = _require_finite_matrix("atom_v", atom_v)
    if atom_u.shape[0] != atom_v.shape[0]:
        raise ValueError("atom_u and atom_v must have the same atom count")
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("atom scaling must be finite and positive")
    bases = torch.einsum("no,ni->noi", atom_u.double(), atom_v.double())
    return (bases * float(scaling)).reshape(atom_u.shape[0], -1).transpose(0, 1).contiguous()


def deterministic_top_k_coefficients(coefficients: Tensor, top_k: int) -> tuple[Tensor, list[int]]:
    """Zero all but the largest-magnitude coefficients with stable tie breaks."""

    if not isinstance(coefficients, Tensor) or coefficients.ndim != 1:
        raise ValueError("coefficients must be a rank-one tensor")
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError("coefficients must contain only finite values")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    selected_count = min(top_k, coefficients.numel())
    magnitudes = coefficients.detach().abs().cpu().tolist()
    selected = sorted(
        range(coefficients.numel()),
        key=lambda index: (-magnitudes[index], index),
    )[:selected_count]
    result = torch.zeros_like(coefficients)
    if selected:
        index = torch.tensor(selected, device=coefficients.device)
        result[index] = coefficients[index]
    return result, selected


def _matrix_error_record(target: Tensor, reconstructed: Tensor) -> dict[str, float]:
    residual = target - reconstructed
    target_squared_norm = float(torch.sum(target.square()))
    residual_squared_norm = float(torch.sum(residual.square()))
    absolute_error = math.sqrt(residual_squared_norm)
    target_norm = math.sqrt(target_squared_norm)
    if target_squared_norm == 0.0:
        relative_error = 0.0 if residual_squared_norm == 0.0 else math.inf
        explained_energy = 1.0 if residual_squared_norm == 0.0 else -math.inf
    else:
        relative_error = absolute_error / target_norm
        explained_energy = 1.0 - residual_squared_norm / target_squared_norm
    return {
        "target_squared_frobenius_norm": target_squared_norm,
        "residual_squared_frobenius_norm": residual_squared_norm,
        "target_frobenius_norm": target_norm,
        "absolute_frobenius_error": absolute_error,
        "relative_frobenius_error": relative_error,
        "explained_energy": explained_energy,
    }


def solve_matrix_projection(
    target_update: Tensor,
    atom_u: Tensor,
    atom_v: Tensor,
    *,
    atom_scaling: float = 1.0,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[Tensor, dict[str, Any]]:
    """Solve the deterministic float64 least-squares atom coefficients.

    Returns the full oracle coefficient vector and a JSON-friendly record for
    both the all-atom and magnitude-top-k reconstructions.
    """

    target = _require_finite_matrix("target_update", target_update).double()
    if atom_u.shape[1] != target.shape[0] or atom_v.shape[1] != target.shape[1]:
        raise ValueError(
            "atom and target dimensions differ: "
            f"u={tuple(atom_u.shape)}, v={tuple(atom_v.shape)}, target={tuple(target.shape)}"
        )
    design = atom_design_matrix(atom_u, atom_v, scaling=atom_scaling)
    target_vector = target.reshape(-1)
    solver_solution = torch.linalg.lstsq(design, target_vector.unsqueeze(1)).solution[:, 0]
    if not bool(torch.isfinite(solver_solution).all()):
        raise RuntimeError("least-squares projection produced non-finite coefficients")

    # The oracle solve is deliberately float64, but AtomLinear stores its live
    # coefficient row in float32.  Quantize before deriving either reconstruction
    # or the top-k mask so the saved diagnostics describe the update that is
    # actually evaluated (including deterministic tie breaks after rounding).
    solution = solver_solution.to(dtype=torch.float32)

    reconstructed = (design @ solution.double()).reshape_as(target)
    top_coefficients, selected = deterministic_top_k_coefficients(solution, top_k)
    top_reconstructed = (design @ top_coefficients.double()).reshape_as(target)
    singular_values = torch.linalg.svdvals(design)
    largest = float(singular_values[0]) if singular_values.numel() else 0.0
    tolerance = max(design.shape) * torch.finfo(design.dtype).eps * largest
    numerical_rank = int((singular_values > tolerance).sum()) if largest else 0
    smallest_retained = (
        float(singular_values[numerical_rank - 1]) if numerical_rank else 0.0
    )
    retained_condition = largest / smallest_retained if smallest_retained else None
    full_rank = numerical_rank == design.shape[1]
    condition_number = retained_condition if full_rank else None

    record: dict[str, Any] = {
        "target_shape": list(target.shape),
        "atom_count": int(atom_u.shape[0]),
        "design_shape": list(design.shape),
        "numerical_rank": numerical_rank,
        "rank_tolerance": float(tolerance),
        "singular_values": [float(value) for value in singular_values],
        "condition_number": condition_number,
        "condition_number_finite": condition_number is not None,
        "retained_subspace_condition_number": retained_condition,
        "coefficient_dtype": str(solution.dtype),
        "solver_coefficients_float64": [float(value) for value in solver_solution],
        "coefficients": [float(value) for value in solution],
        "solver_coefficient_l2_norm": float(solver_solution.norm()),
        "coefficient_l2_norm": float(solution.norm()),
        "all_atoms": _matrix_error_record(target, reconstructed),
        "top_k": {
            "k": min(top_k, int(atom_u.shape[0])),
            "selected_atom_indices": selected,
            "coefficients": [float(value) for value in top_coefficients],
            **_matrix_error_record(target, top_reconstructed),
        },
    }
    return solution, record


def aggregate_layer_errors(
    layer_records: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, float]:
    """Aggregate layer errors by squared target norm, as preregistered."""

    if field not in {"all_atoms", "top_k"}:
        raise ValueError("field must be 'all_atoms' or 'top_k'")
    if not layer_records:
        raise ValueError("at least one layer record is required")
    target_squared = 0.0
    residual_squared = 0.0
    for name, record in layer_records.items():
        try:
            selected = record[field]
            target_value = float(selected["target_squared_frobenius_norm"])
            residual_value = float(selected["residual_squared_frobenius_norm"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid projection record for layer {name!r}") from error
        if target_value < 0 or residual_value < 0:
            raise ValueError("squared norms must be non-negative")
        target_squared += target_value
        residual_squared += residual_value
    if target_squared == 0.0:
        relative_error = 0.0 if residual_squared == 0.0 else math.inf
        explained_energy = 1.0 if residual_squared == 0.0 else -math.inf
    else:
        relative_error = math.sqrt(residual_squared / target_squared)
        explained_energy = 1.0 - residual_squared / target_squared
    return {
        "target_squared_frobenius_norm": target_squared,
        "residual_squared_frobenius_norm": residual_squared,
        "relative_frobenius_error": relative_error,
        "explained_energy": explained_energy,
    }


def _atom_state_layers(atom_state: Mapping[str, Any]) -> OrderedDict[str, tuple[Tensor, Tensor]]:
    u_keys = tuple(sorted(key for key in atom_state if key.endswith(".atom_u")))
    if not u_keys:
        raise ValueError("atom state contains no atom_u tensors")
    layers: OrderedDict[str, tuple[Tensor, Tensor]] = OrderedDict()
    for u_key in u_keys:
        prefix = u_key.removesuffix(".atom_u")
        v_key = prefix + ".atom_v"
        u = atom_state[u_key]
        v = atom_state.get(v_key)
        if not isinstance(u, Tensor) or not isinstance(v, Tensor):
            raise ValueError(f"atom state is missing tensor pair for {prefix!r}")
        layers[prefix] = (u, v)
    unexpected_vectors = sorted(
        key
        for key in atom_state
        if key.endswith(".atom_v") and key.removesuffix(".atom_v") not in layers
    )
    if unexpected_vectors:
        raise ValueError(f"unpaired atom_v tensors: {unexpected_vectors}")
    return layers


def project_lora_state_onto_atoms(
    lora_state: Mapping[str, Any],
    atom_state: Mapping[str, Any],
    *,
    lora_scaling: float = 1.0,
    atom_scaling: float = 1.0,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[OrderedDict[str, Tensor], dict[str, Any]]:
    """Project every path-aligned effective LoRA matrix into an atom state."""

    updates = effective_lora_updates(lora_state, scaling=lora_scaling)
    atoms = _atom_state_layers(atom_state)
    if tuple(updates) != tuple(atoms):
        raise ValueError(
            "LoRA and atom target paths differ: "
            f"lora={tuple(updates)}, atoms={tuple(atoms)}"
        )
    coefficients: OrderedDict[str, Tensor] = OrderedDict()
    layers: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for name in updates:
        solution, layer = solve_matrix_projection(
            updates[name],
            atoms[name][0],
            atoms[name][1],
            atom_scaling=atom_scaling,
            top_k=top_k,
        )
        coefficients[name] = solution
        layers[name] = layer
    return coefficients, {
        "layers": layers,
        "all_atoms": aggregate_layer_errors(layers, field="all_atoms"),
        "top_k": aggregate_layer_errors(layers, field="top_k"),
    }


def _load_compact_payload(directory: str | Path, component: str) -> Mapping[str, Any]:
    if component not in {"adapter", "atoms", "coefficients", "heads"}:
        raise ValueError(f"unsupported compact component {component!r}")
    path = Path(directory) / f"{component}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("component") != component
    ):
        raise ValueError(f"invalid compact component payload: {path}")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"compact component has no state_dict: {path}")
    return payload


def load_compact_component(directory: str | Path, component: str) -> OrderedDict[str, Any]:
    """Load and validate one compact component without accepting base weights."""

    payload = _load_compact_payload(directory, component)
    state = payload["state_dict"]
    return OrderedDict(state)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_adapter_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for prefix in EXPECTED_STATE_PREFIXES:
        shapes[f"{prefix}.lora_a.weight"] = (4, 128)
        shapes[f"{prefix}.lora_b.weight"] = (128, 4)
    return shapes


def _expected_atom_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for prefix in EXPECTED_STATE_PREFIXES:
        shapes[f"{prefix}.atom_u"] = (DEFAULT_CAPACITY, 128)
        shapes[f"{prefix}.atom_v"] = (DEFAULT_CAPACITY, 128)
    return shapes


def _expected_coefficient_shapes() -> dict[str, tuple[int, ...]]:
    return {
        f"{prefix}.coefficients": (1, DEFAULT_CAPACITY)
        for prefix in EXPECTED_STATE_PREFIXES
    }


def _expected_head_shapes(task: str) -> dict[str, tuple[int, ...]]:
    return {f"heads.{task}.weight": (2, 128), f"heads.{task}.bias": (2,)}


def _validate_exact_component(
    directory: Path,
    component: str,
    *,
    expected_metadata: Mapping[str, Any],
    expected_tensor_shapes: Mapping[str, tuple[int, ...]],
    expected_extra_state_tasks: Sequence[str] | None = None,
) -> OrderedDict[str, Any]:
    payload = _load_compact_payload(directory, component)
    if payload.get("metadata") != dict(expected_metadata):
        raise ValueError(f"{component} checkpoint metadata is not exact: {directory}")
    state = payload["state_dict"]
    expected_extra_keys = (
        {f"{prefix}._extra_state" for prefix in EXPECTED_STATE_PREFIXES}
        if expected_extra_state_tasks is not None
        else set()
    )
    expected_keys = set(expected_tensor_shapes) | expected_extra_keys
    if set(state) != expected_keys:
        raise ValueError(
            f"{component} checkpoint state keys differ: "
            f"missing={sorted(expected_keys - set(state))}, "
            f"unexpected={sorted(set(state) - expected_keys)}"
        )
    for key, shape in expected_tensor_shapes.items():
        value = state[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{component} tensor {key!r} has an invalid shape")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{component} tensor {key!r} contains non-finite values")
    if expected_extra_state_tasks is not None:
        expected_tasks = tuple(expected_extra_state_tasks)
        for key in expected_extra_keys:
            extra = state[key]
            if not isinstance(extra, Mapping) or tuple(extra.get("task_ids", ())) != expected_tasks:
                raise ValueError(f"{component} extra state {key!r} has the wrong task IDs")
    return OrderedDict(state)


def _validate_checkpoint_bundle(
    record: Mapping[str, Any],
    directory: Path,
    *,
    expected_metadata: Mapping[str, Any],
    expected_components: Mapping[str, Mapping[str, tuple[int, ...]]],
    extra_state_tasks: Sequence[str] | None = None,
    name: str,
) -> None:
    expected_parameters = {
        component: sum(math.prod(shape) for shape in shapes.values())
        for component, shapes in expected_components.items()
    }
    _validate_compact_checkpoint(
        record,
        directory,
        required_components=frozenset(expected_components),
        expected_tensor_parameters=expected_parameters,
        expected_metadata=expected_metadata,
        name=name,
    )
    for component, shapes in expected_components.items():
        _validate_exact_component(
            directory,
            component,
            expected_metadata=expected_metadata,
            expected_tensor_shapes=shapes,
            expected_extra_state_tasks=(
                extra_state_tasks if component == "atoms" else None
            ),
        )


def validate_cross_transfer_summary(summary_path: str | Path) -> Mapping[str, Any]:
    """Load a complete crossed-transfer summary before projection orchestration."""

    path = Path(summary_path)
    summary = read_json(summary_path)
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "validation_cross_transfer",
        "status": "complete",
        "predeclared_before_live_training": True,
        "targets": list(H1_TASKS),
        "seeds": list(H1_CONFIRMATORY_SEEDS),
        "cell_count": len(H1_TASKS) * len(H1_CONFIRMATORY_SEEDS),
    }
    if any(summary.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError("crossed-transfer summary violates the locked complete-grid schema")
    stored_root = summary.get("output_root")
    if not isinstance(stored_root, str) or not _same_path(stored_root, path.parent):
        raise ValueError("crossed-transfer summary output_root does not contain the summary")
    expected_protocol_path = Path(stored_root) / CROSS_TRANSFER_PROTOCOL_FILENAME
    if not isinstance(summary.get("protocol"), str) or not _same_path(
        summary["protocol"], expected_protocol_path
    ):
        raise ValueError("crossed-transfer summary points to the wrong protocol")
    if not expected_protocol_path.is_file():
        raise FileNotFoundError(expected_protocol_path)
    cross_protocol = read_json(expected_protocol_path)
    try:
        baseline = ExperimentConfig.from_mapping(cross_protocol["baseline_config"])
        atoms = ExperimentConfig.from_mapping(cross_protocol["atom_config"])
        core_root = Path(cross_protocol["core_results_root"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("cross-transfer protocol lacks valid locked configurations") from error
    expected_protocol = build_cross_transfer_protocol(
        baseline, atoms, Path(stored_root), core_root
    )
    if cross_protocol != expected_protocol:
        raise ValueError("cross-transfer protocol content differs from its locked design")

    artifact_grid = summary.get("artifacts_by_target_seed")
    if not isinstance(artifact_grid, Mapping) or set(artifact_grid) != set(H1_TASKS):
        raise ValueError("cross-transfer summary lacks the exact target artifact grid")
    cells: list[Mapping[str, Any]] = []
    for task in H1_TASKS:
        seed_entries = artifact_grid[task]
        if not isinstance(seed_entries, Mapping) or set(seed_entries) != {
            str(seed) for seed in H1_CONFIRMATORY_SEEDS
        }:
            raise ValueError(f"cross-transfer summary has an incomplete seed grid for {task}")
        for seed in H1_CONFIRMATORY_SEEDS:
            entry = seed_entries[str(seed)]
            if not isinstance(entry, Mapping):
                raise ValueError(f"cross-transfer artifact entry is invalid for {task}/{seed}")
            cell_path = Path(str(entry.get("cell_result", "")))
            if not cell_path.is_file():
                raise FileNotFoundError(cell_path)
            cell = read_json(cell_path)
            validate_cross_transfer_cell(cell, expected_target=task, expected_seed=seed)
            if cell.get("artifacts") != entry:
                raise ValueError(f"cross-transfer summary/cell artifacts differ for {task}/{seed}")
            for field, value in entry.items():
                artifact = Path(value)
                if field.endswith("_directory"):
                    if not artifact.is_dir():
                        raise FileNotFoundError(artifact)
                elif not artifact.is_file():
                    raise FileNotFoundError(artifact)
            cells.append(cell)
    rebuilt = build_cross_transfer_summary(cells, output_root=Path(stored_root))
    if summary != rebuilt:
        raise ValueError("cross-transfer summary metrics do not recompute from its 15 cells")
    return summary


def apply_projection_coefficients(
    model: torch.nn.Module,
    task: str,
    coefficients: Mapping[str, Tensor],
) -> None:
    """Install one oracle coefficient row into every path-aligned atom layer."""

    layers = OrderedDict(
        (name, module) for name, module in model.named_modules() if isinstance(module, AtomLinear)
    )
    if tuple(layers) != tuple(coefficients):
        raise ValueError(
            "projection coefficient paths do not match model atom paths: "
            f"model={tuple(layers)}, coefficients={tuple(coefficients)}"
        )
    with torch.no_grad():
        for name, layer in layers.items():
            if task not in layer.task_to_index:
                raise ValueError(f"atom layer {name!r} has no coefficient row for {task!r}")
            values = coefficients[name]
            if values.shape != (layer.atom_count,):
                raise ValueError(
                    f"coefficient shape mismatch at {name!r}: "
                    f"{tuple(values.shape)} != {(layer.atom_count,)}"
                )
            layer.coefficients[layer.task_to_index[task]].copy_(
                values.to(dtype=layer.coefficients.dtype, device=layer.coefficients.device)
            )


def _primary_score(evaluation: Mapping[str, Any]) -> float:
    try:
        score = float(evaluation["metrics"]["primary_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("evaluation lacks a primary score") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("primary score must be finite and in [0, 1]")
    return score


def _validate_fresh_lora_record(
    record: Mapping[str, Any],
    *,
    config: ExperimentConfig,
    directory: Path,
    task: str,
    seed: int,
    provenance: Mapping[str, Any],
) -> None:
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "system": "independent_lora",
        "run_kind": "confirmatory",
        "task": task,
        "seed": seed,
        "rank": 4,
        "model": MODEL_NAME,
    }
    if any(record.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("fresh LoRA identity does not match the locked core baseline")
    revision = record.get("model_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("fresh LoRA record lacks the locked model revision")
    if config.seed != seed or config.experiment_name != "independent_lora":
        raise ValueError("fresh LoRA validator received the wrong baseline configuration")
    if record.get("resolved_config") != config.to_dict():
        raise ValueError("fresh LoRA resolved configuration differs from the projection baseline")
    if record.get("target_modules") != list(EXPECTED_TARGET_MODULES):
        raise ValueError("fresh LoRA target paths differ from the locked BERT-tiny paths")
    if record.get("target_dimensions") != EXPECTED_TARGET_DIMENSIONS:
        raise ValueError("fresh LoRA target dimensions differ from the locked BERT-tiny shapes")
    _validate_environment(record, "fresh LoRA")
    stored_provenance = record.get("dataset_provenance")
    if isinstance(stored_provenance, Mapping) and task in stored_provenance:
        stored_provenance = stored_provenance[task]
    if stored_provenance != provenance:
        raise ValueError("fresh LoRA and projection data provenance differ")
    counts = record.get("parameter_counts")
    if not isinstance(counts, Mapping) or any(
        counts.get(key) != value for key, value in EXPECTED_FRESH_PARAMETER_COUNTS.items()
    ):
        raise ValueError("fresh LoRA parameter accounting differs from the locked architecture")
    if record.get("active_adapter_operations_per_token") != 4096:
        raise ValueError("fresh LoRA active adapter operation count is invalid")
    for field in ("best", "final"):
        evaluation = record.get(field)
        if not isinstance(evaluation, Mapping):
            raise ValueError(f"fresh LoRA lacks its {field} evaluation")
        _validate_saved_evaluation(task, evaluation, f"fresh LoRA {field}")
        if evaluation.get("examples") != _selected_validation_examples(
            {task: provenance}, task
        ):
            raise ValueError(f"fresh LoRA {field} uses the wrong validation size")
        _primary_score(evaluation)
    runtime = record.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("fresh LoRA lacks runtime provenance")
    elapsed = runtime.get("elapsed_seconds")
    peak = runtime.get("peak_rss_bytes")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
        or isinstance(peak, bool)
        or not isinstance(peak, int)
        or peak <= 0
    ):
        raise ValueError("fresh LoRA runtime provenance is invalid")
    _validate_checkpoint_bundle(
        record,
        directory,
        expected_metadata={"system": "independent_lora", "seed": seed, "task": task},
        expected_components={
            "adapter": _expected_adapter_shapes(),
            "heads": _expected_head_shapes(task),
        },
        name="fresh LoRA",
    )


def _evaluate_projected_dictionary(
    config: ExperimentConfig,
    *,
    task: str,
    validation_loader: Any,
    fresh_lora_directory: Path,
    dictionary_directory: Path,
    output_directory: Path,
    dictionary_kind: str,
) -> dict[str, Any]:
    set_seed(config.seed)
    model, target_names = build_atom_model(
        config,
        (task,),
        atom_count=DEFAULT_CAPACITY,
        freeze_atoms=True,
    )
    dictionary = copy_frozen_atom_dictionary(model, dictionary_directory)
    head_state = load_compact_component(fresh_lora_directory, "heads")
    expected_heads = _expected_head_shapes(task)
    if set(head_state) != set(expected_heads) or any(
        not isinstance(head_state[key], Tensor)
        or tuple(head_state[key].shape) != shape
        for key, shape in expected_heads.items()
    ):
        raise ValueError("fresh head checkpoint does not contain the exact target head")
    load_adapter_state_dict(model, head_state, include_heads=True, strict=False)
    installed_state = model.state_dict()
    if any(not torch.equal(installed_state[key].cpu(), value.cpu()) for key, value in head_state.items()):
        raise ValueError("fresh LoRA head did not install exactly into the projection model")

    projection_started = time.perf_counter()
    lora_state = load_compact_component(fresh_lora_directory, "adapter")
    atom_state = load_compact_component(dictionary_directory, "atoms")
    coefficients, projection = project_lora_state_onto_atoms(
        lora_state,
        atom_state,
        lora_scaling=config.lora_alpha / config.lora_rank,
        atom_scaling=config.atom_scaling,
        top_k=DEFAULT_TOP_K,
    )
    apply_projection_coefficients(model, task, coefficients)
    projection_seconds = time.perf_counter() - projection_started

    model.set_active_task(task)
    started = time.perf_counter()
    all_atoms = evaluate(
        model,
        validation_loader,
        config.device,
        task_id=None,
        metric_fn=partial(compute_task_metrics, task),
        scalar_metric_name="primary_score",
    )
    all_seconds = time.perf_counter() - started
    model.set_atom_top_k(DEFAULT_TOP_K)
    try:
        started = time.perf_counter()
        top_k = evaluate(
            model,
            validation_loader,
            config.device,
            task_id=None,
            metric_fn=partial(compute_task_metrics, task),
            scalar_metric_name="primary_score",
        )
        top_seconds = time.perf_counter() - started
    finally:
        model.clear_atom_top_k()

    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = save_compact_checkpoint(
        model,
        output_directory,
        metadata={
            "system": f"oracle_{dictionary_kind}_atom_span",
            "seed": config.seed,
            "task": task,
            "derived_not_trained": True,
        },
    )
    counts = categorized_parameter_counts(model)
    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "system": f"oracle_{dictionary_kind}_atom_span",
        "dictionary_kind": dictionary_kind,
        "derived_not_trained": True,
        "task": task,
        "seed": config.seed,
        "target_modules": target_names,
        "target_dimensions": {
            name: [
                model.encoder.get_submodule(name).out_features,
                model.encoder.get_submodule(name).in_features,
            ]
            for name in target_names
        },
        "model": config.base_model,
        "model_revision": getattr(model.encoder.config, "_commit_hash", None),
        "dictionary": dictionary,
        "source_dictionary_directory": str(dictionary_directory),
        "fresh_lora_directory": str(fresh_lora_directory),
        "projection": projection,
        "all_atoms": all_atoms.to_dict(include_outputs=True),
        "top_k": top_k.to_dict(include_outputs=True),
        "top_k_value": DEFAULT_TOP_K,
        "parameter_counts": counts,
        "active_adapter_operations_per_token": {
            "all_atoms": active_adapter_operations(model, active_atoms=DEFAULT_CAPACITY),
            "top_k": active_adapter_operations(model, active_atoms=DEFAULT_TOP_K),
        },
        "checkpoint": checkpoint,
        "inference_seconds": {
            "all_atoms": all_seconds,
            "top_k": top_seconds,
        },
        "projection_seconds": projection_seconds,
        "resolved_config": config.to_dict(),
        "environment": environment_record(),
    }
    write_json(output_directory / "metrics.json", record)
    return record


def _artifact_entry(
    summary: Mapping[str, Any],
    target: str,
    seed: int,
) -> Mapping[str, Any]:
    try:
        target_entries = summary["artifacts_by_target_seed"][target]
        entry = target_entries.get(str(seed), target_entries.get(seed))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"cross-transfer summary lacks artifacts for {target}/seed {seed}") from error
    if not isinstance(entry, Mapping):
        raise ValueError(f"invalid cross-transfer artifact entry for {target}/seed {seed}")
    return entry


def _required_path(entry: Mapping[str, Any], *keys: str) -> Path:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.exists():
                return path
    raise FileNotFoundError(f"none of the required artifact paths exist: {keys!r}")


def _validate_projection_inputs(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    *,
    task: str,
    prepared: PreparedData,
    fresh_lora_directory: Path,
    source_dictionary_directory: Path,
    random_dictionary_directory: Path,
    cross_cell_path: Path,
) -> dict[str, Any]:
    """Validate and bind every upstream artifact used by one oracle cell."""

    if baseline_config.experiment_name != "independent_lora":
        raise ValueError("projection baseline config must select independent_lora")
    if atom_config.experiment_name != "shared_atoms":
        raise ValueError("projection atom config must select shared_atoms")
    baseline_config.validate_h1_contract()
    atom_config.validate_h1_contract()
    if baseline_config.seed != atom_config.seed:
        raise ValueError("baseline and atom projection seeds differ")
    seed = atom_config.seed
    if task not in H1_TASKS or task not in prepared.provenance:
        raise ValueError("projection target or prepared provenance is invalid")

    cross_cell = read_json(cross_cell_path)
    validate_cross_transfer_cell(cross_cell, expected_target=task, expected_seed=seed)
    artifacts = cross_cell["artifacts"]
    exact_paths = {
        "cell_result": cross_cell_path,
        "source_checkpoint_directory": source_dictionary_directory,
        "matched_random_transfer_directory": random_dictionary_directory,
        "strict_core_lora_record": fresh_lora_directory / "metrics.json",
    }
    for field, expected in exact_paths.items():
        if not _same_path(artifacts[field], expected):
            raise ValueError(f"projection input {field} is not bound to its cross-transfer cell")
    if cross_cell.get("resolved_configs") != {
        "baseline": baseline_config.to_dict(),
        "atoms": atom_config.to_dict(),
    }:
        raise ValueError("cross-transfer cell configurations differ from the projection cell")
    target_provenance = {task: prepared.provenance[task]}
    if cross_cell.get("target_dataset_provenance") != prepared.provenance[task]:
        raise ValueError("cross-transfer target data provenance differs from projection data")
    source_tasks = tuple(value for value in H1_TASKS if value != task)
    source_provenance = {value: prepared.provenance[value] for value in source_tasks}

    source_record = read_json(artifacts["source_record"])
    _validate_source_record(
        source_record,
        atom_config,
        source_tasks,
        source_dictionary_directory,
        source_provenance,
    )
    _validate_exact_component(
        source_dictionary_directory,
        "atoms",
        expected_metadata={
            "system": "shared_atoms",
            "seed": seed,
            "tasks": list(source_tasks),
            "atom_count": DEFAULT_CAPACITY,
        },
        expected_tensor_shapes=_expected_atom_shapes(),
        expected_extra_state_tasks=source_tasks,
    )

    learned_record = read_json(artifacts["learned_transfer_record"])
    _validate_frozen_atom_record(
        learned_record,
        atom_config,
        task,
        Path(artifacts["learned_transfer_directory"]),
        target_provenance,
        dictionary_checkpoint=source_dictionary_directory,
    )
    random_record = read_json(artifacts["matched_random_transfer_record"])
    _validate_frozen_atom_record(
        random_record,
        atom_config,
        task,
        random_dictionary_directory,
        target_provenance,
        dictionary_checkpoint=None,
    )
    _validate_exact_component(
        random_dictionary_directory,
        "atoms",
        expected_metadata={
            "experiment": "validation_cross_transfer",
            "system": "matched_random_frozen_atoms",
            "target": task,
            "seed": seed,
        },
        expected_tensor_shapes=_expected_atom_shapes(),
        expected_extra_state_tasks=(task,),
    )
    head_record = read_json(artifacts["head_only_record"])
    _validate_head_only_record(
        head_record,
        baseline_config,
        task,
        Path(artifacts["head_only_directory"]),
        target_provenance,
    )
    fresh_record = read_json(fresh_lora_directory / "metrics.json")
    _validate_fresh_lora_record(
        fresh_record,
        config=baseline_config,
        directory=fresh_lora_directory,
        task=task,
        seed=seed,
        provenance=prepared.provenance[task],
    )

    score_bindings = {
        "fresh_lora": _primary_score(fresh_record["best"]),
        "learned_frozen_atoms": _primary_score(learned_record["top4"]),
        "matched_random_frozen_atoms": _primary_score(random_record["top4"]),
        "head_only": _primary_score(head_record["best"]),
    }
    for system, expected in score_bindings.items():
        observed = float(cross_cell["systems"][system]["primary_score"])
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"cross-transfer {system} score differs from its raw record")
    revisions = {
        fresh_record.get("model_revision"),
        source_record.get("model_revision"),
        learned_record.get("model_revision"),
        random_record.get("model_revision"),
        head_record.get("model_revision"),
        cross_cell.get("model_identity", {}).get("model_revision"),
    }
    if len(revisions) != 1 or None in revisions:
        raise ValueError("projection prerequisites do not share one model revision")
    current_environment = environment_record()
    if any(
        record.get("environment") != current_environment
        for record in (
            fresh_record,
            source_record,
            learned_record,
            random_record,
            head_record,
        )
    ):
        raise ValueError("projection prerequisites were produced in a different environment")
    learned_digest = learned_record.get("dictionary_sha256")
    random_digest = random_record.get("dictionary_sha256")
    if (
        not isinstance(learned_digest, str)
        or not isinstance(random_digest, str)
        or learned_digest == random_digest
    ):
        raise ValueError("learned and random dictionary digests are invalid or aliased")
    return {
        "cross_cell": cross_cell,
        "fresh_record": fresh_record,
        "source_record": source_record,
        "learned_record": learned_record,
        "random_record": random_record,
        "head_record": head_record,
        "learned_dictionary_sha256": learned_digest,
        "random_dictionary_sha256": random_digest,
    }


def run_projection_cell(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    *,
    task: str,
    prepared: PreparedData,
    fresh_lora_directory: Path,
    source_dictionary_directory: Path,
    random_dictionary_directory: Path,
    cross_cell_path: Path,
    protocol_path: Path,
    output_directory: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run or resume both learned- and random-span oracle projections."""

    cell_started = time.perf_counter()
    inputs = _validate_projection_inputs(
        baseline_config,
        atom_config,
        task=task,
        prepared=prepared,
        fresh_lora_directory=fresh_lora_directory,
        source_dictionary_directory=source_dictionary_directory,
        random_dictionary_directory=random_dictionary_directory,
        cross_cell_path=cross_cell_path,
    )
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol_record = read_json(protocol_path)
    protocol_digest = _json_sha256(protocol_record)
    cell_path = output_directory / "cell_result.json"
    if cell_path.is_file() and not force:
        cached = read_json(cell_path)
        validate_projection_cell(
            cached,
            expected_task=task,
            expected_seed=atom_config.seed,
            require_files=True,
            expected_baseline_config=baseline_config,
            expected_atom_config=atom_config,
            expected_provenance=prepared.provenance[task],
            expected_cross_cell=cross_cell_path,
            expected_fresh_directory=fresh_lora_directory,
            expected_source_directory=source_dictionary_directory,
            expected_random_directory=random_dictionary_directory,
            expected_protocol=protocol_path,
        )
        return cached
    seed = atom_config.seed
    cross_cell = inputs["cross_cell"]
    source_tasks = tuple(cross_cell.get("source_tasks", ()))
    fresh_record = inputs["fresh_record"]
    _, validation_loaders = build_loaders(prepared, atom_config, tasks=(task,))
    learned = _evaluate_projected_dictionary(
        atom_config,
        task=task,
        validation_loader=validation_loaders[task],
        fresh_lora_directory=fresh_lora_directory,
        dictionary_directory=source_dictionary_directory,
        output_directory=output_directory / "learned_span",
        dictionary_kind="learned",
    )
    random = _evaluate_projected_dictionary(
        atom_config,
        task=task,
        validation_loader=validation_loaders[task],
        fresh_lora_directory=fresh_lora_directory,
        dictionary_directory=random_dictionary_directory,
        output_directory=output_directory / "random_span",
        dictionary_kind="random",
    )
    expected_labels = list(fresh_record["best"]["labels"])
    for name, system in (("learned", learned), ("random", random)):
        if system["model_revision"] != fresh_record["model_revision"]:
            raise ValueError(f"{name} span model revision differs from fresh LoRA")
        for evaluation in ("all_atoms", "top_k"):
            if system[evaluation]["labels"] != expected_labels:
                raise ValueError(f"{name} {evaluation} validation labels differ from fresh LoRA")
    if learned["dictionary"].get("sha256") != inputs["learned_dictionary_sha256"]:
        raise ValueError("projected learned dictionary digest differs from cross transfer")
    if random["dictionary"].get("sha256") != inputs["random_dictionary_sha256"]:
        raise ValueError("projected random dictionary digest differs from cross transfer")

    fresh_score = _primary_score(fresh_record["best"])
    if fresh_score <= 0.0:
        raise ValueError("fresh LoRA score must be positive for quality retention")
    learned_directory = output_directory / "learned_span"
    random_directory = output_directory / "random_span"
    cross_artifacts = cross_cell["artifacts"]
    artifacts = {
        "cell_result": str(cell_path),
        "protocol": str(protocol_path),
        "cross_transfer_cell": str(cross_cell_path),
        "fresh_lora_directory": str(fresh_lora_directory),
        "fresh_lora_record": str(fresh_lora_directory / "metrics.json"),
        "fresh_lora_adapter": str(fresh_lora_directory / "adapter.pt"),
        "fresh_lora_heads": str(fresh_lora_directory / "heads.pt"),
        "learned_source_directory": str(source_dictionary_directory),
        "learned_source_record": str(cross_artifacts["source_record"]),
        "learned_source_atoms": str(source_dictionary_directory / "atoms.pt"),
        "random_dictionary_directory": str(random_dictionary_directory),
        "random_dictionary_record": str(cross_artifacts["matched_random_transfer_record"]),
        "random_dictionary_atoms": str(random_dictionary_directory / "atoms.pt"),
        "learned_projection_directory": str(learned_directory),
        "learned_projection_record": str(learned_directory / "metrics.json"),
        "learned_projection_atoms": str(learned_directory / "atoms.pt"),
        "learned_projection_coefficients": str(learned_directory / "coefficients.pt"),
        "learned_projection_heads": str(learned_directory / "heads.pt"),
        "random_projection_directory": str(random_directory),
        "random_projection_record": str(random_directory / "metrics.json"),
        "random_projection_atoms": str(random_directory / "atoms.pt"),
        "random_projection_coefficients": str(random_directory / "coefficients.pt"),
        "random_projection_heads": str(random_directory / "heads.pt"),
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "complete",
        "task": task,
        "seed": seed,
        "source_tasks": list(source_tasks),
        "cross_transfer_cell": str(cross_cell_path),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "fresh_lora": {
            "directory": str(fresh_lora_directory),
            "record": str(fresh_lora_directory / "metrics.json"),
            "score": fresh_score,
            "persistent_parameters": fresh_record["parameter_counts"][
                "persistent_adaptation_parameters"
            ],
            "checkpoint_bytes": fresh_record["checkpoint"]["total_bytes"],
        },
        "systems": {
            "learned_span": learned,
            "random_span": random,
        },
        "quality_retention": {
            "learned_all_atoms": _primary_score(learned["all_atoms"]) / fresh_score,
            "learned_top_k": _primary_score(learned["top_k"]) / fresh_score,
            "random_all_atoms": _primary_score(random["all_atoms"]) / fresh_score,
            "random_top_k": _primary_score(random["top_k"]) / fresh_score,
        },
        "dataset_provenance": prepared.provenance[task],
        "baseline_resolved_config": baseline_config.to_dict(),
        "atom_resolved_config": atom_config.to_dict(),
        "locked_budget": {
            "capacity": DEFAULT_CAPACITY,
            "top_k": DEFAULT_TOP_K,
            "lora_rank": baseline_config.lora_rank,
            "lora_scaling": baseline_config.lora_alpha / baseline_config.lora_rank,
            "atom_scaling": atom_config.atom_scaling,
            "validation_examples": len(expected_labels),
        },
        "model_identity": {
            "model": MODEL_NAME,
            "model_revision": fresh_record["model_revision"],
            "target_dimensions": EXPECTED_TARGET_DIMENSIONS,
        },
        "dictionary_digests": {
            "learned": inputs["learned_dictionary_sha256"],
            "random": inputs["random_dictionary_sha256"],
        },
        "parameter_accounting": {
            "fresh_lora_persistent": fresh_record["parameter_counts"][
                "persistent_adaptation_parameters"
            ],
            "learned_span_persistent": learned["parameter_counts"][
                "persistent_adaptation_parameters"
            ],
            "random_span_persistent": random["parameter_counts"][
                "persistent_adaptation_parameters"
            ],
            "learned_span_marginal_coefficients_and_head": 290,
            "random_span_marginal_coefficients_and_head": 290,
        },
        "checkpoint_bytes": {
            "fresh_lora": fresh_record["checkpoint"]["total_bytes"],
            "learned_span": learned["checkpoint"]["total_bytes"],
            "random_span": random["checkpoint"]["total_bytes"],
        },
        "active_adapter_operations_per_token": {
            "fresh_lora": fresh_record["active_adapter_operations_per_token"],
            "learned_all8": learned["active_adapter_operations_per_token"]["all_atoms"],
            "learned_top4": learned["active_adapter_operations_per_token"]["top_k"],
            "random_all8": random["active_adapter_operations_per_token"]["all_atoms"],
            "random_top4": random["active_adapter_operations_per_token"]["top_k"],
        },
        "runtime": {
            "cell_elapsed_seconds": time.perf_counter() - cell_started,
            "learned_projection_seconds": learned["projection_seconds"],
            "random_projection_seconds": random["projection_seconds"],
            "learned_inference_seconds": learned["inference_seconds"],
            "random_inference_seconds": random["inference_seconds"],
        },
        "environment": environment_record(),
        "artifacts": artifacts,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(cell_path, record)
    validate_projection_cell(
        record,
        expected_task=task,
        expected_seed=seed,
        require_files=True,
        expected_baseline_config=baseline_config,
        expected_atom_config=atom_config,
        expected_provenance=prepared.provenance[task],
        expected_cross_cell=cross_cell_path,
        expected_fresh_directory=fresh_lora_directory,
        expected_source_directory=source_dictionary_directory,
        expected_random_directory=random_dictionary_directory,
        expected_protocol=protocol_path,
    )
    return record


def _validate_saved_evaluation(task: str, evaluation: Mapping[str, Any], name: str) -> None:
    predictions = evaluation.get("predictions")
    labels = evaluation.get("labels")
    examples = evaluation.get("examples")
    if (
        not isinstance(predictions, Sequence)
        or isinstance(predictions, (str, bytes))
        or not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or isinstance(examples, bool)
        or not isinstance(examples, int)
        or examples <= 0
        or len(predictions) != examples
        or len(labels) != examples
    ):
        raise ValueError(f"{name} lacks complete raw predictions and labels")
    recomputed = compute_task_metrics(
        task,
        torch.tensor(predictions, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
    for metric, value in recomputed.items():
        observed = float(evaluation.get("metrics", {}).get(metric, math.nan))
        if not math.isclose(observed, float(value), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name} metric {metric!r} does not match raw outputs")


def validate_projection_cell(
    cell: Mapping[str, Any],
    *,
    expected_task: str | None = None,
    expected_seed: int | None = None,
    require_files: bool = True,
    expected_baseline_config: ExperimentConfig | None = None,
    expected_atom_config: ExperimentConfig | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
    expected_cross_cell: Path | None = None,
    expected_fresh_directory: Path | None = None,
    expected_source_directory: Path | None = None,
    expected_random_directory: Path | None = None,
    expected_protocol: Path | None = None,
) -> None:
    """Validate a completed oracle cell before resume or aggregation.

    The optional ``expected_*`` arguments enforce the specification's strict reuse
    rule: a cached cell may only be resumed when its resolved configuration,
    dataset provenance, and every referenced component path match the run being
    requested. A filename alone is not evidence that a cell is reusable.
    """

    if cell.get("schema_version") != SCHEMA_VERSION or cell.get("status") != "complete":
        raise ValueError("projection cell is not a complete current-schema record")
    task = cell.get("task")
    seed = cell.get("seed")
    if task not in H1_TASKS or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("projection cell has an invalid task or seed")
    if expected_task is not None and task != expected_task:
        raise ValueError("projection cell task does not match requested task")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("projection cell seed does not match requested seed")
    expected_sources = [value for value in H1_TASKS if value != task]
    if cell.get("source_tasks") != expected_sources:
        raise ValueError("projection source tasks do not exactly exclude the target")
    if not isinstance(cell.get("dataset_provenance"), Mapping):
        raise ValueError("projection cell lacks dataset provenance")
    for config_name in ("baseline_resolved_config", "atom_resolved_config"):
        try:
            config = ExperimentConfig.from_mapping(cell[config_name])
            config.validate_h1_contract()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"projection cell has invalid {config_name}") from error
        if config.seed != seed:
            raise ValueError(f"projection cell {config_name} has the wrong seed")
    for config_name, expected_config in (
        ("baseline_resolved_config", expected_baseline_config),
        ("atom_resolved_config", expected_atom_config),
    ):
        if expected_config is not None and _json_sha256(cell[config_name]) != _json_sha256(
            expected_config.to_dict()
        ):
            raise ValueError(f"projection cell {config_name} differs from the requested config")
    if expected_provenance is not None and _json_sha256(
        cell["dataset_provenance"]
    ) != _json_sha256(dict(expected_provenance)):
        raise ValueError("projection cell dataset provenance differs from the prepared selection")
    for label, stored, expected_path in (
        ("cross transfer cell", cell.get("cross_transfer_cell"), expected_cross_cell),
        ("protocol", cell.get("protocol"), expected_protocol),
        (
            "fresh LoRA directory",
            cell.get("fresh_lora", {}).get("directory"),
            expected_fresh_directory,
        ),
        (
            "learned source directory",
            cell.get("artifacts", {}).get("learned_source_directory"),
            expected_source_directory,
        ),
        (
            "random dictionary directory",
            cell.get("artifacts", {}).get("random_dictionary_directory"),
            expected_random_directory,
        ),
    ):
        if expected_path is None:
            continue
        if stored is None or Path(stored) != Path(expected_path):
            raise ValueError(f"projection cell {label} differs from the requested component")
    systems = cell.get("systems")
    if not isinstance(systems, Mapping) or set(systems) != {"learned_span", "random_span"}:
        raise ValueError("projection cell systems are incomplete")
    labels_by_system: list[list[int]] = []
    for system_name, system in systems.items():
        if not isinstance(system, Mapping):
            raise ValueError(f"invalid projection system {system_name}")
        expected_kind = "learned" if system_name == "learned_span" else "random"
        if system.get("dictionary_kind") != expected_kind:
            raise ValueError(f"projection system {system_name} has the wrong dictionary kind")
        if system.get("model_revision") != cell["systems"]["learned_span"].get(
            "model_revision"
        ):
            raise ValueError("learned and random projection model revisions differ")
        if not isinstance(system.get("environment"), Mapping):
            raise ValueError(f"projection system {system_name} lacks environment provenance")
        for evaluation_name in ("all_atoms", "top_k"):
            evaluation = system.get(evaluation_name)
            if not isinstance(evaluation, Mapping):
                raise ValueError(f"projection system {system_name} lacks {evaluation_name}")
            _validate_saved_evaluation(task, evaluation, f"{system_name} {evaluation_name}")
            labels_by_system.append(list(evaluation["labels"]))
        projection = system.get("projection")
        if not isinstance(projection, Mapping):
            raise ValueError(f"projection system {system_name} lacks matrix diagnostics")
        for field in ("all_atoms", "top_k"):
            values = projection.get(field)
            if not isinstance(values, Mapping):
                raise ValueError(f"projection system {system_name} lacks {field} errors")
            for metric in (
                "target_squared_frobenius_norm",
                "residual_squared_frobenius_norm",
                "relative_frobenius_error",
                "explained_energy",
            ):
                value = float(values.get(metric, math.nan))
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {system_name} {field} {metric}")
        if require_files:
            checkpoint = system.get("checkpoint", {}).get("paths", {})
            if not isinstance(checkpoint, Mapping) or not checkpoint:
                raise ValueError(f"projection system {system_name} lacks compact checkpoint paths")
            for component, path_value in checkpoint.items():
                path = Path(path_value)
                if component not in {"atoms", "coefficients", "heads"} or not path.is_file():
                    raise ValueError(f"missing/invalid {system_name} checkpoint component: {path}")
    if any(labels != labels_by_system[0] for labels in labels_by_system[1:]):
        raise ValueError("projection cell evaluations do not use identical validation labels")
    if require_files:
        for name in ("cross_transfer_cell",):
            if not Path(cell[name]).is_file():
                raise ValueError(f"projection prerequisite file is missing: {cell[name]}")
        fresh = cell.get("fresh_lora", {})
        if not Path(fresh.get("record", "")).is_file():
            raise ValueError("projection fresh-LoRA record is missing")


def summarize_projection_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate all 15 projection cells and apply the locked decision rules."""

    if len(cells) != len(H1_TASKS) * len(H1_CONFIRMATORY_SEEDS):
        raise ValueError("projection summary requires all 15 target/seed cells")
    for cell in cells:
        validate_projection_cell(cell, require_files=False)
    identities = {(str(cell["task"]), int(cell["seed"])) for cell in cells}
    expected = {(task, seed) for task in H1_TASKS for seed in H1_CONFIRMATORY_SEEDS}
    if identities != expected:
        raise ValueError("projection cells do not form the complete target/seed grid")

    def score(cell: Mapping[str, Any], system: str, evaluation: str) -> float:
        return _primary_score(cell["systems"][system][evaluation])

    fresh_scores = [float(cell["fresh_lora"]["score"]) for cell in cells]
    learned_all_scores = [score(cell, "learned_span", "all_atoms") for cell in cells]
    learned_top_scores = [score(cell, "learned_span", "top_k") for cell in cells]
    random_all_scores = [score(cell, "random_span", "all_atoms") for cell in cells]
    random_top_scores = [score(cell, "random_span", "top_k") for cell in cells]

    per_target: dict[str, Any] = {}
    for task in H1_TASKS:
        selected = [cell for cell in cells if cell["task"] == task]
        fresh_mean = sum(float(cell["fresh_lora"]["score"]) for cell in selected) / len(selected)
        learned_mean = sum(score(cell, "learned_span", "all_atoms") for cell in selected) / len(
            selected
        )
        random_mean = sum(score(cell, "random_span", "all_atoms") for cell in selected) / len(
            selected
        )
        per_target[task] = {
            "fresh_lora_mean": fresh_mean,
            "learned_span_all_atoms_mean": learned_mean,
            "random_span_all_atoms_mean": random_mean,
            "learned_quality_retention": learned_mean / fresh_mean,
            "learned_advantage_over_random": learned_mean - random_mean,
        }

    def weighted_error(system: str, field: str) -> dict[str, float]:
        target_squared = 0.0
        residual_squared = 0.0
        for cell in cells:
            values = cell["systems"][system]["projection"][field]
            target_squared += float(values["target_squared_frobenius_norm"])
            residual_squared += float(values["residual_squared_frobenius_norm"])
        return {
            "target_squared_frobenius_norm": target_squared,
            "residual_squared_frobenius_norm": residual_squared,
            "relative_frobenius_error": math.sqrt(residual_squared / target_squared),
            "explained_energy": 1.0 - residual_squared / target_squared,
        }

    fresh_mean = sum(fresh_scores) / len(fresh_scores)
    learned_all_mean = sum(learned_all_scores) / len(learned_all_scores)
    learned_top_mean = sum(learned_top_scores) / len(learned_top_scores)
    random_all_mean = sum(random_all_scores) / len(random_all_scores)
    random_top_mean = sum(random_top_scores) / len(random_top_scores)
    learned_error = weighted_error("learned_span", "all_atoms")
    random_error = weighted_error("random_span", "all_atoms")
    retention = learned_all_mean / fresh_mean
    minimum_target_retention = min(
        float(record["learned_quality_retention"]) for record in per_target.values()
    )
    quality_advantage = learned_all_mean - random_all_mean
    checks = {
        "aggregate_quality_retention": retention >= QUALITY_RETENTION_THRESHOLD,
        "every_target_retention": minimum_target_retention >= PER_TARGET_RETENTION_THRESHOLD,
        "lower_reconstruction_error_than_random": (
            learned_error["relative_frobenius_error"]
            < random_error["relative_frobenius_error"]
        ),
        "quality_advantage_over_random": quality_advantage >= RANDOM_QUALITY_MARGIN,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment": "oracle_lora_update_projection",
        "seeds": list(H1_CONFIRMATORY_SEEDS),
        "tasks": list(H1_TASKS),
        "cell_count": len(cells),
        "thresholds": {
            "aggregate_quality_retention": QUALITY_RETENTION_THRESHOLD,
            "per_target_quality_retention": PER_TARGET_RETENTION_THRESHOLD,
            "quality_advantage_over_random": RANDOM_QUALITY_MARGIN,
            "learned_error_must_be_strictly_lower_than_random": True,
        },
        "aggregate": {
            "fresh_lora_mean": fresh_mean,
            "learned_span_all_atoms_mean": learned_all_mean,
            "learned_span_top_k_mean": learned_top_mean,
            "random_span_all_atoms_mean": random_all_mean,
            "random_span_top_k_mean": random_top_mean,
            "learned_all_atoms_quality_retention": retention,
            "learned_all_atoms_advantage_over_random": quality_advantage,
            "minimum_target_quality_retention": minimum_target_retention,
            "learned_span_all_atoms_reconstruction": learned_error,
            "random_span_all_atoms_reconstruction": random_error,
        },
        "per_target": per_target,
        "decision_checks": checks,
        "strong_learned_span_support": all(checks.values()),
        "cells": list(cells),
    }


def render_projection_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    verdict = "PASS" if summary["strong_learned_span_support"] else "FAIL"
    lines = [
        "# Oracle held-out LoRA-to-atom-span projection",
        "",
        f"Strong learned-span support: **{verdict}**.",
        "",
        "## Aggregate results",
        "",
        "| Fresh LoRA | Learned all 8 | Learned top 4 | Random all 8 | "
        "Learned retention | Learned - random |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {aggregate['fresh_lora_mean']:.6f} | "
        f"{aggregate['learned_span_all_atoms_mean']:.6f} | "
        f"{aggregate['learned_span_top_k_mean']:.6f} | "
        f"{aggregate['random_span_all_atoms_mean']:.6f} | "
        f"{aggregate['learned_all_atoms_quality_retention']:.3%} | "
        f"{aggregate['learned_all_atoms_advantage_over_random']:+.6f} |",
        "",
        "| Span | Relative Frobenius error | Explained energy |",
        "|---|---:|---:|",
        f"| Learned | {aggregate['learned_span_all_atoms_reconstruction']['relative_frobenius_error']:.6f} | "
        f"{aggregate['learned_span_all_atoms_reconstruction']['explained_energy']:.3%} |",
        f"| Random | {aggregate['random_span_all_atoms_reconstruction']['relative_frobenius_error']:.6f} | "
        f"{aggregate['random_span_all_atoms_reconstruction']['explained_energy']:.3%} |",
        "",
        "## Per-target all-eight oracle quality",
        "",
        "| Target | Fresh LoRA | Learned span | Random span | Retention |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in H1_TASKS:
        row = summary["per_target"][task]
        lines.append(
            f"| {task.upper()} | {row['fresh_lora_mean']:.6f} | "
            f"{row['learned_span_all_atoms_mean']:.6f} | "
            f"{row['random_span_all_atoms_mean']:.6f} | "
            f"{row['learned_quality_retention']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Locked decision checks",
            "",
            "| Check | Result |",
            "|---|---|",
        ]
    )
    for name, passed in summary["decision_checks"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "All-eight projection is the span-coverage result. Top-4 is a matched-active-compute "
            "diagnostic. Oracle coefficients are derived from target LoRA weights and are not a "
            "deployable generator.",
            "",
        ]
    )
    return "\n".join(lines)


def run_projection_suite(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    *,
    cross_transfer_root: str | Path,
    independent_root: str | Path,
    output_root: str | Path,
    force: bool = False,
) -> tuple[dict[str, Any], Path, Path]:
    """Run/resume the full target-by-seed oracle projection grid."""

    baseline_config.validate_h1_contract()
    atom_config.validate_h1_contract()
    cross_root = Path(cross_transfer_root)
    cross_summary_path = cross_root / "cross_transfer_summary.json"
    cross_summary = validate_cross_transfer_summary(cross_summary_path)
    output = Path(output_root)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "oracle_lora_update_projection",
        "status": "locked_before_live_evaluation",
        "tasks": list(H1_TASKS),
        "seeds": list(H1_CONFIRMATORY_SEEDS),
        "capacity": DEFAULT_CAPACITY,
        "top_k": DEFAULT_TOP_K,
        "target_update": "effective rank-4 LoRA scaling * B @ A",
        "atom_basis": "atom_scaling * outer(atom_u[k], atom_v[k])",
        "solver": "deterministic float64 least squares",
        "primary_evaluation": "all eight atoms with the fresh LoRA target head",
        "top_k_evaluation": "largest-magnitude four oracle coefficients",
        "thresholds": {
            "aggregate_quality_retention": QUALITY_RETENTION_THRESHOLD,
            "per_target_quality_retention": PER_TARGET_RETENTION_THRESHOLD,
            "quality_advantage_over_random": RANDOM_QUALITY_MARGIN,
            "learned_error_must_be_strictly_lower_than_random": True,
        },
        "baseline_config": baseline_config.to_dict(),
        "atom_config": atom_config.to_dict(),
        "cross_transfer_summary": str(cross_summary_path),
        "independent_root": str(Path(independent_root)),
        "output_root": str(output),
    }
    protocol_path = output / PROTOCOL_FILENAME
    if protocol_path.is_file():
        if read_json(protocol_path) != protocol:
            raise ValueError(f"existing oracle projection protocol differs: {protocol_path}")
    else:
        write_json(protocol_path, protocol)
    cells: list[dict[str, Any]] = []
    prepared_by_seed: dict[int, PreparedData] = {}
    for seed in H1_CONFIRMATORY_SEEDS:
        baseline_seed = baseline_config.with_overrides(seed=seed)
        atom_seed = atom_config.with_overrides(seed=seed)
        for task in H1_TASKS:
            destination = output / f"target_{task}" / f"seed_{seed}"
            cell_path = destination / "cell_result.json"
            if cell_path.is_file() and not force:
                print(f"Skipping completed oracle projection: target={task}, seed={seed}.", flush=True)
                cell = read_json(cell_path)
                validate_projection_cell(
                    cell,
                    expected_task=task,
                    expected_seed=seed,
                    require_files=True,
                )
                cells.append(cell)
                continue
            if seed not in prepared_by_seed:
                prepared_by_seed[seed] = prepare_data(atom_seed, tasks=H1_TASKS)
            entry = _artifact_entry(cross_summary, task, seed)
            source_directory = _required_path(
                entry,
                "source_checkpoint_directory",
                "source_learned_atoms_directory",
            )
            random_directory = _required_path(
                entry,
                "random_transfer_directory",
                "matched_random_transfer_directory",
            )
            cross_cell = _required_path(entry, "cell_result", "cell_result_path")
            fresh_directory = Path(independent_root) / f"seed_{seed}" / task
            if not fresh_directory.is_dir():
                raise FileNotFoundError(fresh_directory)
            print(f"Projecting held-out LoRA: target={task}, seed={seed}.", flush=True)
            cells.append(
                run_projection_cell(
                    baseline_seed,
                    atom_seed,
                    task=task,
                    prepared=prepared_by_seed[seed],
                    fresh_lora_directory=fresh_directory,
                    source_dictionary_directory=source_directory,
                    random_dictionary_directory=random_directory,
                    cross_cell_path=cross_cell,
                    protocol_path=protocol_path,
                    output_directory=destination,
                    force=force,
                )
            )
    summary = summarize_projection_cells(cells)
    summary["cross_transfer_summary"] = str(cross_summary_path)
    summary["protocol"] = str(protocol_path)
    summary["baseline_resolved_config"] = baseline_config.to_dict()
    summary["atom_resolved_config"] = atom_config.to_dict()
    summary["environment"] = environment_record()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(output / "oracle_projection_summary.json", summary)
    report_path = output / "oracle_projection_report.md"
    report_path.write_text(render_projection_markdown(summary), encoding="utf-8", newline="\n")
    return summary, summary_path, report_path


__all__ = [
    "DEFAULT_CAPACITY",
    "DEFAULT_TOP_K",
    "SCHEMA_VERSION",
    "aggregate_layer_errors",
    "apply_projection_coefficients",
    "atom_design_matrix",
    "deterministic_top_k_coefficients",
    "load_compact_component",
    "project_lora_state_onto_atoms",
    "render_projection_markdown",
    "run_projection_cell",
    "run_projection_suite",
    "solve_matrix_projection",
    "summarize_projection_cells",
    "validate_cross_transfer_summary",
    "validate_projection_cell",
]
