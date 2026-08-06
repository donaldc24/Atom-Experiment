"""Experiment B: crossed held-out validation of a frozen atom dictionary.

For every target task and confirmatory seed, this module trains an eight-atom
dictionary on the other four tasks, freezes it, and fits only a new target
coefficient row and classification head.  The resulting 5 x 3 grid is a
predeclared validation experiment, separate from the original H1 decision.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch import nn

from cgmoe_h1.config import H1_CONFIRMATORY_SEEDS, H1_TASKS, ExperimentConfig
from cgmoe_h1.experiments import (
    PreparedData,
    _run_metadata,
    build_atom_model,
    build_head_only_model,
    build_loaders,
    prepare_data,
    run_shared_atoms,
    save_compact_checkpoint,
)
from cgmoe_h1.followups_transfer import (
    assert_transfer_trainable_contract,
    copy_frozen_atom_dictionary,
)
from cgmoe_h1.metrics import compute_task_metrics
from cgmoe_h1.models.atoms import coefficient_l1_regularization, iter_atom_layers
from cgmoe_h1.models.injection import extract_adapter_state_dict, load_adapter_state_dict
from cgmoe_h1.training.trainer import create_adamw_optimizer, evaluate, train_single_task
from cgmoe_h1.utils.parameters import active_adapter_operations, categorized_parameter_counts
from cgmoe_h1.utils.reproducibility import set_seed
from cgmoe_h1.utils.runtime import RuntimeMonitor
from cgmoe_h1.utils.serialization import read_json, write_json


CROSS_TRANSFER_SEEDS = H1_CONFIRMATORY_SEEDS
CROSS_TRANSFER_TARGETS = H1_TASKS
PRIMARY_RETENTION_THRESHOLD = 0.95
TARGET_RETENTION_DIAGNOSTIC_THRESHOLD = 0.90
CONTROL_ADVANTAGE_DIAGNOSTIC_THRESHOLD = 0.005
MARGINAL_PARAMETER_DIAGNOSTIC_THRESHOLD = 0.10
SCHEMA_VERSION = 1
SUMMARY_FILENAME = "cross_transfer_summary.json"
REPORT_FILENAME = "cross_transfer_report.md"
PROTOCOL_FILENAME = "cross_transfer_protocol.json"
DEFAULT_OUTPUT_ROOT = Path("results/atom_validation/cross_transfer")
DEFAULT_CORE_RESULTS_ROOT = Path("results")


CellRunner: TypeAlias = Callable[
    [ExperimentConfig, ExperimentConfig, str, Path, Path, bool],
    dict[str, Any],
]


def _validate_locked_configs(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
) -> None:
    baseline_config.validate_h1_contract()
    atom_config.validate_h1_contract()
    if baseline_config.experiment_name != "independent_lora":
        raise ValueError("baseline config must select independent_lora")
    if atom_config.experiment_name != "shared_atoms":
        raise ValueError("atom config must select shared_atoms")
    if baseline_config.tasks != H1_TASKS or atom_config.tasks != H1_TASKS:
        raise ValueError(f"cross-transfer task order must be exactly {H1_TASKS!r}")
    if baseline_config.confirmatory_seeds != CROSS_TRANSFER_SEEDS:
        raise ValueError(f"confirmatory seeds must be exactly {CROSS_TRANSFER_SEEDS!r}")
    if atom_config.confirmatory_seeds != CROSS_TRANSFER_SEEDS:
        raise ValueError(f"confirmatory seeds must be exactly {CROSS_TRANSFER_SEEDS!r}")


def _optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.AdamW:
    return create_adamw_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        epsilon=config.adam_epsilon,
    )


def _parameter_digest(model: nn.Module, *, trainable: bool) -> str:
    digest = hashlib.sha256()
    matched = 0
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad != trainable:
            continue
        if not trainable and not name.endswith((".atom_u", ".atom_v")):
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}".encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        matched += 1
    if matched == 0:
        kind = "trainable" if trainable else "atom"
        raise ValueError(f"model has no {kind} parameters to hash")
    return digest.hexdigest()


def _primary_score(record: Mapping[str, Any], field: str) -> float:
    try:
        value = float(record[field]["metrics"]["primary_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"record has no {field!r} primary score") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"primary score must be finite and in [0, 1], got {value}")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty score sequence")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cannot aggregate non-finite scores")
    return math.fsum(values) / len(values)


def _validate_raw_evaluation(
    evaluation: Mapping[str, Any],
    name: str,
    *,
    task: str | None = None,
    expected_examples: int | None = None,
) -> None:
    try:
        examples = int(evaluation["examples"])
        predictions = evaluation["predictions"]
        labels = evaluation["labels"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} evaluation lacks raw outputs") from error
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise ValueError(f"{name} predictions must be a sequence")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise ValueError(f"{name} labels must be a sequence")
    if examples <= 0 or len(predictions) != examples or len(labels) != examples:
        raise ValueError(
            f"{name} raw output lengths must equal examples: "
            f"{len(predictions)}, {len(labels)}, {examples}"
        )
    if expected_examples is not None and examples != expected_examples:
        raise ValueError(
            f"{name} examples={examples} does not match locked count {expected_examples}"
        )
    if task is not None:
        recomputed = compute_task_metrics(task, predictions, labels)
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{name} evaluation lacks metrics")
        for metric, expected in recomputed.items():
            try:
                observed = float(metrics[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{name} lacks metric {metric!r}") from error
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{name} metric {metric!r} does not match raw outputs: "
                    f"{observed} != {expected}"
                )


def _runtime_dict(record: Mapping[str, Any]) -> dict[str, Any]:
    runtime = record.get("runtime", {})
    if not isinstance(runtime, Mapping):
        return {"elapsed_seconds": None, "peak_rss_bytes": None}
    return {
        "elapsed_seconds": runtime.get("elapsed_seconds"),
        "peak_rss_bytes": runtime.get("peak_rss_bytes"),
    }


def _selected_validation_examples(
    provenance: Mapping[str, Any],
    task: str,
) -> int:
    try:
        split = provenance[task]["validation"]
        selected_count = int(split["selected_count"])
        row_ids = split["selected_row_ids"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{task} provenance lacks validation row selection") from error
    if isinstance(split.get("selected_count"), bool) or selected_count <= 0:
        raise ValueError(f"{task} validation selected_count is invalid")
    if not isinstance(row_ids, Sequence) or isinstance(row_ids, (str, bytes)):
        raise ValueError(f"{task} validation row ids must be a sequence")
    if any(isinstance(row_id, bool) or not isinstance(row_id, int) for row_id in row_ids):
        raise ValueError(f"{task} validation row ids must be integers")
    if len(row_ids) != selected_count or len(set(row_ids)) != selected_count:
        raise ValueError(f"{task} validation row selection is inconsistent")
    return selected_count


def _validate_environment(record: Mapping[str, Any], name: str) -> None:
    environment = record.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError(f"{name} lacks environment provenance")
    required = {"python", "platform", "cpu_threads", "cuda_available", "packages"}
    if not required.issubset(environment):
        raise ValueError(f"{name} environment provenance is incomplete")
    if not isinstance(environment["python"], str) or not isinstance(
        environment["platform"], str
    ):
        raise ValueError(f"{name} Python/platform provenance is invalid")
    if (
        isinstance(environment["cpu_threads"], bool)
        or not isinstance(environment["cpu_threads"], int)
        or environment["cpu_threads"] <= 0
    ):
        raise ValueError(f"{name} CPU provenance is invalid")
    if not isinstance(environment["cuda_available"], bool):
        raise ValueError(f"{name} CUDA provenance is invalid")
    if not isinstance(environment["packages"], Mapping):
        raise ValueError(f"{name} package provenance must be an object")


def _validate_common_metadata(
    record: Mapping[str, Any],
    config: ExperimentConfig,
    provenance: Mapping[str, Any],
    *,
    name: str,
    expected_target_count: int,
    provenance_exact: bool = True,
) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{name} has an unsupported schema version")
    if record.get("model") != config.base_model:
        raise ValueError(f"{name} model identity does not match the locked config")
    if "model_revision" not in record:
        raise ValueError(f"{name} lacks model revision provenance")
    try:
        stored_config = ExperimentConfig.from_mapping(record["resolved_config"])
        stored_config.validate_h1_contract()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} has an invalid resolved config") from error
    if stored_config.to_dict() != config.to_dict():
        raise ValueError(f"{name} resolved config does not match the locked seed config")
    stored_provenance = record.get("dataset_provenance")
    if not isinstance(stored_provenance, Mapping) or any(
        stored_provenance.get(task) != value for task, value in provenance.items()
    ):
        raise ValueError(f"{name} dataset provenance does not match the locked rows")
    if provenance_exact and set(stored_provenance) != set(provenance):
        raise ValueError(f"{name} dataset provenance has unexpected tasks")

    target_names = record.get("target_modules")
    dimensions = record.get("target_dimensions")
    if not isinstance(target_names, Sequence) or isinstance(target_names, (str, bytes)):
        raise ValueError(f"{name} target_modules must be a sequence")
    if any(not isinstance(target, str) for target in target_names):
        raise ValueError(f"{name} target module names must be strings")
    if len(target_names) != expected_target_count or len(set(target_names)) != len(target_names):
        raise ValueError(f"{name} has the wrong number of unique target modules")
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(target_names):
        raise ValueError(f"{name} target dimensions do not match target modules")
    if expected_target_count:
        suffix_counts = {
            suffix: sum(str(target).endswith(f".{suffix}") for target in target_names)
            for suffix in config.target_modules
        }
        expected_per_suffix = expected_target_count // len(config.target_modules)
        if any(count != expected_per_suffix for count in suffix_counts.values()):
            raise ValueError(f"{name} target module suffixes do not match the locked model")
        try:
            dimensions_valid = all(
                list(dimensions[target]) == [128, 128] for target in target_names
            )
        except TypeError as error:
            raise ValueError(f"{name} target dimensions are invalid") from error
        if not dimensions_valid:
            raise ValueError(f"{name} target dimensions do not match bert-tiny")
    _validate_environment(record, name)


def _validate_exact_counts(
    record: Mapping[str, Any],
    expected: Mapping[str, int],
    *,
    name: str,
) -> None:
    counts = record.get("parameter_counts")
    if not isinstance(counts, Mapping):
        raise ValueError(f"{name} lacks parameter counts")
    for field, value in expected.items():
        if counts.get(field) != value:
            raise ValueError(
                f"{name} parameter_counts.{field}={counts.get(field)!r}; expected {value}"
            )


def _validate_compact_checkpoint(
    record: Mapping[str, Any],
    directory: Path,
    *,
    required_components: frozenset[str],
    expected_tensor_parameters: Mapping[str, int],
    expected_metadata: Mapping[str, Any],
    name: str,
) -> None:
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{name} lacks compact checkpoint metadata")
    paths = checkpoint.get("paths")
    byte_counts = checkpoint.get("bytes_by_component")
    if not isinstance(paths, Mapping) or set(paths) != required_components:
        raise ValueError(f"{name} compact checkpoint components are incomplete")
    if not isinstance(byte_counts, Mapping) or set(byte_counts) != required_components:
        raise ValueError(f"{name} compact checkpoint byte accounting is incomplete")
    if set(expected_tensor_parameters) != required_components:
        raise ValueError(f"{name} expected checkpoint accounting is incomplete")
    if checkpoint.get("format") != "torch.save":
        raise ValueError(f"{name} compact checkpoint format is invalid")
    if not isinstance(checkpoint.get("dtype"), str):
        raise ValueError(f"{name} compact checkpoint dtype is missing")

    actual_total = 0
    for component in sorted(required_components):
        path = Path(paths[component])
        expected_path = directory / f"{component}.pt"
        if path.resolve() != expected_path.resolve():
            raise ValueError(f"{name} {component} checkpoint path is outside its run directory")
        if not path.is_file():
            raise FileNotFoundError(f"missing {name} checkpoint component: {path}")
        actual_bytes = path.stat().st_size
        if byte_counts[component] != actual_bytes or actual_bytes <= 0:
            raise ValueError(f"{name} {component} checkpoint byte count is invalid")
        actual_total += actual_bytes
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{name} {component} checkpoint payload is invalid")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{name} {component} checkpoint schema is invalid")
        if payload.get("component") != component:
            raise ValueError(f"{name} {component} checkpoint label is invalid")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping) or any(
            metadata.get(key) != value for key, value in expected_metadata.items()
        ):
            raise ValueError(f"{name} {component} checkpoint metadata is incompatible")
        state = payload.get("state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"{name} {component} checkpoint state is empty")
        key_matches_component = {
            "adapter": lambda key: ".lora_a." in key or ".lora_b." in key,
            "atoms": lambda key: key.endswith((".atom_u", ".atom_v", "._extra_state")),
            "coefficients": lambda key: key.endswith(".coefficients"),
            "heads": lambda key: key.startswith("heads."),
        }[component]
        if any(not isinstance(key, str) or not key_matches_component(key) for key in state):
            raise ValueError(f"{name} {component} checkpoint has unexpected state keys")
        tensor_parameters = sum(
            value.numel() for value in state.values() if isinstance(value, torch.Tensor)
        )
        if tensor_parameters != expected_tensor_parameters[component]:
            raise ValueError(
                f"{name} {component} checkpoint has {tensor_parameters} tensor "
                f"parameters; expected {expected_tensor_parameters[component]}"
            )
    if checkpoint.get("total_bytes") != actual_total:
        raise ValueError(f"{name} compact checkpoint total byte count is invalid")


def _load_strict_core_lora(
    config: ExperimentConfig,
    target: str,
    core_results_root: Path,
    target_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = (
        core_results_root
        / "independent_lora"
        / f"seed_{config.seed}"
        / target
        / "metrics.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"strict core LoRA prerequisite is missing; run the core H1 grid first: {path}"
        )
    record = read_json(path)
    expected_scalars = {
        "system": "independent_lora",
        "run_kind": "confirmatory",
        "seed": config.seed,
        "task": target,
        "rank": config.lora_rank,
    }
    mismatches = [
        f"{name}={record.get(name)!r} (expected {expected!r})"
        for name, expected in expected_scalars.items()
        if record.get(name) != expected
    ]
    try:
        _validate_common_metadata(
            record,
            config,
            {target: dict(target_provenance)},
            name="strict core LoRA",
            expected_target_count=4,
            provenance_exact=False,
        )
    except (KeyError, TypeError, ValueError) as error:
        mismatches.append(str(error))
    try:
        _validate_exact_counts(
            record,
            {
                "base_trainable_parameters": 0,
                "lora_adapter_parameters": 4096,
                "atom_parameters": 0,
                "coefficient_parameters": 0,
                "head_parameters": 258,
                "uncategorized_trainable_parameters": 0,
                "model_trainable_parameters": 4354,
                "persistent_adaptation_parameters": 4354,
            },
            name="strict core LoRA",
        )
    except ValueError as error:
        mismatches.append(str(error))
    try:
        _validate_raw_evaluation(
            record["best"],
            "strict core best",
            task=target,
            expected_examples=_selected_validation_examples(
                {target: target_provenance}, target
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        mismatches.append(str(error))
    try:
        _validate_compact_checkpoint(
            record,
            path.parent,
            required_components=frozenset({"adapter", "heads"}),
            expected_tensor_parameters={"adapter": 4096, "heads": 258},
            expected_metadata={
                "system": "independent_lora",
                "seed": config.seed,
                "task": target,
            },
            name="strict core LoRA",
        )
    except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
        mismatches.append(str(error))
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(f"strict core LoRA record is incompatible ({path}): {detail}")
    return record, path


def _validate_source_record(
    record: Mapping[str, Any],
    config: ExperimentConfig,
    source_tasks: Sequence[str],
    source_directory: Path,
    provenance: Mapping[str, Any],
) -> None:
    name = "source learned atoms"
    if record.get("system") != "shared_atoms":
        raise ValueError(f"{name} has the wrong system identity")
    if record.get("run_kind") != "validation_cross_transfer_source":
        raise ValueError(f"{name} has the wrong run kind")
    if record.get("seed") != config.seed:
        raise ValueError(f"{name} has the wrong seed")
    if record.get("task_ids") != list(source_tasks):
        raise ValueError(f"{name} task order does not match the held-out design")
    _validate_common_metadata(
        record,
        config,
        provenance,
        name=name,
        expected_target_count=4,
    )
    expected_scalars = {
        "atom_count": 8,
        "active_atoms_during_training": 8,
        "active_atoms_for_evaluation": 4,
        "sparsity_lambda": config.sparsity_lambda,
        "atoms_frozen": False,
        "training_labels_shuffled": False,
    }
    for field, expected in expected_scalars.items():
        if record.get(field) != expected:
            raise ValueError(f"{name} {field} does not match the locked design")
    _validate_exact_counts(
        record,
        {
            "base_trainable_parameters": 0,
            "atom_parameters": 8192,
            "coefficient_parameters": 128,
            "head_parameters": 1032,
            "uncategorized_trainable_parameters": 0,
            "total_persistent_task_parameters": 9352,
        },
        name=name,
    )
    tasks = record.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(source_tasks):
        raise ValueError(f"{name} task evaluations are incomplete")
    for task in source_tasks:
        task_record = tasks[task]
        if not isinstance(task_record, Mapping) or task_record.get("top_k_value") != 4:
            raise ValueError(f"{name} {task} top-k evaluation is invalid")
        for field in ("all_atoms", "top_k"):
            try:
                evaluation = task_record[field]
            except KeyError as error:
                raise ValueError(f"{name} {task} lacks {field} evaluation") from error
            _validate_raw_evaluation(
                evaluation,
                f"{name} {task} {field}",
                task=task,
                expected_examples=_selected_validation_examples(provenance, task),
            )
    _validate_compact_checkpoint(
        record,
        source_directory,
        required_components=frozenset({"atoms", "coefficients", "heads"}),
        expected_tensor_parameters={
            "atoms": 8192,
            "coefficients": 128,
            "heads": 1032,
        },
        expected_metadata={
            "system": "shared_atoms",
            "seed": config.seed,
            "tasks": list(source_tasks),
            "atom_count": config.atom_count,
        },
        name=name,
    )


def _validate_head_only_record(
    record: Mapping[str, Any],
    config: ExperimentConfig,
    target: str,
    output_directory: Path,
    provenance: Mapping[str, Any],
) -> None:
    name = "head-only target"
    expected_scalars = {
        "experiment": "validation_cross_transfer",
        "system": "head_only",
        "run_kind": "validation",
        "seed": config.seed,
        "task": target,
        "marginal_new_parameters": 258,
        "total_with_dictionary_parameters": 258,
    }
    if any(record.get(field) != expected for field, expected in expected_scalars.items()):
        raise ValueError(f"{name} identity or parameter totals are incompatible")
    _validate_common_metadata(
        record,
        config,
        provenance,
        name=name,
        expected_target_count=0,
    )
    _validate_exact_counts(
        record,
        {
            "base_trainable_parameters": 0,
            "lora_adapter_parameters": 0,
            "atom_parameters": 0,
            "coefficient_parameters": 0,
            "head_parameters": 258,
            "uncategorized_trainable_parameters": 0,
            "model_trainable_parameters": 258,
            "persistent_adaptation_parameters": 258,
        },
        name=name,
    )
    for field in ("best", "final"):
        try:
            evaluation = record[field]
        except KeyError as error:
            raise ValueError(f"{name} lacks {field} evaluation") from error
        _validate_raw_evaluation(
            evaluation,
            f"{name} {field}",
            task=target,
            expected_examples=_selected_validation_examples(provenance, target),
        )
    _validate_compact_checkpoint(
        record,
        output_directory,
        required_components=frozenset({"heads"}),
        expected_tensor_parameters={"heads": 258},
        expected_metadata={
            "experiment": "validation_cross_transfer",
            "system": "head_only",
            "target": target,
            "seed": config.seed,
        },
        name=name,
    )


def _validate_frozen_atom_record(
    record: Mapping[str, Any],
    config: ExperimentConfig,
    target: str,
    output_directory: Path,
    provenance: Mapping[str, Any],
    *,
    dictionary_checkpoint: Path | None,
) -> None:
    learned = dictionary_checkpoint is not None
    system = "learned_frozen_atoms" if learned else "matched_random_frozen_atoms"
    name = f"{system} target"
    expected_scalars = {
        "experiment": "validation_cross_transfer",
        "system": system,
        "run_kind": "validation",
        "seed": config.seed,
        "task": target,
        "dictionary_frozen": True,
        "matched_initialization_seed": config.seed,
        "primary_evaluation": "top4",
        "marginal_new_parameters": 290,
        "reused_dictionary_parameters": 8192,
        "total_with_dictionary_parameters": 8482,
    }
    if any(record.get(field) != expected for field, expected in expected_scalars.items()):
        raise ValueError(f"{name} identity or parameter totals are incompatible")
    _validate_common_metadata(
        record,
        config,
        provenance,
        name=name,
        expected_target_count=4,
    )
    _validate_exact_counts(
        record,
        {
            "base_trainable_parameters": 0,
            "lora_adapter_parameters": 0,
            "atom_parameters": 8192,
            "coefficient_parameters": 32,
            "head_parameters": 258,
            "uncategorized_trainable_parameters": 0,
            "model_trainable_parameters": 290,
            "persistent_adaptation_parameters": 8482,
        },
        name=name,
    )
    for field in ("all8", "top4", "final"):
        try:
            evaluation = record[field]
        except KeyError as error:
            raise ValueError(f"{name} lacks {field} evaluation") from error
        _validate_raw_evaluation(
            evaluation,
            f"{name} {field}",
            task=target,
            expected_examples=_selected_validation_examples(provenance, target),
        )

    dictionary = record.get("dictionary")
    if not isinstance(dictionary, Mapping) or dictionary.get("parameter_count") != 8192:
        raise ValueError(f"{name} dictionary metadata is invalid")
    if learned:
        expected_atom_path = dictionary_checkpoint / "atoms.pt"
        if dictionary.get("source") != "learned_on_other_four_tasks":
            raise ValueError(f"{name} does not identify the learned source dictionary")
        if Path(str(dictionary.get("checkpoint"))).resolve() != expected_atom_path.resolve():
            raise ValueError(f"{name} points to the wrong learned source dictionary")
        if dictionary.get("tensor_count") != 8:
            raise ValueError(f"{name} learned dictionary tensor count is invalid")
    elif dictionary.get("source") != "deterministic_random_initialization" or dictionary.get(
        "checkpoint"
    ) is not None:
        raise ValueError(f"{name} random dictionary provenance is invalid")

    for field in ("trainable_initialization_sha256", "dictionary_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} lacks a valid {field}")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{name} has a non-hexadecimal {field}") from error
    if learned and dictionary.get("sha256") != record["dictionary_sha256"]:
        raise ValueError(f"{name} dictionary digests disagree")
    trainable_names = record.get("trainable_parameter_names")
    if not isinstance(trainable_names, Sequence) or isinstance(
        trainable_names, (str, bytes)
    ):
        raise ValueError(f"{name} trainable parameter names are invalid")
    if len(trainable_names) != 6:
        raise ValueError(f"{name} must train exactly four coefficient tensors and two head tensors")
    if sum(str(value).endswith(".coefficients") for value in trainable_names) != 4:
        raise ValueError(f"{name} trainable coefficient set is invalid")
    if sum(str(value).startswith(f"heads.{target}.") for value in trainable_names) != 2:
        raise ValueError(f"{name} trainable head set is invalid")
    if record.get("active_adapter_operations_per_token") != {
        "all8": 8192,
        "top4": 4096,
    }:
        raise ValueError(f"{name} active operation accounting is invalid")
    _validate_compact_checkpoint(
        record,
        output_directory,
        required_components=frozenset({"atoms", "coefficients", "heads"}),
        expected_tensor_parameters={
            "atoms": 8192,
            "coefficients": 32,
            "heads": 258,
        },
        expected_metadata={
            "experiment": "validation_cross_transfer",
            "system": system,
            "target": target,
            "seed": config.seed,
        },
        name=name,
    )


def _run_head_only_target(
    config: ExperimentConfig,
    target: str,
    output_directory: Path,
    prepared: PreparedData,
    *,
    force: bool,
) -> dict[str, Any]:
    metrics_path = output_directory / "metrics.json"
    if metrics_path.is_file() and not force:
        record = read_json(metrics_path)
        _validate_head_only_record(
            record,
            config,
            target,
            output_directory,
            prepared.provenance,
        )
        return record

    set_seed(config.seed)
    train_loaders, validation_loaders = build_loaders(prepared, config, tasks=(target,))
    model = build_head_only_model(config, target)
    optimizer = _optimizer(model, config)
    with RuntimeMonitor() as monitor:
        result = train_single_task(
            model,
            train_loaders[target],
            validation_loaders[target],
            optimizer,
            epochs=config.epochs,
            device=config.device,
            task_id=target,
            metric_fn=partial(compute_task_metrics, target),
            primary_metric="primary_score",
            state_capture_fn=lambda value: extract_adapter_state_dict(
                value, include_heads=True
            ),
            state_restore_fn=lambda value, state: load_adapter_state_dict(
                value, state, include_heads=True
            ),
        )
    runtime = monitor.result()
    counts = categorized_parameter_counts(model)
    metadata = _run_metadata(config, model, (), prepared, "validation")
    checkpoint = save_compact_checkpoint(
        model,
        output_directory,
        metadata={
            "experiment": "validation_cross_transfer",
            "system": "head_only",
            "target": target,
            "seed": config.seed,
        },
    )
    record = {
        **metadata,
        "experiment": "validation_cross_transfer",
        "system": "head_only",
        "task": target,
        "best": result.best_validation.to_dict(include_outputs=True),
        "final": result.final_validation.to_dict(include_outputs=True),
        "history": result.to_dict(),
        "parameter_counts": counts,
        "marginal_new_parameters": counts["head_parameters"],
        "total_with_dictionary_parameters": counts["head_parameters"],
        "checkpoint": checkpoint,
        "runtime": {
            "elapsed_seconds": runtime.elapsed_seconds,
            "peak_rss_bytes": runtime.peak_rss_bytes,
        },
    }
    _validate_head_only_record(
        record,
        config,
        target,
        output_directory,
        prepared.provenance,
    )
    write_json(metrics_path, record)
    return record


def _run_frozen_atom_target(
    config: ExperimentConfig,
    target: str,
    output_directory: Path,
    prepared: PreparedData,
    *,
    dictionary_checkpoint: Path | None,
    force: bool,
) -> dict[str, Any]:
    metrics_path = output_directory / "metrics.json"
    if metrics_path.is_file() and not force:
        record = read_json(metrics_path)
        _validate_frozen_atom_record(
            record,
            config,
            target,
            output_directory,
            prepared.provenance,
            dictionary_checkpoint=dictionary_checkpoint,
        )
        return record

    set_seed(config.seed)
    train_loaders, validation_loaders = build_loaders(prepared, config, tasks=(target,))
    model, target_names = build_atom_model(
        config,
        (target,),
        atom_count=config.atom_count,
        freeze_atoms=True,
    )
    if dictionary_checkpoint is None:
        dictionary = {
            "source": "deterministic_random_initialization",
            "checkpoint": None,
            "seed": config.seed,
            "parameter_count": sum(
                layer.atom_u.numel() + layer.atom_v.numel()
                for layer in iter_atom_layers(model)
            ),
        }
        system = "matched_random_frozen_atoms"
    else:
        dictionary = copy_frozen_atom_dictionary(model, dictionary_checkpoint)
        dictionary["source"] = "learned_on_other_four_tasks"
        system = "learned_frozen_atoms"
    trainable_names = assert_transfer_trainable_contract(model, target)
    trainable_initialization_sha256 = _parameter_digest(model, trainable=True)
    dictionary_sha256 = _parameter_digest(model, trainable=False)
    optimizer = _optimizer(model, config)
    with RuntimeMonitor() as monitor:
        result = train_single_task(
            model,
            train_loaders[target],
            validation_loaders[target],
            optimizer,
            epochs=config.epochs,
            device=config.device,
            task_id=target,
            metric_fn=partial(compute_task_metrics, target),
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
        )
    runtime = monitor.result()
    model.set_active_task(target)
    model.set_atom_top_k(config.active_atoms_for_primary_evaluation)
    try:
        top4 = evaluate(
            model,
            validation_loaders[target],
            config.device,
            task_id=None,
            metric_fn=partial(compute_task_metrics, target),
            scalar_metric_name="primary_score",
        )
    finally:
        model.clear_atom_top_k()

    counts = categorized_parameter_counts(model)
    marginal = counts["coefficient_parameters"] + counts["head_parameters"]
    total = counts["atom_parameters"] + marginal
    if counts["model_trainable_parameters"] != marginal:
        raise AssertionError(
            f"{system} trainable parameters must equal coefficients plus head: "
            f"{counts['model_trainable_parameters']} != {marginal}"
        )
    metadata = _run_metadata(config, model, target_names, prepared, "validation")
    checkpoint = save_compact_checkpoint(
        model,
        output_directory,
        metadata={
            "experiment": "validation_cross_transfer",
            "system": system,
            "target": target,
            "seed": config.seed,
        },
    )
    record = {
        **metadata,
        "experiment": "validation_cross_transfer",
        "system": system,
        "task": target,
        "dictionary": dictionary,
        "dictionary_frozen": True,
        "matched_initialization_seed": config.seed,
        "trainable_initialization_sha256": trainable_initialization_sha256,
        "dictionary_sha256": dictionary_sha256,
        "trainable_parameter_names": trainable_names,
        "primary_evaluation": "top4",
        "all8": result.best_validation.to_dict(include_outputs=True),
        "top4": top4.to_dict(include_outputs=True),
        "final": result.final_validation.to_dict(include_outputs=True),
        "history": result.to_dict(),
        "parameter_counts": counts,
        "marginal_new_parameters": marginal,
        "reused_dictionary_parameters": counts["atom_parameters"],
        "total_with_dictionary_parameters": total,
        "active_adapter_operations_per_token": {
            "all8": active_adapter_operations(model, active_atoms=config.atom_count),
            "top4": active_adapter_operations(
                model, active_atoms=config.active_atoms_for_primary_evaluation
            ),
        },
        "checkpoint": checkpoint,
        "runtime": {
            "elapsed_seconds": runtime.elapsed_seconds,
            "peak_rss_bytes": runtime.peak_rss_bytes,
        },
    }
    _validate_frozen_atom_record(
        record,
        config,
        target,
        output_directory,
        prepared.provenance,
        dictionary_checkpoint=dictionary_checkpoint,
    )
    write_json(metrics_path, record)
    return record


def _artifact_paths(
    cell_directory: Path,
    source_checkpoint_directory: Path,
    core_path: Path,
) -> dict[str, str]:
    learned_directory = cell_directory / "learned_atom_transfer"
    random_directory = cell_directory / "matched_random_transfer"
    head_directory = cell_directory / "head_only"
    return {
        "cell_result": str(cell_directory / "cell_result.json"),
        "source_checkpoint_directory": str(source_checkpoint_directory),
        "source_atoms": str(source_checkpoint_directory / "atoms.pt"),
        "source_record": str(source_checkpoint_directory / "metrics_by_task.json"),
        "learned_transfer_directory": str(learned_directory),
        "learned_transfer_record": str(learned_directory / "metrics.json"),
        "matched_random_transfer_directory": str(random_directory),
        "matched_random_transfer_record": str(random_directory / "metrics.json"),
        "head_only_directory": str(head_directory),
        "head_only_record": str(head_directory / "metrics.json"),
        "strict_core_lora_record": str(core_path),
    }


def run_cross_transfer_cell(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    target: str,
    cell_directory: Path,
    core_results_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run one held-out target/seed cell and return its compact index record."""

    _validate_locked_configs(baseline_config, atom_config)
    if baseline_config.seed != atom_config.seed:
        raise ValueError("paired cross-transfer configs must use the same seed")
    if target not in CROSS_TRANSFER_TARGETS:
        raise ValueError(f"unknown cross-transfer target: {target!r}")
    source_tasks = tuple(task for task in H1_TASKS if task != target)
    started = time.perf_counter()

    source_root = cell_directory / "source_learned_atoms"
    source_prepared = prepare_data(atom_config, tasks=source_tasks)
    source_record = run_shared_atoms(
        atom_config,
        source_root,
        tasks=source_tasks,
        prepared=source_prepared,
        run_kind="validation_cross_transfer_source",
        atom_count=atom_config.atom_count,
        top_k=atom_config.active_atoms_for_primary_evaluation,
        force=force,
    )
    source_checkpoint_directory = source_root / f"seed_{atom_config.seed}"
    source_atoms_path = source_checkpoint_directory / "atoms.pt"
    if not source_atoms_path.is_file():
        raise FileNotFoundError(source_atoms_path)
    _validate_source_record(
        source_record,
        atom_config,
        source_tasks,
        source_checkpoint_directory,
        source_prepared.provenance,
    )

    prepared = prepare_data(atom_config, tasks=(target,))
    fresh_lora, core_path = _load_strict_core_lora(
        baseline_config,
        target,
        core_results_root,
        prepared.provenance[target],
    )
    learned = _run_frozen_atom_target(
        atom_config,
        target,
        cell_directory / "learned_atom_transfer",
        prepared,
        dictionary_checkpoint=source_checkpoint_directory,
        force=force,
    )
    random_control = _run_frozen_atom_target(
        atom_config,
        target,
        cell_directory / "matched_random_transfer",
        prepared,
        dictionary_checkpoint=None,
        force=force,
    )
    head_only = _run_head_only_target(
        baseline_config,
        target,
        cell_directory / "head_only",
        prepared,
        force=force,
    )
    if (
        learned["trainable_initialization_sha256"]
        != random_control["trainable_initialization_sha256"]
    ):
        raise AssertionError(
            "learned and random frozen-atom controls did not start from matched "
            "target coefficient/head initialization"
        )
    identity_records = (source_record, fresh_lora, learned, random_control, head_only)
    if {record.get("model") for record in identity_records} != {atom_config.base_model}:
        raise ValueError("cross-transfer systems do not share the locked model identity")
    revisions = {record.get("model_revision") for record in identity_records}
    if len(revisions) != 1:
        raise ValueError("cross-transfer systems do not share one model revision")
    adapted_dimensions = {
        tuple(
            (name, tuple(dimensions))
            for name, dimensions in sorted(record["target_dimensions"].items())
        )
        for record in (source_record, fresh_lora, learned, random_control)
    }
    if len(adapted_dimensions) != 1:
        raise ValueError("cross-transfer adapter target dimensions differ")

    fresh_score = _primary_score(fresh_lora, "best")
    learned_all8 = _primary_score(learned, "all8")
    learned_top4 = _primary_score(learned, "top4")
    random_all8 = _primary_score(random_control, "all8")
    random_top4 = _primary_score(random_control, "top4")
    head_score = _primary_score(head_only, "best")
    fresh_parameters = int(
        fresh_lora["parameter_counts"]["persistent_adaptation_parameters"]
    )
    learned_marginal = int(learned["marginal_new_parameters"])
    dictionary_parameters = int(learned["reused_dictionary_parameters"])
    source_total = int(
        source_record["parameter_counts"]["total_persistent_task_parameters"]
    )
    artifacts = _artifact_paths(cell_directory, source_checkpoint_directory, core_path)
    cell = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "validation_cross_transfer",
        "status": "complete",
        "seed": atom_config.seed,
        "target": target,
        "source_tasks": list(source_tasks),
        "systems": {
            "fresh_lora": {
                "primary_score": fresh_score,
                "marginal_new_parameters": fresh_parameters,
                "reused_dictionary_parameters": 0,
                "total_with_dictionary_parameters": fresh_parameters,
                "raw_output_record": artifacts["strict_core_lora_record"],
            },
            "learned_frozen_atoms": {
                "all8_score": learned_all8,
                "top4_score": learned_top4,
                "primary_score": learned_top4,
                "marginal_new_parameters": learned_marginal,
                "reused_dictionary_parameters": dictionary_parameters,
                "total_with_dictionary_parameters": int(
                    learned["total_with_dictionary_parameters"]
                ),
                "raw_output_record": artifacts["learned_transfer_record"],
            },
            "head_only": {
                "primary_score": head_score,
                "marginal_new_parameters": int(head_only["marginal_new_parameters"]),
                "reused_dictionary_parameters": 0,
                "total_with_dictionary_parameters": int(
                    head_only["total_with_dictionary_parameters"]
                ),
                "raw_output_record": artifacts["head_only_record"],
            },
            "matched_random_frozen_atoms": {
                "all8_score": random_all8,
                "top4_score": random_top4,
                "primary_score": random_top4,
                "marginal_new_parameters": int(
                    random_control["marginal_new_parameters"]
                ),
                "reused_dictionary_parameters": int(
                    random_control["reused_dictionary_parameters"]
                ),
                "total_with_dictionary_parameters": int(
                    random_control["total_with_dictionary_parameters"]
                ),
                "raw_output_record": artifacts["matched_random_transfer_record"],
            },
        },
        "cell_diagnostics": {
            "learned_retention": learned_top4 / fresh_score,
            "learned_minus_head_only": learned_top4 - head_score,
            "learned_minus_random_frozen": learned_top4 - random_top4,
            "marginal_parameter_ratio": learned_marginal / fresh_parameters,
        },
        "deployment_accounting": {
            "source_four_task_parameters": source_total,
            "target_marginal_parameters": learned_marginal,
            "five_task_total_after_transfer": source_total + learned_marginal,
            "target_total_with_dictionary_parameters": dictionary_parameters
            + learned_marginal,
            "five_independent_lora_parameters": fresh_parameters * len(H1_TASKS),
        },
        "locked_budget": {
            "train_examples_per_task": atom_config.train_examples_per_task,
            "validation_examples_per_task": atom_config.validation_examples_per_task,
            "epochs": atom_config.epochs,
            "atom_count": atom_config.atom_count,
            "all8_active_atoms": atom_config.atom_count,
            "primary_top4_active_atoms": atom_config.active_atoms_for_primary_evaluation,
            "lora_rank": baseline_config.lora_rank,
        },
        "matching_evidence": {
            "seed": atom_config.seed,
            "trainable_initialization_sha256": learned[
                "trainable_initialization_sha256"
            ],
            "learned_dictionary_sha256": learned["dictionary_sha256"],
            "random_dictionary_sha256": random_control["dictionary_sha256"],
        },
        "model_identity": {
            "model": atom_config.base_model,
            "model_revision": learned["model_revision"],
            "target_dimensions": learned["target_dimensions"],
        },
        "runtime": {
            "source_learned_atoms": _runtime_dict(source_record),
            "learned_transfer": _runtime_dict(learned),
            "matched_random_transfer": _runtime_dict(random_control),
            "head_only": _runtime_dict(head_only),
            "strict_core_lora": _runtime_dict(fresh_lora),
            "cell_assembly_elapsed_seconds": time.perf_counter() - started,
        },
        "target_dataset_provenance": prepared.provenance[target],
        "resolved_configs": {
            "baseline": baseline_config.to_dict(),
            "atoms": atom_config.to_dict(),
        },
        "artifacts": artifacts,
    }
    validate_cross_transfer_cell(cell, expected_target=target, expected_seed=atom_config.seed)
    return cell


def validate_cross_transfer_cell(
    record: Mapping[str, Any],
    *,
    expected_target: str | None = None,
    expected_seed: int | None = None,
) -> None:
    """Validate one compact cell index before aggregation or resume."""

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("cross-transfer cell has an unsupported schema version")
    if record.get("experiment") != "validation_cross_transfer":
        raise ValueError("cross-transfer cell has the wrong experiment name")
    if record.get("status") != "complete":
        raise ValueError("cross-transfer cell is not complete")
    target = record.get("target")
    seed = record.get("seed")
    if target not in CROSS_TRANSFER_TARGETS:
        raise ValueError(f"cross-transfer cell has unknown target {target!r}")
    if seed not in CROSS_TRANSFER_SEEDS:
        raise ValueError(f"cross-transfer cell has unknown seed {seed!r}")
    if expected_target is not None and target != expected_target:
        raise ValueError(f"cell target {target!r} does not match {expected_target!r}")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(f"cell seed {seed!r} does not match {expected_seed!r}")
    expected_sources = [task for task in H1_TASKS if task != target]
    if record.get("source_tasks") != expected_sources:
        raise ValueError("cell source tasks are not the locked other-four-task order")
    systems = record.get("systems")
    required_systems = {
        "fresh_lora",
        "learned_frozen_atoms",
        "head_only",
        "matched_random_frozen_atoms",
    }
    if not isinstance(systems, Mapping) or set(systems) != required_systems:
        raise ValueError("cell systems do not match the locked comparison")
    for system, values in systems.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"cell system {system} must be an object")
        raw_score = values.get("primary_score", math.nan)
        if isinstance(raw_score, bool):
            raise ValueError(f"cell system {system} has an invalid primary score")
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"cell system {system} has an invalid primary score")
        for name in (
            "marginal_new_parameters",
            "reused_dictionary_parameters",
            "total_with_dictionary_parameters",
        ):
            value = values.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"cell system {system} has invalid {name}")
        if (
            values["marginal_new_parameters"] + values["reused_dictionary_parameters"]
            != values["total_with_dictionary_parameters"]
        ):
            raise ValueError(f"cell system {system} total parameter accounting is inconsistent")
        if not isinstance(values.get("raw_output_record"), str):
            raise ValueError(f"cell system {system} lacks a raw-output record path")
    if float(systems["fresh_lora"]["primary_score"]) <= 0.0:
        raise ValueError("fresh LoRA score must be positive for retention")
    expected_parameters = {
        "fresh_lora": (4354, 0, 4354),
        "learned_frozen_atoms": (290, 8192, 8482),
        "head_only": (258, 0, 258),
        "matched_random_frozen_atoms": (290, 8192, 8482),
    }
    for system, expected in expected_parameters.items():
        observed = tuple(
            systems[system][field]
            for field in (
                "marginal_new_parameters",
                "reused_dictionary_parameters",
                "total_with_dictionary_parameters",
            )
        )
        if observed != expected:
            raise ValueError(f"cell system {system} violates exact parameter accounting")
    for system in ("learned_frozen_atoms", "matched_random_frozen_atoms"):
        values = systems[system]
        for field in ("all8_score", "top4_score"):
            value = values.get(field)
            if isinstance(value, bool):
                raise ValueError(f"cell system {system} has an invalid {field}")
            score = float(value)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"cell system {system} has an invalid {field}")
        if not math.isclose(
            float(values["primary_score"]),
            float(values["top4_score"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"cell system {system} primary score must be top4")

    expected_budget = {
        "train_examples_per_task": 2000,
        "validation_examples_per_task": 500,
        "epochs": 3,
        "atom_count": 8,
        "all8_active_atoms": 8,
        "primary_top4_active_atoms": 4,
        "lora_rank": 4,
    }
    if record.get("locked_budget") != expected_budget:
        raise ValueError("cell budget does not match the locked cross-transfer design")
    matching = record.get("matching_evidence")
    if not isinstance(matching, Mapping) or matching.get("seed") != seed:
        raise ValueError("cell lacks matched-initialization evidence")
    for field in (
        "trainable_initialization_sha256",
        "learned_dictionary_sha256",
        "random_dictionary_sha256",
    ):
        digest = matching.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"cell matching evidence has an invalid {field}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(f"cell matching evidence has a non-hexadecimal {field}") from error
    identity = record.get("model_identity")
    if not isinstance(identity, Mapping) or identity.get("model") != "prajjwal1/bert-tiny":
        raise ValueError("cell model identity is invalid")
    if "model_revision" not in identity:
        raise ValueError("cell model revision is missing")
    dimensions = identity.get("target_dimensions")
    if not isinstance(dimensions, Mapping) or len(dimensions) != 4:
        raise ValueError("cell target dimensions are invalid")
    try:
        dimensions_valid = all(list(value) == [128, 128] for value in dimensions.values())
    except TypeError as error:
        raise ValueError("cell target dimensions are invalid") from error
    if not dimensions_valid:
        raise ValueError("cell target dimensions are invalid")

    fresh = float(systems["fresh_lora"]["primary_score"])
    learned = float(systems["learned_frozen_atoms"]["primary_score"])
    head = float(systems["head_only"]["primary_score"])
    random_score = float(systems["matched_random_frozen_atoms"]["primary_score"])
    expected_diagnostics = {
        "learned_retention": learned / fresh,
        "learned_minus_head_only": learned - head,
        "learned_minus_random_frozen": learned - random_score,
        "marginal_parameter_ratio": 290 / 4354,
    }
    diagnostics = record.get("cell_diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(expected_diagnostics):
        raise ValueError("cell diagnostics are incomplete")
    for field, expected in expected_diagnostics.items():
        try:
            observed = float(diagnostics[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"cell diagnostic {field} is invalid") from error
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"cell diagnostic {field} does not match system scores")

    expected_deployment = {
        "source_four_task_parameters": 9352,
        "target_marginal_parameters": 290,
        "five_task_total_after_transfer": 9642,
        "target_total_with_dictionary_parameters": 8482,
        "five_independent_lora_parameters": 21770,
    }
    if record.get("deployment_accounting") != expected_deployment:
        raise ValueError("cell deployment accounting is invalid")
    artifacts = record.get("artifacts")
    required_paths = {
        "cell_result",
        "source_checkpoint_directory",
        "source_atoms",
        "source_record",
        "learned_transfer_directory",
        "learned_transfer_record",
        "matched_random_transfer_directory",
        "matched_random_transfer_record",
        "head_only_directory",
        "head_only_record",
        "strict_core_lora_record",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_paths:
        raise ValueError("cell artifact index is incomplete")
    if any(not isinstance(value, str) or not value for value in artifacts.values()):
        raise ValueError("cell artifact paths must be non-empty strings")
    cell_directory = Path(artifacts["cell_result"]).parent
    source_directory = cell_directory / "source_learned_atoms" / f"seed_{seed}"
    expected_artifacts = {
        "cell_result": cell_directory / "cell_result.json",
        "source_checkpoint_directory": source_directory,
        "source_atoms": source_directory / "atoms.pt",
        "source_record": source_directory / "metrics_by_task.json",
        "learned_transfer_directory": cell_directory / "learned_atom_transfer",
        "learned_transfer_record": cell_directory / "learned_atom_transfer" / "metrics.json",
        "matched_random_transfer_directory": cell_directory / "matched_random_transfer",
        "matched_random_transfer_record": cell_directory
        / "matched_random_transfer"
        / "metrics.json",
        "head_only_directory": cell_directory / "head_only",
        "head_only_record": cell_directory / "head_only" / "metrics.json",
    }
    for field, expected in expected_artifacts.items():
        if Path(artifacts[field]) != expected:
            raise ValueError(f"cell artifact {field} violates the locked layout")
    expected_raw_records = {
        "fresh_lora": "strict_core_lora_record",
        "learned_frozen_atoms": "learned_transfer_record",
        "head_only": "head_only_record",
        "matched_random_frozen_atoms": "matched_random_transfer_record",
    }
    for system, artifact in expected_raw_records.items():
        if systems[system]["raw_output_record"] != artifacts[artifact]:
            raise ValueError(f"cell system {system} raw-output path is inconsistent")
    for field in ("runtime", "target_dataset_provenance", "resolved_configs"):
        if not isinstance(record.get(field), Mapping):
            raise ValueError(f"cell {field} must be an object")


def build_cross_transfer_summary(
    cells: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Aggregate the exact 15-cell validation grid and apply preregistered rules."""

    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for cell in cells:
        validate_cross_transfer_cell(cell)
        key = (str(cell["target"]), int(cell["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate cross-transfer cell: target={key[0]}, seed={key[1]}")
        indexed[key] = cell
    expected = {
        (target, seed)
        for target in CROSS_TRANSFER_TARGETS
        for seed in CROSS_TRANSFER_SEEDS
    }
    missing = sorted(expected - set(indexed))
    unexpected = sorted(set(indexed) - expected)
    if missing or unexpected:
        raise ValueError(
            f"cross-transfer grid must contain all 15 locked cells; "
            f"missing={missing}, unexpected={unexpected}"
        )

    def system_score(cell: Mapping[str, Any], system: str, field: str = "primary_score") -> float:
        return float(cell["systems"][system][field])

    by_target: dict[str, Any] = {}
    for target in CROSS_TRANSFER_TARGETS:
        target_cells = [indexed[(target, seed)] for seed in CROSS_TRANSFER_SEEDS]
        fresh = [system_score(cell, "fresh_lora") for cell in target_cells]
        learned_all8 = [
            system_score(cell, "learned_frozen_atoms", "all8_score")
            for cell in target_cells
        ]
        learned_top4 = [
            system_score(cell, "learned_frozen_atoms") for cell in target_cells
        ]
        head = [system_score(cell, "head_only") for cell in target_cells]
        random_control = [
            system_score(cell, "matched_random_frozen_atoms") for cell in target_cells
        ]
        fresh_mean = _mean(fresh)
        if fresh_mean <= 0.0:
            raise ValueError(f"fresh LoRA mean must be positive for target {target}")
        learned_mean = _mean(learned_top4)
        head_mean = _mean(head)
        random_mean = _mean(random_control)
        by_target[target] = {
            "fresh_lora_mean": fresh_mean,
            "learned_all8_mean": _mean(learned_all8),
            "learned_top4_mean": learned_mean,
            "retention": learned_mean / fresh_mean,
            "head_only_mean": head_mean,
            "matched_random_top4_mean": random_mean,
            "learned_minus_head_only": learned_mean - head_mean,
            "learned_minus_random_frozen": learned_mean - random_mean,
            "seed_scores": {
                str(seed): {
                    "fresh_lora": system_score(indexed[(target, seed)], "fresh_lora"),
                    "learned_all8": system_score(
                        indexed[(target, seed)], "learned_frozen_atoms", "all8_score"
                    ),
                    "learned_top4": system_score(
                        indexed[(target, seed)], "learned_frozen_atoms"
                    ),
                    "head_only": system_score(indexed[(target, seed)], "head_only"),
                    "matched_random_top4": system_score(
                        indexed[(target, seed)], "matched_random_frozen_atoms"
                    ),
                }
                for seed in CROSS_TRANSFER_SEEDS
            },
        }

    by_seed: dict[str, Any] = {}
    for seed in CROSS_TRANSFER_SEEDS:
        seed_cells = [indexed[(target, seed)] for target in CROSS_TRANSFER_TARGETS]
        means = {
            "fresh_lora_mean": _mean(
                [system_score(cell, "fresh_lora") for cell in seed_cells]
            ),
            "learned_all8_mean": _mean(
                [
                    system_score(cell, "learned_frozen_atoms", "all8_score")
                    for cell in seed_cells
                ]
            ),
            "learned_top4_mean": _mean(
                [system_score(cell, "learned_frozen_atoms") for cell in seed_cells]
            ),
            "head_only_mean": _mean(
                [system_score(cell, "head_only") for cell in seed_cells]
            ),
            "matched_random_top4_mean": _mean(
                [
                    system_score(cell, "matched_random_frozen_atoms")
                    for cell in seed_cells
                ]
            ),
        }
        means["retention"] = means["learned_top4_mean"] / means["fresh_lora_mean"]
        by_seed[str(seed)] = means

    ordered_cells = [
        indexed[(target, seed)]
        for target in CROSS_TRANSFER_TARGETS
        for seed in CROSS_TRANSFER_SEEDS
    ]
    fresh_mean = _mean([values["fresh_lora_mean"] for values in by_target.values()])
    learned_all8_mean = _mean(
        [values["learned_all8_mean"] for values in by_target.values()]
    )
    learned_top4_mean = _mean(
        [values["learned_top4_mean"] for values in by_target.values()]
    )
    head_mean = _mean([values["head_only_mean"] for values in by_target.values()])
    random_mean = _mean(
        [values["matched_random_top4_mean"] for values in by_target.values()]
    )
    marginal_ratios = [
        float(cell["systems"]["learned_frozen_atoms"]["marginal_new_parameters"])
        / float(cell["systems"]["fresh_lora"]["marginal_new_parameters"])
        for cell in ordered_cells
    ]
    accounting_tuples = {
        (
            int(cell["systems"]["fresh_lora"]["marginal_new_parameters"]),
            int(cell["systems"]["learned_frozen_atoms"]["marginal_new_parameters"]),
            int(cell["systems"]["learned_frozen_atoms"]["reused_dictionary_parameters"]),
            int(cell["systems"]["learned_frozen_atoms"]["total_with_dictionary_parameters"]),
        )
        for cell in ordered_cells
    }
    if len(accounting_tuples) != 1:
        raise ValueError("parameter accounting differs across cross-transfer cells")
    fresh_parameters, learned_marginal, learned_dictionary, learned_total = next(
        iter(accounting_tuples)
    )
    retention = learned_top4_mean / fresh_mean
    primary = {
        "criterion": "aggregate learned-top4 mean / aggregate fresh-LoRA mean >= 0.95",
        "threshold": PRIMARY_RETENTION_THRESHOLD,
        "aggregate_retention": retention,
        "passed": retention >= PRIMARY_RETENTION_THRESHOLD,
    }
    target_retention_passed = all(
        values["retention"] >= TARGET_RETENTION_DIAGNOSTIC_THRESHOLD
        for values in by_target.values()
    )
    learned_minus_head = learned_top4_mean - head_mean
    learned_minus_random = learned_top4_mean - random_mean
    diagnostics = {
        "every_target_seed_mean_retention_at_least_0_90": {
            "threshold": TARGET_RETENTION_DIAGNOSTIC_THRESHOLD,
            "minimum_observed": min(values["retention"] for values in by_target.values()),
            "passed": target_retention_passed,
        },
        "learned_mean_exceeds_head_only_by_0_005": {
            "threshold": CONTROL_ADVANTAGE_DIAGNOSTIC_THRESHOLD,
            "observed_difference": learned_minus_head,
            "passed": learned_minus_head >= CONTROL_ADVANTAGE_DIAGNOSTIC_THRESHOLD,
        },
        "learned_mean_exceeds_random_frozen_by_0_005": {
            "threshold": CONTROL_ADVANTAGE_DIAGNOSTIC_THRESHOLD,
            "observed_difference": learned_minus_random,
            "passed": learned_minus_random >= CONTROL_ADVANTAGE_DIAGNOSTIC_THRESHOLD,
        },
        "marginal_new_parameters_at_most_10_percent_fresh_task_state": {
            "threshold": MARGINAL_PARAMETER_DIAGNOSTIC_THRESHOLD,
            "maximum_observed_ratio": max(marginal_ratios),
            "passed": max(marginal_ratios) <= MARGINAL_PARAMETER_DIAGNOSTIC_THRESHOLD,
        },
    }
    all_diagnostics_passed = all(value["passed"] for value in diagnostics.values())
    strong_reusable_basis_support = bool(primary["passed"] and all_diagnostics_passed)
    artifacts_by_target_seed = {
        target: {
            str(seed): dict(indexed[(target, seed)]["artifacts"])
            for seed in CROSS_TRANSFER_SEEDS
        }
        for target in CROSS_TRANSFER_TARGETS
    }
    first = ordered_cells[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "validation_cross_transfer",
        "status": "complete",
        "predeclared_before_live_training": True,
        "aggregation_rule": (
            "mean over seeds per held-out target, then unweighted mean over the five targets"
        ),
        "targets": list(CROSS_TRANSFER_TARGETS),
        "seeds": list(CROSS_TRANSFER_SEEDS),
        "cell_count": len(ordered_cells),
        "aggregate": {
            "fresh_lora_mean": fresh_mean,
            "learned_all8_mean": learned_all8_mean,
            "learned_top4_mean": learned_top4_mean,
            "head_only_mean": head_mean,
            "matched_random_top4_mean": random_mean,
            "learned_minus_head_only": learned_minus_head,
            "learned_minus_random_frozen": learned_minus_random,
        },
        "primary_strong_transfer": primary,
        "strong_reusable_basis_support": strong_reusable_basis_support,
        "diagnostics": diagnostics,
        "by_target": by_target,
        "by_seed": by_seed,
        "parameter_accounting": {
            "fresh_lora_task_state": fresh_parameters,
            "learned_target_marginal": learned_marginal,
            "learned_dictionary": learned_dictionary,
            "learned_target_total_with_dictionary": learned_total,
            "marginal_ratio": marginal_ratios[0],
        },
        "locked_budget": dict(first["locked_budget"]),
        "artifacts_by_target_seed": artifacts_by_target_seed,
        "protocol": str(Path(output_root) / PROTOCOL_FILENAME),
        "output_root": str(Path(output_root)),
    }


def build_protocol_record(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    output_root: str | Path,
    core_results_root: str | Path,
) -> dict[str, Any]:
    """Return the stable preregistration written before any validation cell runs."""

    _validate_locked_configs(baseline_config, atom_config)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "validation_cross_transfer",
        "status": "locked_before_live_training",
        "targets": list(CROSS_TRANSFER_TARGETS),
        "seeds": list(CROSS_TRANSFER_SEEDS),
        "source_design": "for each target, jointly train N=8 atoms on the other four tasks",
        "target_design": (
            "freeze atom_u/atom_v; train only target coefficients and classification head"
        ),
        "comparators": [
            "strict_core_fresh_rank4_lora",
            "separately_trained_head_only",
            "deterministic_matched_random_frozen_atoms",
        ],
        "evaluations": ["all8", "top4_primary"],
        "primary_strong_transfer": {
            "measure": "aggregate learned-top4 mean / aggregate fresh-LoRA mean",
            "operator": ">=",
            "threshold": PRIMARY_RETENTION_THRESHOLD,
        },
        "diagnostics": {
            "every_target_seed_mean_retention": {
                "operator": ">=",
                "threshold": TARGET_RETENTION_DIAGNOSTIC_THRESHOLD,
            },
            "learned_mean_minus_head_only": {
                "operator": ">=",
                "threshold": CONTROL_ADVANTAGE_DIAGNOSTIC_THRESHOLD,
            },
            "learned_mean_minus_random_frozen": {
                "operator": ">=",
                "threshold": CONTROL_ADVANTAGE_DIAGNOSTIC_THRESHOLD,
            },
            "marginal_new_parameters_over_fresh_task_state": {
                "operator": "<=",
                "threshold": MARGINAL_PARAMETER_DIAGNOSTIC_THRESHOLD,
            },
        },
        "strong_reusable_basis_support": (
            "true only when the primary strong-transfer criterion and all four "
            "control/parameter diagnostics pass"
        ),
        "aggregation_rule": (
            "mean over seeds per held-out target, then unweighted mean over the five targets"
        ),
        "artifact_layout": str(
            Path(output_root) / "target_<task>" / "seed_<seed>"
        ),
        "core_results_root": str(Path(core_results_root)),
        "baseline_config": baseline_config.to_dict(),
        "atom_config": atom_config.to_dict(),
    }


def render_cross_transfer_markdown(summary: Mapping[str, Any]) -> str:
    """Render the completed 15-cell validation report."""

    aggregate = summary["aggregate"]
    primary = summary["primary_strong_transfer"]
    lines = [
        "# Experiment B: Crossed Frozen-Atom Transfer",
        "",
        "Status: **COMPLETE**",
        "",
        "Each target is held out in turn at seeds 17, 29, and 43. Source atoms are "
        "trained on the other four tasks, then frozen; only target coefficients and "
        "a target head are fitted.",
        "",
        "## Primary decision",
        "",
        f"Strong transfer: **{'PASS' if primary['passed'] else 'FAIL'}**. Learned top-4 "
        f"aggregate retention was {float(primary['aggregate_retention']):.2%}; the "
        f"predeclared requirement was >= {float(primary['threshold']):.0%}.",
        "",
        "Control-aware strong reusable-basis support: "
        f"**{'PASS' if summary['strong_reusable_basis_support'] else 'FAIL'}**. "
        "This combined interpretation requires the primary criterion and every "
        "diagnostic below to pass.",
        "",
        "| System | Aggregate mean primary score |",
        "|---|---:|",
        f"| Fresh rank-4 LoRA | {float(aggregate['fresh_lora_mean']):.6f} |",
        f"| Learned frozen atoms, all 8 | {float(aggregate['learned_all8_mean']):.6f} |",
        f"| Learned frozen atoms, top 4 | {float(aggregate['learned_top4_mean']):.6f} |",
        f"| Head only | {float(aggregate['head_only_mean']):.6f} |",
        f"| Matched random frozen atoms, top 4 | "
        f"{float(aggregate['matched_random_top4_mean']):.6f} |",
        "",
        "## By held-out target",
        "",
        "| Target | Fresh LoRA | Learned all-8 | Learned top-4 | Retention | "
        "Head only | Random top-4 | Learned-head | Learned-random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in CROSS_TRANSFER_TARGETS:
        row = summary["by_target"][target]
        lines.append(
            f"| {target} | {float(row['fresh_lora_mean']):.6f} | "
            f"{float(row['learned_all8_mean']):.6f} | "
            f"{float(row['learned_top4_mean']):.6f} | "
            f"{float(row['retention']):.2%} | {float(row['head_only_mean']):.6f} | "
            f"{float(row['matched_random_top4_mean']):.6f} | "
            f"{float(row['learned_minus_head_only']):+.6f} | "
            f"{float(row['learned_minus_random_frozen']):+.6f} |"
        )
    lines.extend(
        [
            "",
            "## By seed",
            "",
            "| Seed | Fresh LoRA | Learned all-8 | Learned top-4 | Retention | "
            "Head only | Random top-4 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in CROSS_TRANSFER_SEEDS:
        row = summary["by_seed"][str(seed)]
        lines.append(
            f"| {seed} | {float(row['fresh_lora_mean']):.6f} | "
            f"{float(row['learned_all8_mean']):.6f} | "
            f"{float(row['learned_top4_mean']):.6f} | "
            f"{float(row['retention']):.2%} | {float(row['head_only_mean']):.6f} | "
            f"{float(row['matched_random_top4_mean']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| Diagnostic | Observed | Required | Result |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, diagnostic in summary["diagnostics"].items():
        observed = next(
            float(diagnostic[key])
            for key in (
                "minimum_observed",
                "observed_difference",
                "maximum_observed_ratio",
            )
            if key in diagnostic
        )
        lines.append(
            f"| `{name}` | {observed:.6f} | "
            f"{float(diagnostic['threshold']):.6f} | "
            f"{'PASS' if diagnostic['passed'] else 'FAIL'} |"
        )
    parameters = summary["parameter_accounting"]
    lines.extend(
        [
            "",
            "## Parameter accounting",
            "",
            "| Quantity | Parameters |",
            "|---|---:|",
            f"| Fresh LoRA target state | {int(parameters['fresh_lora_task_state']):,} |",
            f"| Learned target marginal state | {int(parameters['learned_target_marginal']):,} |",
            f"| Reused frozen dictionary | {int(parameters['learned_dictionary']):,} |",
            f"| Learned target total with dictionary | "
            f"{int(parameters['learned_target_total_with_dictionary']):,} |",
            "",
            "Marginal and total-with-dictionary counts are reported separately. Raw "
            "predictions, labels, runtimes, provenance, and compact checkpoint paths are "
            "indexed by the JSON summary.",
            "",
        ]
    )
    return "\n".join(lines)


def run_validation_cross_transfer(
    baseline_config: ExperimentConfig,
    atom_config: ExperimentConfig,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    core_results_root: str | Path = DEFAULT_CORE_RESULTS_ROOT,
    force: bool = False,
    cell_runner: CellRunner | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Run/resume all 15 cells and write strict aggregate JSON and Markdown."""

    _validate_locked_configs(baseline_config, atom_config)
    destination = Path(output_root)
    core_root = Path(core_results_root)
    runner = cell_runner or run_cross_transfer_cell
    protocol = build_protocol_record(
        baseline_config, atom_config, destination, core_root
    )
    protocol_path = destination / PROTOCOL_FILENAME
    if protocol_path.is_file():
        existing_protocol = read_json(protocol_path)
        if existing_protocol != protocol:
            raise ValueError(
                f"existing cross-transfer protocol differs from the locked design: {protocol_path}"
            )
    else:
        write_json(protocol_path, protocol)
    cells: list[dict[str, Any]] = []
    for target in CROSS_TRANSFER_TARGETS:
        for seed in CROSS_TRANSFER_SEEDS:
            cell_directory = destination / f"target_{target}" / f"seed_{seed}"
            cell_path = cell_directory / "cell_result.json"
            if cell_path.is_file() and not force:
                cell = read_json(cell_path)
                validate_cross_transfer_cell(
                    cell, expected_target=target, expected_seed=seed
                )
                print(
                    f"Skipping completed cross-transfer target={target}, seed={seed}.",
                    flush=True,
                )
            else:
                print(f"Running cross-transfer target={target}, seed={seed}.", flush=True)
                cell = runner(
                    baseline_config.with_overrides(seed=seed),
                    atom_config.with_overrides(seed=seed),
                    target,
                    cell_directory,
                    core_root,
                    force,
                )
                validate_cross_transfer_cell(
                    cell, expected_target=target, expected_seed=seed
                )
                write_json(cell_path, cell)
            cells.append(cell)

    summary = build_cross_transfer_summary(cells, output_root=destination)
    summary_path = write_json(destination / SUMMARY_FILENAME, summary)
    report_path = destination / REPORT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    try:
        temporary.write_text(
            render_cross_transfer_markdown(summary), encoding="utf-8", newline="\n"
        )
        temporary.replace(report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary, summary_path, report_path


__all__ = [
    "CONTROL_ADVANTAGE_DIAGNOSTIC_THRESHOLD",
    "CROSS_TRANSFER_SEEDS",
    "CROSS_TRANSFER_TARGETS",
    "DEFAULT_CORE_RESULTS_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "MARGINAL_PARAMETER_DIAGNOSTIC_THRESHOLD",
    "PRIMARY_RETENTION_THRESHOLD",
    "PROTOCOL_FILENAME",
    "REPORT_FILENAME",
    "SUMMARY_FILENAME",
    "TARGET_RETENTION_DIAGNOSTIC_THRESHOLD",
    "build_protocol_record",
    "build_cross_transfer_summary",
    "render_cross_transfer_markdown",
    "run_cross_transfer_cell",
    "run_validation_cross_transfer",
    "validate_cross_transfer_cell",
]
