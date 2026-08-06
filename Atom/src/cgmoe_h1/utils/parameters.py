"""Exact, identity-aware parameter and checkpoint accounting for H1."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from torch import nn

from cgmoe_h1.models.atoms import AtomLinear, iter_atom_layers
from cgmoe_h1.models.lora import LoRALinear, iter_lora_layers


def _unique_parameters(parameters: Iterable[nn.Parameter]) -> dict[int, nn.Parameter]:
    return {id(parameter): parameter for parameter in parameters}


def parameter_count(parameters: Iterable[nn.Parameter]) -> int:
    """Count scalar values, deduplicating genuinely shared parameters by identity."""

    return sum(parameter.numel() for parameter in _unique_parameters(parameters).values())


def module_parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return parameter_count(parameters)


def _head_parameters(model: nn.Module) -> list[nn.Parameter]:
    heads = getattr(model, "heads", None)
    return list(heads.parameters()) if isinstance(heads, nn.Module) else []


def _lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for layer in iter_lora_layers(model)
        for parameter in layer.adapter_parameters()
    ]


def _atom_vector_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for layer in iter_atom_layers(model)
        for parameter in (layer.atom_v, layer.atom_u)
    ]


def _coefficient_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [layer.coefficients for layer in iter_atom_layers(model)]


def categorized_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return an honest mutually exclusive accounting for one deployed model."""

    all_parameters = _unique_parameters(model.parameters())
    heads = _unique_parameters(_head_parameters(model))
    lora = _unique_parameters(_lora_parameters(model))
    atoms = _unique_parameters(_atom_vector_parameters(model))
    coefficients = _unique_parameters(_coefficient_parameters(model))

    learned_ids = set(heads) | set(lora) | set(atoms) | set(coefficients)
    base = {key: value for key, value in all_parameters.items() if key not in learned_ids}
    uncategorized_trainable = {
        key: value
        for key, value in base.items()
        if value.requires_grad
    }

    counts = {
        "base_model_parameters": parameter_count(base.values()),
        "base_trainable_parameters": parameter_count(
            parameter for parameter in base.values() if parameter.requires_grad
        ),
        "lora_adapter_parameters": parameter_count(lora.values()),
        "atom_parameters": parameter_count(atoms.values()),
        "coefficient_parameters": parameter_count(coefficients.values()),
        "head_parameters": parameter_count(heads.values()),
        "uncategorized_trainable_parameters": parameter_count(
            uncategorized_trainable.values()
        ),
        "model_total_parameters": parameter_count(all_parameters.values()),
        "model_trainable_parameters": parameter_count(
            parameter for parameter in all_parameters.values() if parameter.requires_grad
        ),
    }
    counts["persistent_adaptation_parameters"] = (
        counts["lora_adapter_parameters"]
        + counts["atom_parameters"]
        + counts["coefficient_parameters"]
        + counts["head_parameters"]
        + counts["uncategorized_trainable_parameters"]
    )
    return counts


def independent_parameter_totals(
    models_by_task: Mapping[str, nn.Module] | Iterable[nn.Module],
) -> dict[str, int]:
    """Sum separately deployed LoRA adapters and heads across tasks."""

    models = (
        list(models_by_task.values())
        if isinstance(models_by_task, Mapping)
        else list(models_by_task)
    )
    if not models:
        raise ValueError("at least one independent model is required")
    per_model = [categorized_parameter_counts(model) for model in models]
    base_counts = {entry["base_model_parameters"] for entry in per_model}
    if len(base_counts) != 1:
        raise ValueError(f"independent models have different base sizes: {sorted(base_counts)}")
    return {
        "base_model_parameters": next(iter(base_counts)),
        "base_trainable_parameters": sum(
            entry["base_trainable_parameters"] for entry in per_model
        ),
        "adapter_parameters": sum(
            entry["lora_adapter_parameters"] for entry in per_model
        ),
        "head_parameters": sum(entry["head_parameters"] for entry in per_model),
        "uncategorized_trainable_parameters": sum(
            entry["uncategorized_trainable_parameters"] for entry in per_model
        ),
        "total_persistent_task_parameters": sum(
            entry["persistent_adaptation_parameters"] for entry in per_model
        ),
    }


def shared_atom_parameter_totals(model: nn.Module) -> dict[str, int]:
    counts = categorized_parameter_counts(model)
    return {
        "base_model_parameters": counts["base_model_parameters"],
        "base_trainable_parameters": counts["base_trainable_parameters"],
        "atom_parameters": counts["atom_parameters"],
        "coefficient_parameters": counts["coefficient_parameters"],
        "head_parameters": counts["head_parameters"],
        "uncategorized_trainable_parameters": counts[
            "uncategorized_trainable_parameters"
        ],
        "total_persistent_task_parameters": counts[
            "persistent_adaptation_parameters"
        ],
    }


def checkpoint_bytes(paths: str | Path | Iterable[str | Path]) -> int:
    """Return exact serialized bytes for one file or a set of component files."""

    if isinstance(paths, (str, Path)):
        paths = (paths,)
    total = 0
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        total += path.stat().st_size
    return total


def active_adapter_operations(model: nn.Module, *, active_atoms: int | None = None) -> int:
    """Estimate multiply-add terms per token for the active adapter branches."""

    operations = 0
    for layer in model.modules():
        if isinstance(layer, LoRALinear):
            operations += layer.rank * (layer.in_features + layer.out_features)
        elif isinstance(layer, AtomLinear):
            active = layer.atom_count if active_atoms is None else active_atoms
            if not 1 <= active <= layer.atom_count:
                raise ValueError(
                    f"active_atoms must be in [1, {layer.atom_count}], got {active}"
                )
            operations += active * (layer.in_features + layer.out_features)
    return operations


def assert_frozen_base(model: nn.Module) -> None:
    counts = categorized_parameter_counts(model)
    if counts["base_trainable_parameters"]:
        raise AssertionError(
            f"frozen base has {counts['base_trainable_parameters']} trainable parameters"
        )


def accounting_record(model: nn.Module, **metadata: Any) -> dict[str, Any]:
    return {**categorized_parameter_counts(model), **metadata}


__all__ = [
    "accounting_record",
    "active_adapter_operations",
    "assert_frozen_base",
    "categorized_parameter_counts",
    "checkpoint_bytes",
    "independent_parameter_totals",
    "module_parameter_count",
    "parameter_count",
    "shared_atom_parameter_totals",
]
