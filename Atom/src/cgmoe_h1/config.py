"""Typed loading and validation for H1 experiment configuration files.

The YAML files are deliberately boring data.  This module is the single place
where those values acquire types and where internally inconsistent experiments
are rejected before model or dataset work starts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

H1_TASKS = ("sst2", "mrpc", "rte", "qnli", "qqp")
H1_TARGET_MODULES = ("query", "value")
H1_CONFIRMATORY_SEEDS = (17, 29, 43)
EXPERIMENT_NAMES = frozenset({"independent_lora", "shared_atoms"})


def _is_int(value: object) -> bool:
    """Return whether *value* is an integer, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(name: str, value: object, *, minimum: int | None = None) -> None:
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _require_number(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None:
        invalid = numeric > maximum if maximum_inclusive else numeric >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {maximum}, got {value}")


def _tuple_of_strings(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain at least one non-empty string")
    return result


def _tuple_of_ints(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    for item in result:
        _require_int(f"{name} item", item, minimum=0)
    return result


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Fully resolved settings shared by the two H1 systems.

    General validation permits a task subset or a smaller budget for a marked
    development run.  :meth:`validate_h1_contract` performs the stricter check
    required before a result can be called confirmatory.
    """

    experiment_name: str
    base_model: str
    seed: int
    confirmatory_seeds: tuple[int, ...]
    device: str
    max_length: int
    batch_size: int
    learning_rate: float
    epochs: int
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    learning_rate_schedule: str
    train_examples_per_task: int
    validation_examples_per_task: int
    tasks: tuple[str, ...]
    target_modules: tuple[str, ...]
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    atom_count: int
    active_atoms_during_training: int
    active_atoms_for_primary_evaluation: int
    atom_scaling: float
    sparsity_lambda: float

    def __post_init__(self) -> None:
        # YAML presents sequences as lists.  Normalize them once so every
        # consumer sees an immutable, hashable representation.
        object.__setattr__(
            self,
            "confirmatory_seeds",
            _tuple_of_ints("confirmatory_seeds", self.confirmatory_seeds),
        )
        object.__setattr__(self, "tasks", _tuple_of_strings("tasks", self.tasks))
        object.__setattr__(
            self,
            "target_modules",
            _tuple_of_strings("target_modules", self.target_modules),
        )

        if not isinstance(self.experiment_name, str):
            raise TypeError("experiment_name must be a string")
        if self.experiment_name not in EXPERIMENT_NAMES:
            allowed = ", ".join(sorted(EXPERIMENT_NAMES))
            raise ValueError(f"experiment_name must be one of: {allowed}")
        if not isinstance(self.base_model, str) or not self.base_model:
            raise ValueError("base_model must be a non-empty string")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")

        for name, minimum in (
            ("seed", 0),
            ("max_length", 1),
            ("batch_size", 1),
            ("epochs", 1),
            ("train_examples_per_task", 1),
            ("validation_examples_per_task", 1),
            ("lora_rank", 1),
            ("atom_count", 1),
            ("active_atoms_during_training", 1),
            ("active_atoms_for_primary_evaluation", 1),
        ):
            _require_int(name, getattr(self, name), minimum=minimum)

        _require_number("learning_rate", self.learning_rate, minimum=0.0)
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be > 0")
        _require_number("weight_decay", self.weight_decay, minimum=0.0)
        _require_number(
            "adam_beta1", self.adam_beta1, minimum=0.0, maximum=1.0, maximum_inclusive=False
        )
        _require_number(
            "adam_beta2", self.adam_beta2, minimum=0.0, maximum=1.0, maximum_inclusive=False
        )
        _require_number("adam_epsilon", self.adam_epsilon, minimum=0.0)
        if self.adam_epsilon == 0:
            raise ValueError("adam_epsilon must be > 0")
        _require_number("lora_alpha", self.lora_alpha, minimum=0.0)
        if self.lora_alpha == 0:
            raise ValueError("lora_alpha must be > 0")
        _require_number(
            "lora_dropout", self.lora_dropout, minimum=0.0, maximum=1.0, maximum_inclusive=False
        )
        _require_number("atom_scaling", self.atom_scaling, minimum=0.0)
        if self.atom_scaling == 0:
            raise ValueError("atom_scaling must be > 0")
        _require_number("sparsity_lambda", self.sparsity_lambda, minimum=0.0)

        if self.learning_rate_schedule != "constant":
            raise ValueError("H1 supports only a constant learning-rate schedule")
        if len(set(self.confirmatory_seeds)) != len(self.confirmatory_seeds):
            raise ValueError("confirmatory_seeds must not contain duplicates")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must not contain duplicates")
        unknown_tasks = set(self.tasks).difference(H1_TASKS)
        if unknown_tasks:
            raise ValueError(f"unknown H1 task(s): {', '.join(sorted(unknown_tasks))}")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must not contain duplicates")
        unknown_targets = set(self.target_modules).difference(H1_TARGET_MODULES)
        if unknown_targets:
            raise ValueError(f"unsupported target module(s): {', '.join(sorted(unknown_targets))}")
        if self.active_atoms_during_training > self.atom_count:
            raise ValueError("active_atoms_during_training cannot exceed atom_count")
        if self.active_atoms_for_primary_evaluation > self.active_atoms_during_training:
            raise ValueError(
                "active_atoms_for_primary_evaluation cannot exceed "
                "active_atoms_during_training"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        """Build a configuration from a mapping, rejecting schema drift."""
        if not isinstance(values, Mapping):
            raise TypeError("configuration must be a mapping")
        expected = {field.name for field in fields(cls)}
        supplied = set(values)
        missing = expected.difference(supplied)
        unknown = supplied.difference(expected)
        problems: list[str] = []
        if missing:
            problems.append(f"missing field(s): {', '.join(sorted(missing))}")
        if unknown:
            problems.append(f"unknown field(s): {', '.join(sorted(unknown))}")
        if problems:
            raise ValueError("; ".join(problems))
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-friendly dictionary in declaration order."""
        values = asdict(self)
        values["confirmatory_seeds"] = list(self.confirmatory_seeds)
        values["tasks"] = list(self.tasks)
        values["target_modules"] = list(self.target_modules)
        return values

    def with_overrides(self, **changes: Any) -> "ExperimentConfig":
        """Return a validated development configuration with explicit changes."""
        return replace(self, **changes)

    @property
    def active_atoms(self) -> int:
        """Backward-compatible shorthand used in the original roadmap."""
        return self.active_atoms_for_primary_evaluation

    def validate_h1_contract(self) -> None:
        """Raise if the configuration is not the locked confirmatory contract."""
        expected: dict[str, Any] = {
            "base_model": "prajjwal1/bert-tiny",
            "device": "cpu",
            "tasks": H1_TASKS,
            "train_examples_per_task": 2000,
            "validation_examples_per_task": 500,
            "max_length": 128,
            "batch_size": 8,
            "learning_rate": 0.0003,
            "epochs": 3,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "learning_rate_schedule": "constant",
            "target_modules": H1_TARGET_MODULES,
            "lora_rank": 4,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "atom_count": 8,
            "active_atoms_during_training": 8,
            "active_atoms_for_primary_evaluation": 4,
            "atom_scaling": 1.0,
            "sparsity_lambda": 1e-5,
            "confirmatory_seeds": H1_CONFIRMATORY_SEEDS,
        }
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {value!r})"
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if self.seed not in self.confirmatory_seeds:
            mismatches.append(
                f"seed={self.seed!r} (expected one of {self.confirmatory_seeds!r})"
            )
        if mismatches:
            raise ValueError("configuration violates the locked H1 contract: " + "; ".join(mismatches))


def load_config(
    path: str | Path,
    *,
    require_h1_contract: bool = False,
) -> ExperimentConfig:
    """Load one UTF-8 YAML file into an :class:`ExperimentConfig`."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ValueError(f"configuration file is empty: {config_path}")
    config = ExperimentConfig.from_mapping(raw)
    if require_h1_contract:
        config.validate_h1_contract()
    return config


def format_config(config: ExperimentConfig) -> str:
    """Render a resolved configuration as stable YAML."""
    return yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True).rstrip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a resolved CGMoE H1 configuration")
    parser.add_argument("path", type=Path, help="YAML configuration to load")
    parser.add_argument(
        "--require-h1-contract",
        action="store_true",
        help="reject development overrides and require the locked confirmatory values",
    )
    return parser


def main() -> None:
    """CLI used to validate and print one resolved YAML configuration."""
    args = _build_parser().parse_args()
    print(format_config(load_config(args.path, require_h1_contract=args.require_h1_contract)))


if __name__ == "__main__":  # pragma: no cover - exercised as a command
    main()
