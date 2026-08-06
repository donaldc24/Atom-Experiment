"""Tests for typed configuration and reproducibility helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cgmoe_h1.config import (
    H1_CONFIRMATORY_SEEDS,
    H1_TASKS,
    ExperimentConfig,
    format_config,
    load_config,
)
from cgmoe_h1.utils.reproducibility import make_torch_generator, set_seed

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("filename", "experiment_name"),
    (("baseline.yaml", "independent_lora"), ("atoms.yaml", "shared_atoms")),
)
def test_checked_in_configs_load_and_match_locked_contract(
    filename: str,
    experiment_name: str,
) -> None:
    config = load_config(ROOT / "configs" / filename, require_h1_contract=True)

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == experiment_name
    assert config.tasks == H1_TASKS
    assert config.confirmatory_seeds == H1_CONFIRMATORY_SEEDS
    assert config.target_modules == ("query", "value")
    assert config.active_atoms == 4


def test_resolved_config_format_round_trip(tmp_path: Path) -> None:
    original = load_config(ROOT / "configs" / "baseline.yaml")
    output = tmp_path / "resolved.yaml"
    output.write_text(format_config(original), encoding="utf-8")

    assert load_config(output) == original


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_config_rejects_schema_drift(tmp_path: Path, mutation: str) -> None:
    values = load_config(ROOT / "configs" / "baseline.yaml").to_dict()
    if mutation == "missing":
        values.pop("batch_size")
    else:
        values["mystery_setting"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="field"):
        load_config(path)


def test_general_config_allows_predeclared_ablation_shapes() -> None:
    config = load_config(ROOT / "configs" / "atoms.yaml")

    ablation = config.with_overrides(
        atom_count=2,
        active_atoms_during_training=2,
        active_atoms_for_primary_evaluation=1,
        lora_rank=4,
    )

    assert ablation.atom_count == 2
    with pytest.raises(ValueError, match="locked H1 contract"):
        ablation.validate_h1_contract()


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"active_atoms_during_training": 9}, "cannot exceed atom_count"),
        (
            {
                "active_atoms_during_training": 4,
                "active_atoms_for_primary_evaluation": 5,
            },
            "cannot exceed active_atoms_during_training",
        ),
        ({"tasks": ("sst2", "sst2")}, "duplicates"),
        ({"adam_beta1": 1.0}, "adam_beta1"),
        ({"seed": True}, "seed must be an integer"),
    ),
)
def test_config_rejects_inconsistent_values(changes: dict[str, object], message: str) -> None:
    config = load_config(ROOT / "configs" / "atoms.yaml")

    with pytest.raises((TypeError, ValueError), match=message):
        config.with_overrides(**changes)


def test_development_override_is_valid_but_not_confirmatory() -> None:
    config = load_config(ROOT / "configs" / "baseline.yaml")
    development = config.with_overrides(tasks=("sst2", "mrpc"), train_examples_per_task=500)

    assert development.tasks == ("sst2", "mrpc")
    with pytest.raises(ValueError, match="tasks=.*expected"):
        development.validate_h1_contract()


def test_set_seed_repeats_all_rngs_and_tensor_initialization() -> None:
    set_seed(17)
    first = (
        random.random(),
        np.random.random(4),
        torch.randn(3, 2),
        torch.nn.Linear(3, 2).weight.detach().clone(),
    )
    set_seed(17)
    second = (
        random.random(),
        np.random.random(4),
        torch.randn(3, 2),
        torch.nn.Linear(3, 2).weight.detach().clone(),
    )

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
    assert torch.are_deterministic_algorithms_enabled()


def test_independent_torch_generators_repeat_order() -> None:
    first = torch.randperm(50, generator=make_torch_generator(29))
    second = torch.randperm(50, generator=make_torch_generator(29))

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 3.5])
def test_invalid_seed_is_rejected(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        set_seed(seed)  # type: ignore[arg-type]
