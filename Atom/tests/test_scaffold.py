"""Fast checks for the chunk-1 project scaffold."""

from __future__ import annotations

import importlib
from pathlib import Path

import cgmoe_h1
import pytest
import yaml


def test_package_imports() -> None:
    assert cgmoe_h1.__version__ == "0.1.0"


def test_expected_modules_import() -> None:
    modules = (
        "cgmoe_h1.config",
        "cgmoe_h1.cli",
        "cgmoe_h1.data",
        "cgmoe_h1.metrics",
        "cgmoe_h1.models.classifier",
        "cgmoe_h1.models.lora",
        "cgmoe_h1.models.atoms",
        "cgmoe_h1.models.injection",
        "cgmoe_h1.training.trainer",
        "cgmoe_h1.training.multitask",
        "cgmoe_h1.utils.parameters",
        "cgmoe_h1.utils.reproducibility",
        "cgmoe_h1.utils.serialization",
    )

    for module_name in modules:
        assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize(
    "module_name",
    ("datasets", "evaluate", "sklearn", "torch", "tqdm", "transformers", "yaml"),
)
def test_declared_dependency_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_starter_configs_are_valid() -> None:
    expected_names = {
        "baseline.yaml": "independent_lora",
        "atoms.yaml": "shared_atoms",
    }

    for filename, experiment_name in expected_names.items():
        config = yaml.safe_load((Path("configs") / filename).read_text(encoding="utf-8"))
        assert config["experiment_name"] == experiment_name
        assert config["base_model"] == "prajjwal1/bert-tiny"
