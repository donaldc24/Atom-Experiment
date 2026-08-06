from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cgmoe_h1.models.atoms import TaskContext
from cgmoe_h1.models.classifier import BertTaskClassifier
from cgmoe_h1.models.injection import inject_atoms, inject_lora
from cgmoe_h1.utils.parameters import (
    active_adapter_operations,
    assert_frozen_base,
    categorized_parameter_counts,
    independent_parameter_totals,
    parameter_count,
    shared_atom_parameter_totals,
)


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Module()])
        attention = nn.Module()
        attention.self = nn.Module()
        attention.self.query = nn.Linear(3, 3)
        attention.self.value = nn.Linear(3, 3)
        self.encoder.layer[0].attention = attention

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        values = torch.nn.functional.one_hot(input_ids % 3, num_classes=3).float()
        values = self.encoder.layer[0].attention.self.query(values)
        values = self.encoder.layer[0].attention.self.value(values)
        return SimpleNamespace(last_hidden_state=values)


def _classifier(tasks: tuple[str, ...]) -> BertTaskClassifier:
    return BertTaskClassifier(TinyEncoder(), task_num_labels={task: 2 for task in tasks})


def test_parameter_count_deduplicates_shared_parameters() -> None:
    parameter = nn.Parameter(torch.ones(5))
    assert parameter_count((parameter, parameter)) == 5


def test_exact_lora_accounting() -> None:
    model = _classifier(("a",))
    inject_lora(model.encoder, ("query", "value"), rank=2, alpha=2, expected_count=2)
    counts = categorized_parameter_counts(model)

    assert counts["base_model_parameters"] == 24  # two 3x3+bias linears
    assert counts["lora_adapter_parameters"] == 24  # two * rank * (3+3)
    assert counts["head_parameters"] == 8  # 3x2+bias
    assert counts["persistent_adaptation_parameters"] == 32
    assert counts["base_trainable_parameters"] == 0
    assert active_adapter_operations(model) == 24
    assert_frozen_base(model)


def test_exact_shared_atom_accounting() -> None:
    model = _classifier(("a", "b"))
    inject_atoms(
        model.encoder,
        ("a", "b"),
        ("query", "value"),
        atom_count=4,
        expected_count=2,
    )
    counts = shared_atom_parameter_totals(model)

    assert counts["base_model_parameters"] == 24
    assert counts["atom_parameters"] == 48  # two * 4 * (3+3)
    assert counts["coefficient_parameters"] == 16  # two * 2 tasks * 4
    assert counts["head_parameters"] == 16
    assert counts["total_persistent_task_parameters"] == 80
    assert active_adapter_operations(model, active_atoms=2) == 24


def test_independent_totals_count_base_once_and_task_state_each_time() -> None:
    models = {task: _classifier((task,)) for task in ("a", "b")}
    for model in models.values():
        inject_lora(model.encoder, ("query", "value"), rank=2, alpha=2)

    counts = independent_parameter_totals(models)
    assert counts == {
        "base_model_parameters": 24,
        "base_trainable_parameters": 0,
        "adapter_parameters": 48,
        "head_parameters": 16,
        "uncategorized_trainable_parameters": 0,
        "total_persistent_task_parameters": 64,
    }


def test_active_atom_bound_is_validated() -> None:
    model = _classifier(("a",))
    inject_atoms(model.encoder, ("a",), ("query", "value"), atom_count=2)
    with pytest.raises(ValueError, match="active_atoms"):
        active_adapter_operations(model, active_atoms=3)
