from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from cgmoe_h1.experiments import (
    coefficient_analysis,
    load_compact_checkpoint,
    save_compact_checkpoint,
)
from cgmoe_h1.models.atoms import iter_atom_layers
from cgmoe_h1.models.classifier import BertTaskClassifier
from cgmoe_h1.models.injection import inject_atoms, inject_lora
from cgmoe_h1.utils.runtime import RuntimeMonitor


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


def test_compact_lora_checkpoint_round_trip(tmp_path) -> None:
    model = _classifier(("a",))
    inject_lora(model.encoder, ("query", "value"), rank=2, alpha=2)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    trainable_names = {name for name, value in model.named_parameters() if value.requires_grad}
    record = save_compact_checkpoint(model, tmp_path, metadata={"test": True})
    assert set(record["paths"]) == {"adapter", "heads"}

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    load_compact_checkpoint(model, tmp_path)
    after = dict(model.named_parameters())
    for name in trainable_names:
        assert torch.equal(after[name], before[name])


def test_compact_atom_checkpoint_and_analysis(tmp_path) -> None:
    model = _classifier(("a", "b"))
    inject_atoms(model.encoder, ("a", "b"), ("query", "value"), atom_count=4)
    with torch.no_grad():
        for layer in iter_atom_layers(model):
            layer.coefficients[0].copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
            layer.coefficients[1].copy_(torch.tensor([4.0, 0.0, 2.0, 1.0]))

    analysis = coefficient_analysis(model, ("a", "b"), top_k=2)
    assert analysis["top_atoms_by_task"] == {"a": [0, 1], "b": [0, 2]}
    assert analysis["reused_atoms"] == [0]
    assert analysis["task_exclusive_atoms"] == [1, 2]

    record = save_compact_checkpoint(model, tmp_path, metadata={"test": True})
    assert set(record["paths"]) == {"atoms", "coefficients", "heads"}
    load_compact_checkpoint(model, tmp_path)


def test_runtime_monitor_records_elapsed_and_memory() -> None:
    with RuntimeMonitor(poll_interval=0.001) as monitor:
        _ = torch.ones(1024)
    result = monitor.result()
    assert result.elapsed_seconds >= 0
    assert result.peak_rss_bytes > 0
