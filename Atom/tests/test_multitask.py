from __future__ import annotations

from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from cgmoe_h1.models.atoms import AtomLinear
from cgmoe_h1.training.multitask import (
    UniformTaskBatchIterator,
    assert_shared_atom_gradient_contract,
    collect_shared_atom_diagnostics,
    train_multitask,
)
from cgmoe_h1.training.trainer import train_step


def _batch(task: str, value: int) -> dict[str, object]:
    return {
        "features": torch.tensor([[float(value), 1.0]]),
        "labels": torch.tensor([value % 2]),
        "example_id": f"{task}-{value}",
    }


def test_complete_pass_scheduler_is_exact_and_repeatable() -> None:
    loaders = {
        "a": [_batch("a", value) for value in range(3)],
        "b": [_batch("b", value) for value in range(5)],
        "c": [_batch("c", value) for value in range(2)],
    }
    first = UniformTaskBatchIterator(loaders, seed=17, epoch=2)
    second = UniformTaskBatchIterator(loaders, seed=17, epoch=2)

    first_batches = list(first)
    second_batches = list(second)

    assert Counter(item.task_id for item in first_batches) == {"a": 3, "b": 5, "c": 2}
    assert [item.task_id for item in first_batches] == [item.task_id for item in second_batches]
    assert [item.batch["example_id"] for item in first_batches] == [
        item.batch["example_id"] for item in second_batches
    ]
    assert len(first) == 10


def test_uniform_scheduler_cycles_exhausted_loaders_and_is_balanced() -> None:
    loaders = {
        "a": [_batch("a", 0)],
        "b": [_batch("b", 0), _batch("b", 1)],
        "c": [_batch("c", 0), _batch("c", 1), _batch("c", 2)],
    }
    scheduler = UniformTaskBatchIterator(
        loaders,
        seed=29,
        mode="uniform",
        steps_per_epoch=3_000,
    )

    batches = list(scheduler)
    counts = Counter(item.task_id for item in batches)

    assert len(batches) == 3_000
    assert max(counts.values()) - min(counts.values()) < 100
    assert counts == Counter(scheduler.task_schedule())


class TinyMultitaskClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(2, 2, bias=False)
        self.heads = nn.ModuleDict({"left": nn.Linear(2, 2), "right": nn.Linear(2, 2)})

    def forward(self, features: torch.Tensor, task_id: str) -> torch.Tensor:
        return self.heads[task_id](torch.tanh(self.shared(features)))


def _task_loader(reverse: bool = False) -> DataLoader:
    features = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    if reverse:
        labels = 1 - labels
    rows = [
        {"features": feature, "labels": label, "example_id": index}
        for index, (feature, label) in enumerate(zip(features, labels, strict=True))
    ]
    return DataLoader(rows, batch_size=2, shuffle=False)


def test_multitask_trainer_updates_every_task_and_selects_unweighted_mean() -> None:
    torch.manual_seed(12)
    model = TinyMultitaskClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    loaders = {"left": _task_loader(), "right": _task_loader(reverse=True)}

    result = train_multitask(
        model,
        loaders,
        loaders,
        optimizer,
        epochs=3,
        seed=17,
    )

    assert len(result.history) == 3
    for epoch in result.history:
        assert epoch.global_training.optimizer_steps == 6
        assert epoch.training["left"].batches == 3
        assert epoch.training["right"].batches == 3
        expected = sum(
            validation.metrics["accuracy"] for validation in epoch.validation.values()
        ) / 2
        assert epoch.selection_score == expected
    assert result.best_epoch in {1, 2, 3}
    assert set(result.best_validation) == {"left", "right"}


class TinyAtomClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = AtomLinear(nn.Linear(2, 2), ("left", "right"), atom_count=3)
        self.heads = nn.ModuleDict({"left": nn.Linear(2, 2), "right": nn.Linear(2, 2)})

    def forward(self, features: torch.Tensor, task_id: str) -> torch.Tensor:
        return self.heads[task_id](self.adapter(features, task_id))


def test_atom_gradient_contract_and_diagnostics_are_ready_made() -> None:
    torch.manual_seed(31)
    model = TinyAtomClassifier()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )

    train_step(
        model,
        next(iter(_task_loader())),
        optimizer,
        "cpu",
        task_id="left",
        gradient_callback=assert_shared_atom_gradient_contract,
    )
    diagnostics = collect_shared_atom_diagnostics(model, top_n=2)

    assert diagnostics["atom_gradient_l2_norm_last_step"] > 0
    assert diagnostics["atom_parameter_l2_norm"] > 0
    assert set(diagnostics["tasks"]) == {"left", "right"}
    assert diagnostics["tasks"]["left"]["coefficient_count"] == 3
    assert len(
        diagnostics["tasks"]["left"]["top_used_atoms_by_layer"]["adapter"]
    ) == 2
