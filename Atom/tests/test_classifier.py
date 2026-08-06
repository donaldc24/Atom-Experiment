from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cgmoe_h1.models.classifier import BertTaskClassifier


class SyntheticEncoder(nn.Module):
    def __init__(self, hidden_size: int = 6) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(23, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        hidden = self.projection(self.embedding(input_ids))
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_classifier_output_shape_and_contract_initialization() -> None:
    model = BertTaskClassifier(SyntheticEncoder(), num_labels=3)
    logits = model(
        input_ids=torch.tensor([[1, 2, 3], [3, 4, 5]]),
        attention_mask=torch.ones(2, 3),
    )

    assert logits.shape == (2, 3)
    assert torch.count_nonzero(model.head.bias) == 0
    assert model.bert is model.encoder
    assert model.task_context.current_task_id == "default"


def test_base_is_frozen_and_backward_reaches_head_only() -> None:
    model = BertTaskClassifier(SyntheticEncoder(), num_labels=2)

    model(torch.tensor([[1, 2], [3, 4]])).sum().backward()

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert all(parameter.grad is not None for parameter in model.head.parameters())


def test_multitask_head_and_context_selection() -> None:
    model = BertTaskClassifier(
        SyntheticEncoder(),
        task_num_labels={"sst2": 2, "three_way": 3},
    )
    batch = torch.tensor([[1, 2]])

    assert model(batch, task_id="sst2").shape == (1, 2)
    assert model(batch, task_id="three_way").shape == (1, 3)
    model.set_active_task("sst2", top_k=1)
    assert model(batch).shape == (1, 2)
    assert model.task_context.top_k == 1

    with pytest.raises(KeyError, match="unsupported task ID"):
        model.set_task("missing")


def test_multitask_forward_requires_task_selection() -> None:
    model = BertTaskClassifier(
        SyntheticEncoder(),
        task_num_labels={"sst2": 2, "mrpc": 2},
    )
    with pytest.raises(RuntimeError, match="task_id is required"):
        model(torch.tensor([[1, 2]]))
