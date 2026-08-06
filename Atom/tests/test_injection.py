from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cgmoe_h1.models.atoms import AtomLinear
from cgmoe_h1.models.classifier import BertTaskClassifier
from cgmoe_h1.models.injection import (
    extract_adapter_state_dict,
    inject_atoms,
    inject_lora,
    load_adapter_state_dict,
    resolve_target_linears,
)
from cgmoe_h1.models.lora import LoRALinear


class SyntheticSelfAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.query(hidden) + self.key(hidden) + self.value(hidden)


class SyntheticAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self = SyntheticSelfAttention(hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.self(hidden))


class SyntheticLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = SyntheticAttention(hidden_size)
        self.intermediate = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.intermediate(self.attention(hidden))


class SyntheticBert(nn.Module):
    def __init__(self, hidden_size: int = 4, layer_count: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = nn.Embedding(31, hidden_size)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [SyntheticLayer(hidden_size) for _ in range(layer_count)]
        )
        # Bare-name decoys prove that injection does not use endswith("query").
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


TARGETS = ("attention.self.query", "attention.self.value")


def test_resolver_and_lora_injection_replace_only_exact_targets() -> None:
    encoder = SyntheticBert()
    model = BertTaskClassifier(encoder, num_labels=2)
    key_ids = [id(layer.attention.self.key) for layer in encoder.encoder.layer]
    output_ids = [id(layer.attention.output) for layer in encoder.encoder.layer]
    decoy_ids = (id(encoder.query), id(encoder.value), id(encoder.classifier))

    names = inject_lora(model, TARGETS, rank=4, alpha=4, expected_count=4)

    assert names == [
        "encoder.encoder.layer.0.attention.self.query",
        "encoder.encoder.layer.0.attention.self.value",
        "encoder.encoder.layer.1.attention.self.query",
        "encoder.encoder.layer.1.attention.self.value",
    ]
    assert all(
        isinstance(layer.attention.self.query, LoRALinear)
        and isinstance(layer.attention.self.value, LoRALinear)
        for layer in encoder.encoder.layer
    )
    assert [id(layer.attention.self.key) for layer in encoder.encoder.layer] == key_ids
    assert [id(layer.attention.output) for layer in encoder.encoder.layer] == output_ids
    assert (id(encoder.query), id(encoder.value), id(encoder.classifier)) == decoy_ids
    assert model(torch.tensor([[1, 2, 3]])).shape == (1, 2)


def test_lora_injected_classifier_has_exact_trainable_parameter_count() -> None:
    model = BertTaskClassifier(SyntheticBert(), num_labels=2)
    inject_lora(model, ("query", "value"), rank=4, alpha=4)

    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all("lora_" in name or name.startswith("heads.") for name in trainable)
    # Four 4x4 targets: rank * (d_in + d_out) = 32 each; one 4->2 head + bias = 10.
    assert sum(parameter.numel() for parameter in trainable.values()) == 4 * 32 + 10

    model(torch.tensor([[1, 2], [3, 4]])).sum().backward()
    assert all(parameter.grad is not None for parameter in trainable.values())


def test_atom_injection_uses_shared_context_and_task_specific_rows() -> None:
    model = BertTaskClassifier(
        SyntheticBert(),
        task_num_labels={"sst2": 2, "mrpc": 2},
    )
    names = inject_atoms(
        model,
        task_ids=["sst2", "mrpc"],
        target_suffixes=TARGETS,
        atom_count=8,
        expected_count=4,
    )
    layers = [model.get_submodule(name) for name in names]

    assert all(isinstance(layer, AtomLinear) for layer in layers)
    assert all(layer.task_context is model.task_context for layer in layers)
    atom_ids = [(id(layer.atom_u), id(layer.atom_v)) for layer in layers]

    model.set_active_task("sst2")
    model(torch.tensor([[1, 2]])).sum().backward()
    for layer in layers:
        assert torch.count_nonzero(layer.coefficients.grad[0]) > 0
        assert torch.count_nonzero(layer.coefficients.grad[1]) == 0

    model.zero_grad(set_to_none=True)
    model.set_task("mrpc", top_k=4)
    assert model(torch.tensor([[1, 2]])).shape == (1, 2)
    assert [(id(layer.atom_u), id(layer.atom_v)) for layer in layers] == atom_ids
    assert all(layer._effective_top_k() == 4 for layer in layers)

    with pytest.raises(KeyError, match="unsupported task ID"):
        model.set_active_task("missing")


@pytest.mark.parametrize("kind", ["lora", "atoms"])
def test_compact_adapter_and_head_state_round_trip_excludes_base(kind: str) -> None:
    torch.manual_seed(21)
    original_encoder = SyntheticBert()
    left = BertTaskClassifier(copy.deepcopy(original_encoder), num_labels=2)
    right = BertTaskClassifier(copy.deepcopy(original_encoder), num_labels=2)
    if kind == "lora":
        inject_lora(left, TARGETS, rank=2, alpha=2)
        inject_lora(right, TARGETS, rank=2, alpha=2)
        for module in left.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.lora_b.weight)
    else:
        inject_atoms(left, ["default"], TARGETS, atom_count=3)
        inject_atoms(right, ["default"], TARGETS, atom_count=3)

    batch = torch.tensor([[1, 2, 3]])
    state = extract_adapter_state_dict(left)
    assert state
    assert not any("base.weight" in key or "embeddings" in key for key in state)
    load_adapter_state_dict(right, state)
    torch.testing.assert_close(right(batch), left(batch), rtol=0.0, atol=0.0)


def test_injection_rejects_wrong_suffix_and_target_count() -> None:
    model = SyntheticBert()
    with pytest.raises(ValueError, match="only full attention"):
        resolve_target_linears(model, ("key",))
    with pytest.raises(ValueError, match="expected 5"):
        inject_lora(model, TARGETS, rank=4, alpha=4, expected_count=5)
