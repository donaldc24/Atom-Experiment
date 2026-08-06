from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from cgmoe_h1.models.lora import LoRALinear, lora_parameter_count


def test_lora_initial_output_shape_and_exact_base_match() -> None:
    torch.manual_seed(3)
    base = nn.Linear(5, 7)
    reference = copy.deepcopy(base)
    layer = LoRALinear(base, rank=4, alpha=4, dropout=0.0)
    x = torch.randn(2, 3, 5)

    assert layer(x).shape == (2, 3, 7)
    torch.testing.assert_close(layer(x), reference(x), rtol=0.0, atol=0.0)
    assert torch.count_nonzero(layer.lora_b.weight) == 0


def test_lora_gradient_boundary_and_exact_parameter_count() -> None:
    layer = LoRALinear(nn.Linear(5, 7), rank=4, alpha=4)
    layer(torch.randn(8, 5)).square().mean().backward()

    assert all(not parameter.requires_grad for parameter in layer.base.parameters())
    assert all(parameter.grad is None for parameter in layer.base.parameters())
    assert layer.lora_a.weight.grad is not None
    assert layer.lora_b.weight.grad is not None
    assert layer.adapter_parameter_count() == 4 * 5 + 7 * 4
    assert lora_parameter_count(layer) == 48


def test_lora_formula_includes_alpha_over_rank_scaling() -> None:
    base = nn.Linear(2, 2, bias=False)
    layer = LoRALinear(base, rank=1, alpha=2)
    with torch.no_grad():
        base.weight.zero_()
        layer.lora_a.weight.copy_(torch.tensor([[1.0, 2.0]]))
        layer.lora_b.weight.copy_(torch.tensor([[3.0], [4.0]]))

    output = layer(torch.tensor([[2.0, 1.0]]))
    torch.testing.assert_close(output, torch.tensor([[24.0, 32.0]]))


def test_lora_state_dict_round_trip_preserves_output() -> None:
    torch.manual_seed(11)
    layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)
    with torch.no_grad():
        layer.lora_b.weight.normal_()
    x = torch.randn(5, 4)
    state = copy.deepcopy(layer.state_dict())

    restored = LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored(x), layer(x), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("rank", [0, -1])
def test_lora_rejects_invalid_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(2, 2), rank=rank, alpha=4)
