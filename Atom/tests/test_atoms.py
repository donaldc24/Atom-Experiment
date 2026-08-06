from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from cgmoe_h1.models.atoms import (
    AtomLinear,
    TaskContext,
    atom_parameter_count,
    coefficient_l1_penalty,
    coefficient_l1_regularization,
)


def test_atom_output_shape_initially_near_base_and_exact_parameter_count() -> None:
    torch.manual_seed(5)
    base = nn.Linear(5, 7)
    reference = copy.deepcopy(base)
    layer = AtomLinear(base, task_ids=["a", "b"], atom_count=8)
    x = torch.randn(3, 4, 5)

    output = layer(x, task_id="a")
    assert output.shape == (3, 4, 7)
    assert (output - reference(x)).abs().max() < 1e-3
    expected = 8 * (5 + 7) + 2 * 8
    assert layer.adapter_parameter_count() == expected
    assert atom_parameter_count(layer) == expected
    assert all(not parameter.requires_grad for parameter in layer.base.parameters())


def test_atom_vectorized_formula_is_exact() -> None:
    layer = AtomLinear(nn.Linear(2, 2, bias=False), ["a", "b"], atom_count=2)
    with torch.no_grad():
        layer.base.weight.zero_()
        layer.atom_v.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        layer.atom_u.copy_(torch.tensor([[2.0, 3.0], [5.0, 7.0]]))
        layer.coefficients.copy_(torch.tensor([[11.0, 13.0], [17.0, 19.0]]))

    x = torch.tensor([[2.0, 3.0]])
    # Task a: 11 * 2 * [2,3] + 13 * 3 * [5,7].
    torch.testing.assert_close(layer(x, "a"), torch.tensor([[239.0, 339.0]]))
    torch.testing.assert_close(layer(x, "b"), torch.tensor([[353.0, 501.0]]))


def test_tasks_share_atoms_but_select_distinct_coefficient_rows() -> None:
    layer = AtomLinear(nn.Linear(3, 2), ["sst2", "mrpc"], atom_count=4)
    atom_ids = (id(layer.atom_u), id(layer.atom_v))

    assert layer.coefficient_row("sst2").data_ptr() != layer.coefficient_row("mrpc").data_ptr()
    layer(torch.randn(2, 3), "sst2")
    layer(torch.randn(2, 3), "mrpc")
    assert (id(layer.atom_u), id(layer.atom_v)) == atom_ids


@pytest.mark.parametrize("task_id, inactive_row", [("sst2", 1), ("mrpc", 0)])
def test_gradients_reach_shared_atoms_and_only_active_coefficient_row(
    task_id: str,
    inactive_row: int,
) -> None:
    torch.manual_seed(13)
    layer = AtomLinear(nn.Linear(3, 2), ["sst2", "mrpc"], atom_count=4)
    layer(torch.randn(6, 3), task_id).square().mean().backward()

    assert layer.atom_u.grad is not None and torch.count_nonzero(layer.atom_u.grad) > 0
    assert layer.atom_v.grad is not None and torch.count_nonzero(layer.atom_v.grad) > 0
    assert layer.coefficients.grad is not None
    assert torch.count_nonzero(layer.coefficients.grad[1 - inactive_row]) > 0
    torch.testing.assert_close(
        layer.coefficients.grad[inactive_row],
        torch.zeros_like(layer.coefficients.grad[inactive_row]),
        rtol=0.0,
        atol=0.0,
    )
    assert all(parameter.grad is None for parameter in layer.base.parameters())


def test_l1_regularization_is_exact_finite_and_differentiable() -> None:
    model = nn.Sequential(
        AtomLinear(nn.Linear(2, 2), ["a", "b"], atom_count=2),
        AtomLinear(nn.Linear(2, 2), ["a", "b"], atom_count=2),
    )
    with torch.no_grad():
        model[0].coefficients.copy_(torch.tensor([[1.0, -3.0], [50.0, 60.0]]))
        model[1].coefficients.copy_(torch.tensor([[-5.0, 7.0], [70.0, 80.0]]))

    penalty = coefficient_l1_penalty(model, "a")
    assert penalty.item() == 4.0
    assert torch.isfinite(penalty)
    zero_weight = coefficient_l1_regularization(model, "a", weight=0.0)
    assert zero_weight.item() == 0.0

    coefficient_l1_regularization(model, "a", weight=1e-5).backward()
    torch.testing.assert_close(
        model[0].coefficients.grad[0],
        torch.tensor([2.5e-6, -2.5e-6]),
    )
    assert torch.count_nonzero(model[0].coefficients.grad[1]) == 0


def test_top_k_mask_uses_magnitude_and_lower_index_tie_break() -> None:
    context = TaskContext(["a", "b"])
    layer = AtomLinear(nn.Linear(2, 2), ["a", "b"], atom_count=5, task_context=context)
    with torch.no_grad():
        layer.coefficients[0].copy_(torch.tensor([-9.0, 4.0, -4.0, 1.0, 0.0]))

    assert layer.topk_mask("a", 2).tolist() == [True, True, False, False, False]
    assert layer.topk_mask("a", 9).tolist() == [True] * 5
    assert layer.topk_mask("a", 0).tolist() == [False] * 5

    context.set_active_task("a", top_k=2)
    assert layer._effective_top_k() == 2
    context.clear_top_k()
    assert layer._effective_top_k() is None


def test_context_selection_and_atom_state_round_trip() -> None:
    context = TaskContext(["a", "b"])
    layer = AtomLinear(nn.Linear(3, 2), ["a", "b"], atom_count=3, task_context=context)
    context.set_active_task("b")
    x = torch.randn(4, 3)
    expected = layer(x)
    state = copy.deepcopy(layer.state_dict())

    restored_context = TaskContext(["a", "b"], current_task_id="b")
    restored = AtomLinear(
        nn.Linear(3, 2),
        ["a", "b"],
        atom_count=3,
        task_context=restored_context,
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(restored(x), expected, rtol=0.0, atol=0.0)

    with pytest.raises(KeyError, match="unsupported task ID"):
        layer(x, "missing")
