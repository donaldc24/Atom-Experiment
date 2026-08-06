"""Numerical tests for the oracle atom-span projection."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

from cgmoe_h1.config import H1_CONFIRMATORY_SEEDS, H1_TASKS
from cgmoe_h1.models.atoms import AtomLinear
from cgmoe_h1.validation_projection import (
    aggregate_layer_errors,
    apply_projection_coefficients,
    atom_design_matrix,
    deterministic_top_k_coefficients,
    project_lora_state_onto_atoms,
    solve_matrix_projection,
    summarize_projection_cells,
)


def test_exact_span_projection_recovers_update() -> None:
    atom_u = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    atom_v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[2.0, 0.0], [0.0, -3.0]])

    coefficients, record = solve_matrix_projection(target, atom_u, atom_v, top_k=2)

    assert coefficients.tolist() == pytest.approx([2.0, -3.0], abs=1e-12)
    assert record["all_atoms"]["relative_frobenius_error"] == pytest.approx(0.0, abs=1e-12)
    assert record["all_atoms"]["explained_energy"] == pytest.approx(1.0, abs=1e-12)
    assert record["numerical_rank"] == 2


def test_nonspan_projection_reports_unexplained_update() -> None:
    atom_u = torch.tensor([[1.0, 0.0]])
    atom_v = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    coefficients, record = solve_matrix_projection(target, atom_u, atom_v, top_k=1)

    assert coefficients.tolist() == pytest.approx([0.0], abs=1e-12)
    assert record["all_atoms"]["relative_frobenius_error"] == pytest.approx(1.0)
    assert record["all_atoms"]["explained_energy"] == pytest.approx(0.0)


def test_atom_scaling_is_included_in_oracle_coefficients() -> None:
    atom_u = torch.tensor([[1.0, 2.0]])
    atom_v = torch.tensor([[3.0, 4.0]])
    target = 6.0 * torch.outer(atom_u[0], atom_v[0])

    coefficients, record = solve_matrix_projection(
        target,
        atom_u,
        atom_v,
        atom_scaling=2.0,
        top_k=1,
    )

    assert coefficients.tolist() == pytest.approx([3.0], abs=1e-10)
    assert record["all_atoms"]["relative_frobenius_error"] == pytest.approx(0.0, abs=1e-12)


def test_top_k_ties_use_lower_atom_index() -> None:
    values = torch.tensor([-2.0, 2.0, 1.0])
    pruned, selected = deterministic_top_k_coefficients(values, 1)
    assert selected == [0]
    assert pruned.tolist() == [-2.0, 0.0, 0.0]


def test_projection_top_k_is_selected_after_float32_installation_rounding() -> None:
    atom_u = torch.eye(2, dtype=torch.float64)
    atom_v = torch.eye(2, dtype=torch.float64)
    target = torch.diag(torch.tensor([1.0, 1.0 + 1e-8], dtype=torch.float64))

    coefficients, record = solve_matrix_projection(target, atom_u, atom_v, top_k=1)

    assert coefficients.dtype == torch.float32
    assert coefficients.tolist() == [1.0, 1.0]
    assert record["top_k"]["selected_atom_indices"] == [0]
    assert record["coefficient_dtype"] == "torch.float32"


def test_rank_deficient_design_reports_nonfinite_full_condition() -> None:
    atom_u = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    atom_v = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    _, record = solve_matrix_projection(torch.eye(2), atom_u, atom_v, top_k=1)

    assert record["numerical_rank"] == 1
    assert record["condition_number"] is None
    assert record["condition_number_finite"] is False
    assert record["retained_subspace_condition_number"] == pytest.approx(1.0)


def test_layer_aggregation_weights_by_target_energy() -> None:
    layers = {
        "large": {
            "all_atoms": {
                "target_squared_frobenius_norm": 100.0,
                "residual_squared_frobenius_norm": 1.0,
            }
        },
        "small": {
            "all_atoms": {
                "target_squared_frobenius_norm": 1.0,
                "residual_squared_frobenius_norm": 1.0,
            }
        },
    }
    aggregate = aggregate_layer_errors(layers, field="all_atoms")
    assert aggregate["relative_frobenius_error"] == pytest.approx((2.0 / 101.0) ** 0.5)
    assert aggregate["explained_energy"] == pytest.approx(1.0 - 2.0 / 101.0)


def test_path_aligned_compact_states_project_with_lora_scaling() -> None:
    prefix = "encoder.layer.0.attention.self.query"
    lora = OrderedDict(
        {
            f"{prefix}.lora_a.weight": torch.tensor([[1.0, 0.0]]),
            f"{prefix}.lora_b.weight": torch.tensor([[2.0], [0.0]]),
        }
    )
    atoms = OrderedDict(
        {
            f"{prefix}.atom_u": torch.tensor([[1.0, 0.0]]),
            f"{prefix}.atom_v": torch.tensor([[1.0, 0.0]]),
            f"{prefix}._extra_state": {"task_ids": ("source",)},
        }
    )

    coefficients, record = project_lora_state_onto_atoms(
        lora,
        atoms,
        lora_scaling=0.5,
        atom_scaling=1.0,
        top_k=1,
    )

    assert coefficients[prefix].tolist() == pytest.approx([1.0], abs=1e-12)
    assert record["all_atoms"]["relative_frobenius_error"] == pytest.approx(0.0, abs=1e-12)


def test_atom_design_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same atom count"):
        atom_design_matrix(torch.zeros(2, 3), torch.zeros(1, 3))


def test_apply_projection_coefficients_uses_named_task_row() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = AtomLinear(nn.Linear(2, 2), ("target",), atom_count=2)

    model = Tiny()
    apply_projection_coefficients(
        model,
        "target",
        OrderedDict({"projection": torch.tensor([1.25, -0.5], dtype=torch.float64)}),
    )
    assert model.projection.coefficient_row("target").tolist() == pytest.approx([1.25, -0.5])


def _mock_projection_cell(task: str, seed: int) -> dict:
    projection = {
        "all_atoms": {
            "target_squared_frobenius_norm": 10.0,
            "residual_squared_frobenius_norm": 0.1,
        }
    }
    random_projection = {
        "all_atoms": {
            "target_squared_frobenius_norm": 10.0,
            "residual_squared_frobenius_norm": 2.5,
        }
    }
    evaluation = lambda value: {"metrics": {"primary_score": value}}
    return {
        "task": task,
        "seed": seed,
        "fresh_lora": {"score": 0.8},
        "systems": {
            "learned_span": {
                "all_atoms": evaluation(0.78),
                "top_k": evaluation(0.76),
                "projection": projection,
            },
            "random_span": {
                "all_atoms": evaluation(0.70),
                "top_k": evaluation(0.69),
                "projection": random_projection,
            },
        },
    }


def test_projection_summary_applies_complete_grid_decision(monkeypatch) -> None:
    monkeypatch.setattr(
        "cgmoe_h1.validation_projection.validate_projection_cell",
        lambda *args, **kwargs: None,
    )
    cells = [
        _mock_projection_cell(task, seed)
        for task in H1_TASKS
        for seed in H1_CONFIRMATORY_SEEDS
    ]
    summary = summarize_projection_cells(cells)
    assert summary["cell_count"] == 15
    assert summary["aggregate"]["learned_all_atoms_quality_retention"] == pytest.approx(0.975)
    assert summary["aggregate"]["learned_all_atoms_advantage_over_random"] == pytest.approx(0.08)
    assert summary["strong_learned_span_support"] is True


def test_projection_summary_rejects_incomplete_grid(monkeypatch) -> None:
    monkeypatch.setattr(
        "cgmoe_h1.validation_projection.validate_projection_cell",
        lambda *args, **kwargs: None,
    )
    cells = [
        _mock_projection_cell(task, seed)
        for task in H1_TASKS
        for seed in H1_CONFIRMATORY_SEEDS
    ][:-1]
    with pytest.raises(ValueError, match="all 15"):
        summarize_projection_cells(cells)
