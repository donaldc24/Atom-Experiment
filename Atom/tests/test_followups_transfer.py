"""Offline coverage for the chunk 21-22 follow-up orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import cgmoe_h1.followups_transfer as followups
from cgmoe_h1.config import H1_TASKS, load_config
from cgmoe_h1.models.atoms import AtomLinear


def _independent_record() -> dict:
    return {
        "tasks": {
            task: {
                "best": {"metrics": {"primary_score": 0.80 - index * 0.01}},
                "parameter_counts": {
                    "base_model_parameters": 1000,
                    "lora_adapter_parameters": 100,
                    "head_parameters": 10,
                    "persistent_adaptation_parameters": 110,
                },
            }
            for index, task in enumerate(H1_TASKS)
        }
    }


def _shared_record(tasks: tuple[str, ...], storage: int = 500) -> dict:
    return {
        "tasks": {
            task: {"top_k": {"metrics": {"primary_score": 0.76 - index * 0.01}}}
            for index, task in enumerate(tasks)
        },
        "parameter_counts": {"total_persistent_task_parameters": storage},
        "active_atoms_for_evaluation": 4,
        "active_adapter_operations_per_token": {"top_k": 4096},
    }


def _configs():
    root = Path(__file__).parents[1]
    return load_config(root / "configs/baseline.yaml"), load_config(root / "configs/atoms.yaml")


def test_copy_dictionary_ignores_old_task_state_and_freezes_vectors(tmp_path: Path) -> None:
    source = nn.Module()
    source.layer = AtomLinear(nn.Linear(3, 2), ("old",), atom_count=2)
    with torch.no_grad():
        source.layer.atom_u.fill_(3.0)
        source.layer.atom_v.fill_(5.0)
        source.layer.coefficients.fill_(99.0)
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    torch.save(
        {
            "state_dict": {
                "layer.atom_u": source.layer.atom_u.detach().clone(),
                "layer.atom_v": source.layer.atom_v.detach().clone(),
                "layer._extra_state": {"task_ids": ("old",)},
            }
        },
        checkpoint / "atoms.pt",
    )

    target = nn.Module()
    target.layer = AtomLinear(nn.Linear(3, 2), ("qqp",), atom_count=2)
    coefficients_before = target.layer.coefficients.detach().clone()
    copied = followups.copy_frozen_atom_dictionary(target, checkpoint)

    assert torch.all(target.layer.atom_u == 3.0)
    assert torch.all(target.layer.atom_v == 5.0)
    assert torch.equal(target.layer.coefficients, coefficients_before)
    assert target.layer.atom_u.requires_grad is False
    assert target.layer.atom_v.requires_grad is False
    assert target.layer.coefficients.requires_grad is True
    assert copied["tensor_count"] == 2
    assert copied["parameter_count"] == 10


def test_strong_transfer_requires_both_inclusive_quality_and_fewer_parameters() -> None:
    boundary = followups.evaluate_strong_transfer(0.76, 0.80, 20, 100)
    assert boundary["quality_ratio"] == pytest.approx(0.95)
    assert boundary["strong_transfer"] is True

    same_size = followups.evaluate_strong_transfer(0.80, 0.80, 100, 100)
    assert same_size["quality_passed"] is True
    assert same_size["fewer_new_parameters"] is False
    assert same_size["strong_transfer"] is False


def test_scaling_point_uses_exact_prefix_counts_and_top_k_quality() -> None:
    independent = _independent_record()
    tasks = H1_TASKS[:3]
    point = followups.build_scaling_point(3, tasks, independent, _shared_record(tasks, 240))

    assert point["independent_storage_parameters"] == 330
    assert point["shared_storage_parameters"] == 240
    assert point["relative_storage"] == pytest.approx(240 / 330)
    assert point["mean_quality"] == pytest.approx((0.76 + 0.75 + 0.74) / 3)
    assert point["worst_task_score"] == pytest.approx(0.74)
    assert point["active_capacity"] == 4


def test_scaling_runner_mocks_expensive_training_and_writes_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline, atoms = _configs()
    independent = _independent_record()
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        followups,
        "_load_or_run_core_independent",
        lambda config, root: independent,
    )
    monkeypatch.setattr(
        followups,
        "_load_or_run_core_shared",
        lambda config, root: _shared_record(H1_TASKS, 300),
    )

    def fake_shared(config, output_root, *, tasks, **kwargs):
        selected = tuple(tasks)
        calls.append(selected)
        return _shared_record(selected, 200 + len(selected) * 20)

    monkeypatch.setattr(followups, "run_shared_atoms", fake_shared)
    output = tmp_path / "scaling"
    summary = followups.run_scaling_curve(
        baseline,
        atoms,
        output,
        core_results_root=tmp_path / "core",
    )

    assert calls == [H1_TASKS[:index] for index in range(1, 5)]
    assert [point["task_count"] for point in summary["points"]] == [1, 2, 3, 4, 5]
    assert (output / "scaling_curve.json").is_file()
    assert (output / "scaling_curve.md").is_file()

    calls.clear()
    resumed = followups.run_scaling_curve(
        baseline,
        atoms,
        output,
        core_results_root=tmp_path / "core",
    )
    assert resumed == summary
    assert calls == []


def test_transfer_runner_mocks_expensive_training_and_declares_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline, atoms = _configs()
    prefix_root = tmp_path / "prefixes"
    source_directory = prefix_root / "prefix_4" / "seed_17"
    source_directory.mkdir(parents=True)
    (source_directory / "atoms.pt").write_bytes(b"mock")
    calls: list[str] = []

    monkeypatch.setattr(
        followups,
        "run_shared_atoms",
        lambda *args, **kwargs: {"checkpoint": {"paths": {"atoms": "unused"}}},
    )
    monkeypatch.setattr(followups, "prepare_data", lambda *args, **kwargs: object())

    def fake_target(config, output_directory, *, dictionary_checkpoint, **kwargs):
        system = "learned" if dictionary_checkpoint is not None else "random"
        calls.append(system)
        return {
            "best": {"metrics": {"primary_score": 0.722 if system == "learned" else 0.60}},
            "new_task_parameters": 20,
            "reused_dictionary_parameters": 200,
        }

    monkeypatch.setattr(followups, "run_frozen_atom_target", fake_target)
    monkeypatch.setattr(
        followups,
        "_load_or_run_head_only",
        lambda *args, **kwargs: {
            "best": {"metrics": {"primary_score": 0.62}},
            "parameter_counts": {"head_parameters": 10},
        },
    )
    monkeypatch.setattr(
        followups,
        "_load_or_run_core_independent",
        lambda *args, **kwargs: _independent_record(),
    )

    output = tmp_path / "transfer"
    summary = followups.run_frozen_atom_transfer(
        baseline,
        atoms,
        output,
        core_results_root=tmp_path / "core",
        shared_prefix_root=prefix_root,
    )

    assert calls == ["learned", "random"]
    assert summary["strong_result"]["quality_ratio"] == pytest.approx(0.95)
    assert summary["strong_result"]["strong_transfer"] is True
    assert (output / "frozen_atom_transfer.json").is_file()
    assert (output / "frozen_atom_transfer.md").is_file()
