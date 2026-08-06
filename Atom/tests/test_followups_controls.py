from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch

import cgmoe_h1.followups_controls as controls
from cgmoe_h1.config import ExperimentConfig, H1_TASKS, load_config
from cgmoe_h1.followups_controls import (
    CONTROL_IDS,
    CONTROL_SEED,
    ControlRunContext,
    average_effective_lora_state,
    cosine_effective_lora_similarity,
    deterministic_label_permutation,
    deterministically_shuffle_labels,
    effective_lora_updates,
    load_independent_compact_states,
    nearest_other_task_adapters,
    run_control_suite,
    validate_control_configs,
)
from cgmoe_h1.utils.serialization import read_json


@pytest.fixture
def locked_configs() -> tuple[ExperimentConfig, ExperimentConfig]:
    baseline = load_config("configs/baseline.yaml").with_overrides(seed=CONTROL_SEED)
    atoms = load_config("configs/atoms.yaml").with_overrides(seed=CONTROL_SEED)
    return baseline, atoms


def _state(update: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    rank = update.shape[1]
    return OrderedDict(
        {
            "encoder.layer.0.attention.self.query.lora_a.weight": torch.eye(rank),
            "encoder.layer.0.attention.self.query.lora_b.weight": update.clone(),
        }
    )


def test_all_six_controls_are_stable_and_configs_are_seed17_locked(
    locked_configs: tuple[ExperimentConfig, ExperimentConfig],
) -> None:
    baseline, atoms = locked_configs
    assert CONTROL_IDS == (
        "random_frozen_atoms",
        "average_independent_loras",
        "nearest_other_task_lora",
        "shared_multitask_lora",
        "shared_atoms_shuffled_labels",
        "shared_atoms_no_sparsity",
    )
    validate_control_configs(baseline, atoms)

    with pytest.raises(ValueError, match="locked to seed 17"):
        validate_control_configs(
            baseline.with_overrides(seed=29),
            atoms.with_overrides(seed=29),
        )
    with pytest.raises(ValueError, match="locked H1 contract"):
        validate_control_configs(
            baseline.with_overrides(train_examples_per_task=10),
            atoms,
        )


def test_effective_updates_similarity_and_rank_preserving_average() -> None:
    left_matrix = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    right_matrix = torch.tensor([[0.0, 0.0], [0.0, 3.0]])
    left = _state(left_matrix)
    right = _state(right_matrix)

    update = next(iter(effective_lora_updates(left).values()))
    torch.testing.assert_close(update, left_matrix.double())
    expected_cosine = float(
        torch.dot(left_matrix.flatten(), right_matrix.flatten())
        / (left_matrix.norm() * right_matrix.norm())
    )
    assert cosine_effective_lora_similarity(left, right) == pytest.approx(expected_cosine)

    averaged = average_effective_lora_state({"left": left, "right": right}, rank=2)
    averaged_update = next(iter(effective_lora_updates(averaged).values()))
    torch.testing.assert_close(
        averaged_update,
        ((left_matrix + right_matrix) / 2).double(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert set(averaged) == set(left)
    assert all("base" not in key for key in averaged)


def test_nearest_retrieval_excludes_target_and_breaks_ties_by_task_order() -> None:
    states = {
        "target": _state(torch.tensor([[1.0, 0.0], [0.0, 0.0]])),
        "first": _state(torch.tensor([[2.0, 0.0], [0.0, 0.0]])),
        "second": _state(torch.tensor([[3.0, 0.0], [0.0, 0.0]])),
    }
    retrieval = nearest_other_task_adapters(
        states,
        tasks=("target", "first", "second"),
    )

    assert retrieval["target"]["source_task"] == "first"
    assert "target" not in retrieval["target"]["similarities"]
    assert retrieval["target"]["similarities"]["first"] == pytest.approx(1.0)
    assert retrieval["target"]["similarities"]["second"] == pytest.approx(1.0)


def test_label_shuffle_is_deterministic_and_preserves_the_multiset() -> None:
    labels = [0, 0, 0, 1, 1, 1, 1, 0, 1, 0]
    first = deterministically_shuffle_labels(labels, seed=17, task_index=2)
    second = deterministically_shuffle_labels(labels, seed=17, task_index=2)
    different_task = deterministically_shuffle_labels(labels, seed=17, task_index=3)

    assert first == second
    assert sorted(first) == sorted(labels)
    assert first != different_task
    permutation = deterministic_label_permutation(len(labels), 17, 2)
    assert sorted(permutation) == list(range(len(labels)))


def _write_component(
    directory: Path,
    component: str,
    state: OrderedDict[str, torch.Tensor],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "component": component,
            "metadata": {"seed": 17},
            "state_dict": state,
        },
        directory / f"{component}.pt",
    )


def test_independent_loader_reads_compact_adapter_and_target_head_only(tmp_path: Path) -> None:
    tasks = ("sst2", "mrpc")
    for task_index, task in enumerate(tasks, start=1):
        directory = tmp_path / "seed_17" / task
        _write_component(directory, "adapter", _state(torch.eye(2) * task_index))
        _write_component(
            directory,
            "heads",
            OrderedDict(
                {
                    f"heads.{task}.weight": torch.ones(2, 2),
                    f"heads.{task}.bias": torch.zeros(2),
                }
            ),
        )

    adapters, heads = load_independent_compact_states(tmp_path, tasks=tasks)
    assert set(adapters) == set(tasks)
    assert set(heads) == set(tasks)
    assert all("base" not in key for state in adapters.values() for key in state)
    assert all(key.startswith("heads.") for state in heads.values() for key in state)

    bad_directory = tmp_path / "seed_17" / "bad"
    bad_state = _state(torch.eye(2))
    bad_state["encoder.embeddings.weight"] = torch.ones(2, 2)
    _write_component(bad_directory, "adapter", bad_state)
    _write_component(
        bad_directory,
        "heads",
        OrderedDict({"heads.bad.weight": torch.ones(2, 2)}),
    )
    with pytest.raises(ValueError, match="non-adapter keys"):
        load_independent_compact_states(tmp_path, tasks=("bad",))


def _mock_record(control_id: str, config: ExperimentConfig) -> dict[str, Any]:
    score = (CONTROL_IDS.index(control_id) + 1) / 10
    spec = controls.CONTROL_BY_ID[control_id]
    return {
        "schema_version": 1,
        "control_id": control_id,
        "title": spec.title,
        "rules_out": spec.rules_out,
        "status": "complete",
        "seed": 17,
        "run_kind": "followup_control",
        "locked_budget": {},
        "design": {"mock_compute": True},
        "tasks": {
            task: {
                "primary": {
                    "loss": 1.0,
                    "metrics": {"primary_score": score},
                    "examples": 2,
                    "batches": 1,
                }
            }
            for task in H1_TASKS
        },
        "resolved_config": config.to_dict(),
        "compact_state_only": True,
    }


def test_suite_mock_compute_persists_reports_and_resumes_without_recompute(
    tmp_path: Path,
    locked_configs: tuple[ExperimentConfig, ExperimentConfig],
) -> None:
    baseline, atoms = locked_configs
    calls: list[str] = []

    def executor(control_id: str) -> controls.ControlExecutor:
        def run(context: ControlRunContext) -> dict[str, Any]:
            assert context.prepared is None
            calls.append(control_id)
            return _mock_record(control_id, context.baseline_config)

        return run

    executors = {control_id: executor(control_id) for control_id in CONTROL_IDS}
    summary, summary_path, report_path = run_control_suite(
        baseline,
        atoms,
        tmp_path,
        tmp_path / "missing-core-is-not-read",
        executors=executors,
        validate_core=False,
    )

    assert calls == list(CONTROL_IDS)
    assert summary["status"] == "complete"
    assert summary_path.is_file() and report_path.is_file()
    assert read_json(summary_path)["controls_completed"] == list(CONTROL_IDS)
    report = report_path.read_text(encoding="utf-8")
    assert "Roadmap Chunk 25" in report
    assert "0.1000" in report and "0.6000" in report
    assert all(controls.control_result_path(tmp_path, item).is_file() for item in CONTROL_IDS)

    run_control_suite(
        baseline,
        atoms,
        tmp_path,
        tmp_path / "missing-core-is-not-read",
        executors=executors,
        validate_core=False,
    )
    assert calls == list(CONTROL_IDS)

    run_control_suite(
        baseline,
        atoms,
        tmp_path,
        tmp_path / "missing-core-is-not-read",
        executors=executors,
        validate_core=False,
        force=True,
    )
    assert calls == list(CONTROL_IDS) * 2


@pytest.mark.parametrize(
    ("control_id", "expected"),
    [
        ("random_frozen_atoms", {"freeze_atoms": True}),
        ("shared_atoms_shuffled_labels", {"shuffle_training_labels": True}),
        ("shared_atoms_no_sparsity", {"sparsity_lambda": 0.0}),
    ],
)
def test_atom_control_executors_apply_only_the_declared_perturbation(
    monkeypatch: pytest.MonkeyPatch,
    locked_configs: tuple[ExperimentConfig, ExperimentConfig],
    tmp_path: Path,
    control_id: str,
    expected: dict[str, Any],
) -> None:
    baseline, atoms = locked_configs
    captured: dict[str, Any] = {}

    def fake_run_shared_atoms(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "tasks": {
                task: {
                    "top_k": {"metrics": {"primary_score": 0.5}},
                    "all_atoms": {"metrics": {"primary_score": 0.6}},
                }
                for task in H1_TASKS
            },
            "history": {},
            "parameter_counts": {},
            "checkpoint": {"format": "torch.save"},
            "runtime": {},
            "dataset_provenance": {},
            "coefficient_analysis": {},
        }

    monkeypatch.setattr(controls, "run_shared_atoms", fake_run_shared_atoms)
    context = ControlRunContext(
        baseline,
        atoms,
        tmp_path,
        tmp_path,
        prepared=object(),  # type: ignore[arg-type]
        force=False,
    )
    record = controls.DEFAULT_EXECUTORS[control_id](context)

    for key, value in expected.items():
        assert captured[key] == value
    assert captured["prepared"] is context.prepared
    assert captured["run_kind"] == "followup_control"
    assert record["compact_state_only"] is True
    assert set(record["tasks"]) == set(H1_TASKS)
