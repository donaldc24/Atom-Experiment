"""Mock-compute coverage for roadmap chunks 23 and 24."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import cgmoe_h1.followups_ablations as ablations
from cgmoe_h1.config import ExperimentConfig, load_config
from cgmoe_h1.utils.serialization import write_json

ROOT = Path(__file__).parents[1]
TASK_OFFSETS = {
    "sst2": 0.02,
    "mrpc": 0.01,
    "rte": -0.02,
    "qnli": 0.0,
    "qqp": -0.01,
}


def _scores(mean: float) -> dict[str, float]:
    return {task: mean + TASK_OFFSETS[task] for task in TASK_OFFSETS}


def _independent_record(
    config: ExperimentConfig, rank: int, mean: float
) -> dict[str, Any]:
    task_scores = _scores(mean)
    parameters = 1_290 + 5_120 * rank
    return {
        "system": "independent_lora",
        "seed": 17,
        "rank": rank,
        "tasks": {
            task: {
                "best": {"metrics": {"primary_score": score}},
                "final": {"metrics": {"primary_score": score - 0.005}},
                "active_adapter_operations_per_token": rank * 1_024,
                "resolved_config": config.to_dict(),
            }
            for task, score in task_scores.items()
        },
        "parameter_counts": {"total_persistent_task_parameters": parameters},
        "checkpoint_bytes": parameters * 4 + 1_000,
        "resolved_config": config.to_dict(),
    }


def _utilization(atom_count: int) -> list[int]:
    values = {
        2: [5, 5],
        4: [5, 5, 5, 5],
        6: [5, 2, 4, 4, 5, 0],
        8: [2, 2, 4, 3, 0, 2, 3, 4],
        12: [2, 1, 3, 3, 1, 2, 0, 0, 1, 3, 2, 2],
        16: [0, 0, 1, 3, 3, 1, 0, 1, 1, 1, 3, 2, 1, 1, 1, 1],
    }
    return values[atom_count]


def _shared_record(
    config: ExperimentConfig, atom_count: int, mean: float
) -> dict[str, Any]:
    top_k = min(4, atom_count)
    task_scores = _scores(mean)
    utilization = _utilization(atom_count)
    parameters = 1_290 + 1_044 * atom_count
    reused = [index for index, count in enumerate(utilization) if count >= 2]
    private = [index for index, count in enumerate(utilization) if count == 1]
    dead = [index for index, count in enumerate(utilization) if count == 0]
    return {
        "system": "shared_atoms",
        "seed": 17,
        "atom_count": atom_count,
        "tasks": {
            task: {
                "all_atoms": {"metrics": {"primary_score": score + 0.002}},
                "top_k": {"metrics": {"primary_score": score}},
                "top_k_value": top_k,
            }
            for task, score in task_scores.items()
        },
        "parameter_counts": {"total_persistent_task_parameters": parameters},
        "checkpoint": {"total_bytes": parameters * 4 + 2_000},
        "active_adapter_operations_per_token": {
            "all_atoms": atom_count * 1_024,
            "top_k": top_k * 1_024,
        },
        "coefficient_analysis": {
            "reused_atoms": reused,
            "task_exclusive_atoms": private,
            "dead_atoms": dead,
            "atom_utilization_count": utilization,
        },
        "resolved_config": config.to_dict(),
    }


def _active_record(top_k: int, mean: float) -> dict[str, Any]:
    return {
        "system": "shared_atoms",
        "seed": 17,
        "atom_count": 8,
        "top_k": top_k,
        "tasks": {
            task: {"metrics": {"primary_score": score}}
            for task, score in _scores(mean).items()
        },
        "masks": {},
    }


@pytest.fixture
def configs() -> tuple[ExperimentConfig, ExperimentConfig]:
    return (
        load_config(ROOT / "configs" / "baseline.yaml"),
        load_config(ROOT / "configs" / "atoms.yaml"),
    )


def _write_core(
    tmp_path: Path, baseline: ExperimentConfig, atoms: ExperimentConfig
) -> tuple[Path, Path]:
    independent_root = tmp_path / "core_independent"
    shared_root = tmp_path / "core_shared"
    write_json(
        independent_root / "seed_17" / "metrics_by_task.json",
        _independent_record(baseline, 4, 0.800),
    )
    write_json(
        shared_root / "seed_17" / "metrics_by_task.json",
        _shared_record(atoms, 8, 0.790),
    )
    return independent_root, shared_root


def test_run_ablations_uses_locked_variants_and_writes_answers(
    tmp_path: Path,
    configs: tuple[ExperimentConfig, ExperimentConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, atoms = configs
    independent_root, shared_root = _write_core(tmp_path, baseline, atoms)
    destination = tmp_path / "ablations"
    prepared = object()
    prepare_calls: list[tuple[Any, Any]] = []
    shared_calls: list[dict[str, Any]] = []
    independent_calls: list[dict[str, Any]] = []
    active_calls: list[dict[str, Any]] = []
    atom_means = {2: 0.700, 4: 0.750, 6: 0.780, 12: 0.792, 16: 0.791}
    rank_means = {1: 0.760, 2: 0.780, 8: 0.805}
    active_means = {1: 0.750, 2: 0.780, 4: 0.790, 8: 0.792}

    def fake_prepare(config: ExperimentConfig, *, tasks) -> object:
        prepare_calls.append((config, tasks))
        return prepared

    def fake_shared(config: ExperimentConfig, output_root: Path, **kwargs) -> dict:
        shared_calls.append({"config": config, "output_root": Path(output_root), **kwargs})
        record = _shared_record(config, kwargs["atom_count"], atom_means[kwargs["atom_count"]])
        write_json(Path(output_root) / "seed_17" / "metrics_by_task.json", record)
        return record

    def fake_independent(config: ExperimentConfig, output_root: Path, **kwargs) -> dict:
        independent_calls.append(
            {"config": config, "output_root": Path(output_root), **kwargs}
        )
        record = _independent_record(config, kwargs["rank"], rank_means[kwargs["rank"]])
        write_json(Path(output_root) / "seed_17" / "metrics_by_task.json", record)
        return record

    def fake_evaluate(config: ExperimentConfig, run_directory: Path, **kwargs) -> dict:
        active_calls.append(
            {"config": config, "run_directory": Path(run_directory), **kwargs}
        )
        return _active_record(kwargs["top_k"], active_means[kwargs["top_k"]])

    monkeypatch.setattr(ablations, "prepare_data", fake_prepare)
    monkeypatch.setattr(ablations, "run_shared_atoms", fake_shared)
    monkeypatch.setattr(ablations, "run_independent_lora", fake_independent)
    monkeypatch.setattr(ablations, "evaluate_atom_checkpoint", fake_evaluate)

    summary = ablations.run_ablations(
        baseline,
        atoms,
        destination,
        core_independent_root=independent_root,
        core_shared_root=shared_root,
    )

    assert len(prepare_calls) == 1
    assert prepare_calls[0] == (baseline, baseline.tasks)
    assert [call["atom_count"] for call in shared_calls] == [2, 4, 6, 12, 16]
    assert [call["top_k"] for call in shared_calls] == [2, 4, 4, 4, 4]
    assert [call["rank"] for call in independent_calls] == [1, 2, 8]
    assert [call["config"].atom_count for call in shared_calls] == [2, 4, 6, 12, 16]
    assert [call["config"].active_atoms_during_training for call in shared_calls] == [
        2,
        4,
        6,
        12,
        16,
    ]
    assert [call["config"].active_atoms_for_primary_evaluation for call in shared_calls] == [
        2,
        4,
        4,
        4,
        4,
    ]
    assert [call["config"].lora_rank for call in independent_calls] == [1, 2, 8]
    assert [call["top_k"] for call in active_calls] == [1, 2, 4, 8]
    assert all(call["prepared"] is prepared for call in shared_calls + independent_calls)
    assert all(call["tasks"] == baseline.tasks for call in shared_calls + independent_calls)
    assert all(call["run_kind"] == "followup_ablation_seed17_locked" for call in shared_calls)
    assert all(
        call["run_kind"] == "followup_ablation_seed17_locked"
        for call in independent_calls
    )
    assert all(call["prepared"] is prepared for call in active_calls)
    assert all(call["run_directory"] == shared_root / "seed_17" for call in active_calls)

    atom_answers = summary["atom_count_ablation"]["question_answers"]
    assert atom_answers["quality_saturation"]["atom_count"] == 8
    assert atom_answers["quality_saturation"][
        "conventional_rise_then_plateau_observed"
    ] is True
    assert atom_answers["storage_below_independent_baseline"]["answer"] is True
    reuse = atom_answers["reuse_with_dictionary_size"]
    assert reuse["absolute_count_pattern"] == "non_monotonic"
    assert reuse["reused_atom_count_by_atom_count"] == {
        "2": 2,
        "4": 4,
        "6": 5,
        "8": 7,
        "12": 7,
        "16": 4,
    }
    assert atom_answers["extra_atoms_become_task_private"]["answer"] is True
    assert summary["lora_rank_ablation"]["question_answers"][
        "best_mean_quality_rank"
    ] == 8
    active_answers = summary["active_capacity_ablation"]["question_answers"]
    assert active_answers["smallest_active_atoms_within_0_005_of_best"] == 4
    assert active_answers["active_compute_is_linear_in_k"] is True
    assert summary["active_capacity_ablation"]["training_runs"] == 0

    summary_path = destination / ablations.SUMMARY_FILENAME
    report_path = destination / ablations.REPORT_FILENAME
    strict = json.loads(
        summary_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert strict["seed"] == 17
    assert strict["locked_budget"]["adam_beta1"] == 0.9
    assert strict["locked_budget"]["adam_beta2"] == 0.999
    assert strict["locked_budget"]["adam_epsilon"] == 1e-8
    assert strict["locked_budget"]["weight_decay"] == 0.01
    assert strict["locked_budget"]["lora_alpha"] == 4
    assert strict["locked_budget"]["sparsity_lambda"] == 1e-5
    definitions = strict["atom_count_ablation"]["coefficient_usage_definitions"]
    assert definitions["dead_atom_threshold"] == 1e-6
    assert "not interchangeable" in definitions["distinction"]
    markdown = report_path.read_text(encoding="utf-8")
    assert "Operational near-best selection" in markdown
    assert "conventional rise-then-plateau pattern" in markdown
    assert "Reuse is **non-monotonic**" in markdown
    assert "Top-k unused" in markdown
    assert "A non-dead atom can therefore be top-k unused" in markdown
    assert "Storage below the independent rank-4 baseline: **yes**" in markdown
    assert "Do extra atoms become task-private above N=8? **yes**" in markdown
    assert "| Atoms | Eval top-k | Mean | Worst task" in markdown
    assert "| Rank | Mean | Worst task" in markdown
    assert "| Active atoms | Mean | Worst task" in markdown
    assert "All rows reload the same core 8-atom checkpoint" in markdown
    assert (destination / "atom_count" / "atoms_8" / "seed_17" / "reused_core.json").is_file()
    assert (destination / "lora_rank" / "rank_4" / "seed_17" / "reused_core.json").is_file()
    for atom_count in (2, 4, 6, 12, 16):
        record = json.loads(
            (
                destination
                / "atom_count"
                / f"atoms_{atom_count}"
                / "seed_17"
                / "metrics_by_task.json"
            ).read_text(encoding="utf-8")
        )
        assert record["resolved_config"]["atom_count"] == atom_count
        assert record["resolved_config"]["active_atoms_during_training"] == atom_count
        assert record["resolved_config"][
            "active_atoms_for_primary_evaluation"
        ] == min(4, atom_count)
    for rank in (1, 2, 8):
        record = json.loads(
            (
                destination
                / "lora_rank"
                / f"rank_{rank}"
                / "seed_17"
                / "metrics_by_task.json"
            ).read_text(encoding="utf-8")
        )
        assert record["resolved_config"]["lora_rank"] == rank
        assert all(
            task_record["resolved_config"]["lora_rank"] == rank
            for task_record in record["tasks"].values()
        )


def test_completed_variants_resume_without_compute_or_data_loading(
    tmp_path: Path,
    configs: tuple[ExperimentConfig, ExperimentConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, atoms = configs
    independent_root, shared_root = _write_core(tmp_path, baseline, atoms)
    destination = tmp_path / "ablations"
    atom_means = {2: 0.700, 4: 0.750, 6: 0.780, 12: 0.792, 16: 0.791}
    rank_means = {1: 0.760, 2: 0.780, 8: 0.805}
    active_means = {1: 0.750, 2: 0.780, 4: 0.790, 8: 0.792}

    monkeypatch.setattr(ablations, "prepare_data", lambda *args, **kwargs: object())

    def fake_shared(config: ExperimentConfig, output_root: Path, **kwargs) -> dict:
        return _shared_record(config, kwargs["atom_count"], atom_means[kwargs["atom_count"]])

    def fake_independent(config: ExperimentConfig, output_root: Path, **kwargs) -> dict:
        return _independent_record(config, kwargs["rank"], rank_means[kwargs["rank"]])

    monkeypatch.setattr(ablations, "run_shared_atoms", fake_shared)
    monkeypatch.setattr(ablations, "run_independent_lora", fake_independent)
    monkeypatch.setattr(
        ablations,
        "evaluate_atom_checkpoint",
        lambda config, run_directory, **kwargs: _active_record(
            kwargs["top_k"], active_means[kwargs["top_k"]]
        ),
    )
    first = ablations.run_ablations(
        baseline,
        atoms,
        destination,
        core_independent_root=independent_root,
        core_shared_root=shared_root,
    )

    # Simulate legacy cached aggregate metadata: capacity arrived through a
    # function argument while resolved_config retained the core defaults.
    for atom_count in (2, 4, 6, 12, 16):
        path = (
            destination
            / "atom_count"
            / f"atoms_{atom_count}"
            / "seed_17"
            / "metrics_by_task.json"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        record["resolved_config"] = atoms.to_dict()
        write_json(path, record)
    for rank in (1, 2, 8):
        path = (
            destination
            / "lora_rank"
            / f"rank_{rank}"
            / "seed_17"
            / "metrics_by_task.json"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        record["resolved_config"] = baseline.to_dict()
        for task_record in record["tasks"].values():
            task_record["resolved_config"] = baseline.to_dict()
        write_json(path, record)

    def unexpected(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("completed ablations must not invoke compute")

    monkeypatch.setattr(ablations, "prepare_data", unexpected)
    monkeypatch.setattr(ablations, "run_shared_atoms", unexpected)
    monkeypatch.setattr(ablations, "run_independent_lora", unexpected)
    monkeypatch.setattr(ablations, "evaluate_atom_checkpoint", unexpected)
    second = ablations.run_ablations(
        baseline,
        atoms,
        destination,
        core_independent_root=independent_root,
        core_shared_root=shared_root,
    )

    assert second == first
    for atom_count in (2, 4, 6, 12, 16):
        path = (
            destination
            / "atom_count"
            / f"atoms_{atom_count}"
            / "seed_17"
            / "metrics_by_task.json"
        )
        assert json.loads(path.read_text(encoding="utf-8"))["resolved_config"][
            "atom_count"
        ] == atom_count
    for rank in (1, 2, 8):
        path = (
            destination
            / "lora_rank"
            / f"rank_{rank}"
            / "seed_17"
            / "metrics_by_task.json"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["resolved_config"]["lora_rank"] == rank
        assert all(
            task_record["resolved_config"]["lora_rank"] == rank
            for task_record in record["tasks"].values()
        )


def test_ablation_runner_rejects_non_seed17_config(
    tmp_path: Path, configs: tuple[ExperimentConfig, ExperimentConfig]
) -> None:
    baseline, atoms = configs
    with pytest.raises(ablations.AblationError, match="locked to seed 17"):
        ablations.run_ablations(
            baseline.with_overrides(seed=29),
            atoms,
            tmp_path,
            core_independent_root=tmp_path / "missing",
            core_shared_root=tmp_path / "missing",
        )


def test_cli_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_ablations.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--core-independent-root" in completed.stdout
    assert "--force" in completed.stdout
