"""Offline, mocked tests for Experiment A's locked validation frontier."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import cgmoe_h1.validation_frontier as frontier
from cgmoe_h1.config import H1_CONFIRMATORY_SEEDS, H1_TASKS, load_config
from cgmoe_h1.metrics import compute_task_metrics


def _configs():
    project = Path(__file__).parents[1]
    return (
        load_config(project / "configs/baseline.yaml"),
        load_config(project / "configs/atoms.yaml"),
    )


def _provenance(seed: int) -> dict:
    result = {}
    for task in H1_TASKS:
        result[task] = {}
        for split, count in (("train", 2000), ("validation", 500)):
            result[task][split] = {
                "dataset_name": "nyu-mll/glue",
                "task_name": task,
                "split": split,
                "source_fingerprint": f"fingerprint-{task}-{split}",
                "requested_limit": count,
                "selected_count": count,
                "seed": seed,
                "selected_row_ids": list(range(count)),
            }
    return result


def _raw(task: str, score_override: float | None = None) -> dict:
    labels = [0] * 500
    predictions = [0] * 500
    metrics = compute_task_metrics(task, predictions, labels)
    if score_override is not None:
        # Summary-only mock cells bypass strict raw-output validation.
        metrics["primary_score"] = score_override
    return {
        "loss": 0.1,
        "examples": 500,
        "predictions": predictions,
        "labels": labels,
        "metrics": metrics,
        "scalar_metric_name": "primary_score",
    }


def _checkpoint(directory: Path, system: str, capacity: int, seed: int) -> dict:
    shapes = frontier._component_shapes(system, capacity)
    paths = {}
    byte_counts = {}
    directory.mkdir(parents=True, exist_ok=True)
    for component, component_shapes in shapes.items():
        path = directory / f"{component}.pt"
        torch.save(
            {
                "schema_version": 1,
                "component": component,
                "metadata": {
                    "experiment": "validation_frontier",
                    "system": system,
                    "capacity": capacity,
                    "seed": seed,
                    "tasks": list(H1_TASKS),
                },
                "state_dict": {
                    f"tensor_{index}": torch.zeros(shape)
                    for index, shape in enumerate(component_shapes)
                },
            },
            path,
        )
        paths[component] = str(path)
        byte_counts[component] = path.stat().st_size
    return {
        "paths": paths,
        "bytes_by_component": byte_counts,
        "total_bytes": sum(byte_counts.values()),
        "format": "torch.save",
        "dtype": "torch.float32",
    }


def _strict_cell(directory: Path, system: str = "shared_atoms", capacity: int = 2, seed: int = 17) -> dict:
    baseline, atoms = _configs()
    config = frontier.capacity_config(
        baseline.with_overrides(seed=seed) if system == "shared_lora" else atoms.with_overrides(seed=seed),
        system,
        capacity,
    )
    checkpoint = _checkpoint(directory, system, capacity, seed)
    counts = frontier._expected_counts(system, capacity)
    return {
        "schema_version": 1,
        "run_kind": "validation_frontier",
        "experiment": "validation_frontier",
        "status": "complete",
        "system": system,
        "capacity": capacity,
        "seed": seed,
        "model": "prajjwal1/bert-tiny",
        "model_revision": "test-revision",
        "resolved_config": config.to_dict(),
        "target_modules": [
            "encoder.layer.0.attention.self.query",
            "encoder.layer.0.attention.self.value",
            "encoder.layer.1.attention.self.query",
            "encoder.layer.1.attention.self.value",
        ],
        "target_dimensions": {
            "encoder.layer.0.attention.self.query": [128, 128],
            "encoder.layer.0.attention.self.value": [128, 128],
            "encoder.layer.1.attention.self.query": [128, 128],
            "encoder.layer.1.attention.self.value": [128, 128],
        },
        "environment": {
            "python": "test", "platform": "test", "cpu_threads": 1,
            "cuda_available": False, "packages": {},
        },
        "dataset_provenance": _provenance(seed),
        "tasks": {task: _raw(task) for task in H1_TASKS},
        "history": {
            "epochs": [
                {"global_training": {"batches": 1250, "optimizer_steps": 1250, "examples": 10000}}
                for _ in range(3)
            ],
            "best_epoch": 1,
            "best_score": 1.0,
        },
        "parameter_counts": {
            "base_model_parameters": 4_385_920,
            "base_trainable_parameters": 0,
            "uncategorized_trainable_parameters": 0,
            **counts,
        },
        "storage": {
            "persistent_parameters": counts["persistent_adaptation_parameters"],
            "persistent_tensor_bytes": 4 * counts["persistent_adaptation_parameters"],
            "checkpoint_bytes": checkpoint["total_bytes"],
            "common_frozen_base_parameters": 4_385_920,
            "common_frozen_base_tensor_bytes": 17_543_680,
        },
        "checkpoint": checkpoint,
        "runtime": {
            "elapsed_seconds": 1.0,
            "training_elapsed_seconds": 0.5,
            "evaluation_elapsed_seconds": 0.5,
            "peak_rss_bytes": 100,
            "inference_seconds_by_task": {task: 0.1 for task in H1_TASKS},
        },
        "active_adapter_operations_per_token": 1024 * capacity,
        "locked_budget": {"optimizer_updates": 3750},
        "artifact_directory": str(directory),
        "reused_atom_artifact": None,
    }


def _summary_cell(system: str, capacity: int, seed: int, score: float, root: Path) -> dict:
    persistent = frontier._expected_counts(system, capacity)["persistent_adaptation_parameters"]
    return {
        "system": system,
        "capacity": capacity,
        "seed": seed,
        "tasks": {task: _raw(task, score) for task in H1_TASKS},
        "storage": {
            "persistent_parameters": persistent,
            "persistent_tensor_bytes": 4 * persistent,
            "checkpoint_bytes": 100 + seed,
        },
        "active_adapter_operations_per_token": 1024 * capacity,
        "runtime": {
            "elapsed_seconds": 1.0,
            "training_elapsed_seconds": 0.5,
            "evaluation_elapsed_seconds": 0.5,
            "peak_rss_bytes": 100,
        },
        "model_revision": "test-revision",
        "dataset_provenance": _provenance(seed),
        "artifact_directory": str(frontier.cell_directory(root, system, capacity, seed)),
    }


def test_strict_cell_validation_checks_revision_rows_shapes_and_components(tmp_path: Path) -> None:
    record = _strict_cell(tmp_path / "cell")
    frontier.validate_frontier_cell(record, directory=tmp_path / "cell")

    broken = deepcopy(record)
    broken["model_revision"] = None
    with pytest.raises(ValueError, match="revision"):
        frontier.validate_frontier_cell(broken, directory=tmp_path / "cell")

    broken = deepcopy(record)
    broken["dataset_provenance"]["qqp"]["train"]["selected_row_ids"] = [1]
    with pytest.raises(ValueError, match="locked rows"):
        frontier.validate_frontier_cell(broken, directory=tmp_path / "cell")


def test_summary_applies_exact_atom_advantage_rule_and_keeps_every_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frontier, "validate_frontier_cell", lambda *args, **kwargs: None)
    cells = []
    for system in frontier.SYSTEMS:
        for capacity in frontier.CAPACITIES:
            for seed in frontier.SEEDS:
                score = 0.70
                if system == "shared_atoms" and capacity == 4:
                    score = 0.705
                cells.append(_summary_cell(system, capacity, seed, score, tmp_path))
    summary = frontier.build_frontier_summary(cells, output_root=tmp_path)

    assert summary["cell_count"] == 24
    assert summary["atom_specific_advantage"]["passed"] is True
    assert summary["atom_specific_advantage"]["qualifying_capacities"] == [4]
    assert summary["matched_capacity_deltas"]["4"]["mean_delta_atom_minus_lora"] == pytest.approx(0.005)
    assert len(summary["results"]["shared_atoms"]["4"]["by_task"]["qqp"]["seed_scores"]) == 3


def test_advantage_fails_if_any_conjunct_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(frontier, "validate_frontier_cell", lambda *args, **kwargs: None)
    cells = [
        _summary_cell(system, capacity, seed, 0.70, tmp_path)
        for system in frontier.SYSTEMS
        for capacity in frontier.CAPACITIES
        for seed in frontier.SEEDS
    ]
    summary = frontier.build_frontier_summary(cells, output_root=tmp_path)
    assert summary["atom_specific_advantage"]["passed"] is False
    assert summary["atom_specific_advantage"]["qualifying_capacities"] == []


def test_exact_and_tolerance_aware_pareto_are_distinct() -> None:
    points = [
        {"id": "small", "mean_primary_score": 0.700, "worst_task_score": 0.600, "persistent_parameters": 100, "active_operations": 10},
        {"id": "large", "mean_primary_score": 0.704, "worst_task_score": 0.604, "persistent_parameters": 120, "active_operations": 12},
    ]
    assert frontier.pareto_frontier(points) == ["small", "large"]
    assert frontier.pareto_frontier(points, quality_tolerance=0.005) == ["small"]


def test_mocked_orchestrator_writes_protocol_first_runs_all_cells_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline, atoms = _configs()
    output = tmp_path / "frontier"
    calls = []
    monkeypatch.setattr(frontier, "validate_frontier_cell", lambda *args, **kwargs: None)

    def fake_runner(baseline_config, atom_config, system, capacity, directory, prepared, force):
        assert (output / frontier.PROTOCOL_FILENAME).is_file()
        assert prepared is None
        calls.append((system, capacity, baseline_config.seed, force))
        return _summary_cell(system, capacity, baseline_config.seed, 0.7, output)

    summary, summary_path, report_path = frontier.run_validation_frontier(
        baseline, atoms, output, atom_reuse_root=tmp_path / "nothing", cell_runner=fake_runner
    )
    assert len(calls) == 24
    assert summary["status"] == "complete"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["cell_count"] == 24
    assert "Matched Shared-LoRA/Atom Frontier" in report_path.read_text(encoding="utf-8")

    calls.clear()
    resumed, _, _ = frontier.run_validation_frontier(
        baseline, atoms, output, atom_reuse_root=tmp_path / "nothing", cell_runner=fake_runner
    )
    assert resumed == summary
    assert calls == []


def test_capacity_configs_lock_alpha_and_all_atom_activation() -> None:
    baseline, atoms = _configs()
    lora = frontier.capacity_config(baseline, "shared_lora", 2)
    atom = frontier.capacity_config(atoms, "shared_atoms", 2)
    assert (lora.lora_rank, lora.lora_alpha) == (2, 2.0)
    assert (atom.atom_count, atom.active_atoms_during_training, atom.active_atoms_for_primary_evaluation) == (2, 2, 2)
