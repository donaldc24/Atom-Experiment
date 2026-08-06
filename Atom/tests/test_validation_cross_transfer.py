"""Offline tests for the crossed frozen-dictionary validation experiment."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import cgmoe_h1.validation_cross_transfer as cross
from cgmoe_h1.config import H1_CONFIRMATORY_SEEDS, H1_TASKS, load_config
from cgmoe_h1.experiments import environment_record
from cgmoe_h1.utils.serialization import write_json


def _artifacts(cell_directory: Path, seed: int, target: str) -> dict[str, str]:
    source = cell_directory / "source_learned_atoms" / f"seed_{seed}"
    learned = cell_directory / "learned_atom_transfer"
    random_control = cell_directory / "matched_random_transfer"
    head = cell_directory / "head_only"
    return {
        "cell_result": str(cell_directory / "cell_result.json"),
        "source_checkpoint_directory": str(source),
        "source_atoms": str(source / "atoms.pt"),
        "source_record": str(source / "metrics_by_task.json"),
        "learned_transfer_directory": str(learned),
        "learned_transfer_record": str(learned / "metrics.json"),
        "matched_random_transfer_directory": str(random_control),
        "matched_random_transfer_record": str(random_control / "metrics.json"),
        "head_only_directory": str(head),
        "head_only_record": str(head / "metrics.json"),
        "strict_core_lora_record": str(
            Path("results")
            / "independent_lora"
            / f"seed_{seed}"
            / target
            / "metrics.json"
        ),
    }


def _cell(
    target: str,
    seed: int,
    root: Path,
    *,
    fresh: float = 0.80,
    learned: float = 0.76,
    learned_all8: float = 0.77,
    head: float = 0.75,
    random_control: float = 0.75,
) -> dict:
    cell_directory = root / f"target_{target}" / f"seed_{seed}"
    artifacts = _artifacts(cell_directory, seed, target)
    return {
        "schema_version": 1,
        "experiment": "validation_cross_transfer",
        "status": "complete",
        "seed": seed,
        "target": target,
        "source_tasks": [task for task in H1_TASKS if task != target],
        "systems": {
            "fresh_lora": {
                "primary_score": fresh,
                "marginal_new_parameters": 4354,
                "reused_dictionary_parameters": 0,
                "total_with_dictionary_parameters": 4354,
                "raw_output_record": artifacts["strict_core_lora_record"],
            },
            "learned_frozen_atoms": {
                "all8_score": learned_all8,
                "top4_score": learned,
                "primary_score": learned,
                "marginal_new_parameters": 290,
                "reused_dictionary_parameters": 8192,
                "total_with_dictionary_parameters": 8482,
                "raw_output_record": artifacts["learned_transfer_record"],
            },
            "head_only": {
                "primary_score": head,
                "marginal_new_parameters": 258,
                "reused_dictionary_parameters": 0,
                "total_with_dictionary_parameters": 258,
                "raw_output_record": artifacts["head_only_record"],
            },
            "matched_random_frozen_atoms": {
                "all8_score": random_control,
                "top4_score": random_control,
                "primary_score": random_control,
                "marginal_new_parameters": 290,
                "reused_dictionary_parameters": 8192,
                "total_with_dictionary_parameters": 8482,
                "raw_output_record": artifacts["matched_random_transfer_record"],
            },
        },
        "cell_diagnostics": {
            "learned_retention": learned / fresh,
            "learned_minus_head_only": learned - head,
            "learned_minus_random_frozen": learned - random_control,
            "marginal_parameter_ratio": 290 / 4354,
        },
        "deployment_accounting": {
            "source_four_task_parameters": 9352,
            "target_marginal_parameters": 290,
            "five_task_total_after_transfer": 9642,
            "target_total_with_dictionary_parameters": 8482,
            "five_independent_lora_parameters": 21770,
        },
        "locked_budget": {
            "train_examples_per_task": 2000,
            "validation_examples_per_task": 500,
            "epochs": 3,
            "atom_count": 8,
            "all8_active_atoms": 8,
            "primary_top4_active_atoms": 4,
            "lora_rank": 4,
        },
        "matching_evidence": {
            "seed": seed,
            "trainable_initialization_sha256": "0" * 64,
            "learned_dictionary_sha256": "1" * 64,
            "random_dictionary_sha256": "2" * 64,
        },
        "model_identity": {
            "model": "prajjwal1/bert-tiny",
            "model_revision": "test-revision",
            "target_dimensions": {
                "encoder.layer.0.attention.self.query": [128, 128],
                "encoder.layer.0.attention.self.value": [128, 128],
                "encoder.layer.1.attention.self.query": [128, 128],
                "encoder.layer.1.attention.self.value": [128, 128],
            },
        },
        "runtime": {},
        "target_dataset_provenance": {},
        "resolved_configs": {},
        "artifacts": artifacts,
    }


def _grid(root: Path, **changes) -> list[dict]:
    return [
        _cell(target, seed, root, **changes)
        for target in H1_TASKS
        for seed in H1_CONFIRMATORY_SEEDS
    ]


def _configs():
    project = Path(__file__).parents[1]
    return (
        load_config(project / "configs/baseline.yaml"),
        load_config(project / "configs/atoms.yaml"),
    )


def test_predeclared_thresholds_are_inclusive_and_separate_from_diagnostics(
    tmp_path: Path,
) -> None:
    summary = cross.build_cross_transfer_summary(_grid(tmp_path), output_root=tmp_path)

    assert summary["cell_count"] == 15
    assert summary["primary_strong_transfer"]["aggregate_retention"] == pytest.approx(0.95)
    assert summary["primary_strong_transfer"]["passed"] is True
    assert all(value["retention"] == pytest.approx(0.95) for value in summary["by_target"].values())
    assert all(value["passed"] for value in summary["diagnostics"].values())
    assert summary["parameter_accounting"] == pytest.approx(
        {
            "fresh_lora_task_state": 4354,
            "learned_target_marginal": 290,
            "learned_dictionary": 8192,
            "learned_target_total_with_dictionary": 8482,
            "marginal_ratio": 290 / 4354,
        }
    )


def test_primary_can_pass_while_control_advantage_diagnostics_fail(tmp_path: Path) -> None:
    cells = _grid(tmp_path, head=0.759, random_control=0.758)
    summary = cross.build_cross_transfer_summary(cells, output_root=tmp_path)

    assert summary["primary_strong_transfer"]["passed"] is True
    assert summary["diagnostics"]["learned_mean_exceeds_head_only_by_0_005"][
        "passed"
    ] is False
    assert summary["diagnostics"]["learned_mean_exceeds_random_frozen_by_0_005"][
        "passed"
    ] is False


def test_summary_rejects_missing_duplicate_or_malformed_cells(tmp_path: Path) -> None:
    cells = _grid(tmp_path)
    with pytest.raises(ValueError, match="all 15"):
        cross.build_cross_transfer_summary(cells[:-1], output_root=tmp_path)

    with pytest.raises(ValueError, match="duplicate"):
        cross.build_cross_transfer_summary(cells + [deepcopy(cells[0])], output_root=tmp_path)

    malformed = deepcopy(cells[0])
    malformed["source_tasks"] = list(H1_TASKS[:4])
    with pytest.raises(ValueError, match="source tasks"):
        cross.validate_cross_transfer_cell(malformed)

    malformed = deepcopy(cells[0])
    malformed["systems"]["learned_frozen_atoms"]["all8_score"] = float("nan")
    with pytest.raises(ValueError, match="all8_score"):
        cross.validate_cross_transfer_cell(malformed)

    malformed = deepcopy(cells[0])
    del malformed["matching_evidence"]
    with pytest.raises(ValueError, match="matched-initialization"):
        cross.validate_cross_transfer_cell(malformed)


def test_orchestrator_uses_mock_compute_writes_paths_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline, atoms = _configs()
    calls: list[tuple[str, int, Path, bool]] = []

    def fake_runner(
        baseline_config,
        atom_config,
        target,
        cell_directory,
        core_results_root,
        force,
    ):
        assert baseline_config.seed == atom_config.seed
        assert (output / cross.PROTOCOL_FILENAME).is_file()
        calls.append((target, atom_config.seed, cell_directory, force))
        return _cell(target, atom_config.seed, tmp_path / "cross")

    output = tmp_path / "cross"
    summary, summary_path, report_path = cross.run_validation_cross_transfer(
        baseline,
        atoms,
        output,
        core_results_root=tmp_path / "core",
        cell_runner=fake_runner,
    )

    assert len(calls) == 15
    assert summary["status"] == "complete"
    assert summary["strong_reusable_basis_support"] is True
    assert json.loads(summary_path.read_text(encoding="utf-8"))["cell_count"] == 15
    assert "Crossed Frozen-Atom Transfer" in report_path.read_text(encoding="utf-8")
    source = summary["artifacts_by_target_seed"]["qqp"]["17"][
        "source_checkpoint_directory"
    ]
    assert source.endswith("target_qqp\\seed_17\\source_learned_atoms\\seed_17")

    calls.clear()
    resumed, _, _ = cross.run_validation_cross_transfer(
        baseline,
        atoms,
        output,
        core_results_root=tmp_path / "core",
        cell_runner=fake_runner,
    )
    assert resumed == summary
    assert calls == []

    protocol_path = output / cross.PROTOCOL_FILENAME
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["primary_strong_transfer"]["threshold"] = 0.96
    write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="protocol differs"):
        cross.run_validation_cross_transfer(
            baseline,
            atoms,
            output,
            core_results_root=tmp_path / "core",
            cell_runner=fake_runner,
        )


def test_strict_core_loader_requires_exact_contract_provenance_and_raw_outputs(
    tmp_path: Path,
) -> None:
    baseline, _ = _configs()
    config = baseline.with_overrides(seed=17)
    target = "sst2"
    provenance = {
        "train": {
            "seed": 17,
            "selected_count": 2,
            "selected_row_ids": [1, 2],
        },
        "validation": {
            "seed": 17,
            "selected_count": 500,
            "selected_row_ids": list(range(500)),
        },
    }
    record = {
        "schema_version": 1,
        "system": "independent_lora",
        "run_kind": "confirmatory",
        "seed": 17,
        "task": target,
        "rank": 4,
        "model": config.base_model,
        "model_revision": "test-revision",
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
        "environment": environment_record(),
        "resolved_config": config.to_dict(),
        "dataset_provenance": {target: provenance},
        "parameter_counts": {
            "base_trainable_parameters": 0,
            "lora_adapter_parameters": 4096,
            "atom_parameters": 0,
            "coefficient_parameters": 0,
            "head_parameters": 258,
            "uncategorized_trainable_parameters": 0,
            "model_trainable_parameters": 4354,
            "persistent_adaptation_parameters": 4354,
        },
        "best": {
            "examples": 500,
            "predictions": [0] * 500,
            "labels": [0] * 500,
            "metrics": {"accuracy": 1.0, "primary_score": 1.0},
        },
    }
    path = tmp_path / "independent_lora" / "seed_17" / target / "metrics.json"
    path.parent.mkdir(parents=True)
    checkpoint_paths: dict[str, str] = {}
    byte_counts: dict[str, int] = {}
    for component in ("adapter", "heads"):
        component_path = path.parent / f"{component}.pt"
        state = (
            {"encoder.test.lora_a.weight": torch.ones(4096)}
            if component == "adapter"
            else {f"heads.{target}.weight": torch.ones(258)}
        )
        torch.save(
            {
                "schema_version": 1,
                "component": component,
                "metadata": {
                    "system": "independent_lora",
                    "seed": 17,
                    "task": target,
                },
                "state_dict": state,
            },
            component_path,
        )
        checkpoint_paths[component] = str(component_path)
        byte_counts[component] = component_path.stat().st_size
    record["checkpoint"] = {
        "paths": checkpoint_paths,
        "bytes_by_component": byte_counts,
        "total_bytes": sum(byte_counts.values()),
        "format": "torch.save",
        "dtype": "torch.float32",
    }
    write_json(path, record)

    loaded, loaded_path = cross._load_strict_core_lora(
        config, target, tmp_path, provenance
    )
    assert loaded == record
    assert loaded_path == path

    record["run_kind"] = "development"
    write_json(path, record)
    with pytest.raises(ValueError, match="incompatible"):
        cross._load_strict_core_lora(config, target, tmp_path, provenance)

    record["run_kind"] = "confirmatory"
    write_json(path, record)
    Path(checkpoint_paths["adapter"]).unlink()
    with pytest.raises(ValueError, match="missing strict core LoRA checkpoint"):
        cross._load_strict_core_lora(config, target, tmp_path, provenance)
