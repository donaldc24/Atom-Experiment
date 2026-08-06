"""Synthetic, deterministic coverage for the final H1 report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from cgmoe_h1.config import H1_CONFIRMATORY_SEEDS, H1_TASKS
from cgmoe_h1.reporting import (
    DEFAULT_OUTPUT_DIR,
    ReportingError,
    analyze_coefficient_payload,
    generate_h1_report,
    render_h1_report,
    summarize_h1,
    summarize_records,
    write_h1_outputs,
)
from cgmoe_h1.utils.serialization import write_json


def test_default_output_dir_is_canonical() -> None:
    assert DEFAULT_OUTPUT_DIR.as_posix() == "results/h1_report"


def _independent_record(seed: int, scores: dict[str, float], parameters: int = 100) -> dict:
    return {
        "system": "independent_lora",
        "seed": seed,
        "tasks": {
            task: {
                "best": {"metrics": {"primary_score": score}, "loss": 0.2},
                "final": {"metrics": {"primary_score": score - 0.01}, "loss": 0.25},
            }
            for task, score in scores.items()
        },
        "parameter_counts": {"total_persistent_task_parameters": parameters},
    }


def _shared_record(
    seed: int,
    all_scores: dict[str, float],
    top_scores: dict[str, float],
    parameters: int = 44,
) -> dict:
    return {
        "system": "shared_atoms",
        "seed": seed,
        "tasks": {
            task: {
                "all_atoms": {"metrics": {"primary_score": all_scores[task]}},
                "top_k": {"metrics": {"primary_score": top_scores[task]}},
            }
            for task in all_scores
        },
        "parameter_counts": {"total_persistent_task_parameters": parameters},
    }


@pytest.fixture
def paired_records() -> tuple[list[dict], list[dict]]:
    independent: list[dict] = []
    shared: list[dict] = []
    offsets = {17: -0.01, 29: 0.0, 43: 0.01}
    baseline = dict(zip(H1_TASKS, (0.90, 0.80, 0.70, 0.85, 0.75), strict=True))
    for seed in H1_CONFIRMATORY_SEEDS:
        independent_scores = {task: score + offsets[seed] for task, score in baseline.items()}
        all_scores = {task: score - 0.005 for task, score in independent_scores.items()}
        top_scores = {task: score - 0.02 for task, score in independent_scores.items()}
        independent.append(_independent_record(seed, independent_scores))
        shared.append(_shared_record(seed, all_scores, top_scores))
    return independent, shared


def _write_tree(root: Path, records: list[dict]) -> None:
    for record in records:
        write_json(root / f"seed_{record['seed']}" / "metrics_by_task.json", record)


def test_seed_first_summary_and_strict_decision(paired_records: tuple[list[dict], list[dict]]) -> None:
    independent, shared = paired_records

    summary = summarize_records(independent, shared)

    assert summary["models"]["independent_lora"]["task_scores"]["sst2"] == pytest.approx(0.90)
    assert summary["models"]["independent_lora"]["mean_score"] == pytest.approx(0.80)
    assert summary["models"]["shared_atoms_all"]["mean_score"] == pytest.approx(0.795)
    assert summary["models"]["shared_atoms_top_k"]["mean_score"] == pytest.approx(0.78)
    assert summary["models"]["independent_lora_final"]["mean_score"] == pytest.approx(0.79)
    assert summary["quality_retention"] == pytest.approx(0.975)
    assert summary["worst_task_gap"] == pytest.approx(0.02)
    assert summary["relative_storage"] == pytest.approx(0.44)
    assert summary["comparison"]["task_gaps"] == pytest.approx(
        {task: 0.02 for task in H1_TASKS}
    )
    assert summary["models"]["independent_lora"]["task_standard_deviations"][
        "sst2"
    ] == pytest.approx((2 / 3 * 0.01**2) ** 0.5)
    assert summary["preregistered_pass"] is True
    assert summary["decision"] == "Supported"


def test_thresholds_are_inclusive_and_any_failure_is_strict_fail() -> None:
    baseline = {task: 1.0 for task in H1_TASKS}
    independent = [_independent_record(seed, baseline, parameters=100) for seed in H1_CONFIRMATORY_SEEDS]
    top = {task: 0.97 for task in H1_TASKS}
    shared = [_shared_record(seed, top, top, parameters=50) for seed in H1_CONFIRMATORY_SEEDS]

    boundary = summarize_records(independent, shared)
    assert boundary["quality_retention"] == pytest.approx(0.97)
    assert boundary["worst_task_gap"] == pytest.approx(0.03)
    assert boundary["relative_storage"] == pytest.approx(0.50)
    assert boundary["passed"] is True

    shared[0]["parameter_counts"]["total_persistent_task_parameters"] = 51
    shared[1]["parameter_counts"]["total_persistent_task_parameters"] = 51
    shared[2]["parameter_counts"]["total_persistent_task_parameters"] = 51
    failed = summarize_records(independent, shared)
    assert failed["thresholds"]["relative_storage"]["passed"] is False
    assert failed["passed"] is False
    assert failed["decision"] == "Not supported"


def test_better_shared_task_does_not_create_a_negative_worst_gap() -> None:
    baseline = {task: 0.5 for task in H1_TASKS}
    better = {task: 0.6 for task in H1_TASKS}
    independent = [_independent_record(seed, baseline) for seed in H1_CONFIRMATORY_SEEDS]
    shared = [_shared_record(seed, better, better) for seed in H1_CONFIRMATORY_SEEDS]

    summary = summarize_records(independent, shared)

    assert all(gap == pytest.approx(-0.1) for gap in summary["comparison"]["task_gaps"].values())
    assert summary["worst_task_gap"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda independent, shared: shared.pop(), "seed"),
        (lambda independent, shared: shared[0].update(system="wrong"), "system"),
        (
            lambda independent, shared: shared[0]["tasks"]["sst2"]["top_k"]["metrics"].update(primary_score=float("nan")),
            "finite",
        ),
        (
            lambda independent, shared: shared[0]["tasks"].pop("qqp"),
            "missing required tasks",
        ),
        (
            lambda independent, shared: shared[0]["parameter_counts"].update(total_persistent_task_parameters=45),
            "differs across seeds",
        ),
    ],
)
def test_invalid_or_unpaired_records_fail_loudly(
    paired_records: tuple[list[dict], list[dict]], mutation, message: str
) -> None:
    independent, shared = deepcopy(paired_records)
    mutation(independent, shared)

    with pytest.raises(ReportingError, match=message):
        summarize_records(independent, shared)


def test_load_render_and_write_required_tables(
    tmp_path: Path, paired_records: tuple[list[dict], list[dict]]
) -> None:
    independent, shared = paired_records
    independent_root = tmp_path / "independent_lora"
    shared_root = tmp_path / "shared_atoms"
    _write_tree(independent_root, independent)
    _write_tree(shared_root, shared)

    summary = summarize_h1(independent_root, shared_root)
    markdown = render_h1_report(summary)
    summary_path, report_path = write_h1_outputs(summary, tmp_path / "report")

    assert "| Model | Mean score | Worst task gap" in markdown
    assert "| Task | Independent LoRA | Shared atoms top-4 | Absolute gap" in markdown
    assert "## Per-seed primary scores" in markdown
    assert "## Across-seed population standard deviation" in markdown
    assert "Decision: **Supported**" in markdown
    assert json.loads(summary_path.read_text(encoding="utf-8"))["preregistered_pass"] is True
    assert report_path.read_text(encoding="utf-8") == markdown


def test_coefficient_payload_reports_top_k_reuse_and_similarity() -> None:
    # Two layers, five tasks, four atoms; deterministic ties favor lower indices.
    matrices = torch.tensor(
        [
            [
                [4.0, 3.0, 0.0, 0.0],
                [4.0, 2.0, 0.0, 0.0],
                [0.0, 4.0, 3.0, 0.0],
                [0.0, 4.0, 2.0, 0.0],
                [0.0, 0.0, 4.0, 3.0],
            ],
            [
                [4.0, 0.0, 0.0, 3.0],
                [4.0, 0.0, 0.0, 2.0],
                [0.0, 4.0, 0.0, 3.0],
                [0.0, 4.0, 0.0, 2.0],
                [0.0, 0.0, 4.0, 3.0],
            ],
        ]
    )
    payload = {"task_ids": list(H1_TASKS), "coefficients": matrices}

    analysis = analyze_coefficient_payload(payload, top_k=2)

    assert analysis["layer_count"] == 2
    assert analysis["reuse"]["atom_slots"] == 8
    assert sum(analysis["reuse"][key] for key in ("dead_atoms", "task_exclusive_atoms", "reused_by_two_or_more_tasks")) == 8
    assert analysis["task_by_atom_usage"]["sst2"] == [2, 1, 0, 1]
    assert analysis["top_atoms_per_task"]["sst2"][0]["atom_index"] == 0
    assert "sst2|mrpc" in analysis["pairwise_coefficient_similarity"]


def test_generate_report_uses_optional_coefficients_pt(
    tmp_path: Path, paired_records: tuple[list[dict], list[dict]]
) -> None:
    independent, shared = paired_records
    independent_root = tmp_path / "independent_lora"
    shared_root = tmp_path / "shared_atoms"
    _write_tree(independent_root, independent)
    _write_tree(shared_root, shared)
    coefficients = torch.arange(1, 1 + len(H1_TASKS) * 8, dtype=torch.float32).reshape(
        len(H1_TASKS), 8
    )
    for seed in H1_CONFIRMATORY_SEEDS:
        torch.save(
            {"task_ids": list(H1_TASKS), "coefficients": coefficients},
            shared_root / f"seed_{seed}" / "coefficients.pt",
        )

    summary, summary_path, report_path = generate_h1_report(
        independent_root,
        shared_root,
        tmp_path / "outputs",
    )

    coefficient_analysis = summary["coefficient_analysis"]
    assert coefficient_analysis["available"] is True
    assert coefficient_analysis["aggregate"]["seeds_analyzed"] == [17, 29, 43]
    assert "Mean task-by-atom top-k usage" in report_path.read_text(encoding="utf-8")
    assert summary_path.is_file()


def test_embedded_json_coefficient_diagnostics_are_accepted(
    paired_records: tuple[list[dict], list[dict]], tmp_path: Path
) -> None:
    independent, shared = paired_records
    diagnostics = {
        "tasks": {
            task: {
                "top_used_atoms_by_layer": {
                    "query": [
                        {"atom_index": 0, "absolute_coefficient": 0.4},
                        {"atom_index": 1, "absolute_coefficient": 0.3},
                    ]
                }
            }
            for task in H1_TASKS
        }
    }
    for record in shared:
        record["coefficient_analysis"] = diagnostics
    independent_root = tmp_path / "independent_lora"
    shared_root = tmp_path / "shared_atoms"
    _write_tree(independent_root, independent)
    _write_tree(shared_root, shared)

    summary = summarize_h1(independent_root, shared_root)

    reuse = summary["coefficient_analysis"]["aggregate"]["reuse_mean_across_seeds"]
    assert reuse["reused_by_two_or_more_tasks"] == 2.0
    assert reuse["task_exclusive_atoms"] == 0.0


def test_live_experiment_coefficient_analysis_schema_is_normalized(
    paired_records: tuple[list[dict], list[dict]], tmp_path: Path
) -> None:
    independent, shared = paired_records
    analysis = {
        "usage_by_task": {
            task: [0.8, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0] for task in H1_TASKS
        },
        "top_atoms_by_task": {task: [0, 1, 2, 3] for task in H1_TASKS},
        "top_k_masks_by_layer": {
            task: {"encoder.layer.0.query": [0, 1, 2, 3]} for task in H1_TASKS
        },
        "pairwise_cosine_similarity": {"sst2:mrpc": 0.99},
        "atom_utilization_count": [5, 5, 5, 5, 0, 0, 0, 0],
        "dead_atoms": [4, 5, 6, 7],
        "task_exclusive_atoms": [],
        "reused_atoms": [0, 1, 2, 3],
    }
    for record in shared:
        record["coefficient_analysis"] = analysis
    independent_root = tmp_path / "independent_lora"
    shared_root = tmp_path / "shared_atoms"
    _write_tree(independent_root, independent)
    _write_tree(shared_root, shared)

    summary = summarize_h1(independent_root, shared_root)

    coefficient = summary["coefficient_analysis"]
    assert coefficient["aggregate"]["task_by_atom_usage_measure"] == "mean_absolute_coefficient"
    assert coefficient["aggregate"]["reuse_unit"] == "atom_index"
    assert coefficient["aggregate"]["reuse_mean_across_seeds"] == pytest.approx(
        {
            "atom_slots": 8.0,
            "dead_atoms": 4.0,
            "task_exclusive_atoms": 0.0,
            "reused_by_two_or_more_tasks": 4.0,
        }
    )
    assert coefficient["aggregate"]["pairwise_coefficient_similarity_mean"][
        "sst2|mrpc"
    ] == pytest.approx(0.99)
    assert "absolute coefficient magnitude" in render_h1_report(summary)
    assert "Top atoms per task and seed" in render_h1_report(summary)


def test_cli_writes_both_outputs(
    tmp_path: Path, paired_records: tuple[list[dict], list[dict]]
) -> None:
    independent, shared = paired_records
    independent_root = tmp_path / "independent_lora"
    shared_root = tmp_path / "shared_atoms"
    output_dir = tmp_path / "output"
    _write_tree(independent_root, independent)
    _write_tree(shared_root, shared)
    environment = os.environ.copy()
    source_root = str(Path(__file__).parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, environment.get("PYTHONPATH", "")) if item
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_h1.py",
            "--independent-root",
            str(independent_root),
            "--shared-root",
            str(shared_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "H1 decision: PASS" in completed.stdout
    assert (output_dir / "h1_summary.json").is_file()
    assert (output_dir / "h1_report.md").is_file()
