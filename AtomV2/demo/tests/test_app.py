import json

import numpy as np
import pytest
import torch
from fastapi import HTTPException

from demo import app as demo_app


def test_discovery_requires_config_and_final_checkpoint(tmp_path, monkeypatch):
    complete = tmp_path / "e4" / "a14"
    (complete / "checkpoints").mkdir(parents=True)
    (complete / "checkpoints" / "final.pt").write_bytes(b"test")
    (complete / "config.json").write_text(json.dumps({
        "arm": "A14", "seed": 1, "experiment": "e4", "smoke": False,
        "protocol_revision": "test-protocol", "micro_steps": 3,
        "total_steps": 30,
    }), encoding="utf-8")
    (complete / "metrics.json").write_text(json.dumps({
        "acc_seen_hard": .99, "acc_unseen_L3_hard": 0, "final_step": 30,
    }), encoding="utf-8")

    incomplete = tmp_path / "e4" / "unfinished"
    incomplete.mkdir(parents=True)
    (incomplete / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(demo_app, "RUNS_ROOT", tmp_path)
    records = demo_app.discover_models()
    assert [record["model_id"] for record in records] == ["e4/a14"]
    assert records[0]["run_label"] == "A14_s1_test-protocol"
    assert records[0]["recommended"] is True


@pytest.mark.parametrize("digits, ops", [
    ([1, 2, 3, 4, 5], ["P1"]),
    ([1, 2, 3, 4, 5, 10], ["P1"]),
    ([1, 2, 3, 4, 5, 6], []),
    ([1, 2, 3, 4, 5, 6], ["P1"] * 7),
])
def test_request_validation(digits, ops):
    request = demo_app.RunRequest(
        model_id="e4/a14", digits=digits, ops=ops, mode="bottleneck")
    with pytest.raises(HTTPException) as error:
        demo_app._validate_request(request)
    assert error.value.status_code == 422


def test_atom_usage_is_segmented_by_stage():
    stages = [
        {"stage": 1, "op": "P1", "routing": [
            {"chosen_atom_id": 2}, {"chosen_atom_id": 2}, {"chosen_atom_id": 16}],
         "applied_atom_ids": [2, 2, 16]},
        {"stage": 2, "op": "P3", "routing": [
            {"chosen_atom_id": 2}, {"chosen_atom_id": 4}, {"chosen_atom_id": 4}],
         "applied_atom_ids": [2, 4, 4]},
    ]
    usage = {item["label"]: item for item in demo_app._atom_usage(stages)}
    assert usage["A2"]["total"] == 3
    assert usage["A2"]["segments"] == [
        {"stage": 1, "op": "P1", "count": 2},
        {"stage": 2, "op": "P3", "count": 1},
    ]
    assert usage["PASS"]["total"] == 1


def _routing_output(choices, selected_logits):
    logits = torch.zeros(1, len(choices), 17)
    for step, (choice, value) in enumerate(zip(choices, selected_logits)):
        logits[0, step, choice] = value
    return {
        "choices": torch.tensor([choices]),
        "route_logits": logits,
    }


def test_single_atom_collapse_prefers_strict_majority():
    selection = demo_app._collapse_route(
        _routing_output([2, 3, 2], [2.0, 8.0, 2.0]), 3, 1.0)
    assert selection["atom_id"] == 2
    assert selection["strategy"] == "majority"
    assert selection["votes"] == 2


def test_single_atom_collapse_uses_most_confident_pick_without_majority():
    selection = demo_app._collapse_route(
        _routing_output([2, 3, 4], [2.0, 8.0, 3.0]), 3, 1.0)
    assert selection["atom_id"] == 3
    assert selection["strategy"] == "highest_step_confidence"
    assert selection["selected_at_micro_step"] == 1


def test_confidence_only_ignores_an_existing_majority():
    selection = demo_app._collapse_route(
        _routing_output([2, 2, 3], [2.0, 2.0, 8.0]), 3, 1.0,
        confidence_only=True)
    assert selection["atom_id"] == 3
    assert selection["strategy"] == "highest_step_confidence"
    assert selection["selected_at_micro_step"] == 2


def test_repeat_majority_preserves_only_winner_multiplicity():
    majority = demo_app._collapse_route(
        _routing_output([1, 1, 2], [3.0, 3.0, 8.0]), 3, 1.0)
    assert majority["atom_id"] == 1
    assert demo_app._collapsed_application_count(
        majority, "repeat_majority") == 2
    assert demo_app._collapsed_application_count(majority, "single") == 1

    no_majority = demo_app._collapse_route(
        _routing_output([1, 2, 3], [3.0, 8.0, 4.0]), 3, 1.0)
    assert demo_app._collapsed_application_count(
        no_majority, "repeat_majority") == 1


def test_atom_profiles_read_full_surface_score_tensor(tmp_path, monkeypatch):
    panel = tmp_path / "e4" / "a14" / "panel" / "final"
    panel.mkdir(parents=True)
    accuracy = np.zeros((16, 16, 1))
    accuracy[3, 7 + 1, 0] = .87  # Surface columns start after seven sub-ops.
    np.savez_compressed(panel / "standalone_closed_map.npz", acc=accuracy)
    (panel / "standalone.json").write_text(json.dumps({
        "3": {"best_candidate": "surf:P2", "best_acc": .87},
    }), encoding="utf-8")
    monkeypatch.setattr(demo_app, "RUNS_ROOT", tmp_path)

    profile = demo_app._atom_profiles("e4/a14")["3"]
    assert profile["surface_scores"][1] == {"op": "P2", "accuracy": .87}
    assert profile["best_surface_op"] == "P2"


def test_operation_guide_is_derived_for_all_surface_ops():
    guide = demo_app.operation_guide()
    assert [item["op"] for item in guide] == [f"P{i}" for i in range(1, 9)]
    p3 = guide[2]
    assert [step["sub_op"] for step in p3["recipe"]] == ["A", "R"]
    assert p3["example_output"] == [0, 8, 6, 4, 2, 0]


def test_atlas_request_uses_same_digit_validation():
    request = demo_app.AtlasRequest(model_id="e4/a14", digits=[1, 2, 3])
    with pytest.raises(HTTPException) as error:
        demo_app.build_route_atlas(request)
    assert error.value.status_code == 422


def test_selected_route_confidence_is_always_present_in_top_three():
    output = _routing_output([2, 3, 2], [2.0, 8.0, 3.0])
    trace = demo_app._route_trace(output, 3, 1.0)
    for route in trace:
        selected = [item for item in route["top_three"]
                    if item["atom_id"] == route["chosen_atom_id"]]
        assert len(selected) == 1


def test_confidence_profile_reports_distribution_quantiles():
    profile = demo_app._confidence_profile(torch.tensor([.1, .2, .8, .9]))
    assert profile["count"] == 4
    assert profile["mean"] == pytest.approx(.5)
    assert profile["median"] == pytest.approx(.5)
    assert profile["p10"] < profile["median"] < profile["p90"]

    empty = demo_app._confidence_profile(torch.tensor([]))
    assert empty["count"] == 0
    assert empty["mean"] is None


@pytest.mark.parametrize("sample_count, random_seed", [
    (0, 7), (1001, 7), (10, -1), (10, 2**32),
])
def test_survey_request_validation(sample_count, random_seed):
    request = demo_app.AtlasSurveyRequest(
        sample_count=sample_count, random_seed=random_seed)
    with pytest.raises(HTTPException) as error:
        demo_app._validate_survey_request(request)
    assert error.value.status_code == 422


def test_survey_uses_only_healthy_models_and_shared_reproducible_inputs(
        monkeypatch):
    records = [
        {"model_id": "healthy", "run_label": "A14_s1", "recommended": True},
        {"model_id": "weak", "run_label": "A14_s2", "recommended": False},
    ]
    captured = []

    monkeypatch.setattr(demo_app, "discover_models", lambda: records)

    def fake_survey(record, digits, source):
        captured.append((record["model_id"], digits.clone(), source.copy()))
        return {"model": record, "operations": []}

    monkeypatch.setattr(demo_app, "_survey_checkpoint", fake_survey)
    report = demo_app.build_atlas_survey(
        demo_app.AtlasSurveyRequest(sample_count=5, random_seed=99))

    assert [item[0] for item in captured] == ["healthy"]
    assert report["healthy_model_count"] == 1
    assert report["total_model_input_cases"] == 5
    assert report["input_generation"]["inputs"] == captured[0][2].tolist()
    expected = np.random.default_rng(99).integers(
        0, 10, size=(5, 6), dtype=np.int64).tolist()
    assert report["input_generation"]["inputs"] == expected
