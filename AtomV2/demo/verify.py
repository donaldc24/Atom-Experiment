"""Read-only acceptance checks for the Atom Playground.

This script performs batched inference against the highest-ranked healthy
checkpoint, then spot-checks one routing sequence against the harness's saved
``traces/`` artifact. It prints results only and writes nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .app import (
    AtlasRequest,
    RUNS_ROOT,
    RunRequest,
    _load_checkpoint,
    _resolve_model,
    _token_tensors,
    build_route_atlas,
    discover_models,
    run_inference,
)
from atomv2 import ops as world_ops
from atomv2.data import build_bundle


@torch.inference_mode()
def _batch_chain(model, cfg, inputs: np.ndarray, chain: tuple[str, ...],
                 mode: str) -> list[float]:
    digits = torch.from_numpy(inputs.copy())
    state = model.code(digits)
    truth = inputs.copy()
    boundary_accuracy = []
    for op in chain:
        token_row, _ = _token_tensors(op)
        tokens = token_row.repeat(len(inputs), 1)
        n_tokens = torch.ones(len(inputs), dtype=torch.int64)
        if mode == "bottleneck":
            output = model(digits, tokens, n_tokens, mode="hard", tau=cfg.tau_end)
        else:
            output = model.execute_from_state(
                state, tokens, n_tokens, start_token_idx=0,
                mode="hard", tau=cfg.tau_end,
            )
        state = output["states"][cfg.micro_steps]
        prediction = model.decoder(state).argmax(dim=-1)
        truth = world_ops.apply_triple(world_ops.SURFACE_TRIPLES[op], truth)
        boundary_accuracy.append(float((prediction.numpy() == truth).all(axis=1).mean()))
        if mode == "bottleneck":
            digits = prediction
    return boundary_accuracy


@torch.inference_mode()
def _trace_spot_check(model, cfg, model_id: str, step: int) -> dict:
    trace_path = (RUNS_ROOT / Path(*model_id.split("/")) / "traces"
                  / f"step{step:06d}.npz")
    if not trace_path.is_file():
        return {"passed": False, "reason": f"missing {trace_path.name}"}

    task_data = next(td for td in build_bundle(cfg).seen_heldout
                     if td.task.task_id == "P3")
    tokens = torch.from_numpy(np.tile(task_data.task.tokens, (1, 1)))
    n_tokens = torch.ones(1, dtype=torch.int64)
    output = model(torch.from_numpy(task_data.x[:1]), tokens, n_tokens,
                   mode="hard", tau=cfg.tau_end)
    observed = output["choices"][0].numpy()
    with np.load(trace_path, allow_pickle=False) as trace:
        logged = trace["seen_heldout/P3/choices"][0]
    return {
        "passed": bool(np.array_equal(observed, logged)),
        "observed": observed.tolist(),
        "logged": logged.tolist(),
        "trace": str(trace_path.relative_to(RUNS_ROOT.parent)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", help="runs/-relative model id; defaults to the top healthy model")
    parser.add_argument("--examples", type=int, default=512)
    args = parser.parse_args()

    models = discover_models()
    model_id = args.model_id or next(
        (item["model_id"] for item in models if item["recommended"]),
        models[0]["model_id"] if models else None,
    )
    if model_id is None:
        print(json.dumps({"passed": False, "error": "no final checkpoints found"}, indent=2))
        return 1

    record, checkpoint_path = _resolve_model(model_id)
    model, cfg, step = _load_checkpoint(
        str(checkpoint_path), checkpoint_path.stat().st_mtime_ns)
    inputs = np.random.default_rng(20260820).integers(
        0, 10, size=(args.examples, 6), dtype=np.int64)

    p3 = _batch_chain(model, cfg, inputs, ("P3",), "bottleneck")[-1]
    dax_bottleneck = _batch_chain(
        model, cfg, inputs, ("P1", "P3"), "bottleneck")[-1]
    dax_raw = _batch_chain(model, cfg, inputs, ("P1", "P3"), "raw")[-1]
    depth4 = _batch_chain(
        model, cfg, inputs, ("P1", "P3", "P4", "P2"), "bottleneck")
    trace_check = _trace_spot_check(model, cfg, model_id, step)
    single_atom = run_inference(RunRequest(
        model_id=model_id, digits=[3, 1, 4, 1, 5, 9],
        ops=["P1", "P3"], mode="bottleneck", atom_budget="single"))
    repeated_winner = run_inference(RunRequest(
        model_id=model_id, digits=[3, 1, 4, 1, 5, 9],
        ops=["P1"], mode="bottleneck", atom_budget="repeat_majority"))
    atlas = build_route_atlas(AtlasRequest(
        model_id=model_id, digits=[3, 1, 4, 1, 5, 9]))

    checks = {
        "singleton_p3_near_99pct": p3 >= 0.98,
        "dax_bottleneck_correct": dax_bottleneck >= 0.95,
        "dax_raw_failure_visible": dax_raw < 0.05,
        "depth4_tracks_each_boundary": min(depth4) >= 0.85,
        "routing_matches_saved_trace": trace_check["passed"],
        "single_atom_applies_once_per_op": all(
            len(stage["applied_atom_ids"]) == 1
            for stage in single_atom["trace"]),
        "routing_atlas_covers_every_surface_op": (
            [item["op"] for item in atlas["operations"]]
            == [f"P{i}" for i in range(1, 9)]
            and all(len(item["routing"]) == cfg.micro_steps
                    for item in atlas["operations"])),
        "atlas_deduplicates_atom_outcomes": all(
            len({result["atom_id"] for result in item["unique_atom_results"]})
            == len(item["unique_atom_results"])
            for item in atlas["operations"]),
        "repeated_winner_preserves_application_count": (
            len(repeated_winner["trace"][0]["applied_atom_ids"])
            == repeated_winner["trace"][0]["collapsed_selection"][
                "applications"]),
    }
    report = {
        "passed": all(checks.values()),
        "model_id": model_id,
        "run_label": record["run_label"],
        "checkpoint_step": step,
        "n_examples": args.examples,
        "metrics": {
            "singleton_P3": p3,
            "P1_then_P3_bottleneck": dax_bottleneck,
            "P1_then_P3_raw": dax_raw,
            "depth4_bottleneck_boundaries": depth4,
            "single_atom_sample": {
                "selected": [
                    stage["collapsed_selection"]["label"]
                    for stage in single_atom["trace"]
                ],
                "final_exact_match": single_atom["ground_truth"][
                    "final_exact_match"],
            },
        },
        "routing_spot_check": trace_check,
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
