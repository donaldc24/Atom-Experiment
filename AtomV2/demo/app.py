"""FastAPI backend for the read-only AtomV2 inference playground.

The demo deliberately imports the experiment harness instead of copying model
or world logic. It loads only ``checkpoints/final.pt`` on CPU and never writes
to ``runs/`` (or anywhere else in the repository).
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


ATOM_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = ATOM_ROOT / "Harness"
RUNS_ROOT = ATOM_ROOT / "runs"
INDEX_PATH = Path(__file__).with_name("index.html")

# The harness is intentionally a source tree rather than an installed package.
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from atomv2 import ops as world_ops  # noqa: E402
from atomv2.config import Config  # noqa: E402
from atomv2.data import PAD_TOKEN, SURFACE_INDEX  # noqa: E402
from atomv2.model import AtomModel  # noqa: E402


Mode = Literal["bottleneck", "raw"]
AtomBudget = Literal[
    "program", "single", "single_confidence", "repeat_majority"
]
OpName = Literal["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]


class RunRequest(BaseModel):
    model_id: str
    digits: list[int]
    ops: list[OpName]
    mode: Mode = "bottleneck"
    atom_budget: AtomBudget = "program"


class AtlasRequest(BaseModel):
    model_id: str
    digits: list[int]


class AtlasSurveyRequest(BaseModel):
    sample_count: int = 300
    random_seed: int = 20250308


app = FastAPI(
    title="Atom Playground",
    description="Read-only CPU inference over existing AtomV2 checkpoints.",
    version="1.0.0",
)

_INFERENCE_LOCK = threading.Lock()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _number_or_none(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def discover_models() -> list[dict]:
    """Scan runs recursively for complete, loadable-looking final runs.

    Discovery reads only config/metrics metadata. A checkpoint is not loaded
    until it is selected for inference.
    """
    records: list[dict] = []
    if not RUNS_ROOT.is_dir():
        return records

    for config_path in RUNS_ROOT.rglob("config.json"):
        run_dir = config_path.parent
        checkpoint_path = run_dir / "checkpoints" / "final.pt"
        if not checkpoint_path.is_file():
            continue

        config = _read_json(config_path)
        if not config:
            continue
        metrics = _read_json(run_dir / "metrics.json")
        try:
            model_id = run_dir.relative_to(RUNS_ROOT).as_posix()
        except ValueError:
            continue

        arm = str(config.get("arm", "unknown"))
        seed = config.get("seed", "?")
        protocol = str(config.get("protocol_revision", "unknown-protocol"))
        seen = _number_or_none(metrics.get("acc_seen_hard"))
        l3 = _number_or_none(metrics.get("acc_unseen_L3_hard"))
        smoke = bool(config.get("smoke", False))
        recommended = (
            not smoke
            and arm in {"A14", "A16"}
            and seen is not None
            and seen >= 0.90
        )
        records.append({
            "model_id": model_id,
            "run_label": f"{arm}_s{seed}_{protocol}",
            "arm": arm,
            "seed": seed,
            "experiment": config.get("experiment"),
            "micro_steps": int(config.get("micro_steps", 3)),
            "protocol_revision": protocol,
            "checkpoint": "final.pt",
            "checkpoint_step": metrics.get("final_step", config.get("total_steps")),
            "atom_profiles_available": (
                run_dir / "panel" / "final" / "standalone_closed_map.npz"
            ).is_file(),
            "seen_acc": seen,
            "l3_acc": l3,
            "smoke": smoke,
            "recommended": recommended,
        })

    # Put healthy A14/A16 result runs first, then sort transparently by the
    # visible seen score. Weak seeds and smoke runs remain available.
    records.sort(key=lambda item: (
        not item["recommended"],
        -(item["seen_acc"] if item["seen_acc"] is not None else -1.0),
        item["smoke"],
        item["model_id"],
    ))
    return records


def _resolve_model(model_id: str) -> tuple[dict, Path]:
    record = next((item for item in discover_models()
                   if item["model_id"] == model_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown model_id")
    checkpoint_path = RUNS_ROOT / Path(*model_id.split("/")) / "checkpoints" / "final.pt"
    return record, checkpoint_path


@lru_cache(maxsize=2)
def _load_checkpoint(checkpoint_path: str, mtime_ns: int):
    """Load and freeze one final checkpoint on CPU.

    ``mtime_ns`` is part of the cache key so replacing a checkpoint cannot
    leave a stale model in a long-running demo process.
    """
    del mtime_ns
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Checkpoint is missing config/model payloads")
    cfg = Config.from_dict(checkpoint["config"])
    model = AtomModel(cfg).cpu()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, int(checkpoint.get("step", cfg.total_steps))


def _validate_digits(digits: list[int]) -> None:
    if len(digits) != 6:
        raise HTTPException(status_code=422, detail="digits must contain exactly 6 integers")
    if any(type(value) is not int or value < 0 or value > 9 for value in digits):
        raise HTTPException(status_code=422, detail="every digit must be an integer from 0 to 9")


def _validate_survey_request(request: AtlasSurveyRequest) -> None:
    if not 1 <= request.sample_count <= 1000:
        raise HTTPException(
            status_code=422, detail="sample_count must be between 1 and 1000")
    if not 0 <= request.random_seed <= 2**32 - 1:
        raise HTTPException(
            status_code=422, detail="random_seed must be between 0 and 4294967295")


def _validate_request(request: RunRequest) -> None:
    _validate_digits(request.digits)
    if not 1 <= len(request.ops) <= 6:
        raise HTTPException(status_code=422, detail="ops must contain between 1 and 6 operations")


def _token_tensors(op: str) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([[SURFACE_INDEX[op], PAD_TOKEN]], dtype=torch.int64)
    n_tokens = torch.ones(1, dtype=torch.int64)
    return tokens, n_tokens


def _route_trace(output: dict, micro_steps: int, tau: float) -> list[dict]:
    logits = output["route_logits"][0, :micro_steps]
    probabilities = F.softmax(logits / max(float(tau), 1e-6), dim=-1)
    choices = output["choices"][0, :micro_steps]
    trace = []
    for micro_step in range(micro_steps):
        choice = int(choices[micro_step].item())
        top_values, top_indices = torch.topk(probabilities[micro_step], k=3)
        top_three = []
        for atom_id, weight in zip(top_indices.tolist(), top_values.tolist()):
            top_three.append({
                "atom_id": atom_id,
                "label": "PASS" if atom_id == 16 else f"A{atom_id}",
                "weight": float(weight),
            })
        trace.append({
            "micro_step": micro_step,
            "chosen_atom_id": choice,
            "chosen": "PASS" if choice == 16 else f"A{choice}",
            "is_pass": choice == 16,
            "top_three": top_three,
        })
    return trace


def _collapse_route(output: dict, micro_steps: int, tau: float,
                    confidence_only: bool = False) -> dict:
    """Collapse a learned token-program to one route without changing weights.

    Normally a strict majority of hard choices wins, with the most confident
    hard choice at any micro-step as fallback. ``confidence_only`` always uses
    the latter rule. The confidence winner is necessarily a route the hard
    program actually selected, not an unobserved fourth option.
    """
    logits = output["route_logits"][0, :micro_steps]
    probabilities = F.softmax(logits / max(float(tau), 1e-6), dim=-1)
    choices = output["choices"][0, :micro_steps].to(torch.int64)
    counts = torch.bincount(choices, minlength=17)
    majority_threshold = micro_steps // 2 + 1
    most_votes, majority_choice = counts.max(dim=0)

    if not confidence_only and int(most_votes.item()) >= majority_threshold:
        selected = int(majority_choice.item())
        strategy = "majority"
        selected_at = None
    else:
        chosen_confidences = probabilities[
            torch.arange(micro_steps), choices]
        selected_at = int(chosen_confidences.argmax().item())
        selected = int(choices[selected_at].item())
        strategy = "highest_step_confidence"

    selected_weights = probabilities[:, selected]
    votes = int(counts[selected].item())
    return {
        "atom_id": selected,
        "label": "PASS" if selected == 16 else f"A{selected}",
        "is_pass": selected == 16,
        "strategy": strategy,
        "votes": votes,
        "votes_possible": micro_steps,
        "vote_share": votes / micro_steps,
        "mean_soft_weight": float(selected_weights.mean().item()),
        "peak_soft_weight": float(selected_weights.max().item()),
        "selected_at_micro_step": selected_at,
    }


def _collapsed_application_count(selection: dict, atom_budget: str) -> int:
    """Number of winner applications for a collapsed execution budget."""
    if atom_budget == "repeat_majority" and selection["strategy"] == "majority":
        return int(selection["votes"])
    return 1


def _atom_usage(stages: list[dict]) -> list[dict]:
    usage: dict[int, list[dict]] = {}
    for stage in stages:
        counts: dict[int, int] = {}
        for atom_id in stage["applied_atom_ids"]:
            counts[atom_id] = counts.get(atom_id, 0) + 1
        for atom_id, count in counts.items():
            usage.setdefault(atom_id, []).append({
                "stage": stage["stage"],
                "op": stage["op"],
                "count": count,
            })
    return [{
        "atom_id": atom_id,
        "label": "PASS" if atom_id == 16 else f"A{atom_id}",
        "is_pass": atom_id == 16,
        "total": sum(segment["count"] for segment in segments),
        "segments": segments,
    } for atom_id, segments in sorted(usage.items())]


def _atom_profiles(model_id: str) -> dict[str, dict]:
    """Read existing standalone one-atom surface scores from panel artifacts."""
    panel_dir = (RUNS_ROOT / Path(*model_id.split("/")) / "panel" / "final")
    tensor_path = panel_dir / "standalone_closed_map.npz"
    summary = _read_json(panel_dir / "standalone.json")
    if not tensor_path.is_file():
        return {}
    try:
        with np.load(tensor_path, allow_pickle=False) as artifact:
            accuracy = np.asarray(artifact["acc"])
    except (OSError, ValueError, KeyError):
        return {}
    # panel.CANDIDATES is sub-ops, then surfaces, then identity. Keep the
    # ordering derived from the harness rather than transcribing column ids.
    surface_offset = len(world_ops.SUBOP_NAMES)
    if accuracy.ndim != 3 or accuracy.shape[0] < 16 or (
            accuracy.shape[1] < surface_offset + len(world_ops.SURFACE_NAMES)):
        return {}

    profiles = {}
    for atom_id in range(16):
        scores = accuracy[
            atom_id,
            surface_offset:surface_offset + len(world_ops.SURFACE_NAMES),
        ].mean(axis=-1)
        best_index = int(scores.argmax())
        profiles[str(atom_id)] = {
            "atom_id": atom_id,
            "label": f"A{atom_id}",
            "surface_scores": [
                {"op": op, "accuracy": float(scores[index])}
                for index, op in enumerate(world_ops.SURFACE_NAMES)
            ],
            "best_surface_op": world_ops.SURFACE_NAMES[best_index],
            "best_surface_accuracy": float(scores[best_index]),
            "best_candidate_overall": summary.get(str(atom_id), {}).get(
                "best_candidate"),
            "best_candidate_accuracy": summary.get(str(atom_id), {}).get(
                "best_acc"),
        }
    return profiles


_SUBOP_GUIDE = {
    "R": ("Reverse", "reverse the six digit positions"),
    "T": ("Rotate left", "rotate left by one position"),
    "W": ("Swap pairs", "swap positions 0↔1, 2↔3, and 4↔5"),
    "I": ("Increment", "add 1 to every digit modulo 10"),
    "N": ("Negate", "replace every digit x with −x modulo 10"),
    "M": ("Multiply ×3", "multiply every digit by 3 modulo 10"),
    "A": ("Add index", "add its zero-based position to every digit modulo 10"),
}


def operation_guide() -> list[dict]:
    example = np.arange(6, dtype=np.int64)[None, :]
    guide = []
    for op in world_ops.SURFACE_NAMES:
        recipe = world_ops.R.SURFACE_RECIPES[op]
        triple = world_ops.SURFACE_TRIPLES[op]
        output = world_ops.apply_triple(triple, example)[0]
        guide.append({
            "op": op,
            "recipe": [
                {"sub_op": sub_op, "name": _SUBOP_GUIDE[sub_op][0],
                 "description": _SUBOP_GUIDE[sub_op][1]}
                for sub_op in recipe
            ],
            "description": (
                f"First {_SUBOP_GUIDE[recipe[0]][1]}, then "
                f"{_SUBOP_GUIDE[recipe[1]][1]}."
            ),
            "example_input": example[0].tolist(),
            "example_output": output.tolist(),
            "canonical_triple": {
                "permutation": list(triple[0]),
                "multiplier": triple[1],
                "offset": list(triple[2]),
            },
            "is_dax": op == "P3",
        })
    return guide


@torch.inference_mode()
def run_inference(request: RunRequest) -> dict:
    """Execute one chain with hard routing and return a JSON-ready trace."""
    _validate_request(request)
    record, checkpoint_path = _resolve_model(request.model_id)
    try:
        model, cfg, checkpoint_step = _load_checkpoint(
            str(checkpoint_path), checkpoint_path.stat().st_mtime_ns)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not load checkpoint: {exc}") from exc

    model_digits = torch.tensor([request.digits], dtype=torch.int64)
    raw_state = model.code(model_digits) if request.mode == "raw" else None
    truth = np.asarray([request.digits], dtype=np.int64)
    stages: list[dict] = []

    for stage_index, op in enumerate(request.ops, start=1):
        tokens, n_tokens = _token_tensors(op)
        if request.mode == "bottleneck":
            output = model(model_digits, tokens, n_tokens, mode="hard", tau=cfg.tau_end)
            boundary_state = output["states"][cfg.micro_steps]
        else:
            output = model.execute_from_state(
                raw_state, tokens, n_tokens, start_token_idx=0,
                mode="hard", tau=cfg.tau_end,
            )
            boundary_state = output["states"][cfg.micro_steps]
            raw_state = boundary_state

        routing = _route_trace(output, cfg.micro_steps, cfg.tau_end)
        collapsed_selection = None
        if request.atom_budget in {
                "single", "single_confidence", "repeat_majority"}:
            collapsed_selection = _collapse_route(
                output, cfg.micro_steps, cfg.tau_end,
                confidence_only=request.atom_budget == "single_confidence")
            selected = collapsed_selection["atom_id"]
            applications = _collapsed_application_count(
                collapsed_selection, request.atom_budget)
            collapsed_selection["applications"] = applications
            start_state = output["states"][0]
            boundary_state = start_state
            if selected != 16:
                for _ in range(applications):
                    boundary_state = model.step_once(boundary_state, selected)
            if request.mode == "raw":
                raw_state = boundary_state

        decoded = model.decoder(boundary_state).argmax(dim=-1)
        decoded_digits = [int(value) for value in decoded[0].tolist()]
        truth = world_ops.apply_triple(world_ops.SURFACE_TRIPLES[op], truth)
        truth_digits = [int(value) for value in truth[0].tolist()]
        matches = [predicted == expected
                   for predicted, expected in zip(decoded_digits, truth_digits)]
        stages.append({
            "stage": stage_index,
            "op": op,
            "decoded_digits": decoded_digits,
            "ground_truth_digits": truth_digits,
            "matches": matches,
            "exact_match": all(matches),
            "routing": routing,
            "collapsed_selection": collapsed_selection,
            "applied_atom_ids": (
                ([collapsed_selection["atom_id"]]
                 * collapsed_selection["applications"])
                if collapsed_selection is not None
                else [route["chosen_atom_id"] for route in routing]
            ),
        })

        # The model's own discrete prediction is the only boundary channel in
        # bottleneck mode: no ground truth is injected.
        if request.mode == "bottleneck":
            model_digits = decoded

    final = stages[-1]
    model_meta = dict(record)
    model_meta["checkpoint_step"] = checkpoint_step
    return {
        "model": model_meta,
        "mode": request.mode,
        "atom_budget": request.atom_budget,
        "input_digits": request.digits,
        "ops": list(request.ops),
        "micro_steps": cfg.micro_steps,
        "soft_weight_temperature": float(cfg.tau_end),
        "beyond_training_distribution": len(request.ops) > 2,
        "trace": stages,
        "final_output_digits": final["decoded_digits"],
        "ground_truth": {
            "boundaries": [stage["ground_truth_digits"] for stage in stages],
            "final_digits": final["ground_truth_digits"],
            "final_matches": final["matches"],
            "final_exact_match": final["exact_match"],
        },
        "atom_usage": _atom_usage(stages),
        "atom_profiles": _atom_profiles(request.model_id),
    }


@torch.inference_mode()
def build_route_atlas(request: AtlasRequest) -> dict:
    """Run every surface op independently from one shared canonical input."""
    _validate_digits(request.digits)
    record, checkpoint_path = _resolve_model(request.model_id)
    try:
        model, cfg, checkpoint_step = _load_checkpoint(
            str(checkpoint_path), checkpoint_path.stat().st_mtime_ns)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not load checkpoint: {exc}") from exc

    digits = torch.tensor([request.digits], dtype=torch.int64)
    source = np.asarray([request.digits], dtype=np.int64)
    operations = []
    for op in world_ops.SURFACE_NAMES:
        tokens, n_tokens = _token_tensors(op)
        output = model(digits, tokens, n_tokens, mode="hard", tau=cfg.tau_end)
        routing = _route_trace(output, cfg.micro_steps, cfg.tau_end)
        decoded = output["logits"].argmax(dim=-1)[0].tolist()
        truth = world_ops.apply_triple(world_ops.SURFACE_TRIPLES[op], source)[0].tolist()
        matches = [int(predicted) == int(expected)
                   for predicted, expected in zip(decoded, truth)]
        unique_atom_ids = list(dict.fromkeys(
            route["chosen_atom_id"] for route in routing
            if not route["is_pass"]))
        unique_atom_results = []
        start_state = output["states"][0]
        for atom_id in unique_atom_ids:
            selected_routes = [
                route for route in routing
                if route["chosen_atom_id"] == atom_id
            ]
            selected_confidences = []
            for route in selected_routes:
                selected = next(
                    candidate for candidate in route["top_three"]
                    if candidate["atom_id"] == atom_id)
                selected_confidences.append(float(selected["weight"]))
            atom_state = model.step_once(start_state, atom_id)
            atom_decoded = model.decoder(atom_state).argmax(dim=-1)[0].tolist()
            atom_matches = [int(predicted) == int(expected)
                            for predicted, expected in zip(atom_decoded, truth)]
            unique_atom_results.append({
                "atom_id": atom_id,
                "label": f"A{atom_id}",
                "decoded_digits": [int(value) for value in atom_decoded],
                "matches": atom_matches,
                "exact_match": all(atom_matches),
                "selected_micro_steps": [
                    route["micro_step"] for route in selected_routes
                ],
                "selected_confidences": selected_confidences,
                "mean_selected_confidence": float(np.mean(selected_confidences)),
                "peak_selected_confidence": float(np.max(selected_confidences)),
            })
        operations.append({
            "op": op,
            "routing": routing,
            "decoded_digits": [int(value) for value in decoded],
            "ground_truth_digits": [int(value) for value in truth],
            "matches": matches,
            "exact_match": all(matches),
            "unique_atom_results": unique_atom_results,
            "pass_selected": any(route["is_pass"] for route in routing),
        })

    model_meta = dict(record)
    model_meta["checkpoint_step"] = checkpoint_step
    return {
        "model": model_meta,
        "input_digits": request.digits,
        "micro_steps": cfg.micro_steps,
        "operations": operations,
        "atom_profiles": _atom_profiles(request.model_id),
        "routing_note": (
            "Each P operation is hard-routed independently as a singleton from "
            "the same canonical encoding; routes may change with the input digits."
        ),
    }


def _confidence_profile(values: torch.Tensor) -> dict:
    """Compact distribution summary for selected-route soft weights."""
    values = values.detach().to(dtype=torch.float64, device="cpu").flatten()
    if values.numel() == 0:
        return {
            "count": 0, "mean": None, "std": None, "min": None,
            "p10": None, "median": None, "p90": None, "max": None,
        }
    quantiles = torch.quantile(
        values, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p10": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
        "max": float(values.max().item()),
    }


@torch.inference_mode()
def _survey_checkpoint(record: dict, digits: torch.Tensor,
                       source: np.ndarray) -> dict:
    """Aggregate all singleton P routes for one checkpoint and input batch."""
    _, checkpoint_path = _resolve_model(record["model_id"])
    try:
        model, cfg, checkpoint_step = _load_checkpoint(
            str(checkpoint_path), checkpoint_path.stat().st_mtime_ns)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load checkpoint {record['run_label']}: {exc}",
        ) from exc

    batch_size = int(digits.shape[0])
    operations = []
    for op in world_ops.SURFACE_NAMES:
        tokens = torch.full(
            (batch_size, 2), PAD_TOKEN, dtype=torch.int64)
        tokens[:, 0] = SURFACE_INDEX[op]
        n_tokens = torch.ones(batch_size, dtype=torch.int64)
        output = model(digits, tokens, n_tokens, mode="hard", tau=cfg.tau_end)
        choices = output["choices"][:, :cfg.micro_steps]
        probabilities = F.softmax(
            output["route_logits"][:, :cfg.micro_steps]
            / max(float(cfg.tau_end), 1e-6),
            dim=-1,
        )
        truth_array = world_ops.apply_triple(
            world_ops.SURFACE_TRIPLES[op], source)
        truth = torch.from_numpy(truth_array).to(dtype=torch.int64)
        program_digits = output["logits"].argmax(dim=-1)
        program_exact = (program_digits == truth).all(dim=1)
        start_state = output["states"][0]

        atoms = []
        for atom_id in range(cfg.n_atoms):
            route_mask = choices.eq(atom_id)
            selected_input_mask = route_mask.any(dim=1)
            selected_inputs = int(selected_input_mask.sum().item())
            if selected_inputs == 0:
                continue

            selected_indices = selected_input_mask.nonzero(
                as_tuple=True)[0]
            atom_state = model.step_once(
                start_state.index_select(0, selected_indices), atom_id)
            atom_digits = model.decoder(atom_state).argmax(dim=-1)
            atom_truth = truth.index_select(0, selected_indices)
            exact = (atom_digits == atom_truth).all(dim=1)
            digit_matches = atom_digits.eq(atom_truth)
            selected_confidences = probabilities[..., atom_id][route_mask]

            by_micro_step = []
            for micro_step in range(cfg.micro_steps):
                step_mask = route_mask[:, micro_step]
                by_micro_step.append({
                    "micro_step": micro_step,
                    "selection_count": int(step_mask.sum().item()),
                    "selection_rate": float(step_mask.float().mean().item()),
                    "confidence": _confidence_profile(
                        probabilities[:, micro_step, atom_id][step_mask]),
                })

            exact_matches = int(exact.sum().item())
            atoms.append({
                "atom_id": atom_id,
                "label": f"A{atom_id}",
                "selected_inputs": selected_inputs,
                "selection_rate": selected_inputs / batch_size,
                "hard_selection_count": int(route_mask.sum().item()),
                "exact_matches": exact_matches,
                "single_atom_accuracy": exact_matches / selected_inputs,
                "digit_accuracy": float(digit_matches.float().mean().item()),
                "confidence": _confidence_profile(selected_confidences),
                "by_micro_step": by_micro_step,
            })

        pass_mask = choices.eq(cfg.n_atoms)
        operations.append({
            "op": op,
            "program_exact_matches": int(program_exact.sum().item()),
            "program_accuracy": float(program_exact.float().mean().item()),
            "pass_selection_count": int(pass_mask.sum().item()),
            "atoms": atoms,
        })

    model_meta = dict(record)
    model_meta["checkpoint_step"] = checkpoint_step
    return {
        "model": model_meta,
        "operations": operations,
    }


def build_atlas_survey(request: AtlasSurveyRequest) -> dict:
    """Survey one-atom outcomes over shared random inputs and healthy seeds."""
    _validate_survey_request(request)
    healthy = [record for record in discover_models() if record["recommended"]]
    if not healthy:
        raise HTTPException(
            status_code=404,
            detail=("No healthy checkpoints found (requires non-smoke A14/A16 "
                    "with seen accuracy >= 90%)."),
        )

    rng = np.random.default_rng(request.random_seed)
    source = rng.integers(
        0, 10, size=(request.sample_count, 6), dtype=np.int64)
    digits = torch.from_numpy(source.copy())
    results = [
        _survey_checkpoint(record, digits, source)
        for record in healthy
    ]
    return {
        "report_type": "atomv2_atlas_survey",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "sample_count_per_model": request.sample_count,
        "random_seed": request.random_seed,
        "input_generation": {
            "generator": "numpy.random.default_rng",
            "range": [0, 9],
            "shape": [request.sample_count, 6],
            "inputs": source.tolist(),
        },
        "healthy_rule": (
            "non-smoke A14/A16 checkpoint with seen accuracy >= 0.90"
        ),
        "healthy_model_count": len(results),
        "total_model_input_cases": len(results) * request.sample_count,
        "models": results,
        "method": {
            "routing": "hard singleton P program from canonical input",
            "atom_trial": (
                "deduplicate selected atom IDs per input and P op; apply each "
                "selected atom independently once to the original canonical state"
            ),
            "accuracy": "six-digit exact match against canonical P truth",
            "confidence": (
                "softmax weight of the atom at every micro-step where it was "
                "the hard selection; repeated selections remain confidence observations"
            ),
        },
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/models")
def models():
    return discover_models()


@app.get("/ops")
def operations():
    return operation_guide()


@app.post("/run")
def run(request: RunRequest):
    # Torch inference is read-only, but serializing calls avoids thread-pool
    # contention and protects the tiny shared LRU cache on a CPU-only server.
    with _INFERENCE_LOCK:
        return run_inference(request)


@app.post("/route-atlas")
def route_atlas(request: AtlasRequest):
    with _INFERENCE_LOCK:
        return build_route_atlas(request)


@app.post("/atlas-survey")
def atlas_survey(request: AtlasSurveyRequest):
    with _INFERENCE_LOCK:
        return build_atlas_survey(request)
