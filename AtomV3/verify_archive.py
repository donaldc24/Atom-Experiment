"""Independent exhaustive verifier and experiment orchestration for AtomV3.

Only this verification module may use the task oracle. Candidate compilation
and the physically paged runtime are separate processes/modules. A candidate
is admitted only if its pre-verification harvesting gate passed, its grammar
is valid, and exhaustive integer execution matches all 10^6 oracle inputs.
Certification applies to the exported integer program, not the neural teacher.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "AtomV2" / "Harness"))
from atomv2 import ops  # noqa: E402 -- quarantined verification oracle

try:
    from .runtime import apply_program, program_from_bytes
except ImportError:
    from runtime import apply_program, program_from_bytes


TOKENS = tuple(f"P{i}" for i in range(1, 9))
EXHAUSTIVE_INPUTS = 10 ** 6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grammar_checks(source, table) -> dict:
    source = np.asarray(source)
    table = np.asarray(table)
    return {
        "source_shape": source.shape == (6,),
        "table_shape": table.shape == (6, 10),
        "source_integer": bool(np.all(source == np.floor(source))),
        "table_integer": bool(np.all(table == np.floor(table))),
        "source_in_range": bool(np.all((source >= 0) & (source < 6))),
        "table_in_range": bool(np.all((table >= 0) & (table < 10))),
        "source_is_permutation": source.shape == (6,) and sorted(source.tolist()) == list(range(6)),
        "each_table_is_digit_permutation": table.shape == (6, 10) and all(
            sorted(row.tolist()) == list(range(10)) for row in table),
    }


def verify_program(source, table, token: str, chunk_size: int = 8192,
                   total_inputs: int = EXHAUSTIVE_INPUTS, truth_fn=None) -> dict:
    """Test all inputs, including after first failure; retain its witness.

    total_inputs can be reduced for unit tests only. Such a result explicitly
    cannot certify an archive entry, even if every checked example passed.
    """
    if token not in TOKENS and truth_fn is None:
        raise ValueError(f"unknown token {token}")
    if not (1 <= total_inputs <= EXHAUSTIVE_INPUTS) or chunk_size < 1:
        raise ValueError("invalid verification domain or chunk size")
    checks = grammar_checks(source, table)
    executable = all(value for key, value in checks.items()
                     if key not in {"source_is_permutation", "each_table_is_digit_permutation"})
    if not executable:
        return {"grammar": checks, "grammar_valid": False,
                "exhaustive_pass": False, "is_complete_domain": False,
                "inputs_checked": 0, "mismatches": None,
                "first_counterexample": None}
    program = {"source": np.asarray(source, dtype=np.uint8),
               "table": np.asarray(table, dtype=np.uint8)}
    mismatches = 0
    digit_mismatches = 0
    first = None
    powers = 10 ** np.arange(5, -1, -1, dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, total_inputs, chunk_size):
        ids = np.arange(start, min(start + chunk_size, total_inputs), dtype=np.int64)
        x = ((ids[:, None] // powers[None, :]) % 10).astype(np.uint8)
        prediction = apply_program(program, x)
        # Concrete independently specified function composition, not the
        # fitted source/LUT or the neural teacher's predictions.
        oracle = ops.SURFACE_FNS[token] if truth_fn is None else truth_fn
        truth = oracle(x.astype(np.int64))
        wrong_digits = prediction != truth
        wrong = wrong_digits.any(axis=1)
        mismatches += int(wrong.sum())
        digit_mismatches += int(wrong_digits.sum())
        if first is None and np.any(wrong):
            i = int(np.flatnonzero(wrong)[0])
            first = {"input_index": int(ids[i]), "input": x[i].tolist(),
                     "prediction": prediction[i].tolist(), "truth": truth[i].tolist()}
    complete = total_inputs == EXHAUSTIVE_INPUTS
    return {"grammar": checks, "grammar_valid": all(checks.values()),
            "exhaustive_pass": complete and mismatches == 0,
            "checked_inputs_pass": mismatches == 0,
            "is_complete_domain": complete, "inputs_checked": total_inputs,
            "mismatches": mismatches, "digit_mismatches": digit_mismatches,
            "exact_accuracy": 1 - mismatches / total_inputs,
            "first_counterexample": first,
            "elapsed_seconds": time.perf_counter() - started,
            "claim_scope": "Exported integer program on all six-digit inputs; not neural teacher"}


def permutation_closure(generators: list[list[int]]) -> dict:
    """Complete finite semigroup closure (a group here because all maps invert)."""
    identity = tuple(range(6))
    perms = [tuple(int(v) for v in p) for p in generators]
    if any(sorted(p) != list(range(6)) for p in perms):
        raise ValueError("closure generators must be six-position permutations")
    seen = {identity}
    queue = deque([identity])
    while queue:
        first = queue.popleft()
        for second in perms:
            composed = tuple(first[second[j]] for j in range(6))
            if composed not in seen:
                seen.add(composed)
                queue.append(composed)
    return {"count": len(seen), "permutations": [list(p) for p in sorted(seen)],
            "contains_reversal": tuple(reversed(range(6))) in seen,
            "scope": "Position permutations only; digit-lookup composition is not enumerated"}


def _real_origin(origin: str, teacher: str) -> bool:
    return teacher.lower() in {"a14", "a33"} and origin.startswith(("a14_", "a33_"))


def build_archive(harvest_files: list[str | Path], outdir: str | Path) -> dict:
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    binary_dir = outdir / "operators"
    binary_dir.mkdir(exist_ok=True)
    diagnostics = []
    entries = []
    teacher_counts = {}
    data_audits = []
    for harvest_file in sorted(map(Path, harvest_files)):
        harvest = json.loads(harvest_file.read_text(encoding="utf-8"))
        origin = harvest["origin"]
        teacher = harvest["teacher"]
        query_info = harvest.get("query_data")
        if query_info:
            query_path = harvest_file.parent / query_info["file"]
            if sha256(query_path) != query_info["sha256"]:
                raise ValueError(f"query-data hash mismatch: {query_path}")
            with np.load(query_path) as query:
                overlap = []
                for k, token in enumerate(TOKENS):
                    cal = {row.tobytes() for row in query["x_cal"][k]}
                    val = {row.tobytes() for row in query["x_val"][k]}
                    overlap.append({"token": token, "overlap": len(cal & val),
                                    "calibration_unique": len(cal), "validation_unique": len(val)})
            data_audits.append({"origin": origin, "tokens": overlap,
                                "disjoint": all(row["overlap"] == 0 for row in overlap),
                                "expected_counts": all(row["calibration_unique"] == 2048
                                                       and row["validation_unique"] == 2048
                                                       for row in overlap)})
        real = _real_origin(origin, teacher)
        n_admitted = 0
        for token in TOKENS:
            record = harvest["programs"][token]
            candidate = harvest_file.parent / record["candidate_file"]
            blob = candidate.read_bytes()
            parsed = program_from_bytes(blob)
            source = parsed["source"].tolist()
            table = parsed["table"].tolist()
            if source != record["source"] or table != record["table"]:
                raise ValueError(f"candidate binary disagrees with fitted record: {candidate}")
            if record.get("sha256") and sha256(candidate) != record["sha256"]:
                raise ValueError(f"candidate hash mismatch: {candidate}")
            verification = verify_program(source, table, token)
            eligible = bool(record["accepted_candidate"] and verification["grammar_valid"]
                            and verification["exhaustive_pass"])
            admitted = real and eligible
            result = {"origin": origin, "teacher": teacher, "token": token,
                      "skill_id": record["skill_id"],
                      "candidate_accepted": bool(record["accepted_candidate"]),
                      "candidate_gates": record["gates"], "verification": verification,
                      "eligible_after_verification": eligible,
                      "admitted_real_archive": admitted,
                      "candidate_sha256": sha256(candidate),
                      "candidate_path": str(candidate.resolve()),
                      "is_control": not real}
            diagnostics.append(result)
            if admitted:
                destination = binary_dir / f"{record['skill_id']}.bin"
                destination.write_bytes(blob)
                entries.append({"skill_id": record["skill_id"], "token": token,
                                "origin": origin, "teacher": teacher,
                                "source": source, "table": table,
                                "path": str(destination), "payload_bytes": len(blob),
                                "sha256": sha256(destination)})
                n_admitted += 1
        if real:
            teacher_counts[origin] = n_admitted
    entries.sort(key=lambda item: (item["origin"], item["token"]))
    by_token = {token: [entry for entry in entries if entry["token"] == token]
                for token in TOKENS}
    priority = lambda entry: (0 if entry["teacher"].lower() == "a33" else 1,
                              entry["origin"], entry["skill_id"])
    primary = {token: min(available, key=priority)
               for token, available in by_token.items() if available}
    initial = [entry for entry in entries if entry["teacher"].lower() == "a33"]
    summary = {
        "schema_version": 1, "entries": entries, "by_token": by_token,
        "primary": primary, "teacher_certified_counts": teacher_counts,
        "acquisition_gate": sum(teacher_counts.get(f"a14_s{seed}", 0) >= 7
                                 for seed in (0, 1, 2)) >= 2,
        "union_eight_tokens_gate": len(primary) == 8,
        "initial_a33_closure": permutation_closure([entry["source"] for entry in initial]),
        "final_real_archive_closure": permutation_closure([entry["source"] for entry in entries]),
        "admitted_operator_count": len(entries),
        "archive_payload_bytes": sum(entry["payload_bytes"] for entry in entries),
        "query_data_audits": data_audits,
        "query_data_gate": len(data_audits) == len(harvest_files) and all(
            audit["disjoint"] and audit["expected_counts"] for audit in data_audits),
        "diagnostics": diagnostics,
        "harvest_provenance": [{"path": str(Path(path).resolve()),
                                "sha256": sha256(Path(path))} for path in harvest_files],
        "interpretation": (
            "Exhaustive verification certifies exported integer programs. It does not "
            "certify their teachers or discover a representation without human bias: "
            "the source/LUT grammar and shared six-digit interface are explicit. "
            "Synthetic controls are diagnosed but never admitted to the real archive."
        ),
    }
    (outdir / "archive.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_runtime_plan(archive: dict, seed: int = 20260913,
                       lengths: tuple = (2, 4, 8, 16, 64, 256),
                       sequences_per_length: int = 100,
                       inputs_per_sequence: int = 32) -> dict:
    if not archive["union_eight_tokens_gate"]:
        raise ValueError("full runtime panel requires all eight certified tokens")
    rng = np.random.default_rng(seed)
    sequences = []
    origin_counters = {token: 0 for token in TOKENS}
    for length in lengths:
        if length == 2:
            token_sequences = list(itertools.product(TOKENS, repeat=2))
        else:
            token_sequences = []
            seen = set()
            while len(token_sequences) < sequences_per_length:
                candidate = tuple(TOKENS[int(i)] for i in rng.integers(0, 8, size=length))
                if candidate not in seen:
                    seen.add(candidate)
                    token_sequences.append(candidate)
        for index, tokens in enumerate(token_sequences):
            tokens = list(tokens)
            chosen = []
            for k, token in enumerate(tokens):
                alternatives = archive["by_token"][token]
                chosen.append(alternatives[origin_counters[token] % len(alternatives)])
                origin_counters[token] += 1
            x = rng.integers(0, 10, size=(inputs_per_sequence, 6), dtype=np.uint8)
            sequences.append({"sequence_id": f"L{length}_{index:03d}",
                              "inputs": x.tolist(), "tokens": tokens,
                              "origins": [entry["origin"] for entry in chosen],
                              "skill_paths": [entry["path"] for entry in chosen]})
    return {"schema_version": 1, "seed": seed, "cache_capacity": 1,
            "sequences_per_length": sequences_per_length,
            "inputs_per_sequence": inputs_per_sequence,
            "sequences": sequences}


def benchmark_archive(archive: dict, outdir: str | Path,
                      python_executable: str = sys.executable) -> dict:
    """Launch a fresh runtime with no neural weights, then verify predictions."""
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plan = build_runtime_plan(archive)
    evaluator_plan_path = outdir / "runtime_evaluator_plan.json"
    plan_path = outdir / "runtime_plan.json"
    output_path = outdir / "runtime_output.json"
    evaluator_plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    deployment_plan = {"schema_version": 1, "cache_capacity": 1,
                       "sequences": [{"sequence_id": s["sequence_id"],
                                      "inputs": s["inputs"],
                                      "skill_paths": s["skill_paths"]}
                                     for s in plan["sequences"]]}
    plan_path.write_text(json.dumps(deployment_plan) + "\n", encoding="utf-8")
    completed = subprocess.run([python_executable, str(Path(__file__).with_name("runtime.py")),
                                "--plan", str(plan_path), "--output", str(output_path)],
                               capture_output=True, text=True, check=True)
    output = json.loads(output_path.read_text(encoding="utf-8"))
    if len(output["sequences"]) != len(plan["sequences"]):
        raise ValueError("runtime omitted sequence outputs")
    per_length = {}
    witnesses = []
    used_origins = set()
    for sequence, result in zip(plan["sequences"], output["sequences"]):
        if sequence["sequence_id"] != result["sequence_id"]:
            raise ValueError("runtime sequence order changed")
        truth = np.asarray(sequence["inputs"], dtype=np.int64)
        for token in sequence["tokens"]:
            truth = ops.SURFACE_FNS[token](truth)
        prediction = np.asarray(result["prediction"])
        correct = (prediction == truth).all(axis=1)
        length = len(sequence["tokens"])
        group = per_length.setdefault(str(length), {"correct": 0, "total": 0,
                                                   "sequences": 0, "mixed_origin_sequences": 0})
        group["correct"] += int(correct.sum())
        group["total"] += len(correct)
        group["sequences"] += 1
        group["mixed_origin_sequences"] += int(len(set(sequence["origins"])) > 1)
        used_origins.update(sequence["origins"])
        if not correct.all() and len(witnesses) < 10:
            i = int(np.flatnonzero(~correct)[0])
            witnesses.append({"sequence_id": sequence["sequence_id"],
                              "input": sequence["inputs"][i],
                              "prediction": prediction[i].tolist(), "truth": truth[i].tolist()})
    for group in per_length.values():
        group["accuracy"] = group["correct"] / group["total"]
    result = {"per_length": per_length, "used_origins": sorted(used_origins),
              "all_lengths_exact": all(g["accuracy"] == 1.0 for g in per_length.values()),
              "counterexamples": witnesses, "runtime": output["runtime"],
              "counters": output["counters"], "elapsed_seconds": output["elapsed_seconds"],
              "memory_before": output["memory_before"], "memory_after": output["memory_after"],
              "memory_note": output["memory_note"], "subprocess_pid": output["pid"],
              "plan_sha256": sha256(plan_path), "output_sha256": sha256(output_path),
              "evaluator_plan_sha256": sha256(evaluator_plan_path),
              "subprocess_stderr": completed.stderr,
              "payload_capacity_gate": output["counters"]["peak_loaded_operator_payload_bytes"] == 66,
              "no_neural_weights_gate": output["runtime"]["neural_weight_bytes_loaded"] == 0}
    (outdir / "runtime_verdict.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def summarize_experiment(archive: dict, runtime: dict | None) -> dict:
    controls = {}
    for teacher in ("oracle", "shuffled", "wrong"):
        rows = [r for r in archive["diagnostics"] if r["teacher"] == teacher]
        controls[teacher] = {"candidates": len(rows),
                             "compiler_accepted": sum(r["candidate_accepted"] for r in rows),
                             "exhaustively_correct": sum(r["verification"]["exhaustive_pass"] for r in rows),
                             "accepted_and_correct": sum(r["eligible_after_verification"] for r in rows),
                             "admitted_real_archive": sum(r["admitted_real_archive"] for r in rows)}
    checks = {
        "a14_acquisition": archive["acquisition_gate"],
        "union_eight": archive["union_eight_tokens_gate"],
        "initial_reversal_absent": not archive["initial_a33_closure"]["contains_reversal"],
        "acquired_reversal_present": archive["final_real_archive_closure"]["contains_reversal"],
        "runtime_all_lengths_exact": bool(runtime and runtime["all_lengths_exact"]),
        "runtime_payload_capacity": bool(runtime and runtime["payload_capacity_gate"]),
        "runtime_no_neural_weights": bool(runtime and runtime["no_neural_weights_gate"]),
        "shuffled_rejected_by_compiler": controls["shuffled"]["candidates"] == 8
            and controls["shuffled"]["compiler_accepted"] == 0,
        "coherent_wrong_proposed_then_rejected": controls["wrong"]["candidates"] == 8
            and controls["wrong"]["compiler_accepted"] > 0
            and controls["wrong"]["exhaustively_correct"] == 0,
    }
    rig = {"exact_control_exports_all": controls["oracle"]["accepted_and_correct"] == 8,
           "controls_excluded": all(c["admitted_real_archive"] == 0 for c in controls.values()),
           "query_data": archive["query_data_gate"]}
    if not all(rig.values()):
        outcome = "INVALID_RIG"
    elif all(checks.values()):
        outcome = "PORTABLE_SKILL_ACQUISITION_SUPPORTED_IN_ATOM_WORLD"
    elif any(count for name, count in archive["teacher_certified_counts"].items()
             if name.startswith("a14_")):
        outcome = "PARTIAL_ACQUISITION_ONLY"
    else:
        outcome = "EXTRACTION_FALSIFIED_AT_THIS_BUDGET"
    return {"outcome": outcome, "checks": checks, "rig_checks": rig,
            "controls": controls, "teacher_certified_counts": archive["teacher_certified_counts"],
            "initial_permutation_closure_size": archive["initial_a33_closure"]["count"],
            "final_permutation_closure_size": archive["final_real_archive_closure"]["count"],
            "archive_payload_bytes": archive["archive_payload_bytes"],
            "archive_operator_count": archive["admitted_operator_count"],
            "runtime": runtime,
            "source_sha256": {name: sha256(Path(__file__).with_name(name))
                              for name in ("verify_archive.py", "runtime.py")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvest", type=Path, nargs="+")
    parser.add_argument("--harvest-root", type=Path)
    parser.add_argument("--outdir", "--output-dir", dest="outdir", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    harvest_files = args.harvest
    if args.harvest_root:
        if harvest_files:
            raise ValueError("choose --harvest or --harvest-root")
        harvest_files = sorted(args.harvest_root.glob("*/programs.json"))
        if not harvest_files:
            raise ValueError("no programs.json files under harvest root")
    if harvest_files:
        archive = build_archive(harvest_files, args.outdir)
    else:
        archive = json.loads((args.outdir / "archive.json").read_text(encoding="utf-8"))
    if args.benchmark or args.harvest_root:
        runtime = benchmark_archive(archive, args.outdir) if archive["union_eight_tokens_gate"] else None
        if runtime is None:
            skipped = {"not_run": True, "reason": "incomplete certified library",
                       "all_lengths_exact": False, "payload_capacity_gate": False,
                       "no_neural_weights_gate": False}
            (args.outdir / "runtime_verdict.json").write_text(
                json.dumps(skipped, indent=2) + "\n", encoding="utf-8")
        summary = summarize_experiment(archive, runtime)
        metadata = args.outdir / "archive.json"
        summary["archive_metadata_bytes"] = metadata.stat().st_size
        (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"outcome": summary["outcome"], "checks": summary["checks"]}))


if __name__ == "__main__":
    main()
