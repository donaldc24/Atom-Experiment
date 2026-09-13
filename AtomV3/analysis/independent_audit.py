"""Post-result independent audit; never imports original truth or executable code.

This supplement does not change the registered experiment. Truth is implemented
below directly from the eight operation definitions. It reads frozen artifacts,
checks saved execution, and independently checks every distinct admitted binary
on the full finite domain. Only the supplementary audit JSON is written.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "AtomV3/results"
VERIFIED = RESULTS / "verified"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(token, x):
    """Direct formulas for a six-item tuple, also valid for six NumPy columns."""
    if token == "P1":
        return tuple((x[5 - j] + 1) % 10 for j in range(6))
    if token == "P2":
        return tuple((x[(j + 1) % 6] + j) % 10 for j in range(6))
    if token == "P3":
        return tuple((x[5 - j] + 5 - j) % 10 for j in range(6))
    if token == "P4":
        return tuple(3 * x[(j + 1) % 6] % 10 for j in range(6))
    if token == "P5":
        return tuple(-x[j ^ 1] % 10 for j in range(6))
    if token == "P6":
        return tuple((3 * x[j] + 3) % 10 for j in range(6))
    if token == "P7":
        return tuple((x[j ^ 1] + j) % 10 for j in range(6))
    if token == "P8":
        return tuple(-x[(j + 1) % 6] % 10 for j in range(6))
    raise ValueError(token)


def closure(permutations):
    """Repeated all-pairs multiplication, independent of registered BFS code."""
    group = {tuple(range(6))} | set(map(tuple, permutations))
    while True:
        expanded = group | {tuple(a[b[j]] for j in range(6)) for a in group for b in group}
        if expanded == group:
            return {"size": len(group), "contains_reversal": (5, 4, 3, 2, 1, 0) in group}
        group = expanded


def audit():
    started = time.perf_counter()
    paths = {name: VERIFIED / (name + ".json") for name in (
        "runtime_evaluator_plan", "runtime_plan", "runtime_output", "archive")}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    plan, request, output, archive = (data[name] for name in paths)
    registration = json.loads((RESULTS / "REGISTRATION.json").read_text())
    source_failures = [path for path, expected in registration["source_sha256"].items()
                       if digest(REPO / path) != expected]
    checkpoint_failures = [path for path, expected in registration["checkpoint_sha256"].items()
                           if digest(REPO / path) != expected]
    artifact_failures, artifact_count = [], 0
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        expected, path = line.split("  ", 1)
        artifact_count += 1
        if digest(RESULTS / path) != expected:
            artifact_failures.append(path)
    assert not source_failures and not checkpoint_failures and not artifact_failures

    by_id = {row["sequence_id"]: row for row in output["sequences"]}
    requests = {row["sequence_id"]: row for row in request["sequences"]}
    entries = {Path(entry["path"]).resolve(): entry for entry in archive["entries"]}
    assert len(by_id) == len(output["sequences"]) == len(plan["sequences"]) == len(requests) == 564
    stats = defaultdict(lambda: {"sequences": 0, "examples": 0, "correct": 0,
                                 "mixed_origin_sequences": 0, "unique_token_sequences": set()})
    origins, errors, steps = set(), [], 0
    for sequence in plan["sequences"]:
        sid, tokens = sequence["sequence_id"], sequence["tokens"]
        expected = []
        for row in sequence["inputs"]:
            state = tuple(row)
            for token in tokens:
                state = truth(token, state)
            expected.append(list(state))
        prediction = by_id[sid]["prediction"]
        correct = sum(p == t for p, t in zip(prediction, expected))
        if len(prediction) != len(expected) or correct != len(expected):
            errors.append(sid)
        req = requests[sid]
        assert set(req) == {"sequence_id", "inputs", "skill_paths"}
        assert req["inputs"] == sequence["inputs"] and req["skill_paths"] == sequence["skill_paths"]
        assert len(sequence["skill_paths"]) == len(sequence["origins"]) == len(tokens)
        for token, origin, path in zip(tokens, sequence["origins"], sequence["skill_paths"]):
            entry = entries[Path(path).resolve()]
            assert entry["token"] == token and entry["origin"] == origin
        origins.update(sequence["origins"])
        row = stats[len(tokens)]
        row["sequences"] += 1
        row["examples"] += len(expected)
        row["correct"] += correct
        row["mixed_origin_sequences"] += int(len(set(sequence["origins"])) > 1)
        row["unique_token_sequences"].add(tuple(tokens))
        steps += len(tokens)
    for row in stats.values():
        row["unique_token_sequences"] = len(row["unique_token_sequences"])

    unique = {}
    for entry in entries.values():
        blob = Path(entry["path"]).read_bytes()
        assert len(blob) == 66 and hashlib.sha256(blob).hexdigest() == entry["sha256"]
        unique.setdefault((entry["token"], blob), []).append(entry["origin"])
    certificates = {}
    for (token, blob), native_origins in unique.items():
        source = list(blob[:6])
        table = np.array(list(blob[6:])).reshape(6, 10)
        mismatches = 0
        for start in range(0, 1_000_000, 32768):
            ids = np.arange(start, min(start + 32768, 1_000_000))
            inputs = np.stack([(ids // 10 ** (5 - j)) % 10 for j in range(6)], axis=1)
            prediction = np.stack([table[j, inputs[:, source[j]]] for j in range(6)], axis=1)
            expected = np.array(truth(token, [inputs[:, j] for j in range(6)])).T
            mismatches += int(np.any(prediction != expected, axis=1).sum())
        certificates[token] = {"origins": native_origins, "inputs_checked": 1_000_000,
                               "mismatches": mismatches, "binary_sha256": hashlib.sha256(blob).hexdigest()}
    initial = closure([entry["source"] for entry in entries.values() if entry["origin"] == "a33_s1"])
    final = closure([entry["source"] for entry in entries.values()])
    assert not errors and all(cert["mismatches"] == 0 for cert in certificates.values())
    return {
        "label": "POST_RESULT_INDEPENDENT_IMPLEMENTATION_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Finite-domain executable correctness and saved deployment evidence; no new training or preregistered claim.",
        "audit_script_sha256": digest(__file__),
        "audited_artifact_sha256": {path.relative_to(REPO).as_posix(): digest(path) for path in paths.values()},
        "registration_sha256": digest(RESULTS / "REGISTRATION.json"),
        "registered_source_files_checked": len(registration["source_sha256"]),
        "registered_checkpoints_checked": len(registration["checkpoint_sha256"]),
        "main_result_hashes_checked": artifact_count,
        "registered_source_hash_mismatches": source_failures,
        "registered_checkpoint_hash_mismatches": checkpoint_failures,
        "main_result_hash_mismatches": artifact_failures,
        "runtime_per_length": dict(stats), "runtime_counterexamples": errors,
        "runtime_operator_calls": steps, "used_origins": sorted(origins),
        "admitted_origin_entries": len(entries), "distinct_token_binaries": len(unique),
        "independent_full_domain_certification": certificates,
        "independent_initial_a33_closure": initial, "independent_final_closure": final,
        "runtime_request_has_no_truth_or_token_metadata": True,
        "resource_scope": {
            "deduplicated_operator_payload_bytes": 66 * len(unique),
            "all_origin_operator_payload_bytes": 66 * len(entries),
            "reported_runtime_peak_rss_bytes": output["memory_after"]["peak_rss_bytes"],
            "reported_peak_operator_payload_bytes": output["counters"]["peak_loaded_operator_payload_bytes"],
            "caveats": [
                "Operator payload is not total process memory or filesystem allocation.",
                "Runtime and neural baseline used different workloads, so their times do not establish a speedup.",
                "Application bytes read are not measurements of physical disk or PCIe traffic.",
                "Exact composition follows from finite-domain correct operators and closure; depth256 validates execution, not new learned reasoning.",
                "P4 is available only from A33; the A14 union alone covers seven tokens.",
            ],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "passed": True,
    }


if __name__ == "__main__":
    result = audit()
    destination = Path(__file__).with_name("independent_audit.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "passed": result["passed"],
                      "runtime_examples": sum(row["examples"] for row in result["runtime_per_length"].values()),
                      "certification_examples": sum(row["inputs_checked"] for row in result["independent_full_domain_certification"].values()),
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
