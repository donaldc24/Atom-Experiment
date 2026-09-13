"""Harvest frozen singleton behaviors into generic portable candidate programs.

No task generator is used for neural teacher queries or compilation. The
explicit --teacher oracle positive control is the only oracle-output path.
Result-bearing runs require the parent's frozen protocol record to exist.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

try:
    from .compiler import fit_program, program_to_bytes
except ImportError:
    from compiler import fit_program, program_to_bytes

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "AtomV2/Harness"
RESULTS = REPO / "AtomV3/results/harvest"
N_CAL = 2048
N_VAL = 2048
QUERY_SEED = 20260912
TOKENS = tuple(f"P{i + 1}" for i in range(8))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_bytes() -> int | None:
    """Process peak working set on Windows; process max RSS on Unix."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        return None


def query_inputs(query_seed: int, token_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample 4096 unique full-domain inputs, split into disjoint fixed halves."""
    rng = np.random.default_rng(np.random.SeedSequence([730031, query_seed, token_index]))
    numbers = rng.choice(10 ** 6, size=N_CAL + N_VAL, replace=False)
    powers = 10 ** np.arange(5, -1, -1, dtype=np.int64)
    x = ((numbers[:, None] // powers[None, :]) % 10).astype(np.int64)
    return x[:N_CAL], x[N_CAL:]


def _load_teacher(teacher: str, seed: int):
    """Return predictor(inputs, opaque token index) and its provenance."""
    sys.path.insert(0, str(HARNESS))
    if teacher in {"oracle", "shuffled", "wrong"}:
        # Explicit controls only: this import and generator are never called
        # while collecting outputs for A14 or A33 teachers.
        from atomv2 import ops

        def oracle_predict(x, token_index):
            output = ops.SURFACE_FNS[TOKENS[token_index]](x)
            if teacher == "wrong":
                output = output[:, [1, 0, 2, 3, 4, 5]]
            return output

        return oracle_predict, {
            "kind": {"oracle": "explicit_generator_positive_control",
                     "shuffled": "explicit_generator_with_shuffled_calibration_labels_negative_control",
                     "wrong": "explicit_generator_with_coherent_wrong_output_swap_control"}[teacher],
            "ground_truth_used_for_control_only": True,
            "oracle_source_sha256": sha256(HARNESS / "atomv2/ops.py"),
        }

    import torch
    torch.set_num_threads(4)
    if teacher == "a14":
        if seed not in (0, 1, 2):
            raise ValueError("A14 teacher seed must be 0, 1, or 2")
        matches = list((REPO / "AtomV2/runs/e4").glob(f"A14_s{seed}_*/checkpoints/final.pt"))
    else:
        if seed != 1:
            raise ValueError("The existing A33 teacher has seed 1 only")
        matches = list((REPO / "AtomV2/runs/e11").glob("A33_s1_*/checkpoints/final.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen {teacher} seed {seed} checkpoint, got {len(matches)}")
    checkpoint = matches[0]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if teacher == "a14":
        from atomv2.config import Config
        from atomv2.model import AtomModel
        model = AtomModel(Config.from_dict(saved["config"]))
    else:
        from atomv2.exchange_model import ConservativeExchangeModel
        model = ConservativeExchangeModel("A33")
    model.load_state_dict(saved["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    @torch.inference_mode()
    def predict(x, token_index):
        outputs = []
        for lo in range(0, len(x), 256):
            digits = torch.from_numpy(x[lo:lo + 256])
            n = len(digits)
            if teacher == "a14":
                tokens = torch.full((n, 2), 8, dtype=torch.int64)
                tokens[:, 0] = token_index
                out = model(digits, tokens, torch.ones(n, dtype=torch.int64), mode="hard")
            else:
                tokens = torch.full((n, 1), token_index, dtype=torch.int64)
                out = model(digits, tokens)
            outputs.append(out["logits"].argmax(-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    return predict, {
        "kind": "frozen_neural_singleton_teacher",
        "checkpoint": checkpoint.relative_to(REPO).as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": int(saved["step"]),
        "ground_truth_used_for_control_only": False,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
    }


def harvest(teacher: str, seed: int, query_seed: int, protocol_path: Path) -> dict:
    """Run the frozen extraction protocol, saving immutable query evidence."""
    if not protocol_path.is_file():
        raise RuntimeError(f"frozen protocol record required before queries: {protocol_path}")
    origin = f"{teacher}_s{seed}"
    if query_seed != QUERY_SEED:
        origin += f"_q{query_seed}"
    outdir = RESULTS / origin
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite existing harvest: {outdir}")
    started = time.perf_counter()
    predict, provenance = _load_teacher(teacher, seed)
    arrays = {key: [] for key in ("x_cal", "y_cal", "x_val", "y_val")}
    programs = {}
    for token_index, token in enumerate(TOKENS):
        x_cal, x_val = query_inputs(query_seed, token_index)
        y_cal, y_val = predict(x_cal, token_index), predict(x_val, token_index)
        if teacher == "shuffled":
            rng = np.random.default_rng(np.random.SeedSequence([911047, query_seed, token_index]))
            y_cal = y_cal[rng.permutation(len(y_cal))]
        fitted = fit_program(x_cal, y_cal, x_val, y_val)
        blob = program_to_bytes(fitted)
        programs[token] = {
            **fitted, "skill_id": f"{origin}_{token}", "token": token,
            "origin": origin, "teacher": teacher,
            "candidate_file": f"candidates/{token}.bin",
            "sha256": hashlib.sha256(blob).hexdigest(),
            "byte_count": len(blob),
        }
        for key, value in (("x_cal", x_cal), ("y_cal", y_cal), ("x_val", x_val), ("y_val", y_val)):
            arrays[key].append(value.astype(np.uint8))
        print(json.dumps({"origin": origin, "token": token,
                          "accepted_candidate": fitted["accepted_candidate"],
                          "validation_agreement": fitted["validation_agreement"]}), flush=True)

    # All candidates are frozen before any separate external verifier can run.
    outdir.mkdir(parents=True)
    (outdir / "candidates").mkdir()
    np.savez_compressed(outdir / "queries.npz", **{key: np.stack(value) for key, value in arrays.items()})
    for token, program in programs.items():
        (outdir / program["candidate_file"]).write_bytes(program_to_bytes(program))
    report = {
        "schema_version": 1, "origin": origin, "teacher": teacher,
        "seed": seed, "query_seed": query_seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": "FROZEN_PROTOCOL_BEHAVIORAL_HARVEST",
        "counts": {"tokens": len(TOKENS), "calibration_per_token": N_CAL,
                   "validation_per_token": N_VAL, "teacher_queries": len(TOKENS) * (N_CAL + N_VAL)},
        "query_data": {"file": "queries.npz", "sha256": sha256(outdir / "queries.npz"),
                       "axis_order": "token, example, digit_position", "token_order": list(TOKENS),
                       "calibration_validation_disjoint_per_token": True,
                       "sampling": "uniform without replacement from 10^6 input lists; fixed 2048/2048 split"},
        "provenance": {**provenance,
                       "protocol_path": str(protocol_path.resolve().relative_to(REPO)),
                       "protocol_sha256": sha256(protocol_path),
                       "harvest_source_sha256": sha256(Path(__file__)),
                       "compiler_source_sha256": sha256(Path(__file__).with_name("compiler.py")),
                       "python": platform.python_version(), "numpy": np.__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "programs": programs,
    }
    report_path = outdir / "programs.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    paths = [outdir / "queries.npz", report_path] + sorted((outdir / "candidates").glob("*.bin"))
    (outdir / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(outdir).as_posix()}\n" for path in paths), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("a14", "a33", "oracle", "shuffled", "wrong"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--query-seed", type=int, default=QUERY_SEED)
    parser.add_argument("--protocol", type=Path, default=REPO / "AtomV3/PROTOCOL.md")
    args = parser.parse_args()
    report = harvest(args.teacher, args.seed, args.query_seed, args.protocol)
    print(json.dumps({"origin": report["origin"],
                      "accepted_candidates": sum(p["accepted_candidate"] for p in report["programs"].values()),
                      "elapsed_seconds": report["elapsed_seconds"], "peak_rss_bytes": report["peak_rss_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
