"""NumPy-only executor for portable 66-byte operators.

Binary schema: six uint8 source indices, then six rows of ten uint8 lookup
values. No neural framework, checkpoint loader, or AtomV2 import is permitted
in this process. Operator payload accounting is separate from interpreter RSS.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


PAYLOAD_BYTES = 66


def program_from_bytes(blob: bytes) -> dict:
    if len(blob) != PAYLOAD_BYTES:
        raise ValueError(f"operator payload must be {PAYLOAD_BYTES} bytes")
    values = np.frombuffer(blob, dtype=np.uint8)
    source, table = values[:6], values[6:].reshape(6, 10)
    if np.any(source >= 6) or np.any(table >= 10):
        raise ValueError("operator source or digit outside its finite domain")
    assert source.base is table.base
    assert source.dtype == np.uint8 and table.dtype == np.uint8
    return {"source": source, "table": table}


def apply_program(program: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 6 or np.any(x < 0) or np.any(x >= 10):
        raise ValueError("inputs must be a batch of six digits in 0..9")
    source = np.asarray(program["source"], dtype=np.uint8)
    table = np.asarray(program["table"], dtype=np.uint8)
    return table[np.arange(6)[None, :], x[:, source]]


class OperatorCache:
    """Bounded resident payload cache; evict before reading a new payload."""

    def __init__(self, capacity: int = 1):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self.counters = {
            "capacity": self.capacity, "loads": 0, "cache_hits": 0,
            "evictions": 0, "calls": 0, "bytes_read": 0,
            "peak_loaded_operator_payload_bytes": 0,
            "current_loaded_operator_payload_bytes": 0,
        }

    def _load(self, path: str | Path) -> dict:
        key = str(Path(path).resolve())
        if key in self._cache:
            self.counters["cache_hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
            self.counters["evictions"] += 1
        self.counters["current_loaded_operator_payload_bytes"] = (
            len(self._cache) * PAYLOAD_BYTES)
        # The prior cache payload has been released before this read. NumPy
        # views retain this bytes object; no duplicate payload array is stored.
        blob = Path(key).read_bytes()
        program = program_from_bytes(blob)
        self._cache[key] = program
        self.counters["loads"] += 1
        self.counters["bytes_read"] += len(blob)
        resident = sum(entry["source"].nbytes + entry["table"].nbytes
                       for entry in self._cache.values())
        self.counters["current_loaded_operator_payload_bytes"] = resident
        self.counters["peak_loaded_operator_payload_bytes"] = max(
            resident, self.counters["peak_loaded_operator_payload_bytes"])
        return program

    def apply(self, path: str | Path, x: np.ndarray) -> np.ndarray:
        program = self._load(path)
        result = apply_program(program, x)
        self.counters["calls"] += 1
        return result

    def clear(self) -> None:
        self._cache.clear()
        self.counters["current_loaded_operator_payload_bytes"] = 0


def apply_paths(x: np.ndarray, skill_paths: list[str | Path],
                cache_capacity: int = 1) -> dict:
    cache = OperatorCache(cache_capacity)
    state = np.asarray(x, dtype=np.uint8)
    for path in skill_paths:
        state = cache.apply(path, state)
    return {"output": state, "counters": dict(cache.counters)}


def process_memory() -> dict:
    """Actual process working-set measurements, never equated to payload."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        count = Counters()
        count.cb = ctypes.sizeof(count)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(Counters), ctypes.c_ulong]
        ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                        ctypes.byref(count), count.cb)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": int(count.WorkingSetSize),
                "peak_rss_bytes": int(count.PeakWorkingSetSize),
                "method": "Windows GetProcessMemoryInfo working set"}
    import resource
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    rss = None
    status = Path("/proc/self/statm")
    if status.exists():
        rss = int(status.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return {"rss_bytes": rss, "peak_rss_bytes": peak,
            "method": "resource.ru_maxrss and /proc/self/statm where available"}


def execute_plan(plan: dict) -> dict:
    """Plan contains inputs and filepaths only, with one batch per sequence."""
    forbidden = [name for name in sys.modules
                 if name == "torch" or name == "atomv2" or name.startswith("atomv2.")]
    if forbidden:
        raise RuntimeError(f"runtime imported forbidden neural/harness modules: {forbidden}")
    before = process_memory()
    cache = OperatorCache(int(plan.get("cache_capacity", 1)))
    output = []
    started = time.perf_counter()
    for index, sequence in enumerate(plan["sequences"]):
        state = np.asarray(sequence.get("inputs", plan.get("inputs")), dtype=np.uint8)
        for path in sequence["skill_paths"]:
            state = cache.apply(path, state)
        output.append({"sequence_id": sequence.get("sequence_id", index),
                       "length": len(sequence["skill_paths"]),
                       "prediction": state.tolist()})
    elapsed = time.perf_counter() - started
    after = process_memory()
    return {
        "schema_version": 1, "pid": os.getpid(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform(), "torch_loaded": False,
                    "harness_loaded": False, "neural_weight_bytes_loaded": 0,
                    "decoded_operator_dtype": "uint8",
                    "payload_views_share_one_bytes_buffer": True},
        "elapsed_seconds": elapsed, "counters": dict(cache.counters),
        "memory_before": before, "memory_after": after,
        "memory_note": (
            "Peak operator payload is explicit cached 66-byte binary data. Process "
            "RSS additionally includes Python, NumPy, plan metadata, input/output "
            "arrays, and allocator overhead; OS file cache is outside this counter."
        ),
        "sequences": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute_plan(json.loads(args.plan.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
