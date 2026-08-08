"""Determinism, environment capture, checksums and RSS monitoring."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def set_threads(num_threads: int, num_interop_threads: int) -> None:
    """Thread counts change reduction order and therefore determinism. Fix and log them."""
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_interop_threads)
    except RuntimeError:
        # Already initialised in this process (e.g. second run in the same interpreter).
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)


def git_diff_sha256() -> str:
    """SHA256 of the full working-tree diff, so a dirty run is still identifiable."""
    try:
        out = subprocess.run(["git", "diff", "HEAD"], capture_output=True, check=True)
        return hashlib.sha256(out.stdout).hexdigest()
    except Exception:
        return "unknown"


def require_clean_tree(allow_dirty: bool) -> str | None:
    """Refuse to train from a dirty tree unless explicitly allowed.

    Every run before the post-review fixes recorded git_dirty=true under a single
    SHA, so the recorded SHA did not identify the source that produced the numbers.
    Returns the diff hash when running dirty (to be written into env.json), or None
    when the tree is clean. See D33.
    """
    info = git_info()
    if not info.get("git_dirty"):
        return None
    if not allow_dirty:
        raise SystemExit(
            "REFUSING TO RUN: the working tree is dirty, so this run's git SHA would "
            "not identify the source that produced it. "
            "Commit your changes, or pass --allow-dirty to record a diff hash in "
            "env.json instead."
        )
    return git_diff_sha256()


def git_info() -> dict:
    def _run(args):
        try:
            return subprocess.run(
                args, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "unknown"

    sha = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "git_sha": sha,
        "git_sha_short": sha[:7] if sha != "unknown" else "unknown",
        "git_dirty": bool(dirty) if dirty != "unknown" else None,
    }


def env_info(cfg) -> dict:
    info = git_info()
    info.update(
        {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "deterministic_algorithms": True,
        }
    )
    return info


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_sha256sums(run_dir: Path) -> None:
    """Checksum every file under run_dir except SHA256SUMS itself."""
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            rel = p.relative_to(run_dir).as_posix()
            lines.append(f"{sha256_file(p)}  {rel}")
    with open(run_dir / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


class RSSMonitor:
    """Peak resident-set tracking. Fails loudly above the ceiling (T8)."""

    def __init__(self, fail_gb: float = 4.0):
        import psutil

        self.proc = psutil.Process()
        self.fail_bytes = int(fail_gb * (1 << 30))
        self.peak = 0

    def sample(self) -> int:
        rss = self.proc.memory_info().rss
        if rss > self.peak:
            self.peak = rss
        if rss > self.fail_bytes:
            raise MemoryError(
                f"peak RSS {rss / (1 << 30):.2f} GB exceeded the "
                f"{self.fail_bytes / (1 << 30):.1f} GB ceiling -- this indicates a "
                "batching or data-loading bug, not a legitimate need."
            )
        return rss

    @property
    def peak_gb(self) -> float:
        return self.peak / (1 << 30)


def param_counts(model) -> dict:
    """Per-H5: composer, atoms and encoder/decoder are always separate line items."""
    def _count(module):
        return sum(p.numel() for p in module.parameters())

    atoms_total = _count(model.atoms)
    keys = model.atoms.keys.numel()
    return {
        "encoder": _count(model.encoder),
        "decoder": _count(model.decoder),
        "composer": _count(model.composer),
        "atoms_total": atoms_total,
        "atoms_each": (atoms_total - keys) // model.atoms.n_atoms,
        "keys": keys,
        "keys_each": keys // model.atoms.n_atoms,
        "total": _count(model),
        "n_atoms": model.atoms.n_atoms,
    }
