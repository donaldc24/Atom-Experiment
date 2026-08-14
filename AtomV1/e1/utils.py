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
    if not info.get("git_source_dirty"):
        return None
    if not allow_dirty:
        raise SystemExit(
            "REFUSING TO RUN: the working tree has uncommitted SOURCE changes, so "
            "this run's git SHA would not identify the source that produced it.\n  "
            + "\n  ".join(source_dirt(
                subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout)[:10])
            + "\nCommit your changes, or pass --allow-dirty to record a diff hash in "
            "env.json instead. (Untracked files under runs/ and results/ are the "
            "batch's own output and are NOT counted -- see D43.)"
        )
    return git_diff_sha256()


def _git_prefix() -> str:
    """This project's root relative to the git toplevel, as a porcelain prefix.

    `git status --porcelain` paths are relative to the git TOPLEVEL, not to this
    project. Since the project moved into a subfolder of the repo (AtomV1/), the
    output filter below must carry that prefix or it matches nothing and D43's
    guard kills every batch after the first run again.
    """
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        rel = Path(__file__).resolve().parents[1].relative_to(Path(top).resolve())
        return "" if str(rel) == "." else rel.as_posix() + "/"
    except Exception:
        return ""


GIT_PREFIX = _git_prefix()

# Paths that hold EXPERIMENT OUTPUT rather than source. A batch writes into these
# as it goes, so untracked files appearing here do not change what produced the
# numbers. Everything else is source. See D43.
OUTPUT_PREFIXES = (GIT_PREFIX + "runs/", GIT_PREFIX + "results/")


def _porcelain_paths(line: str) -> str:
    """Path from one `git status --porcelain` line, handling renames and quoting."""
    path = line[3:] if len(line) > 3 else ""
    if " -> " in path:                      # "R  old -> new"
        path = path.split(" -> ")[-1]
    return path.strip().strip('"')


def source_dirt(porcelain: str) -> list[str]:
    """Porcelain lines that represent a change to SOURCE.

    Untracked entries under `runs/` and `results/` are excluded: during a batch every
    completed run leaves its own artifacts untracked, and treating those as dirt
    means the second run of any batch refuses to start. Modifications to *tracked*
    files are never excluded, anywhere -- a rewritten committed artifact is exactly
    the kind of thing that should stop a run.
    """
    out = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        untracked = line.startswith("??")
        path = _porcelain_paths(line)
        if untracked and path.startswith(OUTPUT_PREFIXES):
            continue
        out.append(line)
    return out


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
    unknown = dirty == "unknown"
    return {
        "git_sha": sha,
        "git_sha_short": sha[:7] if sha != "unknown" else "unknown",
        # Raw working-tree state, recorded as-is so env.json never overstates
        # cleanliness. True mid-batch simply because earlier runs wrote output.
        "git_dirty": bool(dirty) if not unknown else None,
        # The provenance-relevant one: does the recorded SHA identify the SOURCE that
        # produced this run? This is what D33's guard actually cares about.
        "git_source_dirty": bool(source_dirt(dirty)) if not unknown else None,
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
