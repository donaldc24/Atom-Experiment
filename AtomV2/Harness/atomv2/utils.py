"""Determinism, provenance and byte-stable I/O (V1 lineage).

Conventions inherited from AtomV1/e1/utils.py because they earned their place:
  - one master seed per run; every derived stream is a fixed function of it
  - torch deterministic algorithms + pinned thread counts (thread counts change
    reduction order and therefore determinism; D39)
  - byte-stable JSON (sorted keys, LF, trailing newline) so SHA256SUMS mean
    something
  - dirty-git-tree refusal (D33/D43): runs must be attributable to a commit;
    output dirs are exempt so the second run of a batch does not refuse
  - RSS ceiling as a loud batching-bug detector
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import registered as R

HARNESS_ROOT = Path(__file__).resolve().parents[1]   # AtomV2/Harness
V2_ROOT = HARNESS_ROOT.parent                        # AtomV2
RUNS_DIR = V2_ROOT / "runs"
RESULTS_DIR = V2_ROOT / "results"

# Paths (relative to the git toplevel) whose untracked/modified files never
# block a run: they are batch OUTPUT, not source.
OUTPUT_PREFIXES = ("AtomV2/runs", "AtomV2/results")

# AMENDMENT R10: the identity used to decide whether two runs may be POOLED is
# a content fingerprint of the files that can actually change a number - the
# harness modules and the frozen split. The older dirty-tree fingerprint also
# covered incidental files (.claude/*, editor config, docs) and, worse, changed
# representation when an unmodified tree went from dirty-untracked to
# committed, which made byte-identical implementations look like different
# ones. Result identity is a property of the code that runs, not of whether it
# happens to be committed yet.
#
# Deliberately EXCLUDED: tests (never imported by a run), README/REGISTERED
# (documentation), and everything outside AtomV2/Harness. Provenance is still
# recorded in full - git_sha, git_dirty, and the dirty-snapshot fingerprint are
# all still written to env.json. This narrower value governs pooling only.
HARNESS_SOURCE_GLOBS = ("atomv2/*.py", "splits/*.json")


def _hash_source_files(files: dict[str, bytes]) -> str:
    """Fingerprint a {relative_posix_path: bytes} map, order-independent."""
    h = hashlib.sha256()
    h.update(b"atomv2-harness-source-v1\0")
    for rel in sorted(files):
        h.update(rel.encode() + b"\0")
        h.update(hashlib.sha256(files[rel]).hexdigest().encode() + b"\0")
    return h.hexdigest()


def harness_source_sha256() -> str:
    """Content fingerprint of the working tree's harness source."""
    files = {}
    for pattern in HARNESS_SOURCE_GLOBS:
        for p in HARNESS_ROOT.glob(pattern):
            if p.is_file():
                files[p.relative_to(HARNESS_ROOT).as_posix()] = p.read_bytes()
    return _hash_source_files(files)


def harness_source_sha256_at(rev: str) -> str:
    """Same fingerprint for the harness source as committed at `rev`.

    Lets a run's source identity be established from a commit rather than from
    whatever the working tree happens to hold now.
    """
    prefix = "AtomV2/Harness/"
    listing = _git("ls-tree", "-r", "--name-only", rev, "--", prefix)
    files = {}
    for path in listing.splitlines():
        rel = path[len(prefix):]
        if not any(Path(rel).match(g) for g in HARNESS_SOURCE_GLOBS):
            continue
        blob = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=HARNESS_ROOT,
                              capture_output=True, check=True).stdout
        files[rel] = blob
    return _hash_source_files(files)


def _status_path(line: str) -> str:
    """Path out of one `git status --porcelain` line.

    Format is 'XY<space>path'; X is a SPACE for worktree-only modifications,
    so the leading space is load-bearing and must not be stripped upstream.
    """
    return line[3:].split(" -> ")[-1].strip().strip('"')


def _is_output_path(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(rel == pref or rel.startswith(pref + "/")
               for pref in OUTPUT_PREFIXES)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
# Named streams derived from the one master seed. Every consumer of randomness
# asks for a stream by name; nothing shares a stream. Fixed small integers keep
# the derivation readable and stable across code motion.
_STREAMS = {
    "data": 1,          # dataset input sampling
    "probe": 2,         # fresh probe inputs for the panel
    "shuffle": 3,       # epoch shuffling (combined with epoch index)
    "init": 4,          # torch parameter init (via torch.manual_seed)
    "gumbel": 5,        # torch generator for Gumbel noise
    "probe_train": 6,   # decodability probe training (init + example split)
    "e1b_diag": 7,      # E1b fixed diagnostic batch selection
    "e1b_liveness": 8,  # E1b liveness Gumbel draws (indexed by step*draws+d)
}


def stream_seed(master_seed: int, name: str, index: int = 0) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy=master_seed,
                                  spawn_key=(_STREAMS[name], index))


def stream_rng(master_seed: int, name: str, index: int = 0) -> np.random.Generator:
    return np.random.default_rng(stream_seed(master_seed, name, index))


def seed_everything(master_seed: int) -> None:
    import torch
    random.seed(master_seed)
    np.random.seed(master_seed % (2**32))
    torch.manual_seed(stream_seed(master_seed, "init").generate_state(1)[0].item())
    os.environ["PYTHONHASHSEED"] = str(master_seed)
    torch.use_deterministic_algorithms(True)


def set_threads() -> None:
    import torch
    torch.set_num_threads(R.NUM_THREADS)
    try:
        torch.set_num_interop_threads(R.NUM_INTEROP_THREADS)
    except RuntimeError:
        pass  # can only be set once per process; second call is a no-op


# ---------------------------------------------------------------------------
# Byte-stable JSON + checksums
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256sums(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            rel = p.relative_to(run_dir).as_posix()
            lines.append(f"{sha256_of_file(p)}  {rel}")
    with open(run_dir / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------

def _git(*args: str, strip: bool = True) -> str:
    """Run git. strip=False preserves leading whitespace.

    Porcelain status lines are 'XY<space>path', and X is a SPACE for
    worktree-only modifications (' M foo'). Stripping the whole output ate the
    first line's leading space, so path parsing (line[3:]) lost a character on
    that one line - which could misread an output path as source and refuse to
    run, and corrupted the first entry of the dirty-snapshot fingerprint.
    """
    out = subprocess.run(["git", *args], cwd=HARNESS_ROOT, capture_output=True,
                         text=True, check=True).stdout
    return out.strip() if strip else out.rstrip("\n")


def _dirty_source_fingerprint(sha: str, status: str, toplevel: str) -> str:
    """Hash the actual dirty source snapshot, including untracked files.

    `git diff HEAD` omits untracked files, which made the original untracked
    Harness appear as the empty-diff hash. HEAD identifies the clean baseline;
    status records modifications/deletions; current bytes identify every
    present dirty source file. Output artifacts are deliberately excluded.
    """
    h = hashlib.sha256()
    h.update(f"HEAD\0{sha}\0".encode())
    root = Path(toplevel)
    for line in sorted(status.splitlines()):
        rel = _status_path(line).replace("\\", "/")
        if _is_output_path(rel):
            continue
        h.update(line[:2].encode() + b"\0" + rel.encode() + b"\0")
        path = root / Path(rel)
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def git_info() -> dict:
    try:
        sha = _git("rev-parse", "HEAD")
        status = _git("status", "--porcelain", "--untracked-files=all",
                      strip=False)
        toplevel = _git("rev-parse", "--show-toplevel")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": "unknown", "git_sha_short": "nogit",
                "git_dirty": True, "git_source_dirty": True}
    source_dirty = False
    for line in status.splitlines():
        if not _is_output_path(_status_path(line)):
            source_dirty = True
            break
    info = {"git_sha": sha, "git_sha_short": sha[:7], "git_toplevel": toplevel,
            "git_dirty": bool(status), "git_source_dirty": source_dirty}
    if source_dirty:
        info["dirty_source_sha256"] = _dirty_source_fingerprint(
            sha, status, toplevel)
    return info


def require_clean_tree(allow_dirty: bool) -> dict:
    """Refuse to run from a dirty SOURCE tree (D33). Output dirs are exempt.

    With allow_dirty, records the sha256 of the diff so the run stays
    attributable.
    """
    info = git_info()
    if info["git_source_dirty"]:
        if not allow_dirty:
            raise SystemExit(
                "refusing to run: git source tree is dirty. Commit first, or "
                "pass --allow-dirty to record the diff hash instead.")
        # Backward-compatible field name plus the precise new name. Unlike the
        # former git-diff-only value, this includes untracked source files.
        fingerprint = info.get("dirty_source_sha256", "unavailable")
        info["dirty_diff_sha256"] = fingerprint
    return info


def env_info() -> dict:
    import torch
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": True,
        "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Governs pooling (amendment R10); full provenance is in git_info().
        "harness_source_sha256": harness_source_sha256(),
    }


# ---------------------------------------------------------------------------
# RSS ceiling (loud batching-bug detector, V1 lineage)
# ---------------------------------------------------------------------------

def check_rss(limit_gb: float = R.RSS_FAIL_GB) -> float:
    try:
        import psutil
        gb = psutil.Process().memory_info().rss / 2**30
    except ImportError:
        return 0.0
    if gb > limit_gb:
        raise MemoryError(
            f"resident set {gb:.2f} GB exceeds the {limit_gb} GB ceiling - "
            "this is a batching bug, not a hardware limit. Do not raise the "
            "ceiling; find the leak.")
    return gb


class JsonlLogger:
    """Append-only JSONL event log, flushed per record."""

    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8", newline="\n")

    def log(self, **record) -> None:
        json.dump(record, self._f, sort_keys=True, default=_json_default)
        self._f.write("\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
