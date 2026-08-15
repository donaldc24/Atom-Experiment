"""Dataset generation. Deterministic from the master seed; nothing persisted.

Conventions:
  - Inputs drawn uniformly from the full 10^6 input space, unique per task,
    with train and seen_heldout examples of the same task drawn disjointly
    (V1 lineage: byte-key rejection sampling).
  - Each task's rng is an independent named stream of the master seed keyed by
    a STABLE task index (position in the fixed enumeration of all 72 task ids),
    so a task's examples do not depend on split ordering (V1's known wart).
  - Task tokens are OPAQUE ids 0..7. Sub-op structure never appears in inputs,
    tokens, or any training signal. PAD=8 fills the second token slot of
    singleton tasks; it is used only as router control on dead second-token
    steps, while the model runs 3 live micro-steps per REAL token only.
  - Data is regenerated per run from the master seed (V1 convention); a
    data_manifest with per-task sha256 hashes is written per run so drift is
    detectable without persisting arrays.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import ops
from . import registered as R
from . import split as split_mod
from .utils import stream_rng

PAD_TOKEN = R.N_SURFACE          # 8; router-control padding only
SURFACE_INDEX = {p: i for i, p in enumerate(ops.SURFACE_NAMES)}  # P1->0 .. P8->7

# Stable enumeration of every task id that can ever exist in this world:
# 8 singletons then the 64 pairs in itertools.product order. Split-independent.
ALL_TASK_IDS = tuple(ops.SURFACE_NAMES) + split_mod.PAIR_IDS
_TASK_STREAM_INDEX = {tid: i for i, tid in enumerate(ALL_TASK_IDS)}


@dataclass
class Task:
    task_id: str                 # 'P3' or 'P2_P4'
    surface_ops: tuple           # ('P2','P4')
    tokens: np.ndarray           # [2] int64, PAD-filled for singletons
    n_tokens: int                # 1 or 2
    kind: str                    # 'singleton' | 'pair'
    level: str                   # 'train' | 'L1' | 'L2' | 'L3'


@dataclass
class TaskData:
    task: Task
    x: np.ndarray                # [n, 6] int64
    y: np.ndarray                # [n, 6] int64


def make_task(task_id: str, level: str) -> Task:
    surface = ops.task_surface_ops(task_id)
    tokens = np.full(2, PAD_TOKEN, dtype=np.int64)
    for i, p in enumerate(surface):
        tokens[i] = SURFACE_INDEX[p]
    return Task(task_id=task_id, surface_ops=surface, tokens=tokens,
                n_tokens=len(surface),
                kind="singleton" if len(surface) == 1 else "pair", level=level)


def _unique_inputs(rng: np.random.Generator, n: int) -> np.ndarray:
    seen: set[bytes] = set()
    rows = []
    while len(rows) < n:
        batch = rng.integers(0, R.VOCAB, size=(max(n, 64), R.SEQ_LEN), dtype=np.int64)
        for row in batch:
            key = row.tobytes()
            if key not in seen:
                seen.add(key)
                rows.append(row.copy())
                if len(rows) == n:
                    break
    return np.stack(rows)


@dataclass
class Bundle:
    train: list                  # TaskData, 42 tasks x examples_per_train_task
    seen_heldout: list           # TaskData, same 42 tasks, fresh examples
    unseen: dict                 # level -> list[TaskData] for L1/L2/L3
    probe_inputs: np.ndarray     # [n_probe, 6] fresh inputs for the panel
    split: dict                  # the frozen, verified split dict

    @property
    def all_eval_tasks(self) -> list:
        return (self.seen_heldout + self.unseen["L1"]
                + self.unseen["L2"] + self.unseen["L3"])


def build_bundle(cfg) -> Bundle:
    split = split_mod.load_verified()
    n_tr, n_ev = cfg.examples_per_train_task, cfg.examples_per_eval_task

    train, seen_heldout = [], []
    training_ids = list(split["singletons_train"]) + list(split["train_pairs"])
    for tid in training_ids:
        task = make_task(tid, "train")
        rng = stream_rng(cfg.seed, "data", _TASK_STREAM_INDEX[tid])
        xs = _unique_inputs(rng, n_tr + n_ev)
        ys = ops.apply_task(tid, xs)
        train.append(TaskData(task, xs[:n_tr], ys[:n_tr]))
        seen_heldout.append(TaskData(task, xs[n_tr:], ys[n_tr:]))

    unseen = {}
    for level in ("L1", "L2", "L3"):
        unseen[level] = []
        for tid in split["heldout"][level]:
            task = make_task(tid, level)
            rng = stream_rng(cfg.seed, "data", _TASK_STREAM_INDEX[tid])
            xs = _unique_inputs(rng, n_ev)
            unseen[level].append(TaskData(task, xs, ops.apply_task(tid, xs)))

    probe_inputs = _unique_inputs(stream_rng(cfg.seed, "probe"), cfg.n_probe_examples)
    return Bundle(train=train, seen_heldout=seen_heldout, unseen=unseen,
                  probe_inputs=probe_inputs, split=split)


def data_manifest(bundle: Bundle) -> dict:
    """Per-task content hashes: drift in generation is detectable from disk."""
    def h(arr: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

    manifest = {"train": {}, "seen_heldout": {}, "unseen": {},
                "probe_inputs": h(bundle.probe_inputs)}
    for td in bundle.train:
        manifest["train"][td.task.task_id] = {"x": h(td.x), "y": h(td.y), "n": len(td.x)}
    for td in bundle.seen_heldout:
        manifest["seen_heldout"][td.task.task_id] = {"x": h(td.x), "y": h(td.y), "n": len(td.x)}
    for level, tds in bundle.unseen.items():
        for td in tds:
            manifest["unseen"][td.task.task_id] = {
                "level": level, "x": h(td.x), "y": h(td.y), "n": len(td.x)}
    return manifest


def build_epoch_arrays(bundle: Bundle, cfg, include_partials: bool = False) -> dict:
    """Flat training arrays with the P3 presentation-frequency oversampling.

    Uniform over the 42 training tasks within each epoch, except the dax
    singleton's examples appear p3_oversample_factor times (frequency control
    per the Lake & Baroni rationale; the underlying unique example count stays
    at examples_per_train_task like every other task).

    include_partials (E0 oracle arm ONLY): also emit y_partial [n, 2, 6], the
    ground-truth digit lists after each surface token (singleton slot 2 is a
    copy of slot 1 and is masked by n_tokens in the oracle loss).
    """
    xs, ys, toks, ntoks, parts = [], [], [], [], []
    for td in bundle.train:
        reps = cfg.p3_oversample_factor if td.task.task_id == R.DAX else 1
        if include_partials:
            steps = []
            cur = td.x
            for p in td.task.surface_ops:
                cur = ops.SURFACE_FNS[p](cur)
                steps.append(cur)
            while len(steps) < 2:
                steps.append(steps[-1])
            partial = np.stack(steps, axis=1)            # [n, 2, 6]
        for _ in range(reps):
            xs.append(td.x)
            ys.append(td.y)
            toks.append(np.tile(td.task.tokens, (len(td.x), 1)))
            ntoks.append(np.full(len(td.x), td.task.n_tokens, dtype=np.int64))
            if include_partials:
                parts.append(partial)
    arrays = {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "tokens": np.concatenate(toks),
        "n_tokens": np.concatenate(ntoks),
    }
    if include_partials:
        arrays["y_partial"] = np.concatenate(parts)
    return arrays
