"""Task family, split construction and dataset generation for E1.

Split construction is stronger than a naive 40/24 draw. Because 18 of the 28
unordered primitive pairs commute and `sort_asc` absorbs every position-permuting
primitive, only 39 of the 64 ordered pairs are *extensionally distinct*. A naive
split can therefore place `(reverse, increment)` in train and `(increment, reverse)`
in held-out -- the same function under a different name -- so a model that memorised
the training function would score as if it had recombined. The split is built over
**extensional equivalence classes** instead: a class is entirely in train or entirely
in held-out, and every class equal to a length-1 training task is forced into train.
See DECISIONS.md D2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .primitives import (
    IDENTITY_ID,
    K,
    L,
    V,
    apply_composition,
    apply_primitive,
    random_inputs,
)
from .utils import read_json, sha256_bytes, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "splits" / "pairs_split.json"

SIGNATURE_SAMPLES = 20_000
SIGNATURE_SEED = 987


@dataclass(frozen=True)
class Task:
    task_id: str
    primitives: tuple           # ground-truth primitive ids, length 1 or 2
    instruction: tuple          # length-2 instruction tokens fed to the composer
    kind: str                   # "singleton" | "pair"

    @property
    def is_pair(self) -> bool:
        return self.kind == "pair"


def make_task(primitives) -> Task:
    primitives = tuple(int(p) for p in primitives)
    if len(primitives) == 1:
        # Length-1 tasks are (p, identity): identity is a primitive, so no learned
        # halting mechanism is needed and T is always 2.
        return Task(
            task_id=f"p{primitives[0]}",
            primitives=primitives,
            instruction=(primitives[0], IDENTITY_ID),
            kind="singleton",
        )
    return Task(
        task_id=f"p{primitives[0]}_p{primitives[1]}",
        primitives=primitives,
        instruction=primitives,
        kind="pair",
    )


# --------------------------------------------------------------------------
# Extensional equivalence classes
# --------------------------------------------------------------------------

def _signatures():
    rng = np.random.default_rng(SIGNATURE_SEED)
    probe = random_inputs(rng, SIGNATURE_SAMPLES)
    singleton_sig = {p: apply_primitive(p, probe).tobytes() for p in range(K)}
    pair_sig = {(a, b): apply_composition((a, b), probe).tobytes()
                for a in range(K) for b in range(K)}
    return singleton_sig, pair_sig


def equivalence_classes():
    """Return (classes, forced_train_ids, sig_of_class).

    classes: list of lists of (a, b) pairs sharing one extension.
    forced_train_ids: indices of classes extensionally equal to some length-1 task.
    """
    singleton_sig, pair_sig = _signatures()
    by_sig: dict[bytes, list] = {}
    for pair, sig in pair_sig.items():
        by_sig.setdefault(sig, []).append(pair)

    sigs = sorted(by_sig.keys())            # deterministic ordering
    classes = [sorted(by_sig[s]) for s in sigs]
    singleton_sigs = set(singleton_sig.values())
    forced = {i for i, s in enumerate(sigs) if s in singleton_sigs}
    return classes, forced, sigs


# --------------------------------------------------------------------------
# Split construction
# --------------------------------------------------------------------------

def _position_counts(pairs):
    pos1 = np.zeros(K, dtype=int)
    pos2 = np.zeros(K, dtype=int)
    for a, b in pairs:
        pos1[a] += 1
        pos2[b] += 1
    return pos1, pos2


def informative_pairs(pairs, classes, forced):
    """Pairs that actually teach composition.

    A pair is *uninformative* when its extension equals some length-1 task, because
    then one operand's contribution is invisible: `(p, identity)` teaches nothing
    about composing, and neither does `(rotate_left, sort_asc)` -- sort_asc absorbs
    its predecessor, so the result is the same whatever came first. Both cases are
    exactly the classes forced into train, so one test covers them. See DECISIONS.md D14.
    """
    forced_pairs = {tuple(p) for cid in forced for p in classes[cid]}
    return [tuple(p) for p in pairs if tuple(p) not in forced_pairs]


def build_split(n_heldout: int = 24, seed: int = 1234, max_tries: int = 200_000) -> dict:
    """Rejection-sample a class-level split satisfying every constraint."""
    classes, forced, sigs = equivalence_classes()
    free_ids = [i for i in range(len(classes)) if i not in forced]
    rng = np.random.default_rng(seed)

    for attempt in range(max_tries):
        order = rng.permutation(free_ids)
        chosen, total = [], 0
        for cid in order:
            size = len(classes[cid])
            if total + size <= n_heldout:
                chosen.append(int(cid))
                total += size
            if total == n_heldout:
                break
        if total != n_heldout:
            continue

        heldout_ids = set(chosen)
        heldout_pairs = sorted(p for cid in heldout_ids for p in classes[cid])
        train_pairs = sorted(p for cid in range(len(classes))
                             if cid not in heldout_ids for p in classes[cid])

        pos1, pos2 = _position_counts(train_pairs)
        if pos1.min() < 2 or pos2.min() < 2:
            continue

        # D14: the spec's constraint counted every occurrence, including ones with an
        # identity partner. Those satisfy the letter of it while teaching nothing about
        # composition. Require >=2 *informative* occurrences per position instead.
        # p0 (identity) is exempt: every pair containing it is uninformative by
        # definition, so it can never meet the requirement and does not need to.
        info = informative_pairs(train_pairs, classes, forced)
        ipos1, ipos2 = _position_counts(info)
        non_identity = [p for p in range(K) if p != IDENTITY_ID]
        if min(ipos1[p] for p in non_identity) < 2:
            continue
        if min(ipos2[p] for p in non_identity) < 2:
            continue

        heldout_info = informative_pairs(heldout_pairs, classes, forced)
        return {
            "n_primitives": K,
            "seq_len": L,
            "vocab": V,
            "split_seed": seed,
            "attempts": attempt + 1,
            "n_train_pairs": len(train_pairs),
            "n_heldout_pairs": len(heldout_pairs),
            "train_pairs": [list(p) for p in train_pairs],
            "heldout_pairs": [list(p) for p in heldout_pairs],
            "position1_counts_train": pos1.tolist(),
            "position2_counts_train": pos2.tolist(),
            "informative_position1_counts_train": ipos1.tolist(),
            "informative_position2_counts_train": ipos2.tolist(),
            "n_informative_train_pairs": len(info),
            "n_informative_heldout_pairs": len(heldout_info),
            "n_distinct_pair_functions": len(classes),
            "n_distinct_functions_heldout": len(heldout_ids),
            "n_classes_forced_train": len(forced),
            "heldout_class_ids": sorted(heldout_ids),
            "signature_samples": SIGNATURE_SAMPLES,
            "signature_seed": SIGNATURE_SEED,
            "constraints": [
                "each extensional equivalence class lies entirely in train or entirely in heldout",
                "no heldout pair is extensionally equal to any length-1 (singleton) training task",
                "every primitive appears >=2x in position 1 and >=2x in position 2 among train pairs",
                "every non-identity primitive appears >=2x in position 1 and >=2x in position 2 "
                "among INFORMATIVE train pairs (pairs not extensionally equal to a length-1 task)",
            ],
        }

    raise RuntimeError("could not satisfy split constraints; loosen n_heldout or seed")


def verify_split(split: dict) -> list[str]:
    """T3 (static half) + T5. Returns a list of violation strings; empty means pass."""
    problems = []
    train = {tuple(p) for p in split["train_pairs"]}
    heldout = {tuple(p) for p in split["heldout_pairs"]}

    if train & heldout:
        problems.append(f"train/heldout overlap: {sorted(train & heldout)}")
    if len(train) + len(heldout) != K * K:
        problems.append(f"pairs do not partition K^2: {len(train)}+{len(heldout)}")

    singleton_sig, pair_sig = _signatures()
    train_sigs = {pair_sig[p] for p in train} | set(singleton_sig.values())
    for p in sorted(heldout):
        if pair_sig[p] in train_sigs:
            problems.append(f"heldout pair {p} is extensionally equal to a training task")

    pos1, pos2 = _position_counts(sorted(train))
    for p in range(K):
        if pos1[p] < 2:
            problems.append(f"primitive {p} appears {pos1[p]}x in position 1 of train (<2)")
        if pos2[p] < 2:
            problems.append(f"primitive {p} appears {pos2[p]}x in position 2 of train (<2)")

    classes, forced, _ = equivalence_classes()
    ipos1, ipos2 = _position_counts(informative_pairs(sorted(train), classes, forced))
    for p in range(K):
        if p == IDENTITY_ID:
            continue
        if ipos1[p] < 2:
            problems.append(
                f"primitive {p} has {ipos1[p]} INFORMATIVE occurrences in position 1 (<2)")
        if ipos2[p] < 2:
            problems.append(
                f"primitive {p} has {ipos2[p]} INFORMATIVE occurrences in position 2 (<2)")
    return problems


def load_split(path: Path = SPLIT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m e1.make_split` and commit the result."
        )
    return read_json(path)


def split_hash(path: Path = SPLIT_PATH) -> str:
    return sha256_bytes(path.read_bytes())


# --------------------------------------------------------------------------
# Example generation
# --------------------------------------------------------------------------

def _unique_inputs(rng: np.random.Generator, n: int) -> np.ndarray:
    """n distinct sequences from V^L, so train and eval examples of a task never collide."""
    seen, rows = set(), []
    while len(rows) < n:
        block = random_inputs(rng, max(n - len(rows), 64) * 2)
        for row in block:
            key = row.tobytes()
            if key not in seen:
                seen.add(key)
                rows.append(row)
                if len(rows) == n:
                    break
    return np.stack(rows)


@dataclass
class TaskData:
    task: Task
    inputs: np.ndarray      # [n, L]
    targets: np.ndarray     # [n, L]


@dataclass
class Bundle:
    train: list             # list[TaskData] -- 8 singletons + 40 train pairs
    seen_heldout: list      # list[TaskData] -- fresh examples of the 40 train pairs
    unseen: list            # list[TaskData] -- the 24 held-out pairs
    singleton: list         # list[TaskData] -- fresh examples of the 8 singletons
    probe_inputs: np.ndarray
    split: dict

    @property
    def all_eval_tasks(self) -> list:
        return self.seen_heldout + self.unseen


def build_bundle(cfg, split: dict | None = None) -> Bundle:
    split = split or load_split()
    rng = np.random.default_rng(cfg.data_seed)

    n_tr = cfg.examples_per_train_task
    n_ev = cfg.examples_per_eval_task

    train, seen_heldout, unseen, singleton = [], [], [], []

    # Length-1 tasks: in training so single-atom forward passes are in-distribution,
    # which is what makes the standalone-probing metric (M3) meaningful.
    for p in range(K):
        task = make_task((p,))
        xs = _unique_inputs(rng, n_tr + n_ev)
        ys = apply_composition(task.primitives, xs)
        train.append(TaskData(task, xs[:n_tr], ys[:n_tr]))
        singleton.append(TaskData(task, xs[n_tr:], ys[n_tr:]))

    for pair in split["train_pairs"]:
        task = make_task(pair)
        xs = _unique_inputs(rng, n_tr + n_ev)
        ys = apply_composition(task.primitives, xs)
        train.append(TaskData(task, xs[:n_tr], ys[:n_tr]))
        seen_heldout.append(TaskData(task, xs[n_tr:], ys[n_tr:]))

    for pair in split["heldout_pairs"]:
        task = make_task(pair)
        xs = _unique_inputs(rng, n_ev)
        ys = apply_composition(task.primitives, xs)
        unseen.append(TaskData(task, xs, ys))

    probe_rng = np.random.default_rng(cfg.data_seed + 7919)
    probe_inputs = _unique_inputs(probe_rng, cfg.n_probe_examples)

    return Bundle(train, seen_heldout, unseen, singleton, probe_inputs, split)
