"""Split enumerator: derive the 64-cell split from the op algebra and validate.

Nothing about the split's STRUCTURE is transcribed from SplitMath.md - triples,
functional classes, the identity collapse, the merged class, adjacencies, L3
membership, train membership and every coverage number are derived here from
atomv2.ops. The only registered inputs are the choices no formula makes:
which sibling of each shareable adjacency is held out (HELDOUT_L1) and which
unique-adjacency cells form L2 (HELDOUT_L2), from atomv2.registered.

Derivation rules (SplitMath.md "Split Assignment"):
  - EXCLUDED: any pair cell whose triple equals the identity (derived; P5_P5).
  - Merged classes: cells with equal triple_key are one function and travel
    together (derived; P4_P8 / P8_P4 -> train).
  - L3: every pair cell involving the dax primitive (derived from DAX = P3).
  - L1: registered cells; VALIDATED to each have a trained sibling sharing its
    cross-token adjacency.
  - L2: registered cells; VALIDATED to each have an adjacency that appears in
    ZERO training pairs.
  - Train: everything else. Singletons: all 8 train (P3 as singleton only).

Enumerator duties (verified mechanically in validate(), raising on failure):
  - composition rule reproduces direct function composition on random probes
  - every L1 cell's adjacency appears in >= 1 training pair
  - every L2 cell's adjacency appears in 0 training pairs
  - no held-out cell's triple_key matches any training cell or singleton
  - coverage counts per primitive per position (>= 2 required; actual logged)
  - merged classes are assigned as one; excluded cells are exactly the
    identity class; accounting 34 + 8 + 6 + 15 + 1 = 64
The diff against SplitMath.md's transcribed tables lives in tests/test_split.py.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import ops
from . import registered as R
from .utils import sha256_of_file, write_json

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"
SPLIT_PATH = SPLITS_DIR / "split_v2.json"

PAIR_IDS = tuple(
    f"{a}_{b}" for a, b in itertools.product(ops.SURFACE_NAMES, repeat=2)
)
SINGLETON_IDS = ops.SURFACE_NAMES  # all 8 train, including the dax P3


def derive_cells() -> dict[str, dict]:
    """Derive every pair cell's triple, key, adjacency and functional class."""
    cells = {}
    for tid in PAIR_IDS:
        tri = ops.task_triple(tid)
        cells[tid] = {
            "task": tid,
            "triple": {"pi": list(tri[0]), "a": tri[1], "b": list(tri[2])},
            "triple_key": ops.triple_key(tri),
            "adjacency": list(ops.task_adjacency(tid)),
            "subop_chain": list(ops.task_subop_chain(tid)),
        }
    classes = defaultdict(list)
    for tid, c in cells.items():
        classes[c["triple_key"]].append(tid)
    for tid, c in cells.items():
        c["class_members"] = sorted(classes[c["triple_key"]])
    return cells


def assign(cells: dict[str, dict]) -> dict[str, dict]:
    """Apply derivation rules + registered choices to produce the assignment."""
    identity_key = ops.triple_key(ops.IDENTITY_TRIPLE)
    dax = R.DAX
    for tid, c in cells.items():
        first, second = ops.task_surface_ops(tid)
        if c["triple_key"] == identity_key:
            c["split"] = "excluded"
            c["reason"] = "identity collapse: teaches nothing, muddies steps-per-token"
        elif dax in (first, second):
            c["split"] = "L3"
            c["reason"] = f"dax test: {dax} trained only as a singleton"
        elif tid in R.HELDOUT_L1:
            c["split"] = "L1"
            c["reason"] = "registered choice: unseen surface pair, adjacency trained via sibling"
        elif tid in R.HELDOUT_L2:
            c["split"] = "L2"
            c["reason"] = "registered choice: adjacency unique to this cell, untrained"
        else:
            c["split"] = "train"
            c["reason"] = "remainder"
    return cells


def validate(cells: dict[str, dict]) -> dict:
    """Run every enumerator duty; raise AssertionError on any violation.

    Returns the audit dict that gets frozen into the split file.
    """
    rng = np.random.default_rng(20260814)
    x = rng.integers(0, ops.MOD, size=(512, ops.L), dtype=np.int64)

    # Duty: composition rule == direct function composition, all 64 pairs.
    for tid in PAIR_IDS:
        a, b = ops.task_surface_ops(tid)
        direct = ops.SURFACE_FNS[b](ops.SURFACE_FNS[a](x))
        via_triple = ops.apply_triple(ops.task_triple(tid), x)
        assert np.array_equal(direct, via_triple), f"composition rule broken at {tid}"

    by_split = defaultdict(list)
    for tid, c in cells.items():
        by_split[c["split"]].append(tid)
    for v in by_split.values():
        v.sort()

    # Accounting: 34 train + 8 L1 + 6 L2 + 15 L3 + 1 excluded = 64.
    counts = {k: len(v) for k, v in by_split.items()}
    assert counts == {"train": 34, "L1": 8, "L2": 6, "L3": 15, "excluded": 1}, counts
    assert sum(counts.values()) == 64

    # Excluded cells are EXACTLY the identity class - no more, no fewer.
    identity_key = ops.triple_key(ops.IDENTITY_TRIPLE)
    derived_identity = sorted(
        t for t in PAIR_IDS if cells[t]["triple_key"] == identity_key
    )
    assert by_split["excluded"] == derived_identity, (
        by_split["excluded"], derived_identity)

    # Merged classes travel together: equal triple_key => equal split.
    for tid, c in cells.items():
        for m in c["class_members"]:
            assert cells[m]["split"] == c["split"], (
                f"merged class split apart: {tid}={c['split']} {m}={cells[m]['split']}")

    # L3 is EXACTLY every pair involving the dax.
    derived_l3 = sorted(
        t for t in PAIR_IDS if R.DAX in ops.task_surface_ops(t)
    )
    assert by_split["L3"] == derived_l3, (by_split["L3"], derived_l3)

    # Adjacency duties.
    train_adjacencies = {tuple(cells[t]["adjacency"]) for t in by_split["train"]}
    l1_siblings = {}
    for t in by_split["L1"]:
        adj = tuple(cells[t]["adjacency"])
        sibs = [s for s in by_split["train"] if tuple(cells[s]["adjacency"]) == adj]
        assert sibs, f"L1 cell {t} has no trained sibling for adjacency {adj}"
        l1_siblings[t] = {"adjacency": list(adj), "trained_siblings": sibs}
    for t in by_split["L2"]:
        adj = tuple(cells[t]["adjacency"])
        assert adj not in train_adjacencies, (
            f"L2 cell {t} adjacency {adj} appears in training - it is not L2")

    # The doc's structural claim, derived: the ONLY adjacency overlaps come
    # from A being last in both P2 and P7, and T being first in both P2 and P8.
    last_carriers = defaultdict(list)
    first_carriers = defaultdict(list)
    for p in ops.SURFACE_NAMES:
        last_carriers[ops.LAST_SUBOP[p]].append(p)
        first_carriers[ops.FIRST_SUBOP[p]].append(p)
    multi_last = {k: v for k, v in last_carriers.items() if len(v) > 1}
    multi_first = {k: v for k, v in first_carriers.items() if len(v) > 1}
    assert multi_last == {"A": ["P2", "P7"]}, multi_last
    assert multi_first == {"T": ["P2", "P8"]}, multi_first

    # Leakage duty: no held-out cell's triple_key matches any training cell's
    # or any singleton's (surface op trained as a singleton).
    train_keys = {cells[t]["triple_key"] for t in by_split["train"]}
    train_keys |= {ops.triple_key(ops.SURFACE_TRIPLES[p]) for p in SINGLETON_IDS}
    for level in ("L1", "L2", "L3"):
        for t in by_split[level]:
            assert cells[t]["triple_key"] not in train_keys, (
                f"held-out {t} ({level}) is functionally identical to a training task")

    # Coverage duty: every non-dax primitive appears >= 2 times in each
    # position among training pairs (>= 3 is the doc's actual; we log it).
    coverage = {p: [0, 0] for p in ops.SURFACE_NAMES}
    for t in by_split["train"]:
        a, b = ops.task_surface_ops(t)
        coverage[a][0] += 1
        coverage[b][1] += 1
    for p in ops.SURFACE_NAMES:
        if p == R.DAX:
            assert coverage[p] == [0, 0], f"dax {p} appears in a training pair"
        else:
            assert min(coverage[p]) >= 2, f"{p} coverage {coverage[p]} below 2"

    # Dax-choice rationale, validated mechanically: removing P2 from pairs
    # would leave every remaining adjacency unique to one cell (L1 impossible).
    non_p2 = [t for t in PAIR_IDS
              if "P2" not in ops.task_surface_ops(t)
              and R.DAX not in ops.task_surface_ops(t)]
    adj_counts = defaultdict(int)
    for t in non_p2:
        adj_counts[tuple(cells[t]["adjacency"])] += 1
    p2_would_kill_l1 = all(v == 1 for v in adj_counts.values())

    n_distinct = len({c["triple_key"] for c in cells.values()})
    merged = sorted(
        {tuple(c["class_members"]) for c in cells.values() if len(c["class_members"]) > 1}
    )
    return {
        "counts": counts,
        "n_distinct_pair_functions": n_distinct,
        "merged_classes": [list(m) for m in merged],
        "excluded_cells": by_split["excluded"],
        "l1_siblings": l1_siblings,
        "l2_adjacencies": {t: cells[t]["adjacency"] for t in by_split["L2"]},
        "coverage_train_pairs": {p: coverage[p] for p in ops.SURFACE_NAMES},
        "adjacency_overlaps": {"last": multi_last, "first": multi_first},
        "p2_disqualified_as_dax_verified": p2_would_kill_l1,
        "dax": R.DAX,
    }


def build() -> dict:
    """Derive + assign + validate; return the full split dict (not yet frozen)."""
    cells = assign(derive_cells())
    audit = validate(cells)
    by_split = defaultdict(list)
    for tid in PAIR_IDS:  # deterministic order
        by_split[cells[tid]["split"]].append(tid)
    return {
        "format": "atomv2-split-v1",
        "world": {
            "seq_len": ops.L,
            "vocab": ops.MOD,
            "subops": {n: {"pi": list(t[0]), "a": t[1], "b": list(t[2])}
                       for n, t in ops.SUBOP_TRIPLES.items()},
            "surface_recipes": {p: list(r) for p, r in R.SURFACE_RECIPES.items()},
            "surface_triples": {p: {"pi": list(t[0]), "a": t[1], "b": list(t[2])}
                                for p, t in ops.SURFACE_TRIPLES.items()},
            "p8_note": "T and N commute; P8's internal order is unrecoverable. "
                       "The answer key for P8 is the SET {T, N}. Any probe "
                       "touching P8 must be written set-wise.",
        },
        "registered_choices": {
            "dax": R.DAX,
            "heldout_L1": list(R.HELDOUT_L1),
            "heldout_L2": list(R.HELDOUT_L2),
        },
        "singletons_train": list(SINGLETON_IDS),
        "train_pairs": by_split["train"],
        "heldout": {"L1": by_split["L1"], "L2": by_split["L2"], "L3": by_split["L3"]},
        "excluded": by_split["excluded"],
        "cells": {tid: cells[tid] for tid in PAIR_IDS},
        "audit": audit,
    }


def freeze(force: bool = False) -> Path:
    """Write the derived split to disk. Refuses to overwrite unless force."""
    if SPLIT_PATH.exists() and not force:
        existing = load()
        rebuilt = build()
        if json.dumps(existing, sort_keys=True) != json.dumps(rebuilt, sort_keys=True):
            raise RuntimeError(
                f"{SPLIT_PATH} exists and DIFFERS from the current derivation. "
                "The frozen split is pinned by sha256 in every run's split_ref; "
                "regenerating it would orphan committed runs. Use force only "
                "with a recorded decision.")
        return SPLIT_PATH
    write_json(SPLIT_PATH, build())
    return SPLIT_PATH


def load() -> dict:
    with open(SPLIT_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_verified() -> dict:
    """Load the frozen split AND re-derive to confirm it still matches the code.

    Every run goes through this: a run can never train against a split file
    that drifted from the algebra that is supposed to define it.
    """
    split = load()
    rebuilt = build()
    if json.dumps(split, sort_keys=True) != json.dumps(rebuilt, sort_keys=True):
        raise RuntimeError(
            "frozen split_v2.json does not match the current derivation - "
            "either the ops/registered code changed after freezing (record a "
            "decision and re-freeze deliberately) or the file was edited.")
    return split


def split_ref() -> dict:
    return {
        "path": str(SPLIT_PATH),
        "sha256": sha256_of_file(SPLIT_PATH),
        "dax": R.DAX,
        "n_train_pairs": 34,
        "n_heldout": {"L1": 8, "L2": 6, "L3": 15},
    }


ALL_TRAIN_TASKS = None  # populated lazily by data.py from the frozen split


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Derive, validate and freeze the V2 split")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen split (record a decision first)")
    args = ap.parse_args()
    path = freeze(force=args.force)
    s = load_verified()
    print(f"frozen: {path}")
    print(f"sha256: {sha256_of_file(path)}")
    print(json.dumps(s["audit"], indent=2))
