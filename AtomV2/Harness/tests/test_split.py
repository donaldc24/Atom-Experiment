"""Enumerator duties, including the diff against SplitMath.md's tables.

The tables below are TRANSCRIBED from AtomV2/SplitMath.md as registered
expectations. The harness never reads them - split.py derives everything from
the op definitions - so this test is the mechanical "rebuild all 72 triples
from op definitions, diff against this table" duty. If the doc and the algebra
ever disagree, this test fails and the doc is the bug (or the code is - either
way a human looks).
"""
import numpy as np
import pytest

from atomv2 import ops, split
from atomv2 import registered as R

# --- SplitMath.md "The Complete Deck" (sub-ops) ----------------------------
DOC_SUBOPS = {
    "R": ((5, 4, 3, 2, 1, 0), 1, (0, 0, 0, 0, 0, 0)),
    "T": ((1, 2, 3, 4, 5, 0), 1, (0, 0, 0, 0, 0, 0)),
    "W": ((1, 0, 3, 2, 5, 4), 1, (0, 0, 0, 0, 0, 0)),
    "I": ((0, 1, 2, 3, 4, 5), 1, (1, 1, 1, 1, 1, 1)),
    "N": ((0, 1, 2, 3, 4, 5), 9, (0, 0, 0, 0, 0, 0)),
    "M": ((0, 1, 2, 3, 4, 5), 3, (0, 0, 0, 0, 0, 0)),
    "A": ((0, 1, 2, 3, 4, 5), 1, (0, 1, 2, 3, 4, 5)),
}

# --- SplitMath.md "The Math" (surface ops) ---------------------------------
DOC_SURFACE = {
    "P1": ((5, 4, 3, 2, 1, 0), 1, (1, 1, 1, 1, 1, 1)),
    "P2": ((1, 2, 3, 4, 5, 0), 1, (0, 1, 2, 3, 4, 5)),
    "P3": ((5, 4, 3, 2, 1, 0), 1, (5, 4, 3, 2, 1, 0)),
    "P4": ((1, 2, 3, 4, 5, 0), 3, (0, 0, 0, 0, 0, 0)),
    "P5": ((1, 0, 3, 2, 5, 4), 9, (0, 0, 0, 0, 0, 0)),
    "P6": ((0, 1, 2, 3, 4, 5), 3, (3, 3, 3, 3, 3, 3)),
    "P7": ((1, 0, 3, 2, 5, 4), 1, (0, 1, 2, 3, 4, 5)),
    "P8": ((1, 2, 3, 4, 5, 0), 9, (0, 0, 0, 0, 0, 0)),
}

# --- SplitMath.md "The 64 Tasks" (triple_key + split per cell) -------------
DOC_CELLS = {
    "P5_P5": ("012345|1|000000", "excluded"), "P7_P7": ("012345|1|115599", "train"),
    "P3_P1": ("012345|1|123456", "L3"), "P1_P1": ("012345|1|222222", "train"),
    "P3_P3": ("012345|1|555555", "L3"), "P1_P3": ("012345|1|654321", "L3"),
    "P5_P7": ("012345|9|012345", "train"), "P6_P6": ("012345|9|222222", "train"),
    "P7_P5": ("012345|9|907856", "train"),
    "P5_P8": ("032541|1|000000", "train"), "P7_P2": ("032541|1|135795", "train"),
    "P7_P4": ("032541|3|369250", "L1"), "P5_P4": ("032541|7|000000", "train"),
    "P5_P2": ("032541|9|012345", "L1"), "P7_P8": ("032541|9|987650", "train"),
    "P2_P3": ("054321|1|086420", "L3"), "P2_P1": ("054321|1|654321", "L1"),
    "P4_P1": ("054321|3|111111", "train"), "P4_P3": ("054321|3|543210", "L3"),
    "P8_P1": ("054321|9|111111", "train"), "P8_P3": ("054321|9|543210", "L3"),
    "P6_P7": ("103254|3|345678", "train"), "P7_P6": ("103254|3|369258", "L1"),
    "P5_P6": ("103254|7|333333", "L2"), "P6_P5": ("103254|7|777777", "L2"),
    "P6_P2": ("123450|3|345678", "train"), "P2_P6": ("123450|3|369258", "train"),
    "P8_P6": ("123450|7|333333", "train"), "P6_P8": ("123450|7|777777", "L1"),
    "P4_P6": ("123450|9|333333", "L2"), "P6_P4": ("123450|9|999999", "train"),
    "P8_P5": ("214305|1|000000", "L2"), "P2_P7": ("214305|1|115599", "train"),
    "P4_P7": ("214305|3|012345", "train"), "P4_P5": ("214305|7|000000", "train"),
    "P8_P7": ("214305|9|012345", "train"),
    # NOTE: SplitMath.md's G6 group table says P2_P5 -> train, but that row is
    # a typo IN THE DOC: its own Split Assignment section lists P2_P5 as
    # held-out L1 (adjacency (A,N), trained sibling P7_P5), the 34-pair train
    # list omits it, and the 34+8+6+15+1=64 accounting only closes with it
    # held out. Registered value: L1.
    "P2_P5": ("214305|9|907856", "L1"),
    "P8_P8": ("234501|1|000000", "train"), "P2_P2": ("234501|1|135795", "train"),
    "P4_P2": ("234501|3|012345", "L1"), "P2_P4": ("234501|3|369250", "train"),
    "P4_P8": ("234501|7|000000", "train"), "P8_P4": ("234501|7|000000", "train"),
    "P4_P4": ("234501|9|000000", "train"), "P8_P2": ("234501|9|012345", "train"),
    "P2_P8": ("234501|9|987650", "train"),
    "P1_P2": ("432105|1|123456", "L1"), "P3_P2": ("432105|1|444440", "L3"),
    "P3_P4": ("432105|3|296305", "L3"), "P1_P4": ("432105|3|333333", "L2"),
    "P3_P8": ("432105|9|678905", "L3"), "P1_P8": ("432105|9|999999", "train"),
    "P7_P3": ("452301|1|086420", "L3"), "P1_P7": ("452301|1|123456", "train"),
    "P3_P7": ("452301|1|464646", "L3"), "P7_P1": ("452301|1|654321", "train"),
    "P5_P1": ("452301|9|111111", "train"), "P5_P3": ("452301|9|543210", "L3"),
    "P3_P5": ("452301|9|658709", "L3"), "P1_P5": ("452301|9|999999", "train"),
    "P6_P1": ("543210|3|444444", "L2"), "P1_P6": ("543210|3|666666", "train"),
    "P3_P6": ("543210|3|852963", "L3"), "P6_P3": ("543210|3|876543", "L3"),
}


def test_derived_subop_triples_match_doc_deck():
    for name, expected in DOC_SUBOPS.items():
        assert ops.SUBOP_TRIPLES[name] == expected, name


def test_derived_surface_triples_match_doc_table():
    for name, expected in DOC_SURFACE.items():
        assert ops.SURFACE_TRIPLES[name] == expected, name


def test_derived_64_cells_match_doc_table():
    assert len(DOC_CELLS) == 64
    s = split.build()
    for tid, (doc_key, doc_split) in DOC_CELLS.items():
        cell = s["cells"][tid]
        assert cell["triple_key"] == doc_key, (tid, cell["triple_key"], doc_key)
        assert cell["split"] == doc_split, (tid, cell["split"], doc_split)


def test_accounting():
    s = split.build()
    assert len(s["train_pairs"]) == 34
    assert len(s["heldout"]["L1"]) == 8
    assert len(s["heldout"]["L2"]) == 6
    assert len(s["heldout"]["L3"]) == 15
    assert len(s["excluded"]) == 1
    assert len(s["singletons_train"]) == 8
    # 42 training tasks = 34 pairs + 8 singletons
    assert len(s["train_pairs"]) + len(s["singletons_train"]) == 42


def test_dax_is_singleton_only():
    s = split.build()
    assert R.DAX in s["singletons_train"]
    for tid in s["train_pairs"]:
        assert R.DAX not in ops.task_surface_ops(tid)
    # and every pair containing the dax is L3
    for tid in split.PAIR_IDS:
        if R.DAX in ops.task_surface_ops(tid):
            assert s["cells"][tid]["split"] == "L3"


def test_merged_class_travels_together():
    s = split.build()
    assert s["audit"]["merged_classes"] == [["P4_P8", "P8_P4"]]
    assert s["cells"]["P4_P8"]["split"] == s["cells"]["P8_P4"]["split"] == "train"


def test_frozen_split_matches_derivation():
    split.load_verified()  # raises if the frozen file drifted from the algebra


def test_validate_rejects_split_apart_merged_class():
    cells = split.assign(split.derive_cells())
    # sabotage with counts preserved: P8_P4 (train) <-> P2_P1 (L1). P8_P4 held
    # out while its functional twin P4_P8 trains = training on the test set.
    cells["P8_P4"]["split"] = "L1"
    cells["P2_P1"]["split"] = "train"
    with pytest.raises(AssertionError):
        split.validate(cells)


def test_validate_rejects_trained_adjacency_in_l2():
    cells = split.assign(split.derive_cells())
    # sabotage with counts preserved: P2_P1's adjacency (A,R) is trained via
    # P7_P1, so calling it L2 must be rejected.
    cells["P2_P1"]["split"] = "L2"
    cells["P1_P4"]["split"] = "L1"
    with pytest.raises(AssertionError):
        split.validate(cells)
