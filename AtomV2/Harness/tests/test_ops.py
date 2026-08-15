"""Algebra tests: triple extraction, composition rule, degeneracy structure."""
import itertools

import numpy as np
import pytest

from atomv2 import ops
from atomv2 import registered as R

RNG = np.random.default_rng(123)
X = RNG.integers(0, 10, size=(500, 6), dtype=np.int64)


def test_extract_triple_reproduces_every_subop():
    for name, fn in ops.SUBOPS.items():
        tri = ops.SUBOP_TRIPLES[name]
        assert np.array_equal(fn(X), ops.apply_triple(tri, X)), name


def test_composition_rule_matches_direct_composition_for_all_pairs():
    for f, g in itertools.product(ops.SUBOP_NAMES, repeat=2):
        tri = ops.compose_triples(ops.SUBOP_TRIPLES[f], ops.SUBOP_TRIPLES[g])
        direct = ops.SUBOPS[g](ops.SUBOPS[f](X))
        assert np.array_equal(direct, ops.apply_triple(tri, X)), (f, g)


def test_surface_recipes_apply_first_op_first():
    # P1 = R then I: reverse first, then increment.
    x = np.array([[1, 3, 4, 2, 5, 9]])
    assert ops.apply_task("P1", x).tolist() == [[0, 6, 3, 5, 4, 2]]


def test_add_index_is_zero_based():
    x = np.zeros((1, 6), dtype=np.int64)
    assert ops.SUBOPS["A"](x).tolist() == [[0, 1, 2, 3, 4, 5]]


def test_involutions_square_to_identity():
    for name in ("R", "N", "W"):
        tri = ops.compose_triples(ops.SUBOP_TRIPLES[name], ops.SUBOP_TRIPLES[name])
        assert tri == ops.IDENTITY_TRIPLE, name


def test_p8_internal_order_is_unrecoverable():
    tn = ops.compose_triples(ops.SUBOP_TRIPLES["T"], ops.SUBOP_TRIPLES["N"])
    nt = ops.compose_triples(ops.SUBOP_TRIPLES["N"], ops.SUBOP_TRIPLES["T"])
    assert tn == nt  # the answer key for P8 is the SET {T, N}


def test_pointwise_commutation_nm():
    nm = ops.compose_triples(ops.SUBOP_TRIPLES["N"], ops.SUBOP_TRIPLES["M"])
    mn = ops.compose_triples(ops.SUBOP_TRIPLES["M"], ops.SUBOP_TRIPLES["N"])
    assert nm == mn
    # both equal x -> 7x
    assert nm[1] == 7 and nm[0] == tuple(range(6)) and nm[2] == (0,) * 6


def test_im_and_mi_are_distinct():
    im = ops.compose_triples(ops.SUBOP_TRIPLES["I"], ops.SUBOP_TRIPLES["M"])
    mi = ops.compose_triples(ops.SUBOP_TRIPLES["M"], ops.SUBOP_TRIPLES["I"])
    assert im != mi  # do not over-merge


def test_lattice_prefix_values_include_cancellation():
    # task (P8, P5) chains T,N,N,W; the N,N annihilates so prefix 3's value
    # equals prefix 1's value (T(x)) - order-2 cancellations DO fire at depth 2.
    x = X[:10]
    prefixes = ops.lattice_prefix_values("P8_P5", x)
    assert len(prefixes) == 5  # depth 0 + 4 sub-ops
    assert np.array_equal(prefixes[1], prefixes[3])


def test_task_subop_sets_are_setwise():
    assert ops.task_subop_sets("P8") == [frozenset({"T", "N"})]
    assert ops.task_subop_sets("P2_P4") == [frozenset({"T", "A"}),
                                            frozenset({"M", "T"})]


def test_adjacency_map():
    assert ops.task_adjacency("P2_P4") == ("A", "M")
    assert ops.task_adjacency("P8_P5") == ("N", "N")
