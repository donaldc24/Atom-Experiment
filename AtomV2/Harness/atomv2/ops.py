"""The op algebra: concrete sub-op functions and the (pi, a, b) canonical form.

Nothing in this module is transcribed from the SplitMath.md tables. Triples are
EXTRACTED from the concrete functions by probing (the SplitMath.md template
procedure) and compositions are computed by the composition rule; both are
cross-checked against direct function application on random inputs at import
time. The doc's tables exist only in tests/test_split.py as registered
expectations to diff against (an enumerator duty).

Template: output[j] = a * x[pi(j)] + b[j]  (mod 10), pi a permutation of 0..5.
Composition, f first then g:
    pi(j) = pi_f(pi_g(j));  a = a_f * a_g;  b[j] = a_g * b_f[pi_g(j)] + b_g[j].
"""
from __future__ import annotations

import numpy as np

from . import registered as R

L = R.SEQ_LEN
MOD = R.VOCAB

# ---------------------------------------------------------------------------
# Sub-ops as concrete batch functions over int arrays [B, L] (ground truth)
# ---------------------------------------------------------------------------

def _reverse(x):
    return x[:, ::-1].copy()

def _rotate_left(x):
    return np.roll(x, -1, axis=1)

def _swap_pairs(x):
    idx = np.arange(L)
    idx = idx + np.where(idx % 2 == 0, 1, -1)
    return x[:, idx].copy()

def _increment(x):
    return (x + 1) % MOD

def _negate(x):
    return (MOD - x) % MOD

def _multiply_3(x):
    return (3 * x) % MOD

def _add_index(x):
    # ZERO-BASED indexing: position 0 adds 0, position 5 adds 5. The generator,
    # canonical forms, and probes must all agree on this convention.
    return (x + np.arange(L)[None, :]) % MOD

SUBOPS = {
    "R": _reverse,
    "T": _rotate_left,
    "W": _swap_pairs,
    "I": _increment,
    "N": _negate,
    "M": _multiply_3,
    "A": _add_index,
}
SUBOP_NAMES = tuple(SUBOPS)          # ('R','T','W','I','N','M','A')
SURFACE_NAMES = tuple(R.SURFACE_RECIPES)  # ('P1',...,'P8')

# ---------------------------------------------------------------------------
# Canonical triples
# ---------------------------------------------------------------------------

# Units mod 10 and their inverses (a must always be a unit or the map is lossy)
_A_INV = {1: 1, 3: 7, 7: 3, 9: 9}


def extract_triple(fn) -> tuple[tuple, int, tuple]:
    """Derive (pi, a, b) from a concrete function by probing.

    The SplitMath.md procedure: b = fn(zeros); a = fn(ones) - b (must be
    constant across positions); pi from fn(identity list) after removing a, b.
    Verified against fn on random inputs before returning.
    """
    zeros = np.zeros((1, L), dtype=np.int64)
    ones = np.ones((1, L), dtype=np.int64)
    ident = np.arange(L, dtype=np.int64)[None, :]

    b = fn(zeros)[0]
    a_vals = np.unique((fn(ones)[0] - b) % MOD)
    if len(a_vals) != 1:
        raise ValueError(f"multiplier not constant across positions: {a_vals}")
    a = int(a_vals[0])
    if a not in _A_INV:
        raise ValueError(f"a={a} is not a unit mod {MOD}; op is not invertible")
    pi = tuple(int(v) for v in ((fn(ident)[0] - b) * _A_INV[a]) % MOD)
    if sorted(pi) != list(range(L)):
        raise ValueError(f"pi is not a permutation: {pi}")
    b = tuple(int(v) for v in b)

    rng = np.random.default_rng(0)
    x = rng.integers(0, MOD, size=(256, L), dtype=np.int64)
    if not np.array_equal(fn(x), apply_triple((pi, a, b), x)):
        raise ValueError("extracted triple does not reproduce the function")
    return pi, a, b


def apply_triple(triple, x):
    pi, a, b = triple
    return (a * x[:, list(pi)] + np.asarray(b)[None, :]) % MOD


def compose_triples(tf, tg):
    """Triple of (f first, then g)."""
    (pf, af, bf), (pg, ag, bg) = tf, tg
    pi = tuple(pf[pg[j]] for j in range(L))
    a = (af * ag) % MOD
    b = tuple((ag * bf[pg[j]] + bg[j]) % MOD for j in range(L))
    return pi, a, b


def triple_key(triple) -> str:
    pi, a, b = triple
    return f"{''.join(map(str, pi))}|{a}|{''.join(map(str, b))}"


IDENTITY_TRIPLE = (tuple(range(L)), 1, (0,) * L)

# ---------------------------------------------------------------------------
# Derived tables (computed once at import; each derivation is self-verifying)
# ---------------------------------------------------------------------------
SUBOP_TRIPLES = {name: extract_triple(fn) for name, fn in SUBOPS.items()}


def _surface_fn(recipe):
    f_name, g_name = recipe
    def fn(x, _f=SUBOPS[f_name], _g=SUBOPS[g_name]):
        return _g(_f(x))
    return fn

SURFACE_FNS = {p: _surface_fn(rec) for p, rec in R.SURFACE_RECIPES.items()}
SURFACE_TRIPLES = {}
for _p, _rec in R.SURFACE_RECIPES.items():
    _tri = compose_triples(SUBOP_TRIPLES[_rec[0]], SUBOP_TRIPLES[_rec[1]])
    _rng = np.random.default_rng(1)
    _x = _rng.integers(0, MOD, size=(256, L), dtype=np.int64)
    if not np.array_equal(SURFACE_FNS[_p](_x), apply_triple(_tri, _x)):
        raise AssertionError(f"composition rule failed for {_p}")
    SURFACE_TRIPLES[_p] = _tri

# First/last sub-op per surface op - drives the adjacency map (L1 vs L2).
FIRST_SUBOP = {p: rec[0] for p, rec in R.SURFACE_RECIPES.items()}
LAST_SUBOP = {p: rec[1] for p, rec in R.SURFACE_RECIPES.items()}

# ---------------------------------------------------------------------------
# Tasks: singletons 'P3' and ordered pairs 'P2_P4' (P2 first, then P4)
# ---------------------------------------------------------------------------

def task_surface_ops(task_id: str) -> tuple[str, ...]:
    return tuple(task_id.split("_"))


def task_triple(task_id: str):
    ops_ = task_surface_ops(task_id)
    tri = SURFACE_TRIPLES[ops_[0]]
    for p in ops_[1:]:
        tri = compose_triples(tri, SURFACE_TRIPLES[p])
    return tri


def task_subop_chain(task_id: str) -> tuple[str, ...]:
    """The task's hidden sub-op chain, e.g. P2_P4 -> (T, A, M, T)."""
    chain = []
    for p in task_surface_ops(task_id):
        chain.extend(R.SURFACE_RECIPES[p])
    return tuple(chain)


def task_adjacency(task_id: str) -> tuple[str, str]:
    """Cross-token adjacency of Pi_Pj = (last sub-op of Pi, first of Pj)."""
    a, b = task_surface_ops(task_id)
    return (LAST_SUBOP[a], FIRST_SUBOP[b])


def apply_task(task_id: str, x):
    return apply_triple(task_triple(task_id), x)


def lattice_prefix_values(task_id: str, x) -> list[np.ndarray]:
    """Digit lists at every prefix of the task's sub-op chain, depth 0 first.

    These are the "legal states of the world" the trajectory closed-map error
    measures against. Duplicate values along the chain (order-2 cancellations
    like the N,N inside (P8,P5)) are kept as distinct prefixes; they collide in
    value, which is exactly what the metric wants.
    """
    out = [x.copy()]
    cur = x
    for name in task_subop_chain(task_id):
        cur = SUBOPS[name](cur)
        out.append(cur)
    return out


def task_subop_sets(task_id: str) -> list[frozenset]:
    """Per-token sub-op label SETS for the decodability probes.

    Set-wise by construction: P8's set is {T, N} with no order information -
    the answer key for P8 is the SET, not the sequence (H1Experiments.md P8
    caveat). Every probe touching sub-op labels goes through this function.
    """
    return [frozenset(R.SURFACE_RECIPES[p]) for p in task_surface_ops(task_id)]


# Candidate answer key for standalone semantics / atom-centric closed map:
# every named function an atom could coherently be (identity excluded from the
# candidates; an atom matching identity is a dead atom, which the census and
# coverage companions report separately).
CANDIDATE_OPS: dict[str, object] = {}
for _n in SUBOP_NAMES:
    CANDIDATE_OPS[f"sub:{_n}"] = SUBOPS[_n]
for _p in SURFACE_NAMES:
    CANDIDATE_OPS[f"surf:{_p}"] = SURFACE_FNS[_p]
