"""E1 primitive set: K=8 unary transforms on integer sequences of length L=8 over V={0..9}.

All primitives are total functions on the domain and are implemented as vectorised
numpy operations over a batch of shape [B, L].
"""

from __future__ import annotations

import numpy as np

L = 8
V = 10
K = 8


def _identity(x: np.ndarray) -> np.ndarray:
    return x.copy()


def _reverse(x: np.ndarray) -> np.ndarray:
    return x[:, ::-1].copy()


def _increment(x: np.ndarray) -> np.ndarray:
    return (x + 1) % V


def _sort_asc(x: np.ndarray) -> np.ndarray:
    return np.sort(x, axis=1)


def _rotate_left(x: np.ndarray) -> np.ndarray:
    return np.roll(x, -1, axis=1)


def _swap_halves(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:, L // 2:], x[:, : L // 2]], axis=1)


def _double(x: np.ndarray) -> np.ndarray:
    return (2 * x) % V


def _reflect(x: np.ndarray) -> np.ndarray:
    return (V - 1) - x


def _index_shift(x: np.ndarray) -> np.ndarray:
    """x_i -> (x_i + i) mod V. Position-dependent, order-preserving, NON-idempotent.

    The v2 replacement for `sort_asc`. D10/D15 identified `sort_asc` as the main
    driver of the extensional collapse: it is idempotent and order-destroying, so it
    absorbs every position-permuting predecessor and makes six ordered pairs equal to
    its own singleton. Verified on 10,000 inputs: swapping it in raises the number of
    extensionally distinct ordered pairs from **39 to 42** with **zero** T4
    violations. See D40.
    """
    return (x + np.arange(L)[None, :]) % V


# The primitive set is a GENERATION parameter, not a constant. v1 is the set the
# committed E1 battery ran on and must never change -- every v1 artifact and every
# frozen split hash depends on it byte-for-byte. v2 swaps slot 3 only, so primitive
# ids, the identity slot and K are all preserved and the two sets stay comparable.
PRIMITIVE_SETS = {
    "v1": (
        ("identity", _identity),
        ("reverse", _reverse),
        ("increment", _increment),
        ("sort_asc", _sort_asc),
        ("rotate_left", _rotate_left),
        ("swap_halves", _swap_halves),
        ("double", _double),
        ("reflect", _reflect),
    ),
    "v2": (
        ("identity", _identity),
        ("reverse", _reverse),
        ("increment", _increment),
        ("index_shift", _index_shift),      # <- the only difference from v1
        ("rotate_left", _rotate_left),
        ("swap_halves", _swap_halves),
        ("double", _double),
        ("reflect", _reflect),
    ),
}

DEFAULT_SET = "v1"

# Back-compat aliases. Every existing import keeps resolving to the v1 set, which is
# what makes v1 runs reproduce unchanged after this refactor.
PRIMITIVES = PRIMITIVE_SETS[DEFAULT_SET]
PRIMITIVE_NAMES = tuple(name for name, _ in PRIMITIVES)
IDENTITY_ID = 0

assert all(len(s) == K for s in PRIMITIVE_SETS.values())
assert all(s[IDENTITY_ID][0] == "identity" for s in PRIMITIVE_SETS.values())


def primitives_for(pset: str = DEFAULT_SET):
    """The (name, fn) table for a generation."""
    try:
        return PRIMITIVE_SETS[pset]
    except KeyError:
        raise ValueError(
            f"unknown primitive set {pset!r}; known: {sorted(PRIMITIVE_SETS)}"
        ) from None


def primitive_names(pset: str = DEFAULT_SET) -> tuple:
    return tuple(name for name, _ in primitives_for(pset))


def apply_primitive(pid: int, x: np.ndarray, pset: str = DEFAULT_SET) -> np.ndarray:
    """Apply primitive `pid` to a batch [B, L]."""
    return primitives_for(pset)[pid][1](x)


def apply_composition(pids, x: np.ndarray, pset: str = DEFAULT_SET) -> np.ndarray:
    """Apply primitives left-to-right: pids=(i, j) means p_j(p_i(x))."""
    out = x
    for pid in pids:
        out = apply_primitive(pid, out, pset)
    return out


def random_inputs(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, V, size=(n, L), dtype=np.int64)


def check_primitive_independence(n_samples: int = 10_000, seed: int = 0,
                                 pset: str = DEFAULT_SET):
    """T4 -- no primitive is extensionally equal to a composition of <=2 *other* primitives.

    Returns (violations, notes). `violations` is a list of (target_id, composition)
    tuples. `identity` (p0) is excluded as a *target* by construction: it is the
    designated no-op used for length-1 tasks and is necessarily equal to any
    self-inverse primitive composed with itself (reverse@reverse, reflect@reflect,
    swap_halves@swap_halves). See DECISIONS.md D3.
    """
    rng = np.random.default_rng(seed)
    x = random_inputs(rng, n_samples)
    outs = {p: apply_primitive(p, x, pset) for p in range(K)}

    violations = []
    for target in range(K):
        if target == IDENTITY_ID:
            continue
        others = [p for p in range(K) if p != target]
        for a in others:
            if np.array_equal(outs[a], outs[target]):
                violations.append((target, (a,)))
            ya = outs[a]
            for b in others:
                if np.array_equal(apply_primitive(b, ya, pset), outs[target]):
                    violations.append((target, (a, b)))
    notes = {
        "n_samples": n_samples,
        "seed": seed,
        "primitive_set": pset,
        "excluded_targets": [IDENTITY_ID],
    }
    return violations, notes


def check_noncommutativity(n_samples: int = 2_000, seed: int = 1,
                           pset: str = DEFAULT_SET):
    """Informational: fraction of unordered primitive pairs that do not commute."""
    rng = np.random.default_rng(seed)
    x = random_inputs(rng, n_samples)
    noncommuting = 0
    total = 0
    for a in range(K):
        for b in range(a + 1, K):
            total += 1
            ab = apply_composition((a, b), x, pset)
            ba = apply_composition((b, a), x, pset)
            if not np.array_equal(ab, ba):
                noncommuting += 1
    return noncommuting, total


def distinct_pair_functions(n_samples: int = 2_000, seed: int = 2,
                            pset: str = DEFAULT_SET):
    """Informational: how many of the 64 ordered pairs are extensionally distinct.

    Pairs that collapse onto the same function are still legitimate tasks (the
    model must still route correctly), but a large collapse would mean the
    64-pair space is smaller than it looks. Reported in the split manifest.
    """
    rng = np.random.default_rng(seed)
    x = random_inputs(rng, n_samples)
    sigs = {}
    for a in range(K):
        for b in range(K):
            key = apply_composition((a, b), x, pset).tobytes()
            sigs.setdefault(key, []).append((a, b))
    return sigs
