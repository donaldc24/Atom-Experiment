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


PRIMITIVES = (
    ("identity", _identity),
    ("reverse", _reverse),
    ("increment", _increment),
    ("sort_asc", _sort_asc),
    ("rotate_left", _rotate_left),
    ("swap_halves", _swap_halves),
    ("double", _double),
    ("reflect", _reflect),
)

PRIMITIVE_NAMES = tuple(name for name, _ in PRIMITIVES)
IDENTITY_ID = 0

assert len(PRIMITIVES) == K


def apply_primitive(pid: int, x: np.ndarray) -> np.ndarray:
    """Apply primitive `pid` to a batch [B, L]."""
    return PRIMITIVES[pid][1](x)


def apply_composition(pids, x: np.ndarray) -> np.ndarray:
    """Apply primitives left-to-right: pids=(i, j) means p_j(p_i(x))."""
    out = x
    for pid in pids:
        out = apply_primitive(pid, out)
    return out


def random_inputs(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, V, size=(n, L), dtype=np.int64)


def check_primitive_independence(n_samples: int = 10_000, seed: int = 0):
    """T4 -- no primitive is extensionally equal to a composition of <=2 *other* primitives.

    Returns (violations, notes). `violations` is a list of (target_id, composition)
    tuples. `identity` (p0) is excluded as a *target* by construction: it is the
    designated no-op used for length-1 tasks and is necessarily equal to any
    self-inverse primitive composed with itself (reverse@reverse, reflect@reflect,
    swap_halves@swap_halves). See DECISIONS.md D3.
    """
    rng = np.random.default_rng(seed)
    x = random_inputs(rng, n_samples)
    outs = {p: apply_primitive(p, x) for p in range(K)}

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
                if np.array_equal(apply_primitive(b, ya), outs[target]):
                    violations.append((target, (a, b)))
    notes = {
        "n_samples": n_samples,
        "seed": seed,
        "excluded_targets": [IDENTITY_ID],
    }
    return violations, notes


def check_noncommutativity(n_samples: int = 2_000, seed: int = 1):
    """Informational: fraction of unordered primitive pairs that do not commute."""
    rng = np.random.default_rng(seed)
    x = random_inputs(rng, n_samples)
    noncommuting = 0
    total = 0
    for a in range(K):
        for b in range(a + 1, K):
            total += 1
            ab = apply_composition((a, b), x)
            ba = apply_composition((b, a), x)
            if not np.array_equal(ab, ba):
                noncommuting += 1
    return noncommuting, total


def distinct_pair_functions(n_samples: int = 2_000, seed: int = 2):
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
            key = apply_composition((a, b), x).tobytes()
            sigs.setdefault(key, []).append((a, b))
    return sigs
