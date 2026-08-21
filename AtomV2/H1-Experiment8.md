# H1 Experiment 8 - Capacity Rescue: One Route Per Token, More Computation Per Atom

## Why this experiment exists

E7's registered screen ended in ONE-STEP CAPACITY / OPTIMIZATION FAILURE: A18
(A6 with one routing decision per token) reached 14.7% seen accuracy against
the 80% health gate, mapped zero competent atoms, and the interface question
never opened. E7's spec pre-commits the follow-up: increase computation INSIDE
each pageable atom while keeping one routing decision per token, as a separate
registered experiment.

The A18 seed-1 curve is the evidence this design answers:

- seen accuracy plateaued at ~14% from step 11k to 20k (9k flat steps),
- task loss still fell slowly (-0.025/1k steps, decelerating) while
  exact-match accuracy did not respond - the representational-ceiling
  signature, not slow convergence,
- D3: original-input recoverability 1.000, answer recoverability 0.557 -
  the single residual MLP application preserves its input perfectly but
  cannot fully compute a surface operation (two hidden sub-ops including
  structural permutations) in one shot.

E8 therefore asks:

> Is per-atom serial depth the binding constraint on one-route-per-token
> architectures - and does an adequately deep atom then produce the directly
> composable units E7 was built to detect?

## Registered base: A18 (E7's one-step A6)

All E8 arms descend from A18 byte-for-byte: cosine router, 384-dim residual
state, 16 atoms, one routing decision per token, lambda_use = 0, no noise,
sandbox, producer, pool, rent, canonicalization, or semantic supervision.

## Arms

| Arm | Atom internal layers | Steps | Purpose |
|---|---:|---:|---|
| A22 | 1 (= A18) | 40,000 | Budget control: kills or confirms "just train longer" |
| A23 | 2 | 20,000 | Minimal depth rescue |
| A24 | 3 | 20,000 | Depth margin |

Atom depth d means the atom MLP is state -> hidden -> (hidden ->)^{d-1}
state with GELU between layers, applied as the same residual update through
the same step_once path. The whole atom still pages as one block: one route
decision = one atom load, regardless of internal depth. Nothing else changes;
per-atom parameters grow (~148k / ~185k / ~222k), and the composer does not.

## Registered predictions (before first run)

- A22 (40k budget control) stays under 20% seen: the plateau is a ceiling,
  not a slope.
- A23 and/or A24 clear the 80% health gate: depth was the binding constraint.
- Conditional on health, the E7 interface question reopens exactly as
  registered there: ICR on discovered competent pairs decides between
  composable one-step atoms and translator-justified.

## Run order (staged, mirrors E7's discipline)

1. Gate: zero-depth equivalence - the A18 config (atom_layers = 1) replayed
   under this harness reproduces the completed E7 A18 seed-1 step-1/step-50
   records under the frozen gate ladder.
2. Stage screen: A23 seed 1, A24 seed 1 (health gate: seen >= 80%), then A22
   seed 1 (runs its full 40k regardless; its verdict is its final number, no
   early stop - poor optimization is a result).
3. Stage replicate: seeds 0/2 of the winner per E7's selection rules
   (component competence, direct-chain accuracy, ICR, D3 state content -
   never seen/L1 alone). If both A23 and A24 pass health, the SHALLOWER one
   is the registered winner (minimal sufficient capacity); the deeper arm's
   seed 1 stays on record.
4. Every result-bearing run receives the full certified panel + the E7 audit
   (D1-boundary, D2 atom audit, direct/canonicalized chaining, ICR, D3).

## Outcome bins

- CAPACITY WAS THE FLOOR: a deep arm is healthy and ICR >= 0.80 on a
  meaningful set of competent pairs. One-route-per-token composable atoms
  exist; next stop is E7's "replication and lazy-loading memory tests".
- CAPACITY NECESSARY, INTERFACE STILL OPEN: deep arm healthy, ICR < 0.80
  while canonicalized chaining >= 0.80. E7's Stage-2 question (bandwidth /
  replacement / translator) reopens on the healthy deep base.
- STILL CAPACITY-LIMITED: no deep arm reaches health. Per-atom depth at
  these sizes is not sufficient; the one-route contract itself is suspect at
  this scale.
- BUDGET WAS THE ANSWER: A22 reaches 80% at 40k. The E7 screen verdict is
  amended (dated), and depth arms are reread against the long-budget control.

## Implementation gates

1. atom_layers = 1 reproduces the A18 execution path bit-identically (gate).
2. Depth changes the atom MLP only: routing, keys, composer, encoder,
   decoder, state width, residual update, and pass are untouched.
3. One route decision per token, verified as in E7.
4. Diagnostics remain read-only.
5. Streams and seed pairing unchanged.

Any implementation-gate failure invalidates the run; optimization failure is
a result.
