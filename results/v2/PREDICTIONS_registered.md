# v2 battery — registered predictions

**Recorded 2026-08-10, while the battery was running, before any v2 metric was read.**
Written outside the repository because a source edit mid-batch halts the run (D43);
to be committed to `DECISIONS.md` as D44 at the first safe moment.

No v2 artifact, log or metric has been inspected at time of writing.

## Basis

v1 means (5 seeds, 1 split) from `results/archive_perro_v1/summary.csv`. v2 changes
the task family only: `index_shift` replaces `sort_asc` (42 vs 39 distinct ordered-pair
functions), across 3 splits x 3 seeds. Nothing about the training signal changes, so
the headline prediction is that **the v1 result reproduces qualitatively**.

Two opposing effects on within-task accuracy, which is why `acc_seen` carries the
widest intervals:

- `index_shift` (position-wise addition) is **easier to represent** than `sort_asc`
  (global comparison), pushing `acc_seen` and `acc_singleton` up.
- Removing the absorbing primitive leaves **24 distinct functions among the 40 train
  pairs** against v1's 22, pushing `acc_seen` down.

I expect the first to dominate.

## Per-arm predictions (mean over 9 runs)

| arm | `acc_seen` | `acc_unseen` | closed-map (matched) | coverage | teacher-forced | verdict |
|---|---|---|---|---|---|---|
| **A0** oracle | 0.98–1.00 | **0.93–0.98** | 0.02–0.05 | **8/8** | **≥ 0.995** | PASS (ceiling) |
| **A1** naive | 0.72–0.88 | **< 0.01** | 1.3–1.6 | 1/8 | 0.05–0.15 | FAIL |
| **A2** protected | 0.50–0.70 | **< 0.01** | 1.3–1.6 | 1/8 | 0.03–0.10 | FAIL |
| **A3** seq. frozen | 0.60–0.80 | **< 0.02** | 0.55–0.80 | 1/8 | < 0.05 | FAIL |
| **A3b** assigned | 0.50–0.70 | **< 0.03** | 0.55–0.80 | 1/8 | < 0.05 | FAIL |
| **A4** shuffled | **0.0000** | **0.0000** | ~1.2 | 1/8 | 0.000 | FAIL (by design) |

## High-confidence (I would be genuinely surprised to be wrong)

1. **T1 passes**: A0 `M7_acc_teacher_forced` >= 0.99 on all 9 runs.
2. **No unsupervised arm composes**: A1/A2/A3/A3b `acc_unseen` < 0.05 everywhere.
3. **No unsupervised arm builds a library**: closed-map coverage **1/8** for every
   failing arm, on every split. This is the load-bearing v1 finding.
4. **A4 sits exactly at zero**: 0.0000 exact-match on both eval sets, all 9 runs,
   per-token ~0.10, `M5_dead` = 8. (Gate T3.)
5. **Program verdict: FAIL(training-signal)** — the same category as v1.
6. **A2 costs capacity without buying generalization**: `acc_seen`(A2) <
   `acc_seen`(A1) by >= 0.10, with `acc_unseen` unmoved.

## Lower-confidence, and why

- **A3's half-primitive pathology may not recur cleanly.** v1's phase 1 routed
  `(p, identity)` to `(atom_p, atom_p)`, so each atom learned a *square root* of its
  primitive. `reverse` is an involution, which made that especially clean. Under
  `index_shift` (`x_i -> x_i + i`), the square root is "add i/2", which has no exact
  form mod 10 for odd `i`. A3 may therefore behave differently — I expect
  `library_decay` still negative on a majority of seeds, but with lower confidence
  than any prediction above, and a genuinely different A3 signature is the most
  likely place for v2 to surprise.
- **`acc_singleton` should rise across all arms** (v1: A1 0.857, A3 0.898) because
  `sort_asc` was the hardest primitive to represent. Predict >= 0.88 for A1/A3.
- **`M3_align` stays AMBIGUOUS for A1/A2/A3** (0.50–0.85), reproducing the v1 pattern
  where the pre-registered probe reads mid-range on inert libraries (§2.6).

## The new measurement: split variance

v2's whole point. Predictions:

- **Split-to-split spread exceeds seed-to-seed spread on `acc_seen`**, because splits
  change which functions must be generalized to. Expect per-arm `acc_seen` std across
  the 9 runs to be **larger than v1's**, and for a decomposition to attribute most of
  it to split rather than seed.
- **`acc_unseen` shows near-zero variance of either kind** for the unsupervised arms,
  because they sit on the floor. A floor cannot express variance — so v2 will
  *not* answer whether composition is split-sensitive; it will only show that failure
  is not split-specific.
- **A0 `acc_unseen` varies more across splits than v1's ±0.017 seed spread.** Predict
  a pooled std of 0.02–0.06. This is the one number that genuinely measures something
  v1 could not.

## What would overturn something

- Any unsupervised arm reaching coverage >= 4/8 → the v1 diagnosis is split-specific
  and D37's framing needs revisiting.
- A0 teacher-forced < 0.99 → T1 fails, the battery aborts, and **no other arm is
  interpretable**. Most likely cause would be `index_shift` being harder to represent
  than argued, not a training-signal issue.
- A4 above 0.02 → leakage in the v2 splits; the split construction would need auditing
  before any other number is read.
- Any arm's pooled `acc_unseen` std > 0.10 → D42's escalation rule fires and that arm
  goes to 5 seeds before being reported.
