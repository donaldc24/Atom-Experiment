# H1 Experiment 4 - Catching + Throwing: Do the Two Working Pressures Stack?

## Why this experiment exists
E2 and E3 each moved a dial the other could not:

- Noise (A9) taught downstream to READ garbled states: L1 12 -> 30, ablation CV best ever,
  but distance-to-correct-state never moved (1.37, same as A6)
- Sandbox (A12) taught atoms to PRODUCE states closer to truth: target error 1.38 -> 1.12
  (only pressure to ever move it), stable atom identity, but L1 gains modest

One fixed catching, one fixed throwing. E4 asks the obvious question:

> Do the two middle doses stack, interfere, or overdose when combined?

And the registered sub-question:

> With both pressures on, does ANY P3 cell move off exactly zero? Even one cell at >= 1%
> raw would be the first crack in the dax wall across seven attempted pressures.

## Base and harness
- Base: A6 (cosine router, lambda_use = 0), same as E2 and E3
- The two worktrees (Atom-Experiment = E2, Atom-Experiment-e1-runner = E3) merge into one
  harness before anything runs. Both zero-paths re-gated:
  1. state_noise_sigma = 0 AND lambda_sandbox_* = 0 replays A6 seed 0 bit-identical
     (steps 1 and 50), same as both prior gates
  2. Noise-only config reproduces an A9 run's step-1/step-50 records
  3. Sandbox-only config reproduces an A12 run's step-1/step-50 records
- Hard stop on any gate failure

## The treatment: both priors, unchanged, together
- Interface noise exactly as E2 registered: training-only, at nonterminal handoffs,
  re-LayerNormed, own RNG stream
- Sandbox exactly as E3 registered: READ validity + uniqueness on atom MLPs only,
  frozen decoder, own RNG streams
- Interaction rule (registered): the sandbox sees CLEAN states only. z0 remains
  stopgrad of the task forward's pre-noise state, sandbox chains carry no channel
  noise. The combination is literally both prior treatments coexisting, no new
  mechanism, no third pressure

## Arms
| Arm | state_noise (target cosine) | lambda_sandbox (valid/unique) | Why |
|---|---|---|---|
| A14 | 0.990 (A9 dose) | 0.3 (A12 dose) | The headline: both middle doses |
| A15 | 0.999 (A8 dose) | 0.1 (A11 dose) | Insurance: half doses, in case middles jointly overdose |

Seeds 0/1/2 each. Six new runs. A9, A12, and A6 attach as labelled reference rows,
never re-run.

## Step budget: 30k (change from 20k, registered here)
- A12 seed 0 was still climbing at 20k; both treatments stabilized convergence enough
  that longer runs finally pay
- Paired comparison rule: compare treatment arms to references at the 20k checkpoint
  (like for like) AND report 30k finals separately. Never mix budgets in one comparison
- Checkpoint cadence unchanged, panel at 20k and 30k

## Validity
- E1b liveness gate unchanged (the only gate)
- All E2 implementation gates (cosine target, no clean side channel, timing, masking)
- All E3 structural checks (gradient boundaries, no torch RNG consumption)
- Implementation failure invalidates a run; optimization failure is a result

## Measurement
Full certified panel at 20k and 30k, plus:
- E2 noise telemetry and the robustness sweep
- E3 sandbox telemetry (READ, CYCLE observed-only, fingerprint distances, usage EMA)
- Per-P3-cell raw accuracy, all 15 cells, reported individually (the dax-crack check:
  any cell >= 1% hard = first movement ever, named in results either way)
- Headline: repair gap per level, and WHY it moved (raw up vs repair down)

## Outcome bins (registered before launch)
- STACK: L1 above both singles (> 30%), target error at or below A12's 1.12, repair gap
  shrinks via raw rising. The pressures are complementary
- REDUNDANT: combined ~= best single. They were touching the same underlying thing
- INTERFERE: combined below both singles. Noise blurs what the sandbox is pulling toward
  truth; the two pressures fight over the same states
- JOINT OVERDOSE: A14 unstable but A15 healthy and above singles. Doses stack
  nonlinearly; the frontier is finer than one grid step
- DAX CRACK: any P3 cell >= 1% in any healthy arm. Register immediately as the headline
  regardless of other bins


## Amendments owed with this spec
- Cross-battery synthesis doc (E2 + E3 against shared A6), honest bin calls per each
  experiment's own registered taxonomy (A9/A10 sits BETWEEN success and robustness-only)
- Two-worktree note and the harness merge record
- This spec registered dated, predictions frozen before first run