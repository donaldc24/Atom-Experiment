# H1 Experiment 5 - The Producer Branch: Training Atoms to Speak, Not Just Listen

## Why this experiment exists
The E3 sandbox trained atoms to RECEIVE arbitrary states (closure branch: random chain
in, target atom must produce valid output). It moved distance-to-correct-state for the
first time. E4 stacked it with interface noise and produced the best free arm ever
(A14 s0: L1 56%, repair gap 0.41).

But every pressure so far trains the listening side. No pressure has ever trained an
atom to EMIT a state that arbitrary strangers can use. The dax wall (L3 = 0.000 in all
seven pressures tried, repair on those same cells 87-98%) is exactly where emission
under novel context is demanded.

E5 completes the symmetry:

> Can training atoms to produce states that survive random frozen continuations crack
> the dax wall without ever mentioning dax?

## Generality check (registered up front)
The pressure speaks only the architecture's language: atoms, states, chains, frozen
decoder. It never references tasks, primitives, split levels, or P3. It applies to
every atom identically. Oversampling dax cells, level-aware losses, or per-task tuning
would be tailoring; none are present. The dax question is the registered HEADLINE, not
a training signal.

## Base
- A14 (E4 stacked pressures) IF it holds across all seeds as best arm. If E4's pairs
  break seed 0's story, base falls back to the best surviving arm and this section
  gets a dated amendment BEFORE launch
- Base is inherited byte-for-byte: noise at A9 dose, sandbox at A12 dose, cosine
  router, lambda_use = 0, 30k steps

## The treatment: producer branch
New branch from the same stopgrad(z0) the sandbox already uses:

```text
z_out = Apply_i(z0)                    # trainable target atom i EMITS
for k in 1..K branches (independent):
    chain_k = random frozen atoms, length ~ U{1..6}   # frozen params, LIVE activations
    L_k     = READ(chain_k(z_out))     # frozen decoder validity at the end
L_producer = mean_k(L_k)               # gradient flows back THROUGH the frozen
                                       # chains into atom i ONLY
```

- Gradient path (non-negotiable): unlike the receiver branch (stopgrad before target),
  gradient here flows THROUGH frozen downstream activations to reach the producer.
  Frozen parameters, live activations - same mechanism as training through the frozen
  decoder. Unit test must verify: producer atom gets gradient, chain atoms and decoder
  get exactly none
- K = 4 branches per step, registered. Why branching beats one continuation: one
  listener per step lets the producer chase partner-specific slang step to step;
  K simultaneous listeners force the intersection - a state broadly usable, not
  privately usable. Same expected objective, far lower variance, no listener-chasing
- Chains drawn at full length, only first k applied: fixed RNG consumption per step
- Dedicated numpy stream (next free index), no torch RNG consumed
- Loss: L_task + existing E4 terms + lambda_producer * L_producer

## Arms
| Arm | lambda_producer | Everything else |
|---|---|---|
| A16 | 0.1 | = A14 |
| A17 | 0.3 | = A14 |

Seeds 0/1/2. Six new runs. A14 attaches as reference row, never re-run.

## Registered limitation (the honest asterisk)
The producer emits from CLEAN z0 only. In real composition, second-token atoms emit
from mid-computation states. That distribution gap is where partial-but-not-full
success is predicted. "Emit from chain-context too" is the registered E6 dose if L3
moves but does not close. It is NOT folded into E5.

## Gaming audit (named before launch)
The degenerate exit: a producer emitting one fixed survives-anything state - the
one-word language, gamed twice in V1, and K branches INCREASE the pull toward it
(the trivial message satisfies every listener). Guards:
- Task loss (a constant state fails composition on seen tasks immediately)
- Existing uniqueness fingerprints
- NEW telemetry row: per-atom producer-output variance across inputs. A collapsing
  producer must announce itself
- NEW free metric: spread of READ across the K branches for the same state = direct
  gauge of remaining context-sensitivity in production. Log per eval

## Gates
- Zero-path equivalence: lambda_producer = 0 replays an A14 run bit-identical
  (steps 1, 50) same-env, or passes the registered cross-env three-part test
  (step-1 bit-identity, grad bit-identity, <= 1e-6 rel at step 50) per E4-G1
- All E2/E3/E4 implementation gates inherited
- E1b liveness gate unchanged, the only validity gate

## Measurement
Full certified panel at 20k and 30k (verify final-step panel actually fires), plus
E2 noise telemetry, E3 sandbox telemetry, and the two new producer rows above.
Headline reporting:
1. All 15 P3 cells individually. ANY cell >= 1% hard = first crack in the dax wall
   across eight pressures, the headline either way
2. Repair gap per level and WHY it moved (raw up vs repair down)
3. Distance-to-correct-state (does producer training push it below A12's 1.12)

## Outcome bins
- CRACK: any P3 cell >= 1% in a healthy arm. Headline regardless of all else
- PRODUCER WORKS, DAX HOLDS: L1/L2 up, target error down, L3 still zero. Deep result:
  the failure is not closure as we understand it, and the repair-gap metric is
  measuring something we do not yet own. Triggers a diagnosis pause, not E6
- REDUNDANT: ~= A14. Emission was already covered by the stack
- COLLAPSE: producer-output variance falls, one-word language detected. Guards worked,
  dose wrong or mechanism gamed
- OVERDOSE: A17 unhealthy, A16 fine. Frontier finer than the ladder

## Amendments owed with this spec
- E5 registered dated, predictions frozen before first run
- Base-confirmation note once E4 pairs land (or the fallback amendment)
- DECISIONS: producer branch design, K = 4 rationale, gaming guards