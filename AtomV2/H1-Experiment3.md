# H1 Experiment 3 - The Atom Sandbox: Standalone, Closure, Functional Uniqueness

## Why this experiment exists

E2 asked whether corrupting the shared interface would force robust handoff codes.
E3 attacks the same diagnosis - atoms acquire individually recognizable behavior while
producing fragile, partner-specific handoff states - from the other side: instead of
perturbing the channel, it trains each atom directly to satisfy three content-agnostic
properties, in a sandbox that runs beside the normal task path and never touches it:

```text
standalone + closure + functional uniqueness
```

- **Standalone**: every atom must produce a valid state when applied to a clean
  encoder state, with no predecessor required.
- **Closure**: every atom must accept states produced by arbitrary chains of other
  atoms (self-predecessors included), and those predecessors cannot adapt themselves
  to help it.
- **Functional uniqueness**: atoms the real composer actually uses must be
  behaviorally distinct from each other and from pass, so self-sufficiency cannot be
  satisfied by learning identity.

No pressure specifies WHAT any atom should compute. The sandbox is agnostic about
which factorization emerges, how many slots become useful, and which routes implement
any task.

## Registered base: A6

Identical to E2's registered base: the completed E1b A6 condition (cosine router,
`sigma = E1B_SIGMA['A6']`, `lambda_use = 0`, 20k steps, seeds 0/1/2) is the control.
The completed A6 runs are reused, never re-run, behind the same mechanical
equivalence gate: replay the first 50 training steps of A6 seed 0 under the current
harness and require byte-identical step-1/step-50 log records. The A6 config carries
`state_noise_sigma = 0` AND `lambda_sandbox_* = 0`, so one replay certifies that the
sandbox code path is completely inert at lambda zero. All comparisons are paired by
seed.

## The treatment

Per training step, on the SAME batch as the task forward:

```text
NORMAL A6 (untouched):
batch -> encoder -> composer -> atoms -> decoder -> L_task

SANDBOX (same batch):
z0 = stopgrad(code(x))          # the canonical state the task forward produced
   |
   +- one random atom alone ------------------------> validity      (standalone)
   |
   +- random no-grad atom chain (K ~ U{1..6})
   |          |
   |       stopgrad
   |          |
   |    one random target atom ---------------------> validity      (closure)
   |
   +- sampled atom set + pass, on clean z0
              |
         frozen decoder fingerprints
              |
        behavioral distance ------------------------> uniqueness
```

Total loss:

```text
L = L_task
  + lambda_sandbox_valid  * 0.5 * (L_standalone + L_closure)
  + lambda_sandbox_unique * L_unique
lambda_use = 0
```

### Gradient boundaries (non-negotiable)

Sandbox losses update the atom MLP parameters (`w1/b1/w2/b2`) and NOTHING else.
Encoder, decoder, composer, routing keys, and the pass key are frozen during the
construction of every sandbox graph. The decoder in particular is the frozen
validity evaluator and fingerprint reader: it must never learn to interpret bad atom
states; the atoms must become independently readable. Every sandbox application uses
the exact normal atom update `Apply_i(z) = LN(z + A_i(z))` via `model.step_once` -
no separate sandbox atom implementation exists.

V1 retired round-trip/manifold/intermediate supervision from the free arms because
the trainable decoder gamed them. The sandbox's validity criterion is a different
mechanism: it applies to sandbox states only (never the execution trajectory), and
its evaluator is frozen, so the gaming channel - the evaluator adapting to the
states - does not exist. The registration records this distinction deliberately.

### Registered decision: the validity criterion

The harness has no pre-existing validity evaluator, so `L_valid(z)` is registered
here, against the frozen decoder:

```text
L_valid(z) = READ(z) = mean per-position CE of Decoder(z) against its own hard
             readout (-log p_max: z must decode CONFIDENTLY into some digit list)
```

READ is deliberately both content-agnostic and REPRESENTATION-agnostic: it demands
only that the state not be gibberish to the frozen decoder, and never pushes an
atom's output toward any particular latent encoding - an atom may invent whatever
representation it likes.

`CYCLE(z)` - relative MSE from z to `stopgrad(code(readout))`, the oracle's
relative-MSE form - is measured as TELEMETRY ONLY and is never backpropagated: as a
loss it would define what the intermediate representation should look like, which
is exactly the pressure this experiment refuses to apply. (Unit-enforced: the
computed cycle value carries no autograd graph.)

Standalone and closure share this one criterion and are logged separately;
`L_sandbox_valid` is their mean.

### Standalone branch

Uniformly sample one target atom `i` per step; `L_standalone = L_valid(Apply_i(z0))`.

### Random-context / closure branch

Sample `K ~ Uniform{1..6}` (six = the most atom applications A6 can execute on a
two-token task) and a chain of K uniformly random atoms - self-predecessors
included, no exclusions. The chain runs under `torch.no_grad`; the target atom is
tested on the stop-gradded chain output: `L_closure = L_valid(Apply_i(sg(z_ctx)))`.
The chain is always drawn at full length (only the first K applied) so the sandbox
stream's consumption per step is fixed.

### Functional uniqueness branch

Always on CLEAN standalone states, never on chains. Fingerprints are frozen-decoder
softmax outputs `F_i = softmax(Decoder(Apply_i(z0)))` and
`F_pass = softmax(Decoder(z0))`; distance is mean total variation ACROSS THE BATCH
(never one example):

```text
d(i,j) = E_{x,p}[ 0.5 * |F_i(x)_p - F_j(x)_p|_1 ]
L_unique-pair(i,j)  = max(0, m - d(i,j))          m = 0.5
L_nonidentity(i)    = max(0, m - d(i, pass))
```

Per step, 4 atoms are sampled without replacement; all 6 pairs and all 4
nonidentity terms are computed.

### Usage weighting: unused slots may remain unused

Uniqueness pressure applies only to atoms the real A6 composer actually uses:

```text
usage_ema[i] <- 0.99 * usage_ema[i] + 0.01 * (hard picks of atom i / live steps)
w_i          = stopgrad(clamp(usage_ema[i] / CENSUS_EPS, 0, 1))
L_unique     = E_{i,j}[ w_i w_j L_unique-pair ] + E_i[ w_i L_nonidentity ]
```

Frequencies are over ACTIVE ROUTING OPPORTUNITIES (live steps): a pass pick
contributes zero to every atom, so an all-pass batch decays every weight
(`usage_i <- 0.99 * usage_i`). If sandbox training causes pass usage to emerge, a
formerly-used atom's uniqueness pressure fades instead of freezing at a stale
weight. This normalization deliberately differs from the census (which excludes
pass from its denominator); CENSUS_EPS is reused only as the clamp scale. The EMA
warm-starts uniform (1/16 > 2% => every atom starts fully weighted; unused slots
decay out with half-life ~69 steps). The weights are detached buffers: this
gradient NEVER enters routing.

## Arms

| Arm | lambda_sandbox_valid | lambda_sandbox_unique |
|---|---:|---:|
| A11 | 0.1 | 0.1 |
| A12 | 0.3 | 0.3 |
| A13 | 1.0 | 1.0 |

One treatment at three intensities, not a factorial. Everything else is inherited
from A6 (verified mechanically: the resolved configs differ only in `arm`,
`experiment`, `protocol_revision`, and the two lambdas). Seeds 0/1/2, protocol
revision `e3-atom-sandbox`.

## Randomness

All sandbox sampling comes from the dedicated numpy stream `e3_sandbox` (stream 11);
init-calibration and telemetry draws from the indexed `e3_sandbox_eval` (stream 12).
No torch RNG is consumed by the sandbox, so a sandbox-bearing run's routing, data,
init, and (absent) noise draws are bit-identical to its A6 pair.

## Telemetry and calibration

- `init_calibration.json` additionally records raw sandbox magnitudes vs task loss
  before any update, so the dose grid's scale is on record per run.
- `sandbox_telemetry/step*.json` at every scheduled eval (measurement only): per-atom
  standalone READ and CYCLE (CYCLE observed here and in the step records, never
  trained on), random-chain closure validity, the full 16x16 fingerprint distance
  matrix plus distance-to-pass, and the usage EMA/weight snapshot.
- E1b liveness telemetry and the deafness gate are retained unchanged; headline
  evaluation is the untouched clean path.

## Validity

The E1b liveness rule is the only gate. The sandbox carries no gate of its own:
implementation failures (gradient escaping the boundary, telemetry missing,
non-finite losses, nondeterministic clean eval) invalidate a run; sandbox losses
failing to optimize is a RESULT, not an invalid run.

## Structural checklist (unit-enforced, tests/test_e3.py)

1. Sandbox gradients reach atom MLPs only; encoder/decoder/composer/keys/pass key
   get none, and `requires_grad` is restored even on exception.
2. Sandbox applications are bit-identical to `model.step_once` chains.
3. Closure predecessors carry no gradient; the standalone branch trains exactly its
   target atom.
4. An identity-behaving atom trips the nonidentity hinge.
5. Zero-usage atoms feel exactly zero uniqueness pressure (loss and gradient).
6. The usage EMA normalizes over live steps: an all-pass batch decays every atom's
   weight by exactly the EMA decay, and only a batch with no live steps leaves it
   untouched. CYCLE carries no autograd graph.
7. No torch RNG is consumed; draws are deterministic per seed with fixed per-step
   consumption; streams 11/12 are registered.
8. E3 configs are A6 + the two lambdas and nothing else.
9. Telemetry is read-only.

## Batch order (run_e3.py)

1. A6 zero-sandbox equivalence gate (hard stop on failure).
2. A11-A13 smoke runs + implementation gates.
3. A11/A12/A13 x seeds 0/1/2, 20k steps, certified panel + analyze + aggregate,
   with the completed E1b A6 arm attached as the labelled reference row.
