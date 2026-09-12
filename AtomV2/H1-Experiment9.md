# H1 Experiment 9 - Vortex Crystallization: Compress a Three-Step Program into One Atom Load

## Source and hard boundary of the analogy

The source is OpenAI's September 2026 forced Navier-Stokes construction and
its Lean certificates:

- [paper](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf)
- [formalization](https://github.com/openai/NavierStokesAndEuler)
- [physical overview](https://openai.com/index/navier-stokes-solution/)

The theorem constructs, for every positive viscosity, a smooth compactly
forced flow whose kinetic energy (an L2 quantity) stays bounded while its peak
velocity (an L-infinity quantity) becomes unbounded in a shrinking region.
The external force is load-bearing. This experiment therefore borrows one
shape of reasoning only:

> Bounded global energy does not rule out increasing local concentration, and
> smooth external forcing can deliberately create that concentration against
> diffusion.

Navier-Stokes proves nothing about neural networks. Here "energy," "core,"
"viscosity," and "forcing" are operational analogies backed by separately
defined neural measurements.

## Why this is the next Atom experiment

E7 and E8 found that the working unit is a three-update token program. A18's
one-route-per-token student reached 14.7% seen accuracy at 20k; A22 reached
only 20.6% at 40k; deeper one-shot atoms did not rescue it. E8's close-out
therefore licensed a training-dynamics intervention: distill or anneal the
learned three-step program into one load.

E9 asks:

> Can a competent, distributed three-update token program be concentrated
> into one pageable atom application by a bounded teacher force, while the
> global state norm and clipped global gradient remain bounded?

This is the useful application of the analogy. If it works, the deployed
student makes one atom-page decision per token instead of three. The teacher
exists during training only.

## Operational map

| Navier-Stokes construction | E9 operational counterpart |
|---|---|
| Bounded kinetic energy, `||u||_2^2` | Mean squared state norm and clipped global gradient norm |
| Growing local speed, `||u||_infinity` | Peak coordinate energy and peak per-atom gradient share |
| Shrinking intense-flow core | Falling inverse-participation support over coordinates/atoms |
| Positive viscosity | AdamW weight decay, normalization, and global gradient clipping |
| Smooth compact force | Linearly ramped, finite teacher boundary/logit loss |
| Collapsing vortex profile | Three teacher transitions compressed into one student transition |

The map is not an equation-to-equation identification. In particular, AdamW
is not a discretization of viscosity.

## Fixed source teacher

Each seed uses its already-completed `e0/A0-free` final checkpoint.

- Same seed and data world as its student.
- Three routing decisions per token.
- Teacher hard seen accuracy must be at least 90% or that seed cannot enter E9.
- The exact checkpoint is SHA-256 pinned in `crystallization.json`.
- Teacher parameters are frozen; no teacher optimizer exists.
- Every same-shaped parameter is copied for warm-start arms. The only shape
  mismatch is `composer.micro_emb.weight`: the one-step student receives row
  zero, the only micro-step it can execute. Any other mismatch is a hard stop.

The completed teachers are healthy at seeds 0/1/2 (98.1%, 97.2%, and 97.0%
seen respectively), so this gate is known to be satisfiable before E9 runs.

## Arms

All students inherit the A0-free world, architecture, optimizer, data, split,
and 20k budget, except `micro_steps = 1`. Every deployed model therefore makes
exactly one route decision and at most one atom load per real token.

| Arm | Initialization | Training loss | Purpose |
|---|---|---|---|
| A25 | Scratch | Task CE only | Exact one-step/scaled-dot control: can this world learn directly? |
| A26 | Copy paired teacher | Task CE only | Unforced contraction control: are trained weights alone sufficient? |
| A27 | Copy paired teacher | Task CE + ramped state/logit force | Controlled crystallization treatment |

For A27 at zero-based training step `s`:

```text
rho(s) = min((s + 1) / 1000, 1)

L = L_task
  + rho(s) * [1.0 * L_boundary_state + 0.25 * L_teacher_logits]
```

`L_boundary_state` is mean squared error between the student's state after its
one transition for a token and the frozen teacher's state after all three
transitions for that token. It is averaged over live token boundaries.
`L_teacher_logits` is teacher-to-student KL at the final output, divided by the
six output positions. The init receipt reports both raw magnitudes and their
weighted ratio to task loss. No coefficient is changed after a result exists.

The force is finite and ramps continuously from a small value. Calling it
"smooth" describes this schedule; it does not assert PDE smoothness.

## Concentration measurements

At every scheduled evaluation, a fixed task-balanced batch is executed with
hard routing. No diagnostic consumes an RNG stream or writes a gradient.

For a state or token update vector `x`, record:

```text
global energy       E2(x) = sum_j x_j^2
peak                P(x)  = max_j |x_j|
peak energy share   C(x)  = max_j x_j^2 / E2(x)
effective support   N_eff = E2(x)^2 / sum_j x_j^4
```

`N_eff` is the inverse participation ratio: it is near the dimension for
uniform energy and near one for a one-coordinate core. The audit records these
for token-boundary states and complete token updates. It also records route
participation and mechanically verifies one live decision per token.

At every training-log step, the pre-clip gradient is partitioned by the 16
atom transforms. Record total gradient norm, its post-clip bound, maximum atom
share, maximum atom RMS, and effective atom support. These measurements expose
a local core that global `grad_norm` alone would hide. They are descriptive,
not a success gate: E9 must not retrofit a preferred concentration curve after
seeing it.

## Registered predictions

1. A25 does not meet the joint deployment gate. E8 makes direct one-step
   learning unlikely, although its different cosine router means A25 remains
   a necessary exact control.
2. A26 improves early optimization but does not retain both competent seen
   behavior and teacher L1 recombination; the learned three-step parameters do
   not become a one-step program merely by deleting two calls.
3. A27 reaches at least 90% seen accuracy and retains at least 80% of its
   paired teacher's L1 accuracy with exactly one route per token.
4. The global state-energy span remains within 1% (LayerNorm machinery
   control). Peak shares and effective supports are reported as trajectories;
   no directional threshold is registered because this is the first calibrated
   use of these instruments.

## Primary gates and outcome bins

A student is deployment-successful only if all three are true:

- final seen hard accuracy is at least 90%;
- final L1 hard accuracy is at least 80% of the paired teacher's L1;
- the diagnostic mechanically observes exactly one route decision per token.

The seed-1 screen is interpreted in this fixed order:

- **ONE-STEP BASE LEARNS**: A25 passes. The important cause is the exact
  scaled-dot base, not teacher concentration.
- **WARM START SUFFICIENT**: A25 fails and A26 passes. Learned weights carry
  the solution; auxiliary forcing is unnecessary.
- **CONTROLLED CRYSTALLIZATION**: A25/A26 fail and A27 passes. The bounded
  teacher force compresses the program into one load.
- **FORCING HELPS BUT INCOMPLETE**: nobody passes, but A27 exceeds A26 by at
  least 10 absolute points on seen or L1.
- **NO ONE-STEP RESCUE**: none of the above.

The first passing mechanism is the registered winner and is replicated at
seeds 0 and 2. Incomplete/no-rescue outcomes stop the experiment.

## Secondary audits

Every result-bearing run receives the existing certified panel. Full runs also
receive the E7 one-step audit unchanged: discovered atom competence, raw and
canonicalized chaining, Interface Closure Ratio, and state-content probes.
Reporting order is deployment gates, teacher retention, E7 component audit,
then concentration curves. L3 remains diagnostic and never leads the verdict.

## Interpretation guard

Even a clean A27 success establishes only this:

> In this Atom world, a fixed teacher and a bounded auxiliary loss can compress
> a learned three-transition token function into one routed module call.

It would not establish spontaneous concentration, finite-time blowup in SGD,
or any implication from Navier-Stokes to neural-network mathematics. A25 and
A26 exist specifically to keep those stronger stories from slipping into the
result.

## Commands

From `AtomV2/Harness`:

```text
python -m pytest tests/test_e9.py -q
python -m atomv2.run_e9 --plan
python -m atomv2.run_e9 --smoke --allow-dirty
python -m atomv2.run_e9 --stage screen
# only if the screen names a winner:
python -m atomv2.run_e9 --stage replicate --arms <winner>
```
