# H1 Experiment 10 - Reynolds Transport Network

## Status and interpretation boundary

Registered on 2026-09-12 before any E10 smoke or result-bearing run.

Sources:

- https://github.com/openai/NavierStokesAndEuler
- https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf

This is not a discretization of the Navier-Stokes equations and the cited
theorem implies nothing about neural networks. E10 borrows one constructive
mechanism from the paper rather than its conclusion: localized signed pulses
can have zero mean while their quadratic momentum flux is nonzero; transport
can conserve a global quantity; viscosity damps spatial variation.

E9 asked whether a trained three-transition Atom program could be forced into
one routed call. It could not. E10 abandons both the atom library and the
one-call compression target. It asks whether the *update law itself* can make
an opaque token into a reusable compositional operator.

## Model: Reynolds Transport Network (RTN)

The state is a six-cell one-dimensional field `h in R^(6 x 48)`, one cell per
digit position. A digit embedding and positional embedding initialize the
field. The same token-conditioned cell is then applied once for every real
opaque surface token, in order. A shared pointwise decoder reads one digit
from each final cell. The decoder has no cross-position mixing, so information
can change position only through the registered transport mechanism.

For active token `p`, the cell is split into three named operations:

1. **Transport.** `h_bar[j] = sum_i P_p[j,i] h[i]`.
2. **Viscosity, where enabled.**
   `h_visc = h_bar + nu * (roll_left(h_bar) + roll_right(h_bar) - 2 h_bar)`.
3. **Forcing/reaction.** A shared pointwise MLP conditioned on the opaque token
   and fixed output position adds a residual, followed by per-cell LayerNorm.

No recipe, hidden sub-operation label, target intermediate state, paired
teacher, or held-out task identity enters the model or loss.

### Reynolds pulse parameterization

For every token, A30/A31 learn three pairs of six-cell profiles `(a_r,b_r)`.
Each profile is centered across the six positions. The conceptual pulse set
contains `(+a_r,+b_r)` and `(-a_r,-b_r)`: its signed linear mean is exactly
zero, while its cross-covariance is

`S_p = (1/sqrt(3)) * sum_r a_r outer b_r`.

`S_p + 1.5 I` is divided by temperature `0.5` and projected in log space by
32 alternating row/column normalizations. The resulting `P_p` is doubly
stochastic. Therefore transport conserves the spatial sum of every feature
channel, up to floating-point error. This is the discrete incompressibility
contract tested by E10; it is not a claim about physical incompressibility.

The conventional A29 controller has one free 6x6 logit matrix per token,
also with the fixed `1.5 I` bias and temperature `0.5`, but uses row softmax
only. A29 and the Reynolds controller each have exactly 36 learned transport
scalars per token (A30/A31: `2 * 3 * 6`).

## Arms

| arm | transport | projection | viscosity | purpose |
|---|---|---|---:|---|
| A28 | identity | none | 0 | local-reaction negative control |
| A29 | free 6x6 logits | row softmax | 0 | matched conventional transport |
| A30 | centered pulse cross-covariance | doubly stochastic Sinkhorn | 0 | inviscid Reynolds treatment |
| A31 | same as A30 | doubly stochastic Sinkhorn | 0.04 | full transport/diffusion treatment |

Everything outside the table is shared: width 48, token width 24, position
width 12, force hidden width 96, embeddings, pointwise decoder, data, loss,
optimizer, batch order, and budget. A28 intentionally has no transport
parameters; A29/A30/A31 have matched transport parameter counts.

## Data and optimization

E10 reuses the frozen V2 split and deterministic generator. The eight surface
tokens remain opaque. Training contains the same eight singleton and 34 pair
tasks, with the existing P3 presentation-frequency rule. The loss is only
per-position digit cross entropy on the final answer.

- AdamW, learning rate `5e-4`, betas `(0.9,0.95)`, weight decay `1e-4`.
- Linear warmup for 300 steps, then constant learning rate.
- Batch size 128; global gradient clip 1.0.
- Fixed 6,000-step budget; no early stopping; final checkpoint is the result.
- Screen seed 1. Smoke is 300 steps with reduced data and is never evidence.

## Evaluation

The normal V2 seen-heldout, L1, L2, and L3 sets are evaluated per task and as
macro means. Two new extrapolation panels are generated without training on
them:

- 64 three-token sequences sampled without replacement from the 512 possible
  sequences by fixed selection seed 20260912;
- 64 four-token sequences sampled without replacement from the 4096 possible
  sequences by the same fixed rule;
- 100 fresh inputs per sequence, deterministically derived from the run seed.

The extrapolation manifest stores selected sequences and hashes of inputs and
targets. A hidden-noise panel adds Gaussian noise with sigma 0.15 after every
nonterminal token on all non-excluded pair tasks. The same indexed noise is
used across paired arms. No noise appears in training.

The final audit records every token transport matrix, row/column sum error,
mass-conservation residual on a fixed random field, pulse centering error,
transport entropy and peak, high-frequency energy before/after viscosity,
and state-energy span over a fixed 16-token rollout.

## Frozen gates

`BASE_PASS` requires all of:

- seen hard accuracy >= 0.90;
- L1 >= 0.75;
- L2 >= 0.60;
- L3 >= 0.60;
- three-token extrapolation >= 0.60;
- four-token extrapolation >= 0.50.

`REYNOLDS_PASS` additionally requires maximum mass-conservation residual <=
`1e-5`, and clean long-horizon mean `(triple + quad)/2` no more than 0.03
below A29. This prevents calling a constraint successful merely because it
holds algebraically while destroying the task.

`VISCOSITY_PASS` requires A31 to pass the base and conservation gates, improve
noisy-pair accuracy over A30 by at least 0.05, and lose no more than 0.03 on
clean long-horizon mean. A learned or post-hoc viscosity is forbidden.

## Screen outcomes, in fixed order

1. **LOCAL_REACTION_SUFFICIENT** - A28 passes `BASE_PASS`; transport was not
   required in this world.
2. **VISCOUS_REYNOLDS_FLOW** - A31 passes `VISCOSITY_PASS`.
3. **INVISCID_REYNOLDS_FLOW** - A30 passes `REYNOLDS_PASS`.
4. **STRUCTURED_FLOW_ONLY_WITH_VISCOSITY** - A31 passes base and conservation,
   but the registered A31-vs-A30 viscosity contrast does not pass.
5. **DIRECT_TRANSPORT_ONLY** - A29 passes base while neither structured arm
   reaches its applicable success gate.
6. **NO_FLOW_SUCCESS** - none of the above.

For outcomes 2-4, the first qualifying structured arm (A31 for outcomes 2/4,
A30 for outcome 3) is replicated at seeds 0 and 2. The replicated claim holds
if `BASE_PASS` succeeds in at least two of three seeds and conservation holds
in every structured run. Other outcomes stop after the screen.

## Registered predictions

1. A28 fails, especially on tasks containing position permutations.
2. A29 passes the base gate: sequential token-level transport is the missing
   inductive bias, providing a positive conventional control.
3. A30 passes `REYNOLDS_PASS`: zero-mean pulses can parameterize a useful
   conservative transport operator through their quadratic stress.
4. A31 passes the base gate and improves noisy-pair accuracy, but the fixed
   five-point `VISCOSITY_PASS` margin is deliberately demanding.
5. A30/A31 satisfy the conservation threshold; A29 need not.

## Interpretation guard

A positive A30 result means only that this discrete, low-rank quadratic
parameterization learned reusable conservative operators in the Atom task.
A positive A31 contrast means only that fixed heat diffusion improved the
registered hidden-noise panel without unacceptable clean loss. Neither is a
Navier-Stokes solver, evidence of singularity or turbulence in a neural net,
nor a transfer of any theorem from the cited work.
