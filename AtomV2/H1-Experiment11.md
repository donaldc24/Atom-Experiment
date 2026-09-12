# H1 Experiment 11 - Canonical Conservative Exchange Network

## Status and interpretation boundary

Registered on 2026-09-12 before any E11 smoke or result-bearing run.

Sources motivating the analogy:

- https://github.com/openai/NavierStokesAndEuler
- https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf

E11 is not a Navier-Stokes solver, and no theorem about fluids is transferred
to learning. It tests a narrower engineering idea: globally conservative
motion can be assembled from local pairwise exchanges; a quadratic interaction
of zero-mean signed profiles can control those exchanges; and stable
composition may require a canonical interface at every operator boundary.

## Why E11 follows E10

E10's screen was a useful failure. Its trained Reynolds/Sinkhorn arms achieved
seen accuracy near 0.4 but collapsed on long compositions. More importantly,
their singleton mean was about 0.176 while trained-pair performance was much
higher: token two learned to exploit token one's private hidden state. The
iterative projection also missed the registered conservation gate after its
logits sharpened. E11 addresses those two observed mechanisms directly rather
than adding another loss to the same model.

## Model: a categorical fluid

The state is `q in [0,1]^(6 x 10)`. Each of the six cells holds a distribution
over digits and each row sums to one. An input list is its exact one-hot state.
For each opaque token, three operations occur in order.

### 1. Exact pairwise transport

Eight stages alternate the disjoint ring matchings

- `(0,1), (2,3), (4,5)`;
- `(1,2), (3,4), (5,0)`.

For every matched pair `(i,j)` and gate `g in [0,1]`, all ten channels update
as

`[q_i';q_j'] = [[1-g,g],[g,1-g]] [q_i;q_j]`.

Every block is exactly doubly stochastic for all finite gate logits. It
therefore conserves the spatial sum of every digit channel without a Sinkhorn
iteration, convergence tolerance, entropy penalty, or sharpness tradeoff. A
finite reachability audit performed before registration found that this fixed
schedule reaches all `6! = 720` permutations when gates are binary.

A32 learns one free logit for each token/stage/edge. A33/A34 learn two
six-position profiles for each token and stage, center each profile, and derive
the `(i,j)` gate from

`sigmoid(-2 + 4 * (a_i b_j + a_j b_i)/2)`.

The conceptual signed ensemble `(a,b),(-a,-b)` has exactly zero linear mean
while retaining the same quadratic stress. Pulse values are unrestricted, so
gates can approach either binary limit.

### 2. Optional viscosity

A34 alone applies periodic heat diffusion

`q_j <- 0.92 q_j + 0.04 q_(j-1) + 0.04 q_(j+1)`.

It is another nonnegative conservative map. A32/A33 have viscosity zero.

### 3. Categorical reaction and canonicalization

Each opaque token and output position owns a learned row-stochastic 10x10
reaction kernel. Multiplying a cell distribution by this kernel maps a digit
distribution to another digit distribution without cross-position mixing.
Reaction logits use a fixed identity bias 2.0 plus learned noise of scale 0.02.

After every nonterminal token, rows are normalized and projected to their
argmax one-hot digit in the forward pass. Training uses the standard
straight-through estimator: the forward value is hard and the backward
derivative is that of the soft distribution. Consequently downstream tokens
cannot consume a producer-specific hidden dialect. They receive exactly the
same six one-hot digits as a fresh model input. The loss is final-answer digit
negative log likelihood only; no intermediate label or supervision is used.

## Arms

| arm | exchange gates | boundary | viscosity | purpose |
|---|---|---|---:|---|
| A32 | direct logits | categorical hard canonicalization | 0 | positive scaffold control |
| A33 | centered quadratic pulses | same | 0 | main Reynolds-exchange treatment |
| A34 | centered quadratic pulses | same | 0.04 | conditional viscosity treatment |

All other mechanics are shared. A32 is not parameter matched: its role is to
show whether the exact-exchange/canonical scaffold can solve the task at all.
A33 tests whether the more constrained quadratic pulse controller can match it.

## Data and fixed curriculum

The frozen V2 split and deterministic data generator are reused; tokens remain
opaque. E11 deliberately removes the old P3 presentation oversampling and
samples tasks uniformly within each curriculum pool.

- Phase 1: exactly 4,000 steps drawn uniformly from the eight singleton tasks.
- Phase-1 gate: at the final fixed checkpoint, every held-out singleton task
  must have at least 0.99 exact-list accuracy. A failing arm stops; there is no
  repeated-look extension and no Phase 2 for that arm.
- Phase 2: exactly 4,000 further steps. Every batch contains 64 uniformly
  sampled singleton examples and 64 uniformly sampled examples from the 34
  trained pairs. The optimizer is continuous across phases.
- AdamW; learning rate 0.01; betas `(0.9,0.99)`; weight decay 0; 200-step
  linear warmup; global gradient clip 5.0; screen seed 1.
- Full training/evaluation sizes remain 1,000/400 examples per registered task.
  Smoke uses reduced data and 150+150 steps, ignores the Phase-1 stop solely to
  exercise Phase-2 code, and is never evidence.

Sampling indices are generated before training from named, arm-independent
streams and their hashes are stored, so A32/A33/A34 see identical seed-paired
presentations.

## Fixed evaluation

Standard exact-list accuracy is reported separately for held-out singleton,
trained-pair, L1, L2, and L3 tasks. Long panels contain 64 fixed, unique opaque
token sequences at each length 3, 4, 8, and 16, with 100 fresh deterministic
inputs per sequence. Selection is arm-independent and uses seed 20260912;
input/target hashes are stored.

The following audits are fixed:

- maximum spatial digit-channel mass residual after every exchange stage;
- effective composite transport row/column sums and argmax permutation;
- centered/signed-pulse mean error and learned gate sharpness;
- interface closure: compare ordinary two-token execution with decoding the
  first token, restarting from those one-hot digits, and applying token two;
- sigma-0.15 Gaussian noise immediately before boundary canonicalization;
- symmetric per-tensor INT8 and INT4 quantization of learned weights;
- one fixed dropped boundary cell (replaced by uniform) and one fixed corrupted
  cell (replaced by a deterministic wrong digit), with exact-list and per-cell
  accuracy reported;
- clean length-8 and length-16 rollouts.

## Frozen gates and outcomes

`PRIMARY_PASS` requires all of:

- final held-out singleton minimum >= 0.99;
- trained-pair mean >= 0.95;
- L1, L2, and L3 each >= 0.90;
- length 3/4/8/16 >= 0.85/0.80/0.65/0.45;
- exchange mass residual <= 1e-6;
- interface prediction agreement exactly 1.0.

`REYNOLDS_PASS` requires A33 `PRIMARY_PASS` and its mean over the four long
panels to be no more than 0.03 below A32. If it passes, A34 is run; otherwise
the experiment stops after A32/A33.

The robustness composite is the unweighted mean of Gaussian noisy-pair exact,
INT4 pair exact, dropped-cell pair exact, corrupted-cell pair exact, length-8,
and length-16 accuracy. INT8 is reported but excluded to avoid counting two
versions of the same perturbation. `VISCOSITY_PASS` requires A34
`PRIMARY_PASS`, a four-long-panel clean mean no more than 0.03 below A33, and a
robustness-composite gain of at least 0.03.

Outcome order:

1. `VISCOUS_CANONICAL_REYNOLDS` - A34 passes the viscosity contrast; winner A34.
2. `INVISCID_CANONICAL_REYNOLDS` - A33 passes but A34 does not; winner A33.
3. `DIRECT_CANONICAL_EXCHANGE_ONLY` - A32 passes but A33 does not.
4. `SINGLETON_CRYSTALLIZATION_FAILED` - neither screen arm passes Phase 1.
5. `NO_REUSABLE_EXCHANGE` - all other failures.

A structured winner is replicated at seeds 0 and 2. The claim holds when
`PRIMARY_PASS` succeeds in at least two of the three total seeds, conservation
passes at all three, and interface agreement is exactly 1.0 at all three.

## Registered predictions

1. A32 and A33 pass the fixed singleton crystallization gate.
2. A32 passes `PRIMARY_PASS`, establishing the categorical exact-exchange
   scaffold as a positive control.
3. A33 passes `REYNOLDS_PASS`: the quadratic centered-pulse gates match direct
   exchanges while retaining algebraic conservation.
4. Boundary canonicalization keeps singleton performance above 0.99 during
   pair training and gives exact explicit-restart agreement.
5. A34 mechanically damps variation but does not achieve the demanding 0.03
   robustness gain; the predicted structured winner is A33.

## Interpretation guard

A positive result establishes only that this small categorical conservative
operator learned reusable transformations in the synthetic Atom world. The
terms pulse, stress, transport, conservation, and viscosity name construction
analogies. They do not establish a fluid simulation, a singularity mechanism,
or any implication for the Navier-Stokes existence and smoothness problem.
