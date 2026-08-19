# E1b - Does Free Routing Still Fake It When the Router Stays Awake?

## Why this experiment exists

Every free-routing result so far has an asterisk: in the existing runs, the router's
task-loss gradient collapsed at roughly 2k steps and routing was effectively frozen for
most of training. E1b removes that asterisk by replacing the saturating router with one
whose logit geometry is bounded, then tests three preregistered levels of persistent
exploration.

The primary question is:

> Does pseudo-compositionality persist when the free-routing learning channel remains
> measurably live?

The secondary question is:

> If behavior improves, does it improve as persistent routing exploration increases?

E1b does **not** prescribe whether atoms should represent hidden sub-operations, surface
operations, or some other factorization. It asks only whether the learned pieces are
independently meaningful, reusable, and composable. There is no rent intervention here:
`lambda_use = 0` in every free arm.

## Registered router

Let `q` be the composer's 32-dimensional query and `k_j` the key for route `j`, where
there are 17 routes: 16 atoms and one pass route.

### Bounded base logits

For every routing decision:

```text
q_hat   = q / (||q||_2 + epsilon)
k_hat_j = k_j / (||k_j||_2 + epsilon)
z_j     = alpha * dot(q_hat, k_hat_j)
```

Use the same fixed numerical `epsilon` in every arm. Because the normalized dot product
is in `[-1, 1]`, every base logit is in `[-alpha, alpha]`; learned query/key norm growth
can no longer create an unbounded logit gap.

### Hard forward choice and straight-through backward path

Training uses:

```text
g_j   ~ Gumbel(0, 1)
hard  = one_hot(argmax_j(z_j + sigma * g_j))
soft  = softmax((z + sigma * g) / tau_backward)
route = hard + soft - stop_gradient(soft)
```

Registered common constants:

- `alpha = 3.683854...` in every arm. This fixes the learned cosine-logit geometry.
- `sigma` is the only arm-specific router constant. It controls forward exploration.
- `tau_backward = 1.0`. This controls the straight-through surrogate gradient.
- No temperature annealing.
- Training forward routing remains hard top-1.
- Hard evaluation is deterministic `argmax(z)` with no Gumbel noise and is the headline
  result.
- Soft evaluation uses `softmax(z / 1.0)` with no Gumbel noise. This replaces the old
  use of `tau_end`. The E1b soft-hard gap is reported, but is not numerically
  apples-to-apples with A1's annealed-temperature soft-hard gap.

Forward noise and backward temperature are deliberately named separately because they
control different behaviors. The cosine scale and backward temperature are fixed so that
the only A5/A6/A7 configuration difference is Gumbel noise `sigma`. The sampled
straight-through proxy necessarily sees that same arm-specific Gumbel draw, but its
temperature and the learned logit geometry do not change across arms.

## Deriving the exploration levels

With 17 choices, the largest possible base-logit separation places one route at
`+alpha` and all 16 alternatives at `-alpha`. Under Gumbel-max sampling with noise
scale `sigma`, the largest possible marginal probability of selecting that preferred
route is:

```text
p_max = exp(2 * alpha / sigma) / (exp(2 * alpha / sigma) + 16)
```

First register A5 as the low-exploration anchor with `p_max = 0.99` and `sigma = 1`.
This fixes the shared cosine scale:

```text
alpha = 0.5 * ln(16 * p_max / (1 - p_max))
      = 0.5 * ln(16 * 0.99 / 0.01)
      = 3.683854...
```

Then keep `alpha` fixed and solve for each arm's Gumbel scale:

```text
sigma = 2 * alpha / ln(16 * p_max / (1 - p_max))
```

`p_max` therefore caps the **base/marginal hard-choice probability**, not the probability
in an individual post-Gumbel soft sample. Gumbel noise is unbounded, so an individual
sampled surrogate can still be arbitrarily close to one.

A two-token program makes six routing decisions. At maximum confidence, the probability
that at least one decision differs from the deterministic preferred route is:

```text
P(any exploratory decision in six) = 1 - p_max^6
```

| Arm | `p_max` | Shared `alpha` | Derived `sigma` | Nonpreferred choice per decision at max confidence | At least one nonpreferred choice in six decisions |
|---|---:|---:|---:|---:|---:|
| A5 - low exploration | `0.9900` | `3.683854` | `1.000000` | `1.00%` | `1 - 0.99^6 = 5.85%` |
| A6 - moderate exploration | `0.982593...` | `3.683854` | `1.082544` | `1.74%` | `1 - 0.982593...^6 = 10.00%` |
| A7 - high exploration | `0.9000` | `3.683854` | `1.482492` | `10.00%` | `1 - 0.90^6 = 46.86%` |

A6 uses `p_max = 0.90^(1/6) = 0.982593...`; displayed values are rounded only for
readability. The implementation must derive `sigma` from the full-precision registered
values rather than copying the rounded table.

These percentages describe maximally separated logits. Less-confident states will be
more variable. A7 is intentionally a strong partner-unpredictability condition, not
merely a technical anti-saturation fix.

## Deltas from the certified rig

| # | Change | Why |
|---|---|---|
| 1 | L2-normalize queries and keys | Prevent learned norm growth from producing unbounded base-logit gaps |
| 2 | Replace dot-product scaling and temperature annealing with one shared fixed cosine scale | Bound learned logit geometry identically in every arm |
| 3 | Set only Gumbel `sigma` from each arm's registered exploration target | Make forward exploration explicit while holding learned logits fixed |
| 4 | Fix straight-through temperature at `tau_backward = 1.0` | Keep the surrogate definition stable; do not conflate it with forward exploration |
| 5 | Define deterministic hard eval and no-noise soft eval separately | Preserve a clear headline metric and make the changed soft-eval semantics explicit |
| 6 | Add preregistered liveness and variability telemetry | Verify the premise instead of assuming it |

This is one conceptual anti-saturation routing-stack intervention, not one literal code
change. All non-router model, data, optimizer, batching, checkpoint, evaluation-set, and
training-budget values remain identical to the certified E1 configuration. All free arms
use `lambda_use = 0`.

## Arms and run order

1. **A5-oracle regression check, seed 0:** forced oracle routing under the normalized
   routing stack and shared cosine scale. This must retain the certified oracle task behavior:
   approximately perfect accuracy and closed-map error near `0.015`.
2. **A5/A6/A7 free-routing smoke tests:** one short run of each arm through at least the
   first two scheduled liveness evaluations. These are implementation/liveness checks,
   not results and cannot be used to tune the registered constants.
3. **A5 - low exploration:** free routing, seeds `0/1/2`, 20k steps.
4. **A6 - moderate exploration:** free routing, seeds `0/1/2`, 20k steps.
5. **A7 - high exploration:** free routing, seeds `0/1/2`, 20k steps.

The oracle arm is only a broad architectural regression check. Forced routing bypasses
the actual Gumbel selection used by the free arms, so it does not certify free-router
liveness. The free smoke tests and telemetry do that.

Total result-bearing runs: nine free runs. The one oracle run is a regression check and
is not included in comparisons among A5/A6/A7.

Within each seed, A5/A6/A7 use the same initialization, data order, and underlying
unit-Gumbel random stream; only the registered `sigma` scales those draws. Analyze the
three arms as paired seed-level comparisons where possible.

## Liveness telemetry and validity gates

Use one fixed diagnostic batch at every 1k-step evaluation. For stochastic measurements,
evaluate the same examples under eight independently seeded Gumbel draws. Log both the
median and the full range across draws. Telemetry must distinguish the following.

### 1. Base-router geometry

- Maximum and quantiles of `softmax(z / sigma)` route probability.
- Fraction above `0.999` and `0.999999`, and fraction numerically equal to one.
- Top-1 minus top-2 base-logit gap.
- Frobenius norm of the base-softmax Jacobian `diag(p) - p p^T`.
- Raw, pre-normalization query and key norm quantiles.
- Maximum absolute base logit.

**Implementation invariant:** each arm's maximum base probability must be no greater than
its registered `p_max + 1e-6`. Violation means the implementation is wrong and the run is
invalid.

Do not apply this cap to post-Gumbel surrogate probabilities; individual Gumbel samples
are not bounded by `p_max`.

### 2. Actual learning signal

- Task-loss gradient norm into all router parameters.
- Task-loss gradient norm into composer parameters and into route-key parameters,
  reported separately.
- Atom gradient norm.
- Router-gradient / atom-gradient ratio.
- All absolute norms as well as ratios, so a vanishing denominator cannot fake liveness.

Historical A1 gradient values are descriptive references only. Before launch, record the
exact source artifact/checkpoint for every quoted A1 reference value.

**Deafness rule:** from steps 2k through 18k, a run is invalid if, for two consecutive
scheduled evaluations, both of these hold on the median of the eight diagnostic draws:

```text
router task-gradient norm < 1e-8
router-gradient / atom-gradient ratio < 1e-3
```

The conjunction prevents a small atom-gradient denominator from manufacturing a pass.
Steps after 18k are excluded from this validity rule because task gradients may
legitimately vanish at convergence, but they remain logged.

If raw query or key norms grow by more than 10x their step-1k median for two consecutive
evaluations, flag the run for a normalization-Jacobian audit. This is an interpretive
warning rather than automatic invalidation unless the deafness rule also fires.

### 3. Two different kinds of routing diversity

- **Stochastic execution diversity:** for each fixed input, number of unique hard route
  sequences and route-disagreement rate across the eight Gumbel draws.
- **Deterministic content-conditional diversity:** with Gumbel disabled, number of unique
  hard route sequences observed for a task across its fixed evaluation examples.
- Per-task deterministic routing entropy.
- Atom/pass load, route census, steps per token, and pass rate.
- Pass probability, pass rank, and pass-to-winner logit gap.

“Programs per task” means the number of unique deterministic six-choice hard route
sequences observed for a two-token task across the same fixed number of evaluation
examples. Singleton tasks analogously use three-choice sequences. Sample count must be
identical across arms and seeds.

**Validity rule:** each seed-run is judged independently. If a run violates either the
implementation invariant or the deafness rule, that run is invalid. If any seed in an arm
fails the deafness rule, that arm has failed E1b's verified-liveness premise: report all
telemetry and valid seed-runs descriptively, but do not make an arm-level compositionality
claim or silently replace the failed seed.

## What we measure

Run the full certified panel. Headline comparisons are:

1. **Repair gap**: canonical-repair accuracy minus raw accuracy, per split level. A1's L3
   gap was approximately 98 percentage points. Does verified-live routing shrink it?
2. **Raw hard accuracy**: seen held-out and unseen L1/L2/L3, never averaged across levels.
3. **Closed-map error and trajectory**: A1 ended near `1.05` and rose during training;
   oracle was near `0.015`.
4. **Standalone semantics, ablation consistency, census, steps per token, and pass rate**.
5. **Seed stability**: A1's L1 result varied by approximately 15 points across seeds.
6. **Deterministic content-conditional programs per task**.
7. **Stochastic execution diversity on identical inputs**.
8. **Canonical substitution/repair and downstream repaired accuracy**.
9. **Soft-hard gap**, with the explicit warning that E1b soft evaluation has different
   semantics from A1 soft evaluation.

Primary comparisons are each of A5/A6/A7 against A1. The secondary comparison is the
ordered exploration response A5 -> A6 -> A7. With only three seeds per arm, report all
seed-level values and descriptive intervals; do not turn absence of statistical power
into a categorical equivalence claim.


## What E1b can decide

- **A5 approximately matches A1 while passing liveness:** router freeze was not sufficient
  to explain pseudo-compositionality. The defensible claim is that pseudo-compositionality
  persists under this architecture, objective, and data regime with a verified-live
  router.
- **A5 is materially healthier than A1:** the anti-saturation routing stack improves the
  outcome. This does not by itself identify whether the cause was restored learning,
  persistent exploration, or both.
- **A6/A7 improve monotonically over A5:** partner unpredictability is a plausible active
  pressure against co-adaptation, not merely telemetry insurance. A later ablation must
  isolate forward exploration from backward-channel liveness.
- **A7 degrades while A5/A6 do not:** excessive persistent exploration harms optimization;
  this does not vindicate frozen routing.
- **All valid arms retain a large repair gap and poor L3:** fixing routing alone does not
  fix state production or closure. The later closure/rent experiment remains necessary.
- **Any arm fails its liveness gate:** that arm cannot answer the architectural question.

E2 or any rent/closure redesign starts only after these registered results land. E1b tests
the router and the exploration dose; it does not modify the atom objective or tell the
atoms what kind of reusable factorization to discover.
