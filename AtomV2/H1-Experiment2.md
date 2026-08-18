# H1 Experiment 2 - Can Interface Noise Force a Shared Atom Language?

## Why this experiment exists

E1b answered its registered routing question cleanly. All A5/A6/A7 runs kept the
router's learning channel alive, yet raw L3 remained at the floor and canonical repair
still restored most downstream performance. Moderate routing exploration in A6 also
improved standalone atom semantics in every paired seed, but did not make the atoms
compose.

The strongest surviving diagnosis is:

> Atoms can acquire individually recognizable behavior while still producing fragile,
> partner-specific handoff states.

Experiment 2 asks whether a generic perturbation to the shared interface can make those
private handoff codes too unreliable to use. It adds noise to states passed between
computation steps and supplies no correct intermediate state, hidden sub-operation,
atom identity, route, atom count, or factorization target.

The primary question is:

> Does training through a noisy shared interface improve clean recombination and closure
> without specifying what any atom should represent?

The secondary question is:

> Is there a noise level that preserves A6's improvement in standalone semantics while
> reducing partner dependence and the canonical repair gap?

## Relationship to the end goal

The eventual system should hold a large atom library outside peak resident memory and
load only a small query-specific subset. Experiment 2 does not yet optimize that
resident set. It tests the prior requirement: independently selected atoms must exchange
states through a shared interface reliably enough to compose.

This experiment remains agnostic about:

- Whether atoms become hidden sub-operations, surface operations, or another reusable
  factorization.
- How many of the 16 available slots become useful.
- Which routes implement any opaque task.
- How many distinct atoms should ultimately be resident for a query.

The only added pressure is that an inter-step message must survive small, unstructured
perturbations.

## Registered base: A6

The control is the completed E1b A6 condition:

```text
router                  cosine
router alpha            3.6838542861871852
router Gumbel sigma     1.082543687...
tau_backward            1.0
lambda_use              0
micro-steps per token   3
training steps          20,000
seeds                   0, 1, 2
```

A6 is selected instead of A5 because moderate routing exploration increased standalone
semantics over A5 in all three paired seeds while leaving mean seen accuracy essentially
unchanged. A7 crossed into destructive exploration.

Registered A6 reference results:

| Metric | A6 mean +/- sample SD |
|---|---:|
| Seen hard accuracy | `71.56% +/- 26.87` |
| L1 hard accuracy | `12.36% +/- 18.16` |
| L2 hard accuracy | `20.40% +/- 8.82` |
| L3 hard accuracy | `0.01% +/- 0.01` |
| Canonically repaired L3 | `86.97% +/- 10.01` |
| Seen closed-map error | `1.0964 +/- 0.0314` |
| Correct-target L3 error | `1.4476 +/- 0.0029` |
| Standalone semantics | `0.3919 +/- 0.2381` |
| Ablation CV | `1.0711 +/- 0.2745` |
| Steps per token / pass rate | `3.0 / 0.0` |
| Stochastic route disagreement | `53.29% +/- 3.98` |

All comparisons are paired by seed. Report every seed; do not select only the A6 seed
that achieved high seen accuracy.

## The one treatment: Gaussian noise at inter-step handoffs

Let the clean state produced by a live micro-step be:

```text
s_clean = LayerNorm(s + selected_delta)
```

If another live computation step follows for that example, transmit:

```text
epsilon       ~ Normal(0, state_noise_sigma^2 * I)
s_transmitted = LayerNorm(s_clean + epsilon)
```

The next composer decision and the next selected atom both receive
`s_transmitted`. The producer output `s_clean` is retained only for diagnostics; it
is not a hidden clean side channel into the next computation.

Registered mechanics:

- Noise is enabled during training only.
- Headline evaluation is clean: `state_noise_sigma = 0`.
- Noise is applied after every live, nonterminal computation step.
- This includes the boundary between token 1 and token 2.
- No noise is applied after an example's final live step; its final clean state goes to
  the decoder.
- Dead/padded steps receive no noise.
- Noise is applied regardless of whether the selected route was an atom or pass. Pass
  cannot become a way to avoid interface corruption.
- Noise is i.i.d. across batch item, sequence position, feature, and handoff.
- Reapply the existing per-position non-affine LayerNorm after adding noise. Noise may
  change direction but cannot create an amplitude side channel.
- Evaluation, repair, probes, ablations, and the certified panel use clean states unless
  explicitly labeled as a noise-robustness diagnostic.

For a LayerNorm state with approximately unit feature variance, the nominal expected
cosine between the clean and noisy-renormalized state is:

```text
target_cosine approximately 1 / sqrt(1 + state_noise_sigma^2)
```

Therefore:

```text
state_noise_sigma = sqrt(1 / target_cosine^2 - 1)
```

This relationship is the registration receipt for the noise levels. Actual cosine after
per-position LayerNorm is logged and verified on a fixed diagnostic batch.

## Arms

| Arm | Description | Target clean/noisy cosine | Derived `state_noise_sigma` |
|---|---|---:|---:|
| A6-reference | Existing A6; no state noise | `1.000` | `0.000000000` |
| A8 | Mild interface noise | `0.999` | `0.044754933` |
| A9 | Moderate interface noise | `0.990` | `0.142492283` |
| A10 | Strong interface noise | `0.950` | `0.328684105` |

Each treatment arm runs seeds `0/1/2` for 20k steps. Total new result-bearing runs:
nine.

The noise levels are geometric interface perturbations, not claims about task difficulty.
Repeated perturbations accumulate through a program, so even A8 is not assumed to be
behaviorally negligible.

## Randomness and paired comparisons

Within a seed, A6/A8/A9/A10 must use:

- Identical model initialization.
- Identical data and batch order.
- Identical underlying routing-Gumbel stream.
- A separate dedicated RNG stream for interface noise.

Adding interface-noise draws must not advance or otherwise alter the routing, data, or
initialization RNG streams. This preserves paired seed-level comparisons.

### Reusing the existing A6 control

The existing A6 runs may be used as the control only if all of the following hold:

1. The new `state_noise_sigma = 0` code path bypasses noise generation completely.
2. A no-noise equivalence test shows bit-identical forward outputs, routes, and losses
   relative to the E1b A6 implementation on fixed inputs and RNG state.
3. No non-noise model, optimizer, data, or evaluation behavior changes.

If any condition fails, rerun A6 seeds `0/1/2` under the Experiment 2 harness before
interpreting treatment comparisons.

## Micro-step decision: remain at three

Experiment 2 keeps exactly three micro-steps per token.

Reasons:

1. The experiment is intended to isolate interface noise.
2. Increasing micro-steps simultaneously increases model capacity, routing decisions,
   Gumbel exposure, and the number of noisy handoffs.
3. Every existing free arm uses all three steps and never selects pass. Without a new
   cost, additional steps are likely to become additional mandatory computation rather
   than optional capacity.
4. A6 already contains atoms with strong surface-operation-like standalone semantics,
   so there is no current evidence that three steps are an insufficient factorization
   budget.
5. More steps create more places for co-adapted fragments and step-smearing to hide.

This does not assert that three is optimal. If interface noise produces a valid arm with
better clean closure and recombination, a separately preregistered capacity experiment
may compare additional micro-step budgets. It must not be folded retrospectively into
Experiment 2.

## Complete delta from A6

| # | Change | Why |
|---|---|---|
| 1 | Add training-only Gaussian noise at nonterminal state handoffs | Make fragile private interface codes unreliable |
| 2 | Re-normalize the corrupted state with existing non-affine LayerNorm | Keep state scale fixed and prevent amplitude coding |
| 3 | Give interface noise an independent RNG stream | Preserve paired routing/data randomness |
| 4 | Add noise-specific telemetry and robustness evaluation | Verify the treatment and distinguish robustness from closure |

Everything else remains identical to A6:

- Same encoder, atoms, composer, decoder, and 384-dimensional interface.
- Same 16 atom slots and pass route.
- Same A6 cosine router and Gumbel exploration.
- Same three micro-steps per token.
- Same data, split, optimizer, batch size, and 20k-step budget.
- Same deterministic hard headline evaluation.
- Same full metric panel.
- `lambda_use = 0`.
- No closure loss, consistency loss, intermediate target, decode/re-encode bottleneck,
  atom dropout, resident-set rent, composer-visibility change, or route planning.

## Smoke tests and implementation gates

Run one smoke test for each of A8/A9/A10 before full runs. Smoke tests may verify
implementation but may not be used to change registered noise levels or predictions.

Required checks:

1. **A6 equivalence:** the zero-noise path passes the reuse conditions above.
2. **Noise isolation:** changing the interface-noise seed changes transmitted states but
   not the pre-noise route Gumbel draws.
3. **No clean side channel:** the next composer and atom receive only the transmitted
   state.
4. **Timing:** noise occurs after each live nonterminal step, including token boundaries,
   and nowhere else.
5. **Masking:** singleton and padded examples receive the correct number of noise events.
6. **Pass behavior:** pass receives the same channel noise as atom routes during training.
7. **Noisy-state scale:** per-position mean remains approximately zero and variance
   approximately one after re-normalization.
8. **Nominal cosine:** on a fixed canonical diagnostic batch with at least eight noise
   draws, the observed median clean/transmitted cosine is within `0.005` of the arm's
   registered target.
9. **Clean evaluation:** repeated clean hard evaluation is deterministic and contains no
   interface-noise draw.

An implementation-gate failure invalidates the run. Optimization failure under correctly
implemented noise is a result, not an invalid run.

## Router liveness remains a validity gate

Retain all E1b liveness telemetry:

- Base route-probability quantiles and registered A6 confidence cap.
- Softmax-Jacobian norm.
- Query/key norms and logit gaps.
- Router, composer, key, and atom gradient norms.
- Router/atom gradient ratio.
- Stochastic route disagreement and unique programs.

Apply the same E1b deafness rule from steps 2k through 18k:

```text
router task-gradient norm < 1e-8
AND
router-gradient / atom-gradient ratio < 1e-3
for two consecutive scheduled evaluations
```

A failed liveness premise invalidates architectural interpretation for that run.

## New noise telemetry

Log at every 1k-step evaluation:

- Clean/transmitted cosine quantiles by arm and handoff index.
- Relative L2 perturbation before and after re-normalization.
- Per-position transmitted-state mean and variance.
- Clean producer-state closed-map error.
- Transmitted-state closed-map error.
- Route flip rate: fraction of deterministic route choices that change when the same
  clean state is replaced by its noisy transmitted version.
- Decoder/output disagreement between repeated noise draws on a fixed batch.
- Gradient norms into the state producer under interface noise.

Keep clean producer metrics separate from transmitted-state metrics. Otherwise an arm
could appear to produce worse states merely because the diagnostic measured the injected
noise instead of the atom's output.

## Registered robustness evaluation

At the final checkpoint, run an evaluation-only interface-noise sweep on every arm,
including the A6 reference:

```text
target cosine: 1.000, 0.999, 0.990, 0.950
```

For noisy evaluation points:

- Use deterministic base routing logits plus the injected state-noise draw; state noise
  may legitimately change later deterministic choices.
- Use the same fixed examples and eight registered noise draws per example.
- Report mean accuracy, route-flip rate, and prediction disagreement.
- Keep clean hard accuracy at target cosine `1.000` as the only headline task result.

The sweep asks whether training noise created a generally robust interface or merely
overfit one noise magnitude.

## Full measurement panel

Run the complete certified panel and report:

1. Seen hard accuracy.
2. Unseen hard accuracy separately for L1/L2/L3.
3. Canonical repaired accuracy and repair gap separately for every level.
4. Generic closed-map error and correct-target error over training.
5. Canonical route agreement and route KL under substitution.
6. Standalone semantics and best-matching candidate functions.
7. Ablation CV and compensation-routing probes.
8. Partner variance and input variance in cross-context transfer.
9. Atom census, steps per token, pass rate, and route load.
10. Deterministic programs per task and routing entropy.
11. Stochastic route disagreement and unique route sequences.
12. Task/sub-operation leakage and transfer probes.
13. Clean-state diversity guards: per-feature variance, effective rank, pairwise state
    distance, and decoded-output diversity.
14. The registered noise-robustness sweep.

The diversity guards detect a trivial strategy in which all atoms emit nearly the same
noise-resistant state.

## What counts as improvement

### H1-relevant success

The desired pattern is:

```text
clean raw unseen accuracy rises
repaired accuracy stays high
repair gap shrinks because raw accuracy rose
correct-target state error falls
standalone semantics stays high
ablation and partner dependence fall
```

The strongest evidence would be the same paired seeds showing both healthier states and
better clean recombination.

### Robustness-only result

If noisy-evaluation accuracy improves but clean recombination, repair gap, and
correct-target error do not, the model learned a noise-resistant interface without
learning a more composable one. This is useful engineering information but does not
support H1.

### Fake repair-gap improvement

This pattern is failure:

```text
raw accuracy unchanged
repaired accuracy falls
repair gap becomes numerically smaller
```

A smaller gap counts only when driven primarily by higher raw accuracy.

### Optimization failure

If seen accuracy falls as noise increases, the noise exceeded what the system can absorb.
Lower generic closed-map error in an underfit arm is not evidence of closure; compare
correct-target error, repaired competence, and clean task performance.

### Robust pidgin

If noise robustness improves but ablation CV, partner variance, and canonical repair gap
remain high, the atoms learned a more redundant private code rather than a shared
composable language.

## Predictions - complete before launch

Registered qualitative lean:

- A8 may be too weak to change the clean solution materially.
- A9 is the most plausible region for improving interface robustness while preserving
  A6's standalone-semantic gains.
- A10 may damage optimization, especially because A6 already experiences substantial
  routing variability.
- Noise alone is unlikely to close L3 completely. A productive partial result would be
  a paired reduction in correct-target error and repair gap with preserved repaired
  competence.

## What Experiment 2 can decide

- **Clean recombination and closure improve with a non-destructive noise level:** fragile
  private handoffs were a causal part of pseudo-compositionality, and generic interface
  robustness is a viable factorization pressure.
- **Standalone semantics remains high but closure does not improve:** independently
  meaningful atoms are still insufficient; a stronger public-interface or
  partner-agreement mechanism is required.
- **Only noisy-evaluation robustness improves:** the system learned a robust pidgin.
- **All treatment arms lose seen performance:** interface noise compounds A6's
  optimization instability and is not a viable pressure at these scales.
- **A10 fails while A8/A9 remain healthy:** there is a usable robustness/optimization
  frontier rather than a monotonic benefit from noise.
- **A valid treatment materially improves H1 metrics:** preregister the separate
  micro-step capacity experiment before changing the three-step budget.

## Explicitly outside Experiment 2

Experiment 2 does not:

- Penalize micro-step count.
- Charge distinct resident atoms.
- Reward cross-task atom reuse directly.
- Remove current-state access from the composer.
- Plan a token's routes upfront.
- Decode and re-encode at atom or token boundaries.
- Train a manifold discriminator, projector, or state-validity loss.
- Add intermediate supervision or oracle routing.
- Specify any atom's semantics or the number of atoms that should emerge.

Those remain separate causal interventions. If noise alone is insufficient, the next
highest-priority architectural comparison is current adaptive routing versus
token-boundary route planning, because E1b showed malformed states can alter downstream
route selection.
