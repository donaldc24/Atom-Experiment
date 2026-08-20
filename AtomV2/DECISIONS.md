# DECISIONS
Key decisions as they are made during experiments

## Amendment 1: Calibration Verdict Redefined
E0 formally failed 2 of the 4 registered checks, but after verifying the E0 A0-Oracle passed cleanly and as expected and doing full audit to verify no architecural or leakage issues in A0-Free the real failure was in predicition not machinery. Re pointed the gap criteria at specifically L3 as that is where the real leakage would have shown

## Amendment 2: A0-Free promoted from anomaly to finding
With Pseudo-compositionality (dissociation 0.798, closed-map 1.05 rising through training in the slow-drift shape, standalone 0.096, steps 3.0 with pass never used) combined with seed brittleness observation and the results showing Redhardts results, connected task support lets a task conditional monolith generalize at L1/L2 without any reusable structure

## Amendment 3: L3 Interpretation Corrected
Originally thought L3 would require sub-op granularity but the Oracle L3 = 1.0 actually shows it demands closure.

## Amendment 4: Panel Changes
- Decodability relabled task-identity leakage
- Transfer split sub op probes added as the true granularity instrument
- Canonical Substitution Test added with both readouts (routing agreement, and repair rate) and both conditioning variants
- Transplant metrc decomposed into partner variance vs input variance with conditional routing rationale

## Amendment 5: A1 scored
The prediction vs actual table since A0-Free was just A1, also updated my predicition for E1 A2-4

## Amendment E4-G1 (2026-08-19): A12 gate failed as written, satisfied in purpose
E4's sandbox-only replay gate demands bit-identity against the completed E3
A12 run, but E3 trained in the second worktree under torch 2.7.1+cu118 /
numpy 1.25.2 while the merged harness runs torch 2.9.0+cpu / numpy 2.2.4 -
kernel rounding differs at the ulp level across builds, so strict bit-identity
is unattainable regardless of code fidelity. Evidence backing the substitution
(D18 lineage): harness source byte-identical for every sandbox module (empty
git diff), step-1 records bit-identical, step-50 records within one float32
ulp (~2.4e-7), and grad norms bit-identical at both steps (the
weights-identical witness: d(mean)/dx = 1/N regardless of the forward
reduction's summation order). Substituted operational form registered as
E4_GATE_REL_TOL = 1e-6 in registered.py; applies ONLY when the reference
env.json documents a different torch build. The cross-build caveat travels
with every A14/A12 comparison; A9/A6 references are same-build and stay under
the strict gate.

## E5 registration (2026-08-19): producer branch design, K = 4, gaming guards
Owed with H1-Experiment5.md, recorded before the first E5 run.
- Producer branch design: z_out = Apply_i(z0) from the SAME stopgrad(z0) the
  sandbox already uses, target atom i drawn uniformly per step. K independent
  frozen continuation chains (length ~ U{1..6}, drawn at full length, only
  the first k applied - fixed RNG consumption) run from z_out;
  L_producer = mean_k READ(chain_k(z_out)) against the frozen decoder.
  Gradient mechanism: the emission is built with atom MLPs trainable
  (encoder/decoder/composer/keys frozen, the existing sandbox boundary); the
  chains and READs are built with EVERY parameter frozen, so z_out is the
  only gradient-bearing leaf downstream and gradient flows THROUGH the
  frozen activations into the producer atom only. Chain atoms and the
  decoder receive exactly zero gradient (unit-enforced in tests/test_e5.py).
- K = 4 rationale: one listener per step lets the producer chase partner-
  specific slang step to step; K simultaneous listeners force the
  intersection - a state broadly usable, not privately usable. Same expected
  objective, far lower variance, no listener-chasing.
- Gaming guards (named before launch): task loss (a constant state fails
  composition on seen tasks immediately), inherited E3 uniqueness
  fingerprints, NEW per-atom producer-output-variance telemetry row with the
  z0 variance as reference floor (a collapsing producer must announce
  itself), NEW branch-READ-spread free metric (context-sensitivity remaining
  in production), logged per eval.
- Streams: 'e5_producer' = 13 (training), 'e5_producer_eval' = 14 (init
  calibration + telemetry), numpy-only; zero torch RNG consumed, so a
  producer-bearing run's routing/noise/sandbox/init draws stay bit-identical
  to its A14 pair.

## E5 base + environment note (2026-08-19)
- Base: E5 launches on A14 with E4 seeds 0 and 1 complete (A14 best-arm
  story intact at both); the seed-2 and A15 pairs are still training on the
  second machine. Per the spec's base clause, if the completed E4 pairs
  break seed 0's story, this receives a dated amendment and E5's
  interpretation is re-based before results are read.
- Environment: E5 trains on hostname Perro (python 3.13.1, torch 2.9.0+cpu,
  numpy 2.2.4); every prior battery trained on Perrito (python 3.11.5, same
  torch/numpy builds for e1b/e2/e4). The zero-path gate decides mechanically
  whether A14's records reproduce here; its record
  (results/e5/e5_a14_equivalence.json) documents the outcome and any
  environment difference, per the cross-env three-part test the E5 spec
  registered (E4-G1 lineage). Aggregation never pools across hostnames; A14
  attaches as a labelled reference row only.

## Amendment E5-G1 (2026-08-19): A14 zero-path gate failed as written, satisfied in purpose - PENDING RATIFICATION
The E5 zero-path gate as registered demands step-1 bit-identity and grad-norm
bit-identity even in its cross-env form (E4-G1). Both are unattainable here
because the environment difference is the MACHINE itself: E5 runs on Perro
(Intel, python 3.13.1) against A14 records from Perrito (AMD, python 3.11.5),
identical torch 2.9.0+cpu / numpy 2.2.4 builds. Backward-pass reduction
kernels dispatch differently per CPU microarchitecture, so grad norms differ
at the ulp level regardless of code fidelity.

Evidence recorded in results/e5/e5_a14_equivalence.json:
- step-1 FORWARD records bit-identical (loss, loss_task, batch_acc,
  loss_sandbox_valid, loss_sandbox_unique): identical weights, data and RNG
  streams - the code-fidelity witness available cross-machine.
- The only step-1 mismatch is grad_norm at 7.3e-8 relative (backward
  reduction order); by step 50 the worst drift is 1.5e-7 relative. All four
  mismatching records are within E4_GATE_REL_TOL = 1e-6.
- Unit-level structure (tests/test_e5.py): the A16/A17 configs differ from
  A14 in nothing but identity strings + lambda_producer; the producer
  consumes zero torch RNG and a dedicated numpy stream, so the zero-path
  cannot differ by construction.

Substituted operational form (applies ONLY when the environment difference
includes a different hostname/processor): step-1 non-grad records
bit-identical, every mismatching record within E4_GATE_REL_TOL relative,
environment difference documented in the gate record. When only the build
differs, the stricter E4-G1 form (grad norms bit-identical) still applies.

Consequence that travels with every E5 result: A16/A17 (Perro) vs A14
(Perrito) comparisons are NOT bit-paired by seed - environmental ulp drift
compounds over 30k steps, so seed pairs are same-seed draws, not identical
trajectories up to the treatment. This is the same caveat class E4 accepted
for A14/A12 under E4-G1. Made at launch by the harness operator; if vetoed,
the alternatives are (a) re-run E5 on Perrito after E4 completes, or (b)
train a same-env A14 reference on Perro - the A16/A17 runs themselves remain
valid under either choice.

## A17 CUT (2026-08-20)
Structural unreachability cannot be fixed by dose. A17 (the higher
lambda_producer cell) is cut without being run; the dated rationale is D1's
finding that the dax wall is not a dose-reachable failure (see below and
results/d1/). No result from A17 would have changed the mechanism.

## Gate-ladder freeze (2026-08-20)
The equivalence-gate ladder is frozen, permanently, in this order of
preference: (1) strict bit-identity; (2) E4-G1 three-part cross-BUILD form
(step-1 bit-identical, drift within E4_GATE_REL_TOL, grad norms
bit-identical, build difference documented); (3) E5-G1 forward-only
cross-MACHINE form (step-1 non-grad records bit-identical, all mismatches
within tolerance, hostname/processor difference documented). No new gate
forms are minted mid-flight; a comparison that cannot pass any rung is not
run.

## E5 discipline break, named (2026-08-20)
E5 launched without registered predictions - the only battery to do so. The
cost: the A16~=A14 REDUNDANT outcome cannot be scored against a prior, so
E5 contributes mechanism evidence (producer READ pressure does not move dax)
but no forecast-calibration evidence. Noted as the discipline break it was.

## A16 seeds 0-1 vs A14: REDUNDANT bin confirmed on local pairs (2026-08-20)
A16 s0 seen 96.3 / L1 52.8 vs A14 s0 95.7 / 56.5; A16 s1 97.8 / 36.7 vs
A14 s1 98.9 / 43.2; dax 0.0000 / 0.0025 vs 0.0000 / 0.0025. Combined ~= best
single at both seeds; producer pressure adds nothing on top of A14's stack.
Seed 2 is not present in this repo (trained on Perro); its line lands with
the E5 sync.

## D1 verdict (2026-08-20): CONDITIONING INNOCENT - and the repair metric is
## no longer mysterious
Read-only diagnostic over A6/A14/A16, all local seeds (results/d1/).
- P-A: token-1 execution of every dax cell is BIT-IDENTICAL to its singleton
  (routing and produced state, verified per cell, zero violations across 8
  runs x 63 cells). The conditioning suspect had no object to act on: the R8
  encoder is digit-only, and the premise "task tokens are baked into h0" is
  false for this architecture.
- P-B: the P3-first boundary state decodes to the correct intermediate at
  97.3-100.0% (exactly singleton accuracy - it IS the singleton state) while
  raw composition sits at 0. The state is right and readable by the frozen
  decoder; the consumer cannot use it. Conditioning innocent; closure story
  confirmed on the CONSUMER side.
- P-C: dax boundary states sit 15-25% farther from the trained-consumption
  pool than trained/L1/L2 states in every run - and L1 ~= L2 ~= trained, so
  consumption-manifold geometry explains the L3 wall but NOT the L1/L2
  ordering.
- Mechanism resolution: every program's valid input domain is {canonical
  codes} (its token-1 role trains it on enc-states extensively) UNION
  {products of its trained partners}. Canonical repair works because
  enc(P3(x)) looks like an h0 - in-domain for every program via its token-1
  role. Dax fails because P3's products are in NEITHER set for any consumer.
  The repair metric was never mysterious: it swaps an out-of-domain private
  state for a universally in-domain canonical one.
- Fix direction this licenses: emission toward the canonical family (or
  consumer exposure to unfamiliar products) - architectural/curricular, not
  another dose.

## D2 registration (2026-08-20): Atom Factorization Audit (read-only)
Registered BEFORE any probe ran. The question is causal atom factorization,
not task accuracy. Definition fixed up front:

> An atom is factorized if it implements the same primitive transformation
> across inputs and contexts, and those atoms can be recombined to execute
> novel programs without relying on surface-specific co-adaptation.

Semantics are defined on the readout channel, dialect-independently:
atom i "implements op f" on state h iff decode(A_i(h)) == f(decode(h)),
where decode is the model's OWN frozen decoder readout. Probes, all
@no_grad on final checkpoints (roster: A6 s0-2, A14 s0-2, A16 s0-1):

- P-1 operator signature: one application on canonical states, match rates
  against all 7 sub-ops + 8 surface ops + identity. Sharp = best rate
  >= 0.95 (D2_SHARP); mapping threshold for P-3 = 0.90 (D2_MAP_THRESHOLD).
- P-2 context invariance: the same test with h drawn from other programs'
  states - singleton mid-token states (micro-steps 1, 2) and token-boundary
  states of trained pairs, L1, L2, L3. Invariant = every context group
  >= 0.80 (D2_INVARIANT_MIN) for that atom's P-1 best op.
- P-3 composer-free recomposition (the guillotine): execute
  encoder -> mapped atoms -> decoder from the empirically discovered
  mapping only - no composer, no tokens, no routing - at both
  granularities (sub-op chains; surface chains). Coverage reported; novel
  recomposition "works" = forced accuracy >= 0.50 (D2_RECOMB_WORKS) on the
  unseen level in question.
- P-4 atom swaps: for any two atoms sharing a sharp signature, substitute
  one for the other in P-3 programs and in forced replay of the routed
  model's own recorded programs. Interchangeable = no collapse. Absence of
  duplicates is recorded, not scored.
- P-5 selective ablation: existing panel ablation damage regrouped into an
  atom x primitive matrix (tasks containing vs not containing the atom's
  P-1 op). Selective = in/out mean-damage ratio >= 3 (D2_SELECTIVE_RATIO).

Verdict rule, fixed now: FACTORIZED requires sharp identity + context
invariance + selective damage + composer-free novel recomposition (swaps
where duplicates exist). High routed accuracy with collapsed forced
programs reads as surface-conditioned routing programs, not atom
factorization - the D1-suggested alternative. Ground-truth generator use is
legal (read-only diagnosis; the quarantine covers the training path only).

## D2-A1 (2026-08-20): dual-reference correction after a measurement confound
Discovered AFTER the first D2 pass produced numbers (recorded honestly): the
registered readout-channel definition decode(A_i(h)) == f(decode(h)) is
confounded at canonical states by round-trip infidelity - decode(code(x))
== x only ~0.86 on A16_s0 (the decoder never trains on raw z0; it trains on
post-program states). Verified directly: atom 13 vs ground truth P3(x) is
1.000 while the readout-referenced rate equals the 0.860 round-trip rate
exactly. The atom was perfect; the probe was penalizing decoder misreads.
Correction: P-1 sharpness and the P-3 mapping are judged GROUND-TRUTH-
referenced (decode(A_i(h)) == f(truth content); at canonical states truth =
x, at pair boundaries truth = Pa(x)), matching the certified panel probe.
Readout-referenced rates stay reported beside them. Mid-token states carry
no ground-truth content (programs stage freely), so P-2's invariance verdict
is judged on the truth-referenced groups (canonical + boundaries); mid-token
rates are descriptive. Thresholds unchanged. P-3/P-4/P-5 already compared
against ground truth and are untouched.

## D2-A2 (2026-08-20): roster extension
A15 s0-2 added to the D2 roster (its runs landed with the E4-complete sync
after D2 first ran). Read-only audit, same probes and thresholds; the
light-dose arm tests whether specialist formation tracks dose and run
health. No other change.

## D3 registration (2026-08-20): boundary position probe (read-only)
Registered BEFORE running. Question: do boundary states carry the FINISHED
answer in a strange code (the atoms did the positional work in-state), or a
rough draft plus editor (digits still at original positions; the decoder's
attention performs the permutation at readout)? A linear probe cannot
compute a permutation across positions, so wherever the digits physically
sit is the answer.

Mechanics: token-1 execution of any pair is bit-identical to its singleton
(D1), so singleton states stand in for all pair boundaries. Per checkpoint:
run the 8 singletons hard; for canonical z0 and states after micro-steps
1/2/3, slice the state per position ([6 x 64]) and fit per-position linear
probes (closed-form ridge to one-hot, lambda = 1.0, first 70% train / 30%
held-out, N = 400 per task) with two targets:
  y_moved[j] = Pa(x)[j]   (finished-answer content at position j)
  y_orig[j]  = x[j]       (original-position content; any per-position
                           bijection of it - value maps are probe-invisible)
Families by net token-1 permutation: reverse (P1, P3), rotate (P2, P4,
P8), swap (P5, P7); P6 (identity permutation) is the machinery control
where both targets must score equally. z0 probes (target x[j]) are the
encoder-localization baseline; if they sit near chance the code is not
positionally organized and the question dissolves (recorded as
DELOCALIZED).

Verdict per family at the boundary (step 3), fixed now: IN-STATE if
moved - orig >= +0.20; EDITOR if orig - moved >= +0.20; DELOCALIZED if
max(moved, orig) < 0.30; MIXED otherwise. Chance = 0.10. Mid-token steps
are descriptive (programs stage freely). Roster: the D2 roster incl. A15
(11 checkpoints).

## D3-A1 (2026-08-20): holography controls after the dichotomy dissolved
First D3 pass: moved AND orig both probe at ~1.00 from every position slice
on every checkpoint (z0 localization 1.00). The registered IN-STATE/EDITOR
dichotomy presupposed positional exclusivity; the data suggest a
holographic code instead - every slice may linearly carry everything. Two
controls added BEFORE reading further:
  (a) off-position probe: decode x[k] from slice j for k != j. ~1.0 =
      holographic (position slices are not content-exclusive and the
      "where do digits sit" question dissolves at slice granularity);
  (b) head-only readout: decoder head applied per-slice WITHOUT the
      decoder's transformer layer (its cross-position attention bypassed),
      on z0 and boundary states, vs the full decoder. If head-only
      collapses while the full decoder succeeds, the decoder's
      cross-position machinery is load-bearing at readout (an editor
      exists architecturally); if head-only holds, readout is per-slice
      and the finished answer is linearly in-slice.
Note recorded either way: the 1.00 moved-probe already proves a LINEAR
per-position readout of the finished answer exists at the boundary.
