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
