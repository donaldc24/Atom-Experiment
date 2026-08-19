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
