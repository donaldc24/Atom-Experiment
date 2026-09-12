# Registered Decisions — Atom V2 (H1)

Fixed **2026-08-14, before the first real run**. Values live in code in
[atomv2/registered.py](atomv2/registered.py); if this document and the code
disagree, the code is authoritative and the document is the bug (V1 standing
rule). Amendments are allowed only during Experiment 0 calibration, must be
recorded here with a reason, and are frozen once E0 passes. No tuning against
Experiment 1 results, ever.

The spec is [../H1Experiments.md](../H1Experiments.md) +
[../SplitMath.md](../SplitMath.md). This file records only what the spec left
open, with the rationale for each choice.

## R1 — Census ε = 2%, denominator = atom-selections only

An atom is **in use** if it receives **more than 2% of all hard-routing
atom-selections** (pass picks excluded from the denominator) on the
seen_heldout eval set.

- Pass is excluded from the denominator deliberately: pass usage is a separate
  logged number with its own story (step-smearing detection). Letting it dilute
  the denominator would make ε mean different things in high-pass and low-pass
  regimes, and the λ sweep will produce both.
- 2% is an order of magnitude above the hard-routing noise floor of a converged
  router, and 2–3× below the weakest legitimate atom in the most fragmented
  plausible world (sub-op factorization: the rarest sub-op atom carries ~5–7%
  of atom-selections).
- The census always logs **two** numbers: atoms-in-use AND steps-per-token.
  Neither alone carries the verdict.

## R2 — Closed-map error: trajectory headline + atom-centric panel

**Headline (time-resolved, every 1,000 steps):** during hard-routed eval
forwards, after every live micro-step, the relative L2 distance from the
current state to the **nearest encoding of the task's sub-op-lattice prefix
states** — all prefixes of the task's canonical sub-op chain including depth 0,
encoded in the canonical task-independent content space. Reported per task;
the eval cadence is the time-resolved curve.

Companions (logged with it, per the Learning-1 lesson that a manifold metric
read alone is gameable by doing nothing):

- **distance-to-target** at the final step (an all-pass trajectory scores ~0 on
  the headline but stays far from the target — covered by a test),
- the **prefix-visit histogram** (which lattice depths the trajectory actually
  visits).

**Panel (final + 5k/10k/15k checkpoints):** the V1-style atom-centric probe —
each atom applied once to encoded probe inputs, relative L2 against the
encodings of all 16 candidate targets (7 sub-ops + 8 surface ops + identity),
probed in the single canonical content context; matched one-to-one
assignment + coverage companions (V1 D24 lineage).

## R3 — Ablation: zero-delta intercept at application

Registered F2 mechanism: the composer still **selects** the atom; the delta is
zeroed at application. With the non-affine state norm this is exactly "this
atom becomes pass" (covered by a bitwise test). Routing untouched — the metric
answers "what does this atom's computation contribute, holding everything else
fixed", which is the only question under which F2's damage-variance is
well-defined.

- One atom at a time only; pairwise ablation is future work.
- Damage is measured against the **same run's** unablated per-task accuracy.
- Reported on tasks where the atom's task-level usage ≥ 0.5 (V1 lineage).
- Raw tuples always saved: (atom, task, level, acc_delta,
  routing_entropy_during_ablation).
- **Compensation probe** (routing-mask, composer re-routes): final checkpoint
  only, observational, non-gating. The per-atom gap between intercept-damage
  and mask-damage is a redundancy signal, one of the sprawl signatures at
  low λ.
- E0's oracle arm audits the instrument against ground truth before it ever
  testifies about a free run (atom i must damage exactly the tasks containing
  P(i+1), uniformly).

## R4 — P3 oversampling = presentation frequency

P3 keeps **1,000 unique training examples** like every other task; "oversampled
to ~7,000 examples" is read as presentation frequency — P3's examples appear
7× per epoch in the training array. Rationale: the doc places oversampling
under *Task mixing*, states "1,000 examples per training task" universally, and
its own arithmetic (~55 epochs over ~48k examples at 20k steps × batch 128)
closes exactly under this reading: 41×1000 + 7×1000 = 48,000 presentations.

## R5 — Oracle mechanics (E0 only, quarantined in atomv2/oracle.py)

- Forced routing `[atom, pass, pass]` per token; atom index = surface-op index
  (P1 → atom 0 … P8 → atom 7). Atoms 8–15 stay idle.
- State supervision at token-block boundaries: target = **current** digit-only
  encoder's canonical embedding of the partial-composition digits,
  stop-grad (shapes the atoms, not the encoder). Relative MSE, weight **40.0**
  (V1 D18 lineage).
- Intermediate decode CE (pairs), weight **1.0**; routing CE (composer imitates
  the schedule), weight **1.0**.
- The module is imported only when `cfg.forced_routing` is true, which only
  `config_for_arm("A0-oracle")` sets. A test asserts no oracle loss term ever
  appears in a free arm's training log.

## R6 — E0 pattern thresholds

Qualitative pattern match, NOT number match (the world changed: length 6 not 8,
16 slots not 8, micro-steps added): oracle L1 > 0.70; free L1 < 0.05; gap ≥
0.50. The closed-map direction gate uses final distance-to-target on
seen_heldout (oracle < free), not the nearest-prefix headline that an all-pass
system can game. Both remain reported. Mechanically checked in
`aggregate.py::e0_verdict`; `run_e1` refuses to start without a passing verdict
for the current protocol revision.

## R7 — Architectural details the spec implied but did not pin

- **State norm:** per-position LayerNorm **without affine parameters** applied
  after the encoder and after every atom application. Non-affine ⇒ idempotent
  ⇒ pass is exactly identity and a no-delta step can bypass the norm bitwise
  (hard/forced modes do). `code()` and `step_once()` are the canonical
  accessors every probe goes through (V1 D26 lesson).
- **Weight decay** applies to all parameters uniformly (the doc registers
  "AdamW, weight decay 0.01" with no exceptions; V1's atom-exempt group existed
  for freeze semantics V2 doesn't have).
- **Rent** is charged on the same soft distribution the straight-through
  sample uses (post-Gumbel softmax), summed over live micro-steps, atoms only.
- **Encoder input:** 6 digit positions only; its flattened 6x64 state is a
  canonical content code. Opaque task tokens are routing control, not state.
- **Data streams:** one master seed per run; named, independent SeedSequence
  streams for data / probe / shuffle / init / gumbel / probe-train. Task
  datasets are keyed by a split-independent task index. Both E0 arms and all
  E1 arms share identical data at the same seed (tested).
- **Decodability probes:** linear probes on detached features, example-level
  70/30 split, each probe paired with an h0 no-control leakage floor and a
  shuffled-label chance floor. Sub-op labels are SETS
  (P8 = {T, N}), enforced by `ops.task_subop_sets` being the only label source.

## R8 — E0 control/data separation amendment (2026-08-14)

Recorded after the failed pre-amendment A0-oracle seed 0 and before any E1
result or replacement E0 run. That run reached hard L1 **0.10375**; a
read-only replay of the same checkpoint with the correct routes forced at eval
reached **0.71156**. The audit found that `code(x, full_task_pair)` exposed the
partner token to both atoms and router and made oracle targets pair-contextual,
unlike V1's canonical `code(x)`.

- `code(x)`, atoms, state norm, and decoder are now digit-only. At each live
  micro-step the memoryless composer receives the current content state, the
  **active opaque surface token only**, and micro-step-within-token 0..2.
  It never receives the partner token or absolute token position.
- Oracle targets are stop-grad `code(partial_digits)`. Closed-map targets and
  standalone probes use the same canonical content space. The former eight
  singleton encoding contexts collapse to one.
- The surface-level oracle schedule remains `[atom(P_t), pass, pass]`. It is a
  ceiling for reusable surface closed maps, learned hard routing, pass,
  codec, and metrics; it is not evidence of hidden-sub-op discovery.
- L3 is a Dax-style test of transferring a singleton-learned program into pair
  composition. It does **not** uniquely prove sub-op factorization: a reusable
  P3 surface atom can pass it. Granularity is adjudicated jointly by census,
  steps/token, standalone semantics, and surface-vs-sub-op probes.
- Data, split, seeds, optimizer, training budget, losses, oracle schedule, and
  numeric accuracy thresholds are unchanged. The protocol revision is
  `e0a-control-separated`; aggregation and the E1 gate reject other revisions.
- Calibration repairs made before the replacement run: save every registered
  5k/10k/15k panel checkpoint; fail loudly if one is missing; hash untracked
  source in dirty-run provenance; reject duplicate arm/seed aggregation.
- Under the registered 2% census, a perfect surface oracle reports **7** atoms
  in use because singleton-only P3 has share 1/76. The audit now separately
  checks the mechanically expected threshold mask and that all 8 oracle atoms
  were observed, instead of demanding the mathematically impossible count 8.

This amendment is authorized by the E0 debugging clause. It freezes if the
replacement E0 passes. The failed run is retained as pre-amendment evidence
and must never be pooled with amended runs.

## R9 — A1 is not re-run in the E1 battery (2026-08-15)

E0's **A0-free arm is configurationally identical to E1's A1**. Verified
mechanically: the two resolved configs differ only in the `arm` and
`experiment` strings — same λ_use = 0, same free routing, same data, split,
seeds, schedule, and losses. Re-running A1 would spend compute reproducing an
existing result and would attach two arm names to one condition.

- The E1 battery is `("A2", "A3", "A4")` (`registered.E1_BATTERY_ARMS`).
  `run_e1` refuses `--arms A1` with a pointer to where that data lives.
- `LAMBDA_GRID` is **unchanged**; A1 remains registered as the λ=0 cell and
  `config_for_arm("A1")` still resolves (tests construct it).
- E1 aggregation carries the λ=0 numbers as a **labelled reference row**
  (`lambda_zero_reference.json`, and a `A1=A0-free (ref, from e0)` row in
  `summary.md`), never pooled into the battery — no table may imply it was run
  under E1.

Consequence to state plainly when reporting E1: the λ=0 cell was run under the
E0 protocol revision, so it is a reference point, not a within-battery arm.

## R10 — Pooling identity is a harness-source content fingerprint (2026-08-15)

The prior guard compared a dirty-tree snapshot fingerprint that (a) also
covered files which cannot change a number — `.claude/*`, editor config, docs —
and (b) **changed representation when an unmodified tree went from
dirty-untracked to committed**. The five completed E0 runs recorded
`dirty_source_sha256 = 46c8ed00…` at dirty `d354ffd`; committing the harness as
`ff6a1b3` meant any further run recorded a clean `git_sha` instead, so
byte-identical implementations were unpoolable. Result identity is a property
of the code that runs, not of whether it happens to be committed yet.

- Pooling now keys on `harness_source_sha256`: a content fingerprint over
  `atomv2/*.py` and `splits/*.json` — the modules a run imports plus the frozen
  split. Computed identically for clean and dirty trees.
- **Deliberately excluded:** tests (never imported by a run), README/REGISTERED
  (documentation), everything outside `AtomV2/Harness`.
- **Nothing is discarded.** `git_sha`, `git_dirty`, `git_source_dirty` and the
  dirty-snapshot fingerprint are all still written to `env.json`. R10 narrows
  only the value that governs pooling, and the dirty-tree *refusal* (D33) is
  untouched.
- `collect()` **refuses** runs lacking the key rather than falling back to the
  old scheme, so the two schemes can never be silently mixed.
- Pre-R10 runs are repaired with `python -m atomv2.backfill --rev <commit>
  --reason "…"`, which computes the fingerprint from the harness source *as
  committed* at an explicit revision and marks it
  `harness_source_provenance.recorded_by = "backfill"` — a backfilled value is
  never indistinguishable from one a run recorded about itself. It rewrites
  provenance fields in `env.json` only, then regenerates `SHA256SUMS`; no
  artifact any metric derives from is touched.

Evidence required before backfilling, and the evidence used here: A0-free seed
2 was re-run from `ff6a1b3` after being interrupted at step 14,800, and its
train_log reproduced the interrupted run's losses **bit-for-bit** over all
14,800 shared steps. That establishes the committed source is behaviourally
identical to the source that produced the batch, so recording `ff6a1b3` as the
source identity of all six runs is an observation, not an assumption.

Known limitation: the fingerprint covers file contents, not the interpreter or
library versions. Those remain recorded separately in `env.json` (python,
torch, numpy, platform, thread counts) and the hostname guard still applies.

## R11 — E1 is no longer gated on the E0 verdict (2026-08-15)

**No E0 threshold is moved.** E0 still fails `free_L1_floor` (0.807 vs < 0.05)
and `gap_unmistakable` (0.193 vs ≥ 0.50), and that verdict stands on the record
exactly as computed. What changes is only whether that failure *blocks* E1.

Rationale, from what E0 actually established:

- **The instrument works** — which is what E0 was for. Its audit passed on all
  three oracle seeds: `usage_matches_ground_truth = 1.0`, ablation CV = 0.0,
  census = the predicted 7, all 8 forced atoms observed, idle atoms idle.
- **The ceiling is reachable**: oracle L1 = 1.0000 > 0.70, so a composing
  solution exists in this world — the precondition that makes a null result in
  E1 readable rather than ambiguous.
- **A leaking world is excluded**: the closed-map direction check passed, and
  the formal leak audit passed all six checks with negative controls proving it
  can fail.
- **The two failures are one fact about the free arm's absolute L1 level**,
  which is a *finding about this architecture*, not a broken rig. Amendment R8
  separated control from data, making per-token program composition available,
  so free routing recombines trained tokens by construction. The "free arm sits
  at the floor" premise was inherited from V1, whose composer had memory and
  whose state carried task identity.
- **The dissociation the protocol wanted does exist** — at L3, where oracle
  1.0000 against free 0.0108 is a gap of 0.993.

Mechanically: `registered.E1_REQUIRES_E0_PASS = False`. Ungating is not
forgetting — two hard stops remain, and E0's verdict is copied to
`results/e1/e0_context.json` so no E1 number is ever read without its
calibration context:

- E0 must still have **run**. A missing verdict is still a refusal: without
  calibration the instrument is unvalidated and E1 is uninterpretable.
- A **protocol-revision mismatch** is still a refusal (override needs
  `--force --force-reason`).

Reporting requirement: every E1 result must be published alongside the E0
context, stating that calibration failed its free-arm criterion and why that
was judged not to impugn the instrument.

## E9 registration - vortex crystallization (2026-09-11)

Registered before any full E9 run. The motivating Navier-Stokes result is a
**forced** construction with bounded L2 energy and unbounded L-infinity peak;
it is not a theorem about learning. E9 preserves that distinction causally:

- Fixed seed-paired teacher: completed `e0/A0-free` final, three routes per
  token, hard seen accuracy at least 0.90, exact checkpoint SHA-256 recorded.
- All students use one route per token and the A0-free world otherwise.
- A25: scratch, task CE only.
- A26: copy the teacher, task CE only (unforced warm-start control).
- A27: same copy plus a 1,000-step linear ramp of
  `1.0 * boundary_state_MSE + 0.25 * final_logit_KL`.
- Copy rule: every same-shaped tensor is copied; the only allowed mismatch is
  `composer.micro_emb.weight`, where student row 0 receives teacher row 0.
- Fixed 20k budget. Screen seed 1; replicate the first sufficient mechanism at
  seeds 0/2.
- Deployment gate: seen hard >= 0.90, L1 hard >= 0.80 of the paired teacher,
  and exactly one route decision per token.

The full outcome order is frozen in `H1-Experiment9.md` and
`atomv2.run_e9.screen_verdict`: ONE-STEP BASE LEARNS, WARM START SUFFICIENT,
CONTROLLED CRYSTALLIZATION, FORCING HELPS BUT INCOMPLETE, NO ONE-STEP RESCUE.
The first passing mechanism wins; no-rescue/incomplete outcomes stop.

At every eval, task-balanced read-only telemetry records state/update L2
energy, L-infinity peak, peak energy share, inverse-participation effective
coordinate support, route participation, and the one-route invariant. At
every training-log step, pre-clip gradients are partitioned by atom transform
and report global norm, post-clip bound, peak atom share/RMS, and effective
atom support. These concentration curves are descriptive in E9 and cannot be
promoted to success gates after inspection.

## E10 registration - Reynolds Transport Network (2026-09-12)

Registered before smoke. E10 is a new sequential field model, not an Atom arm
and not a Navier-Stokes discretization. A28 has pointwise token reaction only;
A29 adds a conventional row-stochastic 6x6 transport; A30 replaces that matrix
with a parameter-matched centered-pulse cross-covariance followed by a doubly
stochastic Sinkhorn projection; A31 adds fixed periodic viscosity 0.04. The
pointwise decoder cannot mix positions. All arms train for 6k steps on the
frozen V2 singleton/pair data using final CE only. Fixed evaluation adds 64
unseen triples, 64 unseen quads, and sigma-0.15 hidden-noise pairs. Primary
gates, outcome order, and conditional replication are frozen in
`H1-Experiment10.md`; `atomv2.registered` is the executable constant record.
The pre-smoke invariant test fixed Sinkhorn normalization at 32 iterations
after 12 failed the already-frozen `1e-5` column-conservation gate; no gate or
result was inspected or changed.

## E11 registration - Canonical Conservative Exchange Network (2026-09-12)

Registered before any E11 smoke or result-bearing run. E11 responds to two
specific E10 failures: approximate Sinkhorn transport lost conservation as it
sharpened, and trained pairs displaced the singleton meaning of tokens. Each
token now transports a six-cell categorical digit field through eight fixed
alternating ring matchings. Every two-cell update is the doubly stochastic
block `[[1-g,g],[g,1-g]]`, so conservation is algebraic for every gate value.
A32 learns the 24 gates per token directly. A33/A34 derive each gate from a
quadratic interaction of centered signed pulse profiles. No Sinkhorn loop is
used.

The state is six 10-way digit distributions. A learned token- and
position-conditioned row-stochastic 10x10 reaction kernel maps distributions
to distributions. Every nonterminal boundary is normalized and projected to
the argmax one-hot digit in the forward pass, with a straight-through soft
gradient. Thus a downstream token consumes exactly the same canonical
representation used for a fresh digit input; no intermediate target, recipe,
sub-operation label, teacher, or held-out identity is used in training.

Training is task-balanced. Phase 1 is exactly 4,000 singleton-only AdamW steps.
Phase 2 is forbidden unless every held-out singleton task reaches 0.99 exact-
list accuracy at that fixed checkpoint. Passing arms receive exactly 4,000
more steps, with every batch split 64 singleton and 64 trained-pair examples.
There is no P3 oversampling in E11. The learning rate is 0.01, warmup 200,
betas `(0.9,0.99)`, zero weight decay, and gradient clip 5.0.

The result-bearing screen uses seed 1 for A32/A33. Primary success jointly
requires singleton minimum 0.99, trained-pair mean 0.95, L1/L2/L3 each 0.90,
and clean length-3/4/8/16 means of 0.85/0.80/0.65/0.45. It also requires exact-
exchange relative mass residual at most 1e-6 and complete agreement between
ordinary sequential execution and explicit decode/restart execution. A33's
four-length mean may trail A32 by at most 0.03.

A34, fixed periodic viscosity 0.04, is trained only if A33 passes. Robustness
is audited with sigma-0.15 handoff noise, symmetric INT8/INT4 weight
quantization, one dropped boundary cell, and one corrupted boundary cell.
Viscosity succeeds only if A34 passes the primary gates, loses at most 0.03 on
the four-length clean mean, and improves the preregistered six-panel robustness
mean (noise, INT4, dropped, corrupted, length 8, length 16) by at least 0.03.
INT8 is reported but omitted from that mean to avoid double-counting the two
closely related quantization panels.

The outcome order is VISCOUS_CANONICAL_REYNOLDS, INVISCID_CANONICAL_REYNOLDS,
DIRECT_CANONICAL_EXCHANGE_ONLY, SINGLETON_CRYSTALLIZATION_FAILED, then
NO_REUSABLE_EXCHANGE. A structured winner is replicated at seeds 0 and 2; the
claim holds with primary success in at least two of three seeds, conservation
in all three, and exact interface agreement in all three. This remains a
discrete neural analogy, not a Navier-Stokes solver or theorem transfer.

## Errata found while building (doc bugs, not code bugs)

- **SplitMath.md G6 table, row P2_P5**: says `train`, but the doc's own Split
  Assignment section, its 34-pair train list, and the 34+8+6+15+1=64
  accounting all require **held-out L1** (adjacency (A,N), trained sibling
  P7_P5). The registered value is L1; the G6 row is a transcription typo in
  the doc. Caught by the enumerator-duty diff (tests/test_split.py).
