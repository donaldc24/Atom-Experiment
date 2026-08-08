# Claude Code Handoff — Post-Review Batch: E1b Re-runs + R3-fixed + S-arb

Context: an external review found three implementation bugs. All three are now fixed
in the working tree (verified). This document covers: committing the fixes with
provenance, which runs must be re-run and why, two new experimental probes, the
decision table that consumes the results, and documentation requirements.

Read `results/E1_FINDINGS.md`, `e1/DECISIONS.md`, and the E1b sections of the branch
plan before starting. Continue the D-numbering in DECISIONS.md for every deviation.

---

## Part 0 — Provenance first (blocks everything else)

The reviewer flagged: spec files untracked, all runs report `git_dirty=true` under
one SHA. Fix before any training run:

1. `git add` everything relevant (e1/, tests/, splits/, specs, DECISIONS.md);
   commit with a message referencing the three bug fixes; tag `e1b-postfix`.
2. Add a **dirty-tree launch guard** to `train.py` (and `train_e1b.py` if separate):
   at startup, if `git_info()` reports a dirty tree, refuse to run unless
   `--allow-dirty` is passed. When `--allow-dirty` is used, write the full
   `git diff` SHA256 into `env.json` as `dirty_diff_sha256`.
3. Log as a D-entry: what the reviewer found, what changed.

## Part 1 — The three bug fixes (already in tree; verify, do not re-implement)

Confirm each with the test noted, then move on:

- **B1 state_norm optimizer omission** (`model.py::non_atom_parameters`,
  D32): `state_norm` params are now included. Effect of the old bug: in every run
  with `atom_layernorm=True`, LN gains/biases were frozen at init AND their
  gradients inflated the global norm entering `clip_grad_norm_`, silently shrinking
  every other parameter's update (observed grad norms 5–8 vs clip 1.0).
  **Add test:** construct an R1 config model, build the optimizer, assert the set of
  optimizer params equals the set of `model.parameters()` requiring grad.
- **B2 aggregation contamination** (`aggregate.py::collect`): runs with a truthy
  `rung` field are skipped from the E1 table. **Add test:** a synthetic run dir with
  `rung="R2", arm="A1"` must not appear in `collect()`'s output.
- **B3 stochastic eval projection** (`model.py::code_project` +
  `forward`): projection now uses Gumbel only when `self.training`; eval uses
  deterministic straight-through argmax. **Add test:** same input twice through an
  R3-config model in `.eval()` → byte-identical states; in `.train()` with different
  generator states → allowed to differ.

## Part 2 — Blast radius: what is and is not compromised

`state_norm` only exists when `atom_layernorm=True`. Therefore:

| Runs | Status | Action |
|---|---|---|
| **E1 battery (A0–A4, 30 runs)** | **Clean** — no LN, no bug path executes | Keep. Do not re-run for this reason. |
| **E1b R0 (5 seeds)** | Clean — plain A1 config | Keep. Byte-identity with A1 stands. |
| R1 (2 seeds) | Compromised (frozen LN + clipping suppression) | Re-run ×1, low priority |
| R2 w=10 (5 seeds) | Compromised | **Re-run ×3 — mandatory** (verdict anchor) |
| R2 w=1 (3 seeds) | Compromised, and the *marginal* case | Re-run ×1 first; expand to 3 only if outcome differs |
| R2 w=40 (partial) | Compromised | Kill; do not re-run unless the writeup needs the third dose-response point |
| R3 seed 0 | Compromised by B1 **and** its metrics by B3 | Discard. Superseded by R3-fixed below. |

**Important caveat to record:** R3 seed 0's broken-codec diagnosis
(`code_pairwise_mean` 26.5, `code_residual` 1.32) was measured under stochastic eval
projection (B3) and is **suspect until re-measured**. Do not cite it as established.
The warm-start design below is justified by the bootstrap argument independently.

## Part 3 — Run queue (in this order)

All runs: post-fix code, clean tree, standard logging/artifact layout, deterministic
seeds, one job at a time. Expected ~20–30 min/run on this machine.

### 3.1 R2 w=10 × 3 seeds (0,1,2) — verdict anchor
Config unchanged (`config_for_rung("R2", seed, weight=10.0)`).
**Registered prediction (record in DECISIONS.md before launching):** collapse
reproduces — atom_residual_norms ≲0.1, closed_map_coverage=1, acc_unseen ≈ 0. The
collapse mechanism is geometric (nearest valid code to h0 is h0), not a clipping
artifact. If prediction holds, prior R2-w10 conclusions are confirmed on clean code.
If atoms come out ALIVE (norms ≳0.3), stop the queue and flag — the entire R2
verdict reopens.

### 3.2 R2 w=1 × 1 seed (0) — the marginal case
The one result with real probability of flipping: w=1 balanced task loss against the
constraint, so suppressed effective LR could plausibly have tipped it.
- Reproduces collapse (coverage 1, norms ~0.12) → old seeds stand as corroboration;
  move on.
- Atoms alive post-fix → **the window question reopens**; run seeds 1–2 and report
  to the user before proceeding.

### 3.3 R3-fixed × 1 seed — structural enforcement, made trainable
New config knobs (add to `Config` with defaults off; document in DECISIONS.md):
- `codec_pretrain_epochs: int = 10` — **phase 0**: train encoder+decoder ONLY on
  reconstruction (tokens → enc → [state_norm] → dec → tokens, cross-entropy),
  atoms and composer excluded from the optimizer. Purpose: the projection routes
  through the decoder; an untrained codec means atoms learn to write through a
  transcriber that mis-transcribes, and the two corrupt each other's signal.
- `codec_lr_scale: float = 0.1` — **after phase 0**, encoder+decoder(+state_norm)
  join the main optimizer in their own param group at `lr * codec_lr_scale`.
  (Down-weighted rather than frozen: fully frozen risks the codec being unable to
  accommodate the atoms at all; 0.1 makes it a slow-moving substrate.)
- `project_tau_floor: float = 1.0` — the projection's temperature uses
  `max(tau, project_tau_floor)` so the routing anneal (2.0 → 0.5) does not drag the
  projection's gradient channel to near-zero. Routing keeps its own schedule.
Run as `config_for_rung("R3", 0)` + the three knobs, `code_consistency_weight=0`
(bottleneck only — the penalty is the thing that collapses atoms; see R2 sweep).

**Kill rule (register now):** if at epoch 30 the epoch_train_acc curve is linear at
~0.001/4-epochs with no acceleration (the pre-fix signature), stop the run and mark
R3-line INCONCLUSIVE(optimization) rather than riding it to 80.
**Leading indicator:** watch `acc_singleton` (log it per epoch if cheap, else eval at
20/40/60): the singleton anchoring channel should move EARLY if the codec fix
unblocked it. Pair accuracy moving without singletons moving is unexpected — flag it.

### 3.4 S-arb × 1 seed — consistency vs. correctness (new rung)
Tests: does intermediate supervision need to be semantically CORRECT, or merely
CONSISTENT per primitive? A0 supplied (a) on-manifold + (b) consistent-per-primitive
+ (c) correct. E1b tested (a) alone → fails. S-arb tests (a)+(b) without (c).

Implementation:
- New config: `rung="Sarb"`, base = `config_for_rung` style off A1 (no dropout, no
  curriculum), `atom_layernorm=True` for parity with the ladder.
- **Frozen arbitrary targets:** at model init, for each primitive p, draw ONE target
  state `T_p` by encoding a fixed random token sequence (seeded by `split_seed + p`,
  NOT the run seed — targets identical across seeds) through the *initial* encoder,
  then store as a **non-learnable buffer** (`register_buffer`). Norm-match to the
  encoder's typical output norm. They must never receive gradient and never drift.
- **Loss:** replace A0's state-consistency target with `T_{p_1}` — i.e. for each
  training example with first primitive p₁, `h_1` is pulled to `T_{p_1}` by the same
  relative-MSE form as `state_consistency`, weight `state_consistency_weight=40`
  (A0's authorized value). NO intermediate decode supervision, NO forced routing —
  routing stays gumbel. Final task loss unchanged.
- **DECISIONS.md notes (required):** (i) targets frozen and why (learnable targets
  would drift toward true encodings and contaminate the test); (ii) `acc_singleton`
  is no longer a free anchor — the decoder must learn the arbitrary codebook, so
  singleton accuracy is part of what's being learned, not a given; (iii) the
  registered predictions below.

**Registered predictions:** S-arb WORKS (acc_unseen well above 0.5, plausibly near
A0's 0.95 minus some decoder tax) — the mechanism A0 needed was a stable,
predictable input distribution for atom j, which consistency supplies in full. If it
FAILS with the constraint verifiably satisfied (h1 actually near T_{p1} — log the
distance as `loss_state_rel` already does), that is the more surprising and more
interesting outcome: semantic correctness itself is load-bearing.

### 3.5 R1 × 1 seed (0) — housekeeping, run last / when idle
Post-fix learnable LN. Expected: same conclusion (LN alone insufficient), possibly
marginally better numbers. One seed for honesty.

## Part 4 — Decision table (consume results in this order)

After 3.1–3.4 complete, place the outcome in this 2×2 and STOP for user review —
do not launch follow-ups beyond confirmation seeds without sign-off:

| | S-arb works | S-arb fails (constraint satisfied) |
|---|---|---|
| **R3-fixed trains** | Two independent routes to a substrate. Next: ablate which is cleaner, then the H6 battery (E1c-A2) on the winner. | Structure trains but consistency insufficient → correctness is load-bearing. Next: supervision-minimization ladder (S4 first). |
| **R3-fixed dead** | **Consistency suffices, structure unnecessary** — the "convention" result. S-arb becomes the main line; next question is whether the convention can be self-established (e.g. EMA of each atom's own early outputs as its target). | All unsupervised and consistency-only routes closed. Branch C ladder (S4 first) is the program; prospectus §10 formally adopted. |

Confirmation seeds (to 5) for whichever cell's headline claim: run AFTER the user
reviews, while the report is being written.

## Part 5 — Reporting requirements

1. **DECISIONS.md**: D-entries for (a) the review + three fixes + blast radius,
   (b) the R2 re-run predictions, (c) R3-fixed design + kill rule, (d) S-arb design
   + predictions, (e) any deviation during execution.
2. **Aggregation**: E1b runs aggregate SEPARATELY from E1 (B2 fix enforces this).
   Build/extend an `aggregate_e1b.py` (or a `--e1b` mode) that keys on
   `(rung, weight, seed)`, emits its own summary.csv/md, and includes the guard
   metrics (`code_residual`, `code_residual_soft`, `code_soft_hard_gap`,
   `code_decode_diversity`, `code_entropy_frac`, `atom_residual_norms` mean) as
   first-class columns.
3. **E1_REPORT.md**: add a "Post-review revisions" section: the three bugs, which
   results were re-established vs. changed, the T1 framing fix (report BOTH A0
   end-to-end 0.951 and teacher-forced 0.999; state "threshold amended, conclusion
   invariant — every comparison arm is at 0.000"; remove any "all gates passed"
   phrasing), and the reviewer's scoping language adopted for the causal claim:
   "privileged-supervision feasibility" for A0, "task-only variants robustly fail on
   this fixed split" for A1–A3, with fresh-split replication listed as pending.
4. **Status file** for the user: one table — every run in the queue, its state
   (done/running/killed), its one-line outcome, and which decision-table cell the
   evidence currently points to.

## Part 6 — Explicitly deferred (do NOT start)

- Fresh-split A1/A0 replication (queued after this batch; needs a split-seed config
  field — note it in DECISIONS.md as pending).
- w=40 re-run, anti-collapse rung, narrow-substrate rung, dreaming/recombination
  rung, any non-gradient machinery, S4 itself — all gated behind the decision table.
- The H6 battery re-run — gated behind a winning configuration existing.
