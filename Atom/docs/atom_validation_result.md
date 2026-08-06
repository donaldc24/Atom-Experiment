# Atom validation result: the H1 compression is real, but it is not atom-specific

Status as of 2026-08-06. All three experiments in [`atom_validation_spec.md`](atom_validation_spec.md)
are complete at the locked seeds 17, 29, and 43.

| Experiment | Locked question | Decision |
|---|---|---|
| A — matched frontier | Do atoms beat shared multitask LoRA at matched storage and matched active compute? | **FAIL** — no qualifying capacity |
| B — crossed frozen transfer | Does a four-task dictionary give reusable capacity for each held-out fifth task? | **Primary PASS (95.83%), control-aware FAIL** |
| C — oracle span projection | Does the frozen learned span contain the held-out LoRA update at all? | **FAIL** — 2.35% explained energy |

The headline: the H1 storage result survives, but the mechanism does not. Shared adaptation
compresses independent LoRA convincingly; **task-specific atom composition is not what makes it
work**, and the learned dictionary does not contain held-out task structure.

The machine-readable sources are the generated JSON files under `results/atom_validation/`. They
are ignored by Git and must be regenerated locally when absent.

## Experiment A: matched shared-LoRA/atom frontier — FAIL

Twenty-four cells: four capacities `c in [1, 2, 4, 8]` x three seeds x two systems, all on the
same balanced multitask scheduler and 3,750 optimizer updates. At matched `c` both systems execute
`1,024 * c` estimated adapter operations per token.

| c | Shared LoRA | Shared atoms | Mean delta | Storage ratio | Ops equal | Qualifies |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.681516 | 0.670172 | -0.011345 | 1.0086 | yes | FAIL |
| 2 | 0.685303 | 0.672862 | -0.012441 | 1.0120 | yes | FAIL |
| 4 | 0.686498 | 0.677406 | -0.009092 | 1.0149 | yes | FAIL |
| 8 | 0.691089 | 0.679316 | -0.011773 | 1.0169 | yes | FAIL |

An atom-specific advantage required at least one capacity where atoms exceeded shared LoRA by
`0.005` in mean primary score. Atoms lost at **every** capacity, by roughly twice the tolerance in
the opposite direction, and were also worse on the worst task (RTE) in every column. The
consistency across all four capacities and all three seeds matters more than the size of any single
gap: this is not a noise result.

The exact Pareto frontier is `shared_lora` at c1, c2, c4, and c8 — atoms appear nowhere on it.

**The compression claim survives, and gets stronger.** Shared multitask LoRA at c=8 reached
0.691089 with 9,482 persistent parameters, against independent rank-4 LoRA's 0.685209 mean with
21,770. That is *better* quality at 43.6% of the storage. The H1 headline — five tasks can share
persistent adaptation structure at under half the cost — is confirmed and improved. What Experiment
A removes is the attribution: ordinary joint low-rank sharing does the work, and the extra
machinery of a rank-1 dictionary with per-task coefficients costs slightly more storage and returns
slightly less quality.

This directly resolves the open caveat from the H1 chunk-25 controls, where shared multitask LoRA
was the strongest control at a single seed. It now holds at three seeds and four matched
capacities.

## Experiment B: crossed frozen-dictionary transfer — primary PASS, control-aware FAIL

Fifteen cells: each task held out in turn at three seeds. A source dictionary of eight atoms is
trained on the other four tasks, every atom vector is frozen, and only a new coefficient row and
target head are fitted.

| System | Aggregate mean primary score |
|---|---:|
| Fresh rank-4 LoRA | 0.685209 |
| Learned frozen atoms, all 8 | 0.660276 |
| Learned frozen atoms, top 4 | 0.656660 |
| Matched random frozen atoms, top 4 | 0.656644 |
| Head only | 0.655065 |

| Criterion | Observed | Required | Result |
|---|---:|---:|---:|
| Aggregate top-4 retention | 95.83% | >= 95% | PASS |
| Every target three-seed retention | 93.15% | >= 90% | PASS |
| Marginal new state vs fresh LoRA | 6.67% | <= 10% | PASS |
| Learned exceeds head-only | +0.001595 | >= 0.005 | FAIL |
| Learned exceeds random frozen | **+0.000016** | >= 0.005 | FAIL |

This is exactly the outcome the spec anticipated when it warned that "a result may pass the
original retention rule yet fail the stronger control-aware interpretation." The 95.83% retention
is real, but the controls show what produced it: a freshly trained head alone reaches 0.655065, and
the entire learned dictionary adds **0.0016** on top of that. Against a matched *random* frozen
dictionary the margin is 0.000016 — sixteen millionths, which is indistinguishable from zero.

So the retention number is carried almost entirely by the target head and by generic random
projection capacity, not by anything learned from the four source tasks. Per-target retention was
99.17% (MRPC) and 97.74% (RTE) at the top, 93.15% (SST-2) and 93.44% (QQP) at the bottom.

Marginal accounting: 290 new parameters per held-out target against 4,354 for a fresh LoRA, atop a
reused 8,192-parameter dictionary (8,482 total). The parameter efficiency is genuine; the transfer
it is supposed to buy is not.

## Experiment C: oracle LoRA-update projection — FAIL

Experiment C exists to answer *why* B fails: is the frozen span missing the held-out update, or did
coefficient learning simply fail to find an update the span could already express? For every cell
and target module it forms the fresh LoRA's exact effective matrix `B @ A`, then solves float64
least squares against the frozen rank-1 atom bases.

| Span | Relative Frobenius error | Explained energy |
|---|---:|---:|
| Learned | 0.988186 | **2.35%** |
| Random | 0.999737 | 0.05% |

The answer is unambiguous: **the span is missing the update.** A least-squares-optimal projection
of the held-out LoRA update onto the learned dictionary captures 2.35% of its energy. The
dictionary learned from four tasks is very nearly orthogonal to what the fifth task needs.

| Target | Fresh LoRA | Learned span | Random span | Retention |
|---|---:|---:|---:|---:|
| SST-2 | 0.730000 | 0.657333 | 0.656000 | 90.05% |
| MRPC | 0.764936 | 0.748025 | 0.748025 | 97.79% |
| RTE | 0.586041 | 0.578821 | 0.574007 | 98.77% |
| QNLI | 0.677333 | 0.604667 | 0.597333 | 89.27% |
| QQP | 0.667736 | 0.581740 | 0.572218 | 87.12% |

| Check | Observed | Required | Result |
|---|---:|---:|---:|
| Aggregate quality retention | 92.54% | >= 95% | FAIL |
| Every target retention | 87.12% | >= 90% | FAIL |
| Lower reconstruction error than random | 0.9882 vs 0.9997 | strictly lower | PASS |
| Quality advantage over random | +0.004601 | >= 0.005 | FAIL |

The one check that passes is informative. The learned span *is* reliably better than a random span
— 2.35% explained energy against 0.05% is a factor of 45 — so the dictionary did learn something
real and shared. It is simply nowhere near enough to represent a new task. The quality margin over
random, +0.004601, fell just under its 0.005 threshold; treat that specific check as a near-miss
rather than a clean negative, and note that it points the same direction as everything else.

### One honest caveat about C

The oracle scored *lower* than B's trained coefficients (0.634117 all-eight, versus 0.660276 in
Experiment B). This is not a contradiction, and it is worth stating precisely: least-squares
minimizes Frobenius error against the LoRA weight matrix while reusing the fresh LoRA head, whereas
B optimizes coefficients and a fresh head directly against task labels. Weight-space optimality is
not quality optimality.

The consequence is that C is a clean upper bound on **span coverage**, which is what it was
designed to measure and where the result is decisive at 2.35%. It is *not* an upper bound on
achievable quality. The 92.54% retention row should be read as a diagnostic of the projection
procedure, not as a ceiling on what coefficient learning could achieve.

## What this means for the research program

Taking the three experiments together:

1. **Shared adaptation compresses.** Five tasks share persistent structure at ~44% of independent
   LoRA storage with no quality loss — at c=8, with a small gain. This is the durable H1 result.
2. **Atoms are not the reason.** At matched storage and matched active compute, ordinary shared
   multitask LoRA beat rank-1 atom composition at all four capacities and all three seeds.
3. **The dictionary does not generalize.** It contains ~2% of what a held-out task needs, and
   frozen-dictionary transfer is statistically indistinguishable from a random dictionary.

The spec's own interpretation boundary states the consequence: "A failed oracle span test is
evidence to revise the dictionary before building a hypernetwork." That condition is now met.
Building H2 — a hypernetwork generating coefficients over this dictionary — would be generating
coefficients over a basis demonstrated not to span held-out updates. The failure is in the
dictionary, not in the coefficient inference, so better coefficient inference cannot fix it.

Candidate directions, in rough order of how directly they attack the measured failure:

- **Train the dictionary for span coverage, not joint task fit.** Nothing in the current objective
  asks atoms to cover unseen updates; they are optimized only to fit five known tasks jointly. That
  is the most likely root cause of 2.35%.
- **Test more tasks and more diverse tasks.** Four GLUE source tasks on BERT-tiny is a thin and
  highly correlated basis. The span may be narrow because the training distribution is narrow.
- **Reconsider the rank-1 constraint.** Experiment A shows the low-rank bank is the more efficient
  parameterization at every capacity tested; the composition story may not need atoms at all.
- **Scale the base model before drawing architectural conclusions.** All results are BERT-tiny with
  two attention layers and 128 hidden units; a 2-layer model may not have enough structure for a
  shared basis to be meaningful.

## Scope limits

These results are BERT-tiny, five GLUE tasks, query and value projections, 2,000 training rows per
task, and three epochs. They do not test a hypernetwork, routing, wake-dream consolidation,
plugins, language generation, or the 1T-to-10B system claim. Experiment B tests frozen *supervised*
reuse — target labels still optimize the coefficients — and so is not a test of conditional
generation. No significance test was preregistered for any of the three experiments; the decisions
are threshold comparisons on three-seed means. As the spec notes, none of these outcomes alone
speaks to semantic atom identity, which would require normalized functional contribution tests,
causal ablations, and genuinely compositional held-out splits.

## Two implementation fixes made during this run

Experiment C's suite path had never been executed end to end; only its smoke path had. Two defects
surfaced and were fixed before the graded run:

- `run_projection_suite` built the protocol path but never passed it to `run_projection_cell`,
  which required it as a keyword-only argument (`TypeError` on the first cell).
- `validate_projection_cell` accepted only `expected_task`, `expected_seed`, and `require_files`,
  while both call sites passed eight further expectations — resolved configs, dataset provenance,
  and the cross-transfer, fresh-LoRA, source, random, and protocol paths. These are the
  specification's strict-reuse guarantee, so they were implemented rather than dropped: cached
  cells now resume only when configuration, provenance, and every referenced component path match
  the requested run.

Both fixes are in [`validation_projection.py`](../src/cgmoe_h1/validation_projection.py). The full
suite passes at 165 tests. Neither fix touches Experiment A or B, which ran unmodified.

## Reproduction

```powershell
python scripts/run_validation_frontier.py        # Experiment A, 24 cells
python scripts/run_validation_cross_transfer.py  # Experiment B, 15 cells
python scripts/run_validation_projection.py      # Experiment C, 15 cells (requires B)
python -m pytest
```

Every runner resumes by completed atomic record; `--force` is the only way to replace a complete
cell. Experiment C reads Experiment B's frozen dictionaries and must run after it.

Artifacts:

- `results/atom_validation/frontier/{frontier_summary.json,frontier_report.md}`
- `results/atom_validation/cross_transfer/{cross_transfer_summary.json,cross_transfer_report.md}`
- `results/atom_validation/oracle_projection/{oracle_projection_summary.json,oracle_projection_report.md}`
- `results/atom_validation/logs/`: run logs for all three experiments.
