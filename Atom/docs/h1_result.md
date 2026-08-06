# H1 result: compact known-task storage passes; frozen transfer does not

Status as of 2026-08-06:

- The preregistered three-seed H1 comparison is complete and **Supported (PASS)**.
- Frozen-dictionary transfer to held-out QQP is complete and **FAILS** its stronger 95% criterion.
- Seed-17 task-scaling and atom/rank/top-k ablations are complete and exploratory.
- The six chunk-25 controls are complete as a seed-17 diagnostic.

The machine-readable sources for this document are the generated JSON files under `results/`.
Those files are ignored by Git and must be regenerated locally when absent.

## What the core result tests

The locked experiment compares independent rank-4 LoRA adapters with one shared dictionary of
eight rank-1 atoms. Both systems use frozen `prajjwal1/bert-tiny` encoders, the same five GLUE
tasks, 2,000 sampled training examples and up to 500 validation examples per task, three epochs,
and paired seeds 17, 29, and 43. The primary shared result activates the four largest-magnitude
atoms per task and target layer.

A core pass required all of the following:

- Mean-quality retention of at least 97%.
- No per-task absolute score gap greater than 0.03.
- Shared persistent adaptation storage no greater than 50% of independent LoRA storage.

## Core three-seed result: PASS

All locked thresholds passed.

| Model | Mean primary score | Worst task gap | Persistent adaptation parameters | Relative storage | Active capacity |
|---|---:|---:|---:|---:|---:|
| Independent LoRA | 0.6852 | 0.0000 | 21,770 | 100.00% | rank 4 |
| Shared atoms, all 8 | 0.6793 | 0.0168 | 9,642 | 44.29% | 8 atoms |
| Shared atoms, top 4 | 0.6752 | 0.0267 | 9,642 | 44.29% | 4 atoms |

| Locked criterion | Observed | Required | Result |
|---|---:|---:|---:|
| Mean-quality retention | 98.54% | >= 97.00% | PASS |
| Worst per-task gap | 0.0267 | <= 0.0300 | PASS |
| Relative persistent storage | 44.29% | <= 50.00% | PASS |

The paired-seed task table locates the largest gap:

| Task | Independent LoRA | Shared top-4 | Absolute gap | Relative retention |
|---|---:|---:|---:|---:|
| SST-2 | 0.7300 | 0.7300 | 0.0000 | 100.00% |
| MRPC | 0.7649 | 0.7613 | 0.0036 | 99.53% |
| RTE | 0.5860 | 0.5716 | 0.0144 | 97.54% |
| QNLI | 0.6773 | 0.6507 | 0.0267 | 96.06% |
| QQP | 0.6677 | 0.6626 | 0.0052 | 99.23% |

The 97% retention threshold applies to the unweighted aggregate across tasks, not independently to
every row; QNLI has the largest gap and 96.06% per-task retention. Aggregation first averages each
task across seeds and then averages the five task primary scores. MRPC and QQP primary scores blend
accuracy and F1, so this is not a pooled accuracy or example-weighted micro-average.

The result supports the narrow claim that jointly trained, known task adaptations can share
persistent structure. Coefficient diagnostics also show reuse: across seeds, a mean 6.33 of eight
atom indices were selected by at least two tasks, 1.33 were task-exclusive, and none were dead.
Here, reuse means overlap among per-task top-k atom indices; it is not evidence for held-out task
transfer. The result is also not evidence for routing, online learning, or a general mixture-of-
experts system.

The complete generated report is `results/h1_report/h1_report.md`; its strict JSON counterpart is
`results/h1_report/h1_summary.json`.

## Frozen-atom transfer: FAIL

The stronger transfer follow-up trained the dictionary on SST-2, MRPC, RTE, and QNLI, froze only
the learned atom vectors, then trained new QQP coefficients and a QQP head. It used the full locked
2,000/500/three-epoch budget at seed 17. The registered decision required at least 95% of fresh
rank-4 LoRA quality while adding fewer parameters.

| QQP system | Primary score | Fresh-LoRA quality | New parameters |
|---|---:|---:|---:|
| Fresh rank-4 LoRA | 0.6704 | 100.0% | 4,354 |
| Frozen learned atoms + new coefficients/head | 0.6091 | 90.8% | 290 |
| Head only | 0.6260 | 93.4% | 258 |
| Frozen random atoms + new coefficients/head | 0.6267 | 93.5% | 290 |

The learned transfer used only 6.7% as many marginal new parameters, but its 90.85% quality
retention was below 95%. Its 290 new parameters exclude the already learned and reused
8,192-parameter atom dictionary. It also scored below head-only and the seeded random-frozen
dictionary. Therefore the strong transfer claim is not supported by this run. The core H1 result
remains valid because it tests joint compression of known tasks, not transfer to an unseen task.

Artifacts:

- `results/followups/frozen_atom_transfer/frozen_atom_transfer.json`
- `results/followups/frozen_atom_transfer/frozen_atom_transfer.md`
- `results/followups/shared_prefixes/prefix_4/seed_17/`: the four-task source dictionary.

## Task-count scaling

The seed-17 scaling curve uses registered task prefixes in the order SST-2, MRPC, RTE, QNLI,
QQP. Independent storage adds 4,354 parameters per task. Shared storage starts with the fixed
dictionary and then adds 290 parameters per task for coefficients and a head.

| Tasks | Shared mean | Shared worst | Independent mean | Independent storage | Shared storage | Relative storage | Active atoms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6960 | 0.6960 | 0.7180 | 4,354 | 8,482 | 1.948 | 4 |
| 2 | 0.7300 | 0.6980 | 0.7388 | 8,708 | 8,772 | 1.007 | 4 |
| 3 | 0.6867 | 0.5812 | 0.6911 | 13,062 | 9,062 | 0.694 | 4 |
| 4 | 0.6674 | 0.5668 | 0.6868 | 17,416 | 9,352 | 0.537 | 4 |
| 5 | 0.6655 | 0.5668 | 0.6835 | 21,770 | 9,642 | 0.443 | 4 |

The fixed dictionary is uneconomical for a single task and approximately breaks even at two.
Shared storage becomes smaller at three tasks and reaches 44.3% of independent storage at five.
Shared mean quality stays within 0.0220 absolute score of independent LoRA at every prefix, though
this curve is a single-seed diagnostic rather than a new confirmatory test. Because tasks enter in
one fixed order, task count is also confounded with task mix.

Artifacts:

- `results/followups/scaling_curve/scaling_curve.json`
- `results/followups/scaling_curve/scaling_curve.md`
- `results/followups/shared_prefixes/prefix_<1..4>/seed_17/`

## Seed-17 ablations

These ablations keep the sampled data, optimizer, and three-epoch budget fixed. They are
exploratory and do not revise H1's locked thresholds.

### Atom count

| Atoms | Evaluation top-k | Mean | Worst task | Persistent parameters | Reused atoms | Task-private atoms |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.6705 | 0.5740 | 3,378 | 2 | 0 |
| 4 | 4 | 0.6692 | 0.5560 | 5,466 | 4 | 0 |
| 6 | 4 | 0.6697 | 0.5704 | 7,554 | 5 | 0 |
| 8 | 4 | 0.6655 | 0.5668 | 9,642 | 7 | 0 |
| 12 | 4 | 0.6620 | 0.5740 | 13,818 | 7 | 3 |
| 16 | 4 | 0.6583 | 0.5704 | 17,994 | 4 | 9 |

Under the report's explicitly exploratory +/-0.005 saturation rule, N=2 was the smallest tested
dictionary within 0.005 of the best and also produced the best observed grid mean, 0.6705. Every
tested dictionary remained smaller than the 21,770-parameter rank-4 baseline. The reused-atom
counts were non-monotonic as N grew: 2, 4, 5, 7, 7, then 4. Reuse fraction fell for larger
dictionaries and task-private atoms appeared above N=8. Extra dictionary size did not improve mean
quality in this seed; this is not a claim that two atoms are universally optimal.

### Independent LoRA rank

| Rank | Mean | Worst task | Persistent parameters | Operations/token |
|---:|---:|---:|---:|---:|
| 1 | 0.6742 | 0.5812 | 6,410 | 1,024 |
| 2 | 0.6768 | 0.5776 | 11,530 | 2,048 |
| 4 | 0.6835 | 0.5957 | 21,770 | 4,096 |
| 8 | 0.6848 | 0.6101 | 42,250 | 8,192 |

Rank 8 had the best observed mean, but improved on rank 4 by only 0.0013 while nearly doubling
persistent parameters and estimated active adapter operations. The 0.0106 total quality range
shows some rank sensitivity, but no unregistered robustness threshold is applied.

### Active top-k capacity

All rows reload the same core eight-atom checkpoint; changing k affects active compute, not the
9,642 stored parameters.

| Active atoms | Mean | Worst task | Mean change vs k=4 | Operations/token |
|---:|---:|---:|---:|---:|
| 1 | 0.6517 | 0.5776 | -0.0138 | 1,024 |
| 2 | 0.6592 | 0.5668 | -0.0062 | 2,048 |
| 4 | 0.6655 | 0.5668 | 0.0000 | 4,096 |
| 8 | 0.6768 | 0.5668 | +0.0113 | 8,192 |

Estimated adapter operations scale linearly with k; they are not measured latency. Using all eight
atoms gave the best mean; it was the only k within 0.005 of that best result. The preregistered k=4
point matches rank-4 LoRA's estimated adapter operations and trades 0.0113 seed-17 mean score for
half the k=8 estimated adapter operations. Unlike the atom-count ablation, this comparison masks
one fixed N=8 checkpoint rather than retraining the dictionary.

The complete ablation records are
`results/followup_ablations/{h1_followup_ablations.json,h1_followup_ablations.md}`.

## Chunk-25 controls

All six controls completed under the locked seed-17 data selection and three-epoch budget. They
are diagnostics and do not replace or alter the preregistered three-seed H1 decision.

| Control | Mean primary score | What the run establishes |
|---|---:|---|
| Random frozen atoms + trained coefficients/heads | 0.655820 | Competitive, but 0.009669 below learned shared atoms at the same seed |
| Averaged independent LoRA effective updates | 0.639124 | Simple effective-update averaging does not reproduce shared-atom quality |
| Nearest-other-task LoRA retrieval | 0.634651 | Cross-task adapter retrieval does not reproduce shared-atom quality |
| One shared balanced multitask LoRA | 0.685615 | Strongest control; ordinary multitask sharing is a viable alternative explanation |
| Shared atoms with shuffled labels | 0.531454 | Large degradation is consistent with dependence on real task labels |
| Shared atoms without coefficient sparsity | 0.665958 | Essentially unchanged from 0.665488 with registered sparsity at this seed |

The shared multitask LoRA is the most important caveat. At seed 17 it exceeded both shared top-4
atoms (0.665488) and independent LoRA (0.683546) in mean primary score. It also used 5,386
persistent adaptation parameters versus 9,642 for shared atoms. Thus the controls do not support an
atom-specific quality advantage over ordinary balanced multitask sharing. This is a single-seed
follow-up, so it does not overturn the locked PASS; it narrows the interpretation of why sharing
worked.

Random frozen atoms, averaged LoRAs, and nearest-other-task retrieval all scored below learned
shared atoms. Shuffled labels caused the largest drop, which is consistent with learning from task
signal rather than data-independent capacity or leakage. Removing the coefficient L1 penalty
changed the mean by only +0.000469, so the registered sparsity term was not responsible for the
seed-17 quality result. These differences are descriptive; no significance threshold was
preregistered for the controls.

Artifacts:

- `results/controls/seed_17/control_results.json`
- `results/controls/seed_17/control_report.md`
- `results/controls/<control>/seed_17/`: per-control compact checkpoints and records.

Run or resume the suite with:

```powershell
python scripts/run_controls.py
```

## Reproduction commands

```powershell
# Core paired-seed training and preregistered decision
python scripts/run_h1.py
python scripts/summarize_h1.py

# Frozen transfer and task-prefix scaling
python scripts/run_transfer_scaling.py

# Atom-count, LoRA-rank, and active-top-k ablations
python scripts/run_ablations.py

# Completed seed-17 controls
python scripts/run_controls.py

# Offline verification
python -m pytest
```

The runners skip completed records unless `--force` is supplied, so the same commands resume an
interrupted experiment.

## Compact checkpoint and audit trail

The frozen BERT base is never copied into adaptation checkpoints. Independent task directories
contain `adapter.pt` and `heads.pt`; shared directories contain `atoms.pt`, `coefficients.pt`, and
`heads.pt`. They can be reconstructed by loading the named pretrained base and the compact
components.

At each core seed:

| System | Persistent adaptation parameters | Serialized checkpoint bytes |
|---|---:|---:|
| Five independent rank-4 LoRAs | 21,770 | 118,070 |
| Shared eight-atom model | 9,642 | 50,229 |

The parameter decision uses exact scalar counts, not serialized bytes. Byte counts are reported as
a separate implementation diagnostic because file-format overhead differs by component. Every
metrics record also preserves resolved configuration, dataset provenance, task metrics, parameter
categories, runtime, environment, and checkpoint paths.

Persistent adaptation counts exclude the common frozen 4,385,920-parameter BERT base and include
all task heads, coefficients, and stored atom or LoRA tensors. Top-k evaluation changes active
capacity only; it does not remove atoms or reduce the 9,642 shared stored parameters.
