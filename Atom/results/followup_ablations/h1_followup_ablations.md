# H1 Follow-up Ablations - Seed 17

These are exploratory single-seed follow-ups. All five tasks, sampled examples, optimizer settings, and the locked three-epoch budget are unchanged from H1.

## Atom-count ablation

Operational near-best selection (an explicitly exploratory +/-0.005 rule): **2 atoms**; counts within tolerance [2, 4, 6, 8]; best observed mean 0.6705.
This rule does **not show** a conventional rise-then-plateau pattern; it identifies the smallest near-best dictionary.
Storage below the independent rank-4 baseline: **yes**; largest relative storage 82.7%.
Reuse is **non-monotonic** in absolute count and **non-monotonic** by fraction. Reused counts by N: 2 -> 4 -> 5 -> 7 -> 7 -> 4.
Do extra atoms become task-private above N=8? **yes**; maximum additional private atoms 9.
Dead atom means the learned coefficient criterion is <= 1e-06; top-k unused means selected by zero tasks. A non-dead atom can therefore be top-k unused.

| Atoms | Eval top-k | Mean | Worst task | Persistent params | vs rank-4 LoRA | Checkpoint bytes | Reused | Reuse | Private | Dead | Top-k unused | Ops/token |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.6705 | 0.5740 | 3,378 | 15.5% | 25,141 | 2 | 100.0% | 0 | 0 | 0 | 2,048 |
| 4 | 4 | 0.6692 | 0.5560 | 5,466 | 25.1% | 33,589 | 4 | 100.0% | 0 | 0 | 0 | 4,096 |
| 6 | 4 | 0.6697 | 0.5704 | 7,554 | 34.7% | 42,037 | 5 | 83.3% | 0 | 0 | 1 | 4,096 |
| 8 | 4 | 0.6655 | 0.5668 | 9,642 | 44.3% | 50,229 | 7 | 87.5% | 0 | 0 | 1 | 4,096 |
| 12 | 4 | 0.6620 | 0.5740 | 13,818 | 63.5% | 66,869 | 7 | 58.3% | 3 | 0 | 2 | 4,096 |
| 16 | 4 | 0.6583 | 0.5704 | 17,994 | 82.7% | 83,765 | 4 | 25.0% | 9 | 0 | 3 | 4,096 |

### Atom-count task scores

| atom_count | sst2 | mrpc | rte | qnli | qqp |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.7020 | 0.7631 | 0.5740 | 0.6640 | 0.6492 |
| 4 | 0.7080 | 0.7689 | 0.5560 | 0.6640 | 0.6492 |
| 6 | 0.7080 | 0.7582 | 0.5704 | 0.6580 | 0.6539 |
| 8 | 0.7080 | 0.7673 | 0.5668 | 0.6560 | 0.6293 |
| 12 | 0.7140 | 0.7608 | 0.5740 | 0.6360 | 0.6253 |
| 16 | 0.6960 | 0.7553 | 0.5704 | 0.6320 | 0.6376 |

## Independent-LoRA rank ablation

Best observed mean quality used rank **8**. The single-seed quality range was 0.0106; no unregistered pass/fail robustness threshold is applied.

| Rank | Mean | Worst task | Persistent params | vs rank 4 | vs shared N=8 | Checkpoint bytes | Ops/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6742 | 0.5812 | 6,410 | 29.4% | 66.5% | 56,630 | 1,024 |
| 2 | 0.6768 | 0.5776 | 11,530 | 53.0% | 119.6% | 77,110 | 2,048 |
| 4 | 0.6835 | 0.5957 | 21,770 | 100.0% | 225.8% | 118,070 | 4,096 |
| 8 | 0.6848 | 0.6101 | 42,250 | 194.1% | 438.2% | 199,990 | 8,192 |

### Rank task scores

| rank | sst2 | mrpc | rte | qnli | qqp |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6940 | 0.7600 | 0.5812 | 0.6680 | 0.6679 |
| 2 | 0.7140 | 0.7626 | 0.5776 | 0.6700 | 0.6600 |
| 4 | 0.7180 | 0.7596 | 0.5957 | 0.6740 | 0.6704 |
| 8 | 0.7160 | 0.7548 | 0.6101 | 0.6660 | 0.6773 |

## Active-atom capacity ablation

All rows reload the same core 8-atom checkpoint; no retraining occurs. Best observed mean used k=**8**; smallest k within 0.005 was **8**.
Estimated active adapter compute is **linear** in k. Persistent storage is unchanged because every row deploys the full dictionary.

| Active atoms | Mean | Worst task | delta mean vs k=4 | Persistent params | vs rank-4 LoRA storage | Checkpoint bytes | Ops/token | vs rank-4 LoRA compute |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6517 | 0.5776 | -0.0138 | 9,642 | 44.3% | 50,229 | 1,024 | 25.0% |
| 2 | 0.6592 | 0.5668 | -0.0062 | 9,642 | 44.3% | 50,229 | 2,048 | 50.0% |
| 4 | 0.6655 | 0.5668 | +0.0000 | 9,642 | 44.3% | 50,229 | 4,096 | 100.0% |
| 8 | 0.6768 | 0.5668 | +0.0113 | 9,642 | 44.3% | 50,229 | 8,192 | 200.0% |

### Active-capacity task scores

| active_atoms | sst2 | mrpc | rte | qnli | qqp |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6900 | 0.7553 | 0.5776 | 0.6160 | 0.6198 |
| 2 | 0.7040 | 0.7587 | 0.5668 | 0.6380 | 0.6287 |
| 4 | 0.7080 | 0.7673 | 0.5668 | 0.6560 | 0.6293 |
| 8 | 0.7060 | 0.7650 | 0.5668 | 0.6860 | 0.6600 |
