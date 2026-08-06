# H1 Experiment Report

Decision: **Supported**

Preregistered result: **PASS**. A pass requires all three locked thresholds; no narrative interpretation overrides this result.

## Comparison

| Model | Mean score | Worst task gap | Persistent adaptation params | Relative storage | Active rank/atoms |
|---|---:|---:|---:|---:|---:|
| Independent LoRA | 0.6852 | 0.0000 | 21,770 | 100.00% | rank 4 |
| Shared atoms, all 8 | 0.6793 | 0.0168 | 9,642 | 44.29% | 8 |
| Shared atoms, top 4 | 0.6752 | 0.0267 | 9,642 | 44.29% | 4 |

Top-4 evaluation changes active capacity, not stored parameters; all eight learned atoms and every coefficient remain in the shared deployment count.

## Per-task primary comparison

| Task | Independent LoRA | Shared atoms top-4 | Absolute gap | Relative retention |
|---|---:|---:|---:|---:|
| SST-2 | 0.7300 | 0.7300 | 0.0000 | 100.00% |
| MRPC | 0.7649 | 0.7613 | 0.0036 | 99.53% |
| RTE | 0.5860 | 0.5716 | 0.0144 | 97.54% |
| QNLI | 0.6773 | 0.6507 | 0.0267 | 96.06% |
| QQP | 0.6677 | 0.6626 | 0.0052 | 99.23% |

The gap is LoRA minus shared top-4 in absolute score units; a negative value means the shared system scored higher.

## Locked threshold decision

| Criterion | Observed | Required | Result |
|---|---:|---:|---:|
| Quality retention | 98.54% | >= 97.00% | PASS |
| Worst task gap | 0.0267 | <= 0.0300 | PASS |
| Relative storage | 44.29% | <= 50.00% | PASS |

## Per-seed primary scores

| Model | Seed | SST-2 | MRPC | RTE | QNLI | QQP | Seed mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent LoRA | 17 | 0.7180 | 0.7596 | 0.5957 | 0.6740 | 0.6704 | 0.6835 |
| Independent LoRA | 29 | 0.7460 | 0.7729 | 0.5921 | 0.7020 | 0.6745 | 0.6975 |
| Independent LoRA | 43 | 0.7260 | 0.7623 | 0.5704 | 0.6560 | 0.6583 | 0.6746 |
| Shared atoms, all 8 | 17 | 0.7060 | 0.7650 | 0.5668 | 0.6860 | 0.6600 | 0.6768 |
| Shared atoms, all 8 | 29 | 0.7300 | 0.7545 | 0.5632 | 0.6800 | 0.6609 | 0.6777 |
| Shared atoms, all 8 | 43 | 0.7440 | 0.7650 | 0.5776 | 0.6640 | 0.6668 | 0.6835 |
| Shared atoms, top 4 | 17 | 0.7080 | 0.7673 | 0.5668 | 0.6560 | 0.6293 | 0.6655 |
| Shared atoms, top 4 | 29 | 0.7360 | 0.7535 | 0.5776 | 0.6560 | 0.6763 | 0.6799 |
| Shared atoms, top 4 | 43 | 0.7460 | 0.7631 | 0.5704 | 0.6400 | 0.6821 | 0.6803 |

## Across-seed population standard deviation

| Model | SST-2 | MRPC | RTE | QNLI | QQP |
|---|---:|---:|---:|---:|---:|
| Independent LoRA | 0.0118 | 0.0057 | 0.0112 | 0.0189 | 0.0069 |
| Shared atoms, all 8 | 0.0157 | 0.0049 | 0.0061 | 0.0093 | 0.0030 |
| Shared atoms, top 4 | 0.0161 | 0.0058 | 0.0045 | 0.0075 | 0.0236 |

## Independent final-epoch diagnostic

The independent final-epoch mean was 0.6804; the retained best checkpoints, not final epochs, define the baseline above.

## Coefficient reuse

Mean atom-index counts across analyzed seeds:

| Atom indices | Dead atoms | Task-exclusive atoms | Reused by 2+ tasks |
|---:|---:|---:|---:|
| 8.00 | 0.00 | 1.33 | 6.33 |

Mean task-by-atom absolute coefficient magnitude:

| Task | Atom 0 | Atom 1 | Atom 2 | Atom 3 | Atom 4 | Atom 5 | Atom 6 | Atom 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SST-2 | 0.10 | 0.17 | 0.10 | 0.11 | 0.12 | 0.12 | 0.07 | 0.10 |
| MRPC | 0.06 | 0.07 | 0.07 | 0.05 | 0.06 | 0.06 | 0.07 | 0.07 |
| RTE | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.03 | 0.05 | 0.03 |
| QNLI | 0.08 | 0.06 | 0.09 | 0.06 | 0.07 | 0.06 | 0.10 | 0.07 |
| QQP | 0.09 | 0.11 | 0.09 | 0.09 | 0.09 | 0.10 | 0.11 | 0.10 |

Mean pairwise cosine similarity of per-atom absolute coefficient magnitudes:

| Task pair | Similarity |
|---|---:|
| mrpc / qnli | 0.9439 |
| mrpc / qqp | 0.9544 |
| mrpc / rte | 0.9434 |
| qnli / qqp | 0.9357 |
| rte / qnli | 0.9583 |
| rte / qqp | 0.9530 |
| sst2 / mrpc | 0.8713 |
| sst2 / qnli | 0.8791 |
| sst2 / qqp | 0.9056 |
| sst2 / rte | 0.8690 |

Top atoms per task and seed:

| Seed | Task | Top atom indices |
|---:|---|---|
| 17 | SST-2 | 1, 2, 6, 7 |
| 17 | MRPC | 0, 7, 3, 1 |
| 17 | RTE | 6, 2, 7, 5 |
| 17 | QNLI | 6, 2, 0, 3 |
| 17 | QQP | 5, 2, 3, 7 |
| 29 | SST-2 | 5, 3, 4, 7 |
| 29 | MRPC | 4, 1, 0, 6 |
| 29 | RTE | 4, 2, 0, 6 |
| 29 | QNLI | 6, 7, 0, 4 |
| 29 | QQP | 6, 4, 7, 1 |
| 43 | SST-2 | 1, 4, 5, 0 |
| 43 | MRPC | 2, 6, 4, 7 |
| 43 | RTE | 6, 3, 2, 4 |
| 43 | QNLI | 2, 4, 6, 3 |
| 43 | QQP | 1, 6, 7, 5 |

## Conclusion

The shared top-4 system meets every preregistered quality, worst-task, and storage threshold, so H1 is supported under the locked comparison.
