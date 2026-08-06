# H1 Experiment Report

Decision: **Not supported**

Preregistered result: **FAIL**. A pass requires all three locked thresholds; no narrative interpretation overrides this result.

## Comparison

| Model | Mean score | Worst task gap | Persistent adaptation params | Relative storage | Active rank/atoms |
|---|---:|---:|---:|---:|---:|
| Independent LoRA | 0.6835 | 0.0000 | 21,770 | 100.00% | rank 4 |
| Shared atoms, all 8 | 0.6768 | 0.0289 | 9,642 | 44.29% | 8 |
| Shared atoms, top 4 | 0.6655 | 0.0411 | 9,642 | 44.29% | 4 |

Top-4 evaluation changes active capacity, not stored parameters; all eight learned atoms and every coefficient remain in the shared deployment count.

## Per-task primary comparison

| Task | Independent LoRA | Shared atoms top-4 | Absolute gap | Relative retention |
|---|---:|---:|---:|---:|
| SST-2 | 0.7180 | 0.7080 | 0.0100 | 98.61% |
| MRPC | 0.7596 | 0.7673 | -0.0077 | 101.01% |
| RTE | 0.5957 | 0.5668 | 0.0289 | 95.15% |
| QNLI | 0.6740 | 0.6560 | 0.0180 | 97.33% |
| QQP | 0.6704 | 0.6293 | 0.0411 | 93.87% |

The gap is LoRA minus shared top-4 in absolute score units; a negative value means the shared system scored higher.

## Locked threshold decision

| Criterion | Observed | Required | Result |
|---|---:|---:|---:|
| Quality retention | 97.36% | >= 97.00% | PASS |
| Worst task gap | 0.0411 | <= 0.0300 | FAIL |
| Relative storage | 44.29% | <= 50.00% | PASS |

## Per-seed primary scores

| Model | Seed | SST-2 | MRPC | RTE | QNLI | QQP | Seed mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent LoRA | 17 | 0.7180 | 0.7596 | 0.5957 | 0.6740 | 0.6704 | 0.6835 |
| Shared atoms, all 8 | 17 | 0.7060 | 0.7650 | 0.5668 | 0.6860 | 0.6600 | 0.6768 |
| Shared atoms, top 4 | 17 | 0.7080 | 0.7673 | 0.5668 | 0.6560 | 0.6293 | 0.6655 |

## Across-seed population standard deviation

| Model | SST-2 | MRPC | RTE | QNLI | QQP |
|---|---:|---:|---:|---:|---:|
| Independent LoRA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Shared atoms, all 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Shared atoms, top 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Independent final-epoch diagnostic

The independent final-epoch mean was 0.6762; the retained best checkpoints, not final epochs, define the baseline above.

## Coefficient reuse

Mean atom-index counts across analyzed seeds:

| Atom indices | Dead atoms | Task-exclusive atoms | Reused by 2+ tasks |
|---:|---:|---:|---:|
| 8.00 | 0.00 | 0.00 | 7.00 |

Mean task-by-atom absolute coefficient magnitude:

| Task | Atom 0 | Atom 1 | Atom 2 | Atom 3 | Atom 4 | Atom 5 | Atom 6 | Atom 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SST-2 | 0.12 | 0.19 | 0.14 | 0.08 | 0.12 | 0.08 | 0.13 | 0.13 |
| MRPC | 0.09 | 0.07 | 0.07 | 0.08 | 0.03 | 0.07 | 0.06 | 0.08 |
| RTE | 0.02 | 0.03 | 0.04 | 0.03 | 0.02 | 0.03 | 0.05 | 0.03 |
| QNLI | 0.10 | 0.05 | 0.10 | 0.05 | 0.03 | 0.05 | 0.12 | 0.05 |
| QQP | 0.06 | 0.08 | 0.11 | 0.10 | 0.08 | 0.12 | 0.08 | 0.09 |

Mean pairwise cosine similarity of per-atom absolute coefficient magnitudes:

| Task pair | Similarity |
|---|---:|
| mrpc / qnli | 0.9023 |
| mrpc / qqp | 0.9639 |
| mrpc / rte | 0.9405 |
| qnli / qqp | 0.8843 |
| rte / qnli | 0.9396 |
| rte / qqp | 0.9637 |
| sst2 / mrpc | 0.9436 |
| sst2 / qnli | 0.8954 |
| sst2 / qqp | 0.9326 |
| sst2 / rte | 0.9415 |

Top atoms per task and seed:

| Seed | Task | Top atom indices |
|---:|---|---|
| 17 | SST-2 | 1, 2, 6, 7 |
| 17 | MRPC | 0, 7, 3, 1 |
| 17 | RTE | 6, 2, 7, 5 |
| 17 | QNLI | 6, 2, 0, 3 |
| 17 | QQP | 5, 2, 3, 7 |

## Conclusion

The shared top-4 system misses at least one preregistered threshold, so H1 is not supported by this locked comparison.
