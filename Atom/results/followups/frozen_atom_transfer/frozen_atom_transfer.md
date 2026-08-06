# H1 Chunk 21: Frozen-Atom Transfer

Seed: 17; source tasks: sst2, mrpc, rte, qnli; target task: qqp.

| System | QQP primary score | Fresh-LoRA quality | New parameters |
|---|---:|---:|---:|
| Fresh rank-4 LoRA | 0.6704 | 1.000 | 4,354 |
| Frozen learned atoms + new coefficients/head | 0.6091 | 0.908 | 290 |
| Head only | 0.6260 | 0.934 | 258 |
| Frozen random atoms + new coefficients/head | 0.6267 | 0.935 | 290 |

## Decision

Strong frozen-atom transfer: **FAIL**.

The transferred model retained 90.8% of fresh-LoRA quality and used 6.7% as many new parameters.
