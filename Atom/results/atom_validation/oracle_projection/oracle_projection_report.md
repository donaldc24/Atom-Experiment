# Oracle held-out LoRA-to-atom-span projection

Strong learned-span support: **FAIL**.

## Aggregate results

| Fresh LoRA | Learned all 8 | Learned top 4 | Random all 8 | Learned retention | Learned - random |
|---:|---:|---:|---:|---:|---:|
| 0.685209 | 0.634117 | 0.634862 | 0.629517 | 92.544% | +0.004601 |

| Span | Relative Frobenius error | Explained energy |
|---|---:|---:|
| Learned | 0.988186 | 2.349% |
| Random | 0.999737 | 0.053% |

## Per-target all-eight oracle quality

| Target | Fresh LoRA | Learned span | Random span | Retention |
|---|---:|---:|---:|---:|
| SST2 | 0.730000 | 0.657333 | 0.656000 | 90.046% |
| MRPC | 0.764936 | 0.748025 | 0.748025 | 97.789% |
| RTE | 0.586041 | 0.578821 | 0.574007 | 98.768% |
| QNLI | 0.677333 | 0.604667 | 0.597333 | 89.272% |
| QQP | 0.667736 | 0.581740 | 0.572218 | 87.121% |

## Locked decision checks

| Check | Result |
|---|---|
| aggregate quality retention | FAIL |
| every target retention | FAIL |
| lower reconstruction error than random | PASS |
| quality advantage over random | FAIL |

All-eight projection is the span-coverage result. Top-4 is a matched-active-compute diagnostic. Oracle coefficients are derived from target LoRA weights and are not a deployable generator.
