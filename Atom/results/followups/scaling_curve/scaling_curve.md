# H1 Chunk 22: Task-Count Scaling Curve

Locked seed 17; task prefixes follow the registered order.

| Tasks | Shared mean | Shared worst | Independent mean | Independent storage | Shared storage | Relative storage | Active atoms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6960 | 0.6960 | 0.7180 | 4,354 | 8,482 | 1.948 | 4 |
| 2 | 0.7300 | 0.6980 | 0.7388 | 8,708 | 8,772 | 1.007 | 4 |
| 3 | 0.6867 | 0.5812 | 0.6911 | 13,062 | 9,062 | 0.694 | 4 |
| 4 | 0.6674 | 0.5668 | 0.6868 | 17,416 | 9,352 | 0.537 | 4 |
| 5 | 0.6655 | 0.5668 | 0.6835 | 21,770 | 9,642 | 0.443 | 4 |

Independent storage is computed by summing the exact per-task core LoRA counts. Shared storage is the exact persistent dictionary, coefficient, and head count for each prefix model.
