# Experiment A: Matched Shared-LoRA/Atom Frontier

Status: **COMPLETE**

Atom-specific advantage: **FAIL**. Qualifying matched capacities: none.

Scores first average each task across seeds 17, 29, and 43, then average the five task means. Standard deviations below are population standard deviations.

## Frontier summary

| System | c | Mean | Worst task (score) | Parameters | Tensor bytes | Active ops/token | Checkpoint bytes (17/29/43) |
|---|---:|---:|---:|---:|---:|---:|---|
| shared_lora | 1 | 0.681516 | rte (0.580024) | 2,314 | 9,256 | 1,024 | 17758/17758/17758 |
| shared_lora | 2 | 0.685303 | rte (0.583634) | 3,338 | 13,352 | 2,048 | 21918/21918/21918 |
| shared_lora | 4 | 0.686498 | rte (0.581227) | 5,386 | 21,544 | 4,096 | 30110/30110/30110 |
| shared_lora | 8 | 0.691089 | rte (0.592058) | 9,482 | 37,928 | 8,192 | 46494/46494/46494 |
| shared_atoms | 1 | 0.670172 | rte (0.567990) | 2,334 | 9,336 | 1,024 | 21045/21045/21045 |
| shared_atoms | 2 | 0.672862 | rte (0.566787) | 3,378 | 13,512 | 2,048 | 24901/25205/25205 |
| shared_atoms | 4 | 0.677406 | rte (0.569194) | 5,466 | 21,864 | 4,096 | 33349/33653/33653 |
| shared_atoms | 8 | 0.679316 | rte (0.569194) | 9,642 | 38,568 | 8,192 | 50293/50293/50293 |

## Every task and seed score

### Capacity 1

| System | Task | Seed 17 | Seed 29 | Seed 43 | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| shared_lora | sst2 | 0.702000 | 0.726000 | 0.734000 | 0.720667 | 0.013597 |
| shared_lora | mrpc | 0.766537 | 0.752668 | 0.761820 | 0.760341 | 0.005758 |
| shared_lora | rte | 0.559567 | 0.577617 | 0.602888 | 0.580024 | 0.017768 |
| shared_lora | qnli | 0.676000 | 0.694000 | 0.658000 | 0.676000 | 0.014697 |
| shared_lora | qqp | 0.668625 | 0.656041 | 0.686983 | 0.670550 | 0.012705 |
| shared_atoms | sst2 | 0.704000 | 0.726000 | 0.694000 | 0.708000 | 0.013367 |
| shared_atoms | mrpc | 0.765228 | 0.755037 | 0.755777 | 0.758681 | 0.004639 |
| shared_atoms | rte | 0.570397 | 0.548736 | 0.584838 | 0.567990 | 0.014836 |
| shared_atoms | qnli | 0.654000 | 0.686000 | 0.644000 | 0.661333 | 0.017913 |
| shared_atoms | qqp | 0.634506 | 0.664396 | 0.665661 | 0.654854 | 0.014397 |

### Capacity 2

| System | Task | Seed 17 | Seed 29 | Seed 43 | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| shared_lora | sst2 | 0.694000 | 0.752000 | 0.740000 | 0.728667 | 0.024998 |
| shared_lora | mrpc | 0.771534 | 0.760520 | 0.757336 | 0.763130 | 0.006083 |
| shared_lora | rte | 0.577617 | 0.581227 | 0.592058 | 0.583634 | 0.006136 |
| shared_lora | qnli | 0.672000 | 0.702000 | 0.654000 | 0.676000 | 0.019799 |
| shared_lora | qqp | 0.680457 | 0.656460 | 0.688336 | 0.675084 | 0.013556 |
| shared_atoms | sst2 | 0.702000 | 0.734000 | 0.728000 | 0.721333 | 0.013888 |
| shared_atoms | mrpc | 0.763121 | 0.760784 | 0.757883 | 0.760596 | 0.002143 |
| shared_atoms | rte | 0.574007 | 0.548736 | 0.577617 | 0.566787 | 0.012848 |
| shared_atoms | qnli | 0.664000 | 0.688000 | 0.636000 | 0.662667 | 0.021250 |
| shared_atoms | qqp | 0.649173 | 0.655668 | 0.653943 | 0.652928 | 0.002747 |

### Capacity 4

| System | Task | Seed 17 | Seed 29 | Seed 43 | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| shared_lora | sst2 | 0.724000 | 0.730000 | 0.744000 | 0.732667 | 0.008380 |
| shared_lora | mrpc | 0.763655 | 0.758423 | 0.757883 | 0.759987 | 0.002603 |
| shared_lora | rte | 0.563177 | 0.588448 | 0.592058 | 0.581227 | 0.012848 |
| shared_lora | qnli | 0.698000 | 0.704000 | 0.654000 | 0.685333 | 0.022291 |
| shared_lora | qqp | 0.679241 | 0.649607 | 0.690975 | 0.673274 | 0.017408 |
| shared_atoms | sst2 | 0.708000 | 0.742000 | 0.754000 | 0.734667 | 0.019482 |
| shared_atoms | mrpc | 0.768912 | 0.758956 | 0.755502 | 0.761123 | 0.005685 |
| shared_atoms | rte | 0.555957 | 0.563177 | 0.588448 | 0.569194 | 0.013930 |
| shared_atoms | qnli | 0.664000 | 0.678000 | 0.652000 | 0.664667 | 0.010625 |
| shared_atoms | qqp | 0.649173 | 0.648785 | 0.674183 | 0.657380 | 0.011882 |

### Capacity 8

| System | Task | Seed 17 | Seed 29 | Seed 43 | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| shared_lora | sst2 | 0.710000 | 0.754000 | 0.718000 | 0.727333 | 0.019137 |
| shared_lora | mrpc | 0.752941 | 0.767332 | 0.767332 | 0.762535 | 0.006784 |
| shared_lora | rte | 0.599278 | 0.592058 | 0.584838 | 0.592058 | 0.005895 |
| shared_lora | qnli | 0.724000 | 0.712000 | 0.682000 | 0.706000 | 0.017664 |
| shared_lora | qqp | 0.663940 | 0.651872 | 0.686742 | 0.667518 | 0.014459 |
| shared_atoms | sst2 | 0.706000 | 0.730000 | 0.744000 | 0.726667 | 0.015691 |
| shared_atoms | mrpc | 0.764961 | 0.754495 | 0.764961 | 0.761472 | 0.004934 |
| shared_atoms | rte | 0.566787 | 0.563177 | 0.577617 | 0.569194 | 0.006136 |
| shared_atoms | qnli | 0.686000 | 0.680000 | 0.664000 | 0.676667 | 0.009286 |
| shared_atoms | qqp | 0.660036 | 0.660865 | 0.666840 | 0.662580 | 0.003031 |

## Matched-capacity decisions

| c | Mean delta | Worst delta | Storage ratio | Ops equal | Qualifies |
|---:|---:|---:|---:|---:|---:|
| 1 | -0.011345 | -0.012034 | 1.008643 | yes | FAIL |
| 2 | -0.012441 | -0.016847 | 1.011983 | yes | FAIL |
| 4 | -0.009092 | -0.012034 | 1.014853 | yes | FAIL |
| 8 | -0.011773 | -0.022864 | 1.016874 | yes | FAIL |

## Pareto frontiers

- Exact: shared_lora:c1, shared_lora:c2, shared_lora:c4, shared_lora:c8
- Quality tolerance 0.005: shared_lora:c1, shared_lora:c8

## Runtime by cell

| System | c | Seed | Total seconds | Training seconds | Evaluation seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|---:|
| shared_lora | 1 | 17 | 131.264751 | 128.493679 | 2.771073 | 541,245,440 |
| shared_lora | 1 | 29 | 117.016673 | 114.159751 | 2.856921 | 562,290,688 |
| shared_lora | 1 | 43 | 125.265433 | 122.272555 | 2.992878 | 174,874,624 |
| shared_lora | 2 | 17 | 164.426184 | 160.289955 | 4.136229 | 562,503,680 |
| shared_lora | 2 | 29 | 151.738510 | 147.464663 | 4.273847 | 493,252,608 |
| shared_lora | 2 | 43 | 165.837226 | 162.545865 | 3.291361 | 192,630,784 |
| shared_lora | 4 | 17 | 173.349275 | 170.455716 | 2.893558 | 563,499,008 |
| shared_lora | 4 | 29 | 147.341750 | 143.684113 | 3.657637 | 207,835,136 |
| shared_lora | 4 | 43 | 148.441302 | 144.727765 | 3.713537 | 211,906,560 |
| shared_lora | 8 | 17 | 130.390352 | 127.614979 | 2.775374 | 565,825,536 |
| shared_lora | 8 | 29 | 157.883454 | 154.792120 | 3.091334 | 200,183,808 |
| shared_lora | 8 | 43 | 141.479721 | 137.552387 | 3.927334 | 231,247,872 |
| shared_atoms | 1 | 17 | 126.081810 | 123.034279 | 3.047532 | 542,781,440 |
| shared_atoms | 1 | 29 | 131.357486 | 128.382769 | 2.974717 | 494,690,304 |
| shared_atoms | 1 | 43 | 166.468163 | 161.791485 | 4.676679 | 173,858,816 |
| shared_atoms | 2 | 17 | 162.245468 | 158.605844 | 3.639624 | 543,457,280 |
| shared_atoms | 2 | 29 | 195.608612 | 187.871757 | 7.736855 | 493,129,728 |
| shared_atoms | 2 | 43 | 132.537893 | 129.658377 | 2.879516 | 192,434,176 |
| shared_atoms | 4 | 17 | 157.333888 | 153.828917 | 3.504971 | 564,224,000 |
| shared_atoms | 4 | 29 | 138.542809 | 133.201723 | 5.341086 | 190,980,096 |
| shared_atoms | 4 | 43 | 162.064799 | 158.508724 | 3.556075 | 212,590,592 |
| shared_atoms | 8 | 17 | 119.622270 | 116.873531 | 2.748739 | 566,280,192 |
| shared_atoms | 8 | 29 | 154.629239 | 150.930219 | 3.699020 | 186,716,160 |
| shared_atoms | 8 | 43 | 143.478599 | 140.794580 | 2.684019 | 206,364,672 |

The common frozen base is reported separately: 4,385,920 parameters (17,543,680 raw tensor bytes). Raw predictions, labels, histories, provenance, and compact component paths are retained in each cell JSON.
