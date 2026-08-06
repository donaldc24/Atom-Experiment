# Experiment B: Crossed Frozen-Atom Transfer

Status: **COMPLETE**

Each target is held out in turn at seeds 17, 29, and 43. Source atoms are trained on the other four tasks, then frozen; only target coefficients and a target head are fitted.

## Primary decision

Strong transfer: **PASS**. Learned top-4 aggregate retention was 95.83%; the predeclared requirement was >= 95%.

Control-aware strong reusable-basis support: **FAIL**. This combined interpretation requires the primary criterion and every diagnostic below to pass.

| System | Aggregate mean primary score |
|---|---:|
| Fresh rank-4 LoRA | 0.685209 |
| Learned frozen atoms, all 8 | 0.660276 |
| Learned frozen atoms, top 4 | 0.656660 |
| Head only | 0.655065 |
| Matched random frozen atoms, top 4 | 0.656644 |

## By held-out target

| Target | Fresh LoRA | Learned all-8 | Learned top-4 | Retention | Head only | Random top-4 | Learned-head | Learned-random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sst2 | 0.730000 | 0.679333 | 0.680000 | 93.15% | 0.682000 | 0.680000 | -0.002000 | +0.000000 |
| mrpc | 0.764936 | 0.759103 | 0.758579 | 99.17% | 0.755773 | 0.757873 | +0.002805 | +0.000706 |
| rte | 0.586041 | 0.577617 | 0.572804 | 97.74% | 0.570397 | 0.575211 | +0.002407 | -0.002407 |
| qnli | 0.677333 | 0.657333 | 0.648000 | 95.67% | 0.650667 | 0.649333 | -0.002667 | -0.001333 |
| qqp | 0.667736 | 0.627992 | 0.623916 | 93.44% | 0.616487 | 0.620803 | +0.007429 | +0.003113 |

## By seed

| Seed | Fresh LoRA | Learned all-8 | Learned top-4 | Retention | Head only | Random top-4 |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.683546 | 0.657079 | 0.653942 | 95.67% | 0.655458 | 0.657762 |
| 29 | 0.697483 | 0.674085 | 0.669656 | 96.01% | 0.666582 | 0.670974 |
| 43 | 0.674600 | 0.649663 | 0.646381 | 95.82% | 0.643154 | 0.641195 |

## Diagnostics

| Diagnostic | Observed | Required | Result |
|---|---:|---:|---:|
| `every_target_seed_mean_retention_at_least_0_90` | 0.931507 | 0.900000 | PASS |
| `learned_mean_exceeds_head_only_by_0_005` | 0.001595 | 0.005000 | FAIL |
| `learned_mean_exceeds_random_frozen_by_0_005` | 0.000016 | 0.005000 | FAIL |
| `marginal_new_parameters_at_most_10_percent_fresh_task_state` | 0.066605 | 0.100000 | PASS |

## Parameter accounting

| Quantity | Parameters |
|---|---:|
| Fresh LoRA target state | 4,354 |
| Learned target marginal state | 290 |
| Reused frozen dictionary | 8,192 |
| Learned target total with dictionary | 8,482 |

Marginal and total-with-dictionary counts are reported separately. Raw predictions, labels, runtimes, provenance, and compact checkpoint paths are indexed by the JSON summary.
