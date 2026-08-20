# D1 - Dax Diagnosis (read-only)

Premise correction: the R8 encoder is digit-only; h0 carries no task conditioning and token-1 execution of any pair is bit-identical to its singleton (verified mechanically below). The probes therefore attach to the composer's token-2 content input and the token-boundary states - the only places 'a region that never existed in training' can exist.

## P-A routing audit (token-2 program flips vs singleton)

| run | trained | L1 | L2 | L3 (P3 first) | L3 (P3 second) | token1 bit-identical everywhere |
|---|---|---|---|---|---|---|
| A6_s0 | 0.176 | 0.112 | 0.020 | 0.126 | 0.364 | True |
| A6_s1 | 0.147 | 0.248 | 0.273 | 0.178 | 0.709 | True |
| A6_s2 | 0.263 | 0.215 | 0.142 | 0.211 | 0.400 | True |
| A14_s0 | 0.288 | 0.362 | 0.338 | 0.465 | 0.610 | True |
| A14_s1 | 0.174 | 0.109 | 0.108 | 0.193 | 0.432 | True |
| A14_s2 | 0.118 | 0.137 | 0.179 | 0.158 | 0.331 | True |
| A16_s0 | 0.290 | 0.235 | 0.311 | 0.478 | 0.625 | True |
| A16_s1 | 0.206 | 0.215 | 0.185 | 0.324 | 0.325 | True |

## P-B content audit (token-1 boundary decode vs truth)

| run | trained decode | L1 decode | L2 decode | L3 decode (P3 first) | L3 decode (P3 second) | L3 raw acc | singleton P3 acc |
|---|---|---|---|---|---|---|---|
| A6_s0 | 0.734 | 0.582 | 0.959 | 0.973 | 0.740 | 0.0000 | 0.973 |
| A6_s1 | 0.987 | 0.984 | 0.986 | 0.967 | 0.981 | 0.0000 | 0.967 |
| A6_s2 | 0.501 | 0.463 | 0.684 | 0.995 | 0.518 | 0.0002 | 0.995 |
| A14_s0 | 0.970 | 0.977 | 0.966 | 0.996 | 0.970 | 0.0000 | 0.996 |
| A14_s1 | 0.993 | 0.992 | 0.992 | 0.999 | 0.994 | 0.0003 | 0.999 |
| A14_s2 | 0.749 | 0.584 | 0.970 | 1.000 | 0.754 | 0.0005 | 1.000 |
| A16_s0 | 0.980 | 0.988 | 0.977 | 0.999 | 0.980 | 0.0000 | 0.999 |
| A16_s1 | 0.988 | 0.993 | 0.988 | 1.000 | 0.990 | 0.0002 | 1.000 |

## P-C geometry audit (NN distance of boundary states to the trained-consumption pool; canonical distance)

| run | trained NN (LOO) | L1 NN | L2 NN | L3 NN | trained canon | L1 canon | L2 canon | L3 canon |
|---|---|---|---|---|---|---|---|---|
| A6_s0 | 17.064 | 16.839 | 16.816 | 21.225 | 1.315 | 1.317 | 1.314 | 1.350 |
| A6_s1 | 16.827 | 16.542 | 16.537 | 20.707 | 1.296 | 1.299 | 1.293 | 1.337 |
| A6_s2 | 17.903 | 17.719 | 17.659 | 21.682 | 1.317 | 1.315 | 1.319 | 1.347 |
| A14_s0 | 13.628 | 13.225 | 13.567 | 15.816 | 1.019 | 1.010 | 1.041 | 1.042 |
| A14_s1 | 13.627 | 13.597 | 13.244 | 15.596 | 0.982 | 0.978 | 0.979 | 0.995 |
| A14_s2 | 15.634 | 15.622 | 15.018 | 17.946 | 1.030 | 1.042 | 1.021 | 1.044 |
| A16_s0 | 14.271 | 13.959 | 14.167 | 17.297 | 1.083 | 1.081 | 1.100 | 1.127 |
| A16_s1 | 12.905 | 12.883 | 12.803 | 15.561 | 1.042 | 1.033 | 1.041 | 1.080 |

Per-cell, per-seed detail lives in the per-run JSON files beside this summary. Outcome-bin reading belongs to the operator; the registered bins are in the D1 brief.
