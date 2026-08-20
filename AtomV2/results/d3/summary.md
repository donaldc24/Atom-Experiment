# D3 - boundary position probe (read-only)

Registered in DECISIONS.md (2026-08-20). Per-position linear probes on boundary (= singleton final) states: moved = Pa(x)[j] at slice j, orig = x[j] at slice j. Linear probes cannot permute across positions; value maps are probe-invisible bijections. P6 = identity-permutation control.

| run | z0 loc | z0 off-pos | z0 head-only | bnd head-only | bnd full dec | reverse (m/o) | rotate (m/o) | swap (m/o) | calls |
|---|---|---|---|---|---|---|---|---|---|
| A6_s0 | 1.00 | 0.32 | 0.00 | 0.13 | 0.77 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A6_s1 | 1.00 | 0.36 | 0.00 | 0.17 | 0.98 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A6_s2 | 1.00 | 0.33 | 0.00 | 0.13 | 0.58 | 1.00/1.00 | 0.99/1.00 | 0.99/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A14_s0 | 1.00 | 0.41 | 0.12 | 0.52 | 0.98 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A14_s1 | 1.00 | 0.38 | 0.77 | 0.60 | 0.99 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A14_s2 | 1.00 | 0.36 | 0.57 | 0.45 | 0.79 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A15_s0 | 1.00 | 0.31 | 0.00 | 0.31 | 0.78 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A15_s1 | 1.00 | 0.35 | 0.00 | 0.53 | 0.99 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A15_s2 | 1.00 | 0.33 | 0.00 | 0.15 | 0.55 | 1.00/1.00 | 1.00/1.00 | 0.99/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A16_s0 | 1.00 | 0.39 | 0.03 | 0.38 | 0.99 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |
| A16_s1 | 1.00 | 0.40 | 0.56 | 0.52 | 0.99 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | reverse:MIXED, rotate:MIXED, swap:MIXED |

Registered calls: IN-STATE (moved - orig >= 0.20: atoms finished the positional work in-state), EDITOR (orig - moved >= 0.20: draft in place, decoder permutes at readout), DELOCALIZED (both < 0.30: content not positionally organized), MIXED otherwise. Mid-token steps are descriptive and live in the per-run JSONs.
