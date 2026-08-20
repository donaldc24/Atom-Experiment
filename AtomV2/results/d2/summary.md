# D2 - Atom Factorization Audit (read-only)

Registered definition and thresholds in DECISIONS.md (2026-08-20). Semantics are readout-channel: decode(A_i(h)) == f(decode(h)) under the model's own frozen decoder.

## P-1/P-2: sharp signatures and context invariance

| run | sharp atoms (op@rate) | any sub-op? | worst context group (per atom) |
|---|---|---|---|
| A6_s0 | - | False | - |
| A6_s1 | - | False | - |
| A6_s2 | - | False | - |
| A14_s0 | A7=P2@1.00, A9=P7@1.00, A13=P3@1.00, A15=P1@1.00 | False | A7:0.15, A9:0.26, A13:0.00, A15:0.22 |
| A14_s1 | A0=P7@0.99, A1=P6@1.00, A5=P4@0.96, A7=P1@0.99, A12=P3@1.00 | False | A0:0.23, A1:0.12, A5:0.15, A7:0.20, A12:0.00 |
| A14_s2 | A8=P6@1.00, A12=P3@1.00 | False | A8:0.06, A12:0.00 |
| A15_s0 | A1=P1@0.95, A6=P1@0.95, A11=P3@0.96, A13=P3@0.95 | False | A1:0.03, A6:0.05, A11:0.00, A13:0.00 |
| A15_s1 | A0=P5@0.98, A1=P2@0.99, A7=P7@0.97, A9=P1@0.98, A10=P7@0.97, A11=P6@0.99, A13=P3@0.97, A14=P1@0.98 | False | A0:0.09, A1:0.15, A7:0.13, A9:0.10, A10:0.10, A11:0.15, A13:0.00, A14:0.08 |
| A15_s2 | - | False | - |
| A16_s0 | A13=P3@1.00, A15=P1@1.00 | False | A13:0.00, A15:0.25 |
| A16_s1 | A0=P7@0.99, A1=P6@1.00, A7=P1@0.99, A12=P3@1.00 | False | A0:0.20, A1:0.19, A7:0.15, A12:0.00 |

## P-3: composer-free recomposition (the guillotine)

| run | kind | coverage | singletons | trained pairs | L1 | L2 | L3 |
|---|---|---|---|---|---|---|---|
| A6_s0 | sub | 0.00 | - | - | - | - | - |
| A6_s0 | surf | 0.00 | - | - | - | - | - |
| A6_s1 | sub | 0.00 | - | - | - | - | - |
| A6_s1 | surf | 0.00 | - | - | - | - | - |
| A6_s2 | sub | 0.00 | - | - | - | - | - |
| A6_s2 | surf | 0.00 | - | - | - | - | - |
| A14_s0 | sub | 0.00 | - | - | - | - | - |
| A14_s0 | surf | 0.42 | 0.984 | 0.000 | 0.000 | - | 0.000 |
| A14_s1 | sub | 0.00 | - | - | - | - | - |
| A14_s1 | surf | 0.42 | 0.989 | 0.000 | 0.000 | 0.000 | 0.000 |
| A14_s2 | sub | 0.00 | - | - | - | - | - |
| A14_s2 | surf | 0.08 | 0.999 | 0.000 | - | - | 0.000 |
| A15_s0 | sub | 0.00 | - | - | - | - | - |
| A15_s0 | surf | 0.17 | 0.940 | 0.000 | - | 0.000 | 0.000 |
| A15_s1 | sub | 0.00 | - | - | - | - | - |
| A15_s1 | surf | 0.58 | 0.985 | 0.000 | 0.000 | 0.000 | 0.000 |
| A15_s2 | sub | 0.00 | - | - | - | - | - |
| A15_s2 | surf | 0.03 | 0.897 | - | - | - | 0.000 |
| A16_s0 | sub | 0.00 | - | - | - | - | - |
| A16_s0 | surf | 0.17 | 0.971 | 0.000 | 0.000 | - | 0.000 |
| A16_s1 | sub | 0.00 | - | - | - | - | - |
| A16_s1 | surf | 0.28 | 0.997 | 0.000 | 0.000 | 0.000 | 0.000 |

## P-4/P-5: swaps and selectivity

| run | duplicate pairs (mean acc drop) | selective ablation (atom: in/out ratio) |
|---|---|---|
| A6_s0 | none | - |
| A6_s1 | none | - |
| A6_s2 | none | - |
| A14_s0 | none | A7(P2): 1.5, A9(P7): -, A13(P3): 0.1, A15(P1): - |
| A14_s1 | none | A0(P7): 1.1, A1(P6): 1.1, A5(P4): 0.9, A7(P1): -, A12(P3): - |
| A14_s2 | none | A8(P6): 0.7, A12(P3): - |
| A15_s0 | surf:P1 A1<->A6: 0.000, surf:P3 A11<->A13: 0.000 | A1(P1): -, A6(P1): -, A11(P3): -, A13(P3): - |
| A15_s1 | surf:P7 A7<->A10: -0.002, surf:P1 A9<->A14: 0.002 | A0(P5): 1.0, A1(P2): -, A7(P7): -, A9(P1): -, A10(P7): -, A11(P6): -, A13(P3): 0.0, A14(P1): - |
| A15_s2 | none | - |
| A16_s0 | none | A13(P3): 0.1, A15(P1): - |
| A16_s1 | none | A0(P7): 1.2, A1(P6): 1.0, A7(P1): -, A12(P3): - |

Registered verdict rule: FACTORIZED requires sharp identity + context invariance + selective damage + composer-free novel recomposition. High routed accuracy with collapsed forced programs = surface-conditioned routing programs, not atom factorization. Per-run detail in the JSON files beside this summary.
