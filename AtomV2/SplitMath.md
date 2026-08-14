# Split Math

## Universal Template
### Sub-ops
- Structural
	- R - reverse: [1, 3, 4, 2, 5, 9] -> [9, 5, 2, 4, 3, 1]
	- T - rotate_left: [1, 3, 4, 2, 5, 9] -> [3, 4, 2, 5, 9, 1]
	- W - swap_pairs: [1, 3, 4, 2, 5, 9] -> [3, 1, 2, 4, 9, 5]
- Pointwise
	-  I - increment: x -> x+1 mod 10
	- N - negate: x -> 10-x mod 10
	- M - multiply_3: x -> 3x mod 10
- Mixed
	- A - add_index: x_i -> x_i + i mod 10, ZERO-BASED indexing (position 0 adds 0 so it passes through unchanged, position 5 adds 5). The generator, canonical forms, and probes must all agree on this convention

### 3 Parts to Every Sub-Op
- Where it came from?
	- For each output slot, which input slot feeds it
	- x[$\pi$(j)] = lookup which input slot feeds output slot j and grab the digit
- Was it multiplied?
	- Most operations multiply by 1 (not at all really), m multiplys by 3, and N multiplys by 9
	- a = what to multiply by
- What was added?
	- List of 6 numbers, one per position, which is the amount added at each slot
	- b = 6 digit list telling what to add at each slot

### The Template
- For each output slot
	- fetch the digit from the chart $\pi$
	- multiply it by a
	- add the list b
	- and wrap around at 10
- output[j] = a * x[$\pi$(j)] + b[j] where x is input list
- Canonical form is triple: ($\pi$, a, b)
- 2 tasks share a function if and only if their triples match

## Triple For Each Sub-Op
Calculate using the template formula where j is index
- [0, 0, 0, 0, 0, 0]: Output is b
	- R - [0, 0, 0, 0, 0, 0]
	- T - [0, 0, 0, 0, 0, 0]
	- W - [0, 0, 0, 0, 0, 0]
	- I - [1, 1, 1, 1, 1, 1]
	- N - [0, 0, 0, 0, 0, 0]
	- M - [0, 0, 0, 0, 0, 0]
	- A - [0, 1, 2, 3, 4, 5]
- [1, 1, 1, 1, 1, 1]: Output at each position is a + b(j). Subtract b. Result is a
	- R - 1
	- T - 1
	- W - 1
	- I - 1
	- N - 9
	- M - 3
	- A - 1
- [0, 1, 2, 3, 4, 5]: Given a and b get $\pi$
	- R - [5, 4, 3, 2, 1, 0]
	- T - [1, 2, 3, 4, 5, 0]
	- W - [1, 0, 3, 2, 5, 4]
	- I - [0, 1, 2, 3, 4, 5]
	- N - [0, 1, 2, 3, 4, 5]
	- M - [0, 1, 2, 3, 4, 5]
	- A - [0, 1, 2, 3, 4, 5]

## The Complete Deck
Template: output[j] = a · x[$\pi$(j)] + b[j], mod 10
π must be a permutation - every slot 0-5 appearing exactly once

| Op | π (chart) | a (multiplier) | b (add-list) |
|--|--|--|--|
| R | (5, 4, 3, 2, 1, 0) | 1 | [0, 0, 0, 0, 0, 0] |
| T | (1, 2, 3, 4, 5, 0) | 1 | [0, 0, 0, 0, 0, 0] |
| W | (1, 0, 3, 2, 5, 4) | 1 | [0, 0, 0, 0, 0, 0] |
| I | (0, 1, 2, 3, 4, 5) | 1 | [1, 1, 1, 1, 1, 1] |
| N | (0, 1, 2, 3, 4, 5) | 9 | [0, 0, 0, 0, 0, 0] |
| M | (0, 1, 2, 3, 4, 5) | 3 | [0, 0, 0, 0, 0, 0] |
| A | (0, 1, 2, 3, 4, 5) | 1 | [0, 1, 2, 3, 4, 5] |

## Composition Rule
For our Surface Operations we have 2 ops, f first then g. We need to get ( $\pi$, a, b) for each Surface Op.
- $\pi$(j) =  $\pi$_f( $\pi$_g(j)) - Read g's chart first then look that up in f's
- a = a_f * a_g, mod 10
- b[j] = a_g * b_f[ $\pi$_g(j)] + b_g(j), mod 10 - f's offset, fetched thorugh g's chart and scaled by g's multiplier, plus g's own offset

**Surface Ops**
- Surface Operations:
	- P1 - R then I
	- P2 - T then A
	- P3 - A then R
	- P4 - M then T
	- P5 - N then W
	- P6 - I then M
	- P7 - W then A
	- P8 - T then N

### The Math
| Op | π (chart) | a (multiplier) | b (add-list) |
|--|--|--|--|
| P1 | (5, 4, 3, 2, 1, 0) | 1 | [1, 1, 1, 1, 1, 1] |
| P2 | (1, 2, 3, 4, 5, 0) | 1 | [0, 1, 2, 3, 4, 5] |
| P3 | (5, 4, 3, 2, 1, 0) | 1 | [5, 4, 3, 2, 1, 0] |
| P4 | (1, 2, 3, 4, 5, 0) | 3 | [0, 0, 0, 0, 0, 0] |
| P5 | (1, 0, 3, 2, 5, 4) | 9 | [0, 0, 0, 0, 0, 0] |
| P6 | (0, 1, 2, 3, 4, 5) | 3 | [3, 3, 3, 3, 3, 3] |
| P7 | (1, 0, 3, 2, 5, 4) | 1 | [0, 1, 2, 3, 4, 5] |
| P8 | (1, 2, 3, 4, 5, 0) | 9 | [0, 0, 0, 0, 0, 0] |

## The 64 Tasks, Grouped by Chart
10 distinct charts. Within a group every task moves digits identically and differs only in value-work (a and b). Classes come from the full triple_key, never from group membership.

### G1 - Stay-Put - pi = (0,1,2,3,4,5) - 9 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P5_P5 | 1 | [0,0,0,0,0,0] | 012345\|1\|000000 | EXCLUDED (identity) |
| P7_P7 | 1 | [1,1,5,5,9,9] | 012345\|1\|115599 | train |
| P3_P1 | 1 | [1,2,3,4,5,6] | 012345\|1\|123456 | held-out L3 |
| P1_P1 | 1 | [2,2,2,2,2,2] | 012345\|1\|222222 | train |
| P3_P3 | 1 | [5,5,5,5,5,5] | 012345\|1\|555555 | held-out L3 |
| P1_P3 | 1 | [6,5,4,3,2,1] | 012345\|1\|654321 | held-out L3 |
| P5_P7 | 9 | [0,1,2,3,4,5] | 012345\|9\|012345 | train |
| P6_P6 | 9 | [2,2,2,2,2,2] | 012345\|9\|222222 | train |
| P7_P5 | 9 | [9,0,7,8,5,6] | 012345\|9\|907856 | train |

### G2 - Swap-then-Rotate - pi = (0,3,2,5,4,1) - 6 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P5_P8 | 1 | [0,0,0,0,0,0] | 032541\|1\|000000 | train |
| P7_P2 | 1 | [1,3,5,7,9,5] | 032541\|1\|135795 | train |
| P7_P4 | 3 | [3,6,9,2,5,0] | 032541\|3\|369250 | held-out L1 |
| P5_P4 | 7 | [0,0,0,0,0,0] | 032541\|7\|000000 | train |
| P5_P2 | 9 | [0,1,2,3,4,5] | 032541\|9\|012345 | held-out L1 |
| P7_P8 | 9 | [9,8,7,6,5,0] | 032541\|9\|987650 | train |

### G3 - Rotate-then-Reverse - pi = (0,5,4,3,2,1) - 6 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P2_P3 | 1 | [0,8,6,4,2,0] | 054321\|1\|086420 | held-out L3 |
| P2_P1 | 1 | [6,5,4,3,2,1] | 054321\|1\|654321 | held-out L1 |
| P4_P1 | 3 | [1,1,1,1,1,1] | 054321\|3\|111111 | train |
| P4_P3 | 3 | [5,4,3,2,1,0] | 054321\|3\|543210 | held-out L3 |
| P8_P1 | 9 | [1,1,1,1,1,1] | 054321\|9\|111111 | train |
| P8_P3 | 9 | [5,4,3,2,1,0] | 054321\|9\|543210 | held-out L3 |

### G4 - Pair-Swap - pi = (1,0,3,2,5,4) - 4 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P6_P7 | 3 | [3,4,5,6,7,8] | 103254\|3\|345678 | train |
| P7_P6 | 3 | [3,6,9,2,5,8] | 103254\|3\|369258 | held-out L1 |
| P5_P6 | 7 | [3,3,3,3,3,3] | 103254\|7\|333333 | held-out L2 |
| P6_P5 | 7 | [7,7,7,7,7,7] | 103254\|7\|777777 | held-out L2 |

### G5 - Rotate-One - pi = (1,2,3,4,5,0) - 6 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P6_P2 | 3 | [3,4,5,6,7,8] | 123450\|3\|345678 | train |
| P2_P6 | 3 | [3,6,9,2,5,8] | 123450\|3\|369258 | train |
| P8_P6 | 7 | [3,3,3,3,3,3] | 123450\|7\|333333 | train |
| P6_P8 | 7 | [7,7,7,7,7,7] | 123450\|7\|777777 | held-out L1 |
| P4_P6 | 9 | [3,3,3,3,3,3] | 123450\|9\|333333 | held-out L2 |
| P6_P4 | 9 | [9,9,9,9,9,9] | 123450\|9\|999999 | train |

### G6 - Rotate-then-Swap - pi = (2,1,4,3,0,5) - 6 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P8_P5 | 1 | [0,0,0,0,0,0] | 214305\|1\|000000 | held-out L2 |
| P2_P7 | 1 | [1,1,5,5,9,9] | 214305\|1\|115599 | train |
| P4_P7 | 3 | [0,1,2,3,4,5] | 214305\|3\|012345 | train |
| P4_P5 | 7 | [0,0,0,0,0,0] | 214305\|7\|000000 | train |
| P8_P7 | 9 | [0,1,2,3,4,5] | 214305\|9\|012345 | train |
| P2_P5 | 9 | [9,0,7,8,5,6] | 214305\|9\|907856 | train |

### G7 - Rotate-Two - pi = (2,3,4,5,0,1) - 9 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P8_P8 | 1 | [0,0,0,0,0,0] | 234501\|1\|000000 | train |
| P2_P2 | 1 | [1,3,5,7,9,5] | 234501\|1\|135795 | train |
| P4_P2 | 3 | [0,1,2,3,4,5] | 234501\|3\|012345 | held-out L1 |
| P2_P4 | 3 | [3,6,9,2,5,0] | 234501\|3\|369250 | train |
| P4_P8 | 7 | [0,0,0,0,0,0] | 234501\|7\|000000 | train (MERGED with P8_P4, one class) |
| P8_P4 | 7 | [0,0,0,0,0,0] | 234501\|7\|000000 | train (MERGED with P4_P8, one class) |
| P4_P4 | 9 | [0,0,0,0,0,0] | 234501\|9\|000000 | train |
| P8_P2 | 9 | [0,1,2,3,4,5] | 234501\|9\|012345 | train |
| P2_P8 | 9 | [9,8,7,6,5,0] | 234501\|9\|987650 | train |

### G8 - Reverse-then-Rotate - pi = (4,3,2,1,0,5) - 6 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P1_P2 | 1 | [1,2,3,4,5,6] | 432105\|1\|123456 | held-out L1 |
| P3_P2 | 1 | [4,4,4,4,4,0] | 432105\|1\|444440 | held-out L3 |
| P3_P4 | 3 | [2,9,6,3,0,5] | 432105\|3\|296305 | held-out L3 |
| P1_P4 | 3 | [3,3,3,3,3,3] | 432105\|3\|333333 | held-out L2 |
| P3_P8 | 9 | [6,7,8,9,0,5] | 432105\|9\|678905 | held-out L3 |
| P1_P8 | 9 | [9,9,9,9,9,9] | 432105\|9\|999999 | train |

### G9 - Reverse-and-Swap - pi = (4,5,2,3,0,1) - 8 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P7_P3 | 1 | [0,8,6,4,2,0] | 452301\|1\|086420 | held-out L3 |
| P1_P7 | 1 | [1,2,3,4,5,6] | 452301\|1\|123456 | train |
| P3_P7 | 1 | [4,6,4,6,4,6] | 452301\|1\|464646 | held-out L3 |
| P7_P1 | 1 | [6,5,4,3,2,1] | 452301\|1\|654321 | train |
| P5_P1 | 9 | [1,1,1,1,1,1] | 452301\|9\|111111 | train |
| P5_P3 | 9 | [5,4,3,2,1,0] | 452301\|9\|543210 | held-out L3 |
| P3_P5 | 9 | [6,5,8,7,0,9] | 452301\|9\|658709 | held-out L3 |
| P1_P5 | 9 | [9,9,9,9,9,9] | 452301\|9\|999999 | train |

### G10 - Reverse - pi = (5,4,3,2,1,0) - 4 tasks
| task | a | b | triple_key | split |
|--|--|--|--|--|
| P6_P1 | 3 | [4,4,4,4,4,4] | 543210\|3\|444444 | held-out L2 |
| P1_P6 | 3 | [6,6,6,6,6,6] | 543210\|3\|666666 | train |
| P3_P6 | 3 | [8,5,2,9,6,3] | 543210\|3\|852963 | held-out L3 |
| P6_P3 | 3 | [8,7,6,5,4,3] | 543210\|3\|876543 | held-out L3 |

## Split Assignment

### The Adjacency Map (drives L1 vs L2)
Cross-token adjacency of Pi_Pj = (last sub-op of Pi, first sub-op of Pj)
| Primitive | first sub-op | last sub-op |
|--|--|--|
| P1 | R | I |
| P2 | T | A |
| P3 | A | R |
| P4 | M | T |
| P5 | N | W |
| P6 | I | M |
| P7 | W | A |
| P8 | T | N |
Only two overlaps exist: A is the last sub-op of both P2 and P7, and T is the first sub-op of both P2 and P8. These overlaps are the ONLY source of multi-task adjacencies, and multi-task adjacencies are the only thing that makes L1 possible.

### Dax Choice: P3
- P2 is DISQUALIFIED as dax: removing P2 from pairs makes every adjacency unique to one task, every held-out cell becomes L2, L1 and the dissociation gap cease to exist
- P3 preserves the most L1 room (both A-last and T-first overlaps survive) and its sub-ops appear elsewhere: A in P2 and P7, R in P1
- Known limitation: with P3 as dax, R is learned only from P1 contexts (single carrier). Logged, accepted
- P7 was the runner-up: keeps only the T-first overlap, and leaves W with single carrier P5. Less L1 room, same thinness problem

### Assignments
- Singletons: ALL 8 go to train, including P3. P3 trained as singleton only is the entire point of the dax test
- EXCLUDED (1): P5_P5, the identity collapse. Teaches nothing, muddies steps-per-token stats
- Merged class rule: P4_P8 and P8_P4 are one function, they travel together. Assigned to train. Note their recipes differ (adjacencies (T,T) vs (N,M)) even though the function matches, extensional equivalence in the wild
- Held-out L3 (15): every pair cell involving P3. P3_P1, P3_P2, P3_P3, P3_P4, P3_P5, P3_P6, P3_P7, P3_P8, P1_P3, P2_P3, P4_P3, P5_P3, P6_P3, P7_P3, P8_P3
- Held-out L1 (8): each has a trained sibling sharing its adjacency

| L1 cell | adjacency | trained sibling covering it |
|--|--|--|
| P2_P1 | (A,R) | P7_P1 |
| P2_P5 | (A,N) | P7_P5 |
| P7_P4 | (A,M) | P2_P4 |
| P7_P6 | (A,I) | P2_P6 |
| P1_P2 | (I,T) | P1_P8 |
| P5_P2 | (W,T) | P5_P8 |
| P6_P8 | (M,T) | P6_P2 |
| P4_P2 | (T,T) | P4_P8 |
- Held-out L2 (6): each has an adjacency unique to it, so holding it out removes that adjacency from training entirely

| L2 cell | adjacency (untrained) |
|--|--|
| P1_P4 | (I,M) |
| P5_P6 | (W,I) |
| P6_P1 | (M,R) |
| P6_P5 | (M,N) |
| P4_P6 | (T,I) |
| P8_P5 | (N,N) |
- Train (34 pairs): P1_P1, P1_P5, P1_P6, P1_P7, P1_P8, P2_P2, P2_P4, P2_P6, P2_P7, P2_P8, P4_P1, P4_P4, P4_P5, P4_P7, P4_P8, P5_P1, P5_P4, P5_P7, P5_P8, P6_P2, P6_P4, P6_P6, P6_P7, P7_P1, P7_P2, P7_P5, P7_P7, P7_P8, P8_P1, P8_P2, P8_P4, P8_P6, P8_P7, P8_P8

### Accounting
34 train + 8 L1 + 6 L2 + 15 L3 + 1 excluded = 64. Training tasks total 42 (34 pairs + 8 singletons)

### Coverage Check (informative appearances in training pairs)
Every non-dax primitive appears at least 3 times in each position among the 34 training pairs (P5 second position is the minimum at 3). Requirement of at least 2 per position is met everywhere. P3 appears in zero pairs by design

### Enumerator Duties (verify all of this mechanically)
- Rebuild all 72 triples from op definitions, diff against this table
- Verify every L1 cell's adjacency appears in at least one training pair
- Verify every L2 cell's adjacency appears in zero training pairs
- Verify no held-out cell's triple_key matches any training cell or singleton
- Verify coverage counts per primitive per position