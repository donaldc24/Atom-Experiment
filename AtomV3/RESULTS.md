# Skill Forge: results and interpretation

Report written after the experiment on September 12, 2026, America/Chicago. This report does not amend the prospectively frozen protocol or its gates.

## Outcome and proposed direction

**All prespecified success gates passed.** The result supports portable skill acquisition in the Atom six-digit world. It does not validate a general AI architecture or establish field-wide novelty.

The proposed architecture treats neural models as learning machinery whose useful results can outlive their weights. A compiler observes a teacher's behavior, proposes an executable skill, and sends that proposal to an independent verifier. Verified skills enter a common library and become available to other systems without retraining or weight transfer. The long-term ambition is cumulative competence whose permanent representation is executable, inspectable knowledge.

What was built here is the complete narrow extraction, verification, acquisition, and execution path. An autonomous planner, learned program language, and general verification system were not built.

## Why this experiment follows from Atom

The earlier repository review found a sharp distinction between successful behavior and reusable internal computation. In the A14 seed-1 checkpoint, the hardest held-out pair panel produced only 2 correct answers out of 6,000 with the original latent handoff. Decoding an intermediate answer and re-encoding it through the model's own interface produced 5,959 correct answers. The main experiment reproduced these figures.

That suggested an alternative research question: can the useful behavior be recovered outside the neural representation and preserved in a representation with explicit composition rules? The D2 atom-composition failures and the A33 model's partial set of reliable operations motivated this question. Those prior results remain unchanged.

The baseline remains descriptive context. Its pair panel and the compiled runtime panel are different workloads; their accuracies and times are not a matched training or speed benchmark. See [neural_baseline.json](results/neural_baseline.json) and the [earlier review](../review/2026-09-12/fresh_review.md).

## Prospective design and provenance

The [protocol](PROTOCOL.md) and executable sources were frozen before result-bearing extraction. Registration occurred at **2026-09-13 02:26:58.406449 UTC**. Finalization occurred at **02:28:03.084770 UTC**, a wall-clock interval of **64.678 seconds**. Design, implementation, synthetic tests, and the later independent audit are outside that interval.

Protocol SHA-256:

```text
66c98f8c9ed2d9791aaaa8ab6a06dfc4245eed91d0adab83e95e3bfb341a9796
```

[REGISTRATION.json](results/REGISTRATION.json) binds 52 source/data files and four checkpoints. [SHA256SUMS](results/SHA256SUMS) records 112 main result artifacts. The independent audit found no hash mismatches. The original AtomV1/AtomV2 source and checkpoints were not modified.

Real teachers were A14 seeds 0, 1, and 2 and A33 seed 1. The weaker A14 seed 2 was retained. For each of eight operations, every source received the same 2,048 unique calibration inputs and 2,048 unique, disjoint validation inputs. Query seed was 20260912. Each origin therefore supplied 32,768 teacher predictions. No neural weights were updated.

The compiler received input arrays and teacher predictions only. Its supplied grammar was:

```text
output[j] = lookup[j][input[source[j]]]
```

It inferred the six source positions and six ten-entry digit tables using conditional majority counts. It did not receive the hidden recipe, reference implementation, or correct outputs. Candidate gates required a position permutation, bijective digit tables, source predictability margin of at least 0.10, and at least 0.98 exact-list agreement with the teacher on validation data.

A separate verifier checked every candidate, including rejected ones, against the reference operation on all 1,000,000 six-digit inputs. It never repaired a candidate or fed truth back to fitting. Admission required both the frozen compiler gates and exhaustive correctness. Synthetic control programs were excluded from the real archive.

## Extraction and complementary acquisition

| Real source | Exhaustively exact candidates | Admitted under all frozen gates |
|---|---:|---:|
| A14 seed 0 | 8 / 8 | 7 / 8 |
| A14 seed 1 | 8 / 8 | 7 / 8 |
| A14 seed 2 | 7 / 8 | 4 / 8 |
| A33 seed 1 | 6 / 8 | 6 / 8 |
| Total | 29 / 32 | 24 / 32 |

The 24 admitted entries are teacher/operation variants that reduce to **eight distinct binary programs**, not 24 distinct skills. Two A14 sources met the prospectively specified seven-skill threshold. The union of A14 and A33 supplied all eight operations.

All three A14 P4 candidates were exact but failed the teacher-agreement gate. Consequently, the admitted A14 union covered only seven operations. A33 was the sole admitted source for P4. A14 supplied P1 and P3, the reversal-bearing operations absent from the initial A33 archive.

The complete closure of the initial A33 position permutations contained **18 permutations and no reversal**. Therefore, no sequence of those admitted A33 operators, however long, can implement a transformation whose position action is reversal. Acquiring P1/P3 from A14 expanded the closure to **36 permutations, including reversal**. This is a concrete capability gap and its repair. It is not a claim that the system discovered a universal algebra or reached all 720 six-position permutations.

The acquisition procedure uses supplied operation identities and a fixed source preference. It demonstrates interoperable import following an algebraic gap check; it is not an autonomous agent deciding what to learn.

## Execution after removing neural weights

A fresh subprocess imported neither PyTorch nor the Atom harness. It received inputs and binary operator paths, without target answers or a reference implementation. Filenames retain origin/token provenance, but the runtime never interprets those names as operations: behavior comes from the loaded bytes. Execution seed was 20260913. Source variants were cycled to test interoperability across all four real teachers.

| Sequence length | Unique sequences | Exact outputs | Sequences mixing origins |
|---|---:|---:|---:|
| 2 | 64 | 2,048 / 2,048 | 45 |
| 4 | 100 | 3,200 / 3,200 | 99 |
| 8 | 100 | 3,200 / 3,200 | 100 |
| 16 | 100 | 3,200 / 3,200 | 100 |
| 64 | 100 | 3,200 / 3,200 | 100 |
| 256 | 100 | 3,200 / 3,200 | 100 |
| Total | 564 | 18,048 / 18,048 | 544 |

There were 34,928 batch operator calls, equivalent to 1,117,696 individual input-list operator applications. No runtime counterexample was found. Full-domain correctness of each admitted primitive and a complete shared state interface mathematically imply correctness of their compositions. The long-chain panel checks that extraction, origin mixing, loading, and actual execution preserve that property; it does not independently demonstrate learned long-horizon reasoning.

## Controls and independent checks

| Control or check | Result |
|---|---|
| Exact synthetic teacher | All eight candidates accepted and exhaustively correct |
| Shuffled calibration labels | Zero candidates accepted |
| Coherently wrong synthetic teacher | All eight pass teacher agreement; all eight fail exhaustive correctness |
| Synthetic programs in real archive | Zero |
| Calibration/validation overlap | None within any operation/source |
| Equal query inputs across sources | Confirmed |
| Thirteen synthetic implementation tests | All passed |
| Registered sources and checkpoints unchanged | Confirmed |

Main exhaustive checking covered 56 candidates across seven real/control origins: **56 million candidate-input checks**. Correctness here is correctness of the exported integer program, not a certificate for the teacher network.

The implementation tests included arbitrary non-affine digit permutations, a nonseparable function outside the grammar, shuffled labels, validation isolation from fitting, coherent errors, incomplete-domain certificate rejection, exact binary round-trips, cache eviction, and closure checks.

After the main result, a separate audit used only standard-library code and NumPy and independently reimplemented the eight task operations. It reproduced all 18,048 saved answers, independently checked eight distinct binaries on all one million inputs each, and recomputed both permutation closures. It also checked all registered source/checkpoint hashes and all 112 main artifact hashes. [Audit evidence](analysis/independent_audit.json) and [audit script](analysis/independent_audit.py) are explicitly labeled post-result supplements.

## A surprising rejected candidate: exploratory finding

The A14 seed-2 P7 program matched its teacher on only **23.7793%** of held-out six-digit answers, yet matched the correct operation on **all 1,000,000 inputs**. Because that exported program is exact, its teacher agreement on this panel also measures the teacher's exact task accuracy there. Conditional aggregation within the supplied grammar recovered a correct rule despite mostly incorrect complete teacher answers.

This candidate failed both the 0.98 teacher-agreement gate and the 0.10 source-margin gate. It stayed excluded. Its exact program can be found in the preserved rejected-candidate evidence; it was not silently promoted after seeing verification results.

The contrasting A14 seed-2 P2 candidate had similarly low teacher agreement, **22.7539%**, but was wrong on **990,000 inputs**. Low teacher agreement is therefore neither a reason to trust a recovered program nor proof that it is useless. External correctness checks separate these cases.

Five exact real-source candidates were rejected overall: A14 seed-0 P4, seed-1 P4, and seed-2 P4/P6/P7. This suggests that a future prospectively designed extraction study should distinguish fidelity to an imperfect teacher from correctness of a recovered rule. It does not justify changing this run's admission rule. See [candidate diagnostics](results/verified/archive.json) and [seed-2 fitting evidence](results/harvest/a14_s2/programs.json).

## Hardware and actual resource use

Execution used the user's AMD Ryzen 9 6900HX, with eight cores/sixteen logical processors and about 16 GB installed RAM. Available memory at design time was about 1.7 GiB. The installed Python was 3.11.5, NumPy 2.2.4, and PyTorch 2.9.0+cpu. Neural queries ran sequentially with four threads and batch size 256. No GPU execution was used.

| Resource | Measured value or scope |
|---|---:|
| Eight distinct operator payloads | 528 bytes |
| All 24 admitted origin payloads | 1,584 bytes |
| Peak cached operator payload | 66 bytes |
| Runtime peak process working set | 41,181,184 bytes = 39.27 MiB |
| Largest recorded teacher-harvest process working set | 253,116,416 bytes |
| Compiled execution panel | 12.80 seconds |
| Operator loads / evictions | 34,397 / 34,396 |
| Application-level operator bytes read | 2,270,202 |
| Neural parameter bytes loaded in deployment | 0 |

Payload counts exclude Python, NumPy, input/output arrays, plan metadata, object overhead, filesystem allocation, and OS cache. The runtime uses uint8 views sharing a single 66-byte operator buffer and evicts the old operator before loading a new one. **The AI process does not run in 66 bytes of RAM.** Application reads may be served from the OS file cache; they are not measured physical disk traffic.

The reference A14 seed-1 teacher has 2,499,850 parameters occupying 9,999,400 parameter bytes. Comparing that with 528 exported bytes describes representations of this restricted task family. It does not establish general neural-model compression. The 8.51-second neural baseline and 12.80-second compiled runtime cover different workloads and do not establish a speedup.

## What is valuable, and what remains unproven

The useful local result is that **poor neural composability does not imply that useful computation cannot be recovered from the model**. An explicit executable representation recovered exact operations and enabled interoperability between independently trained sources. The complementary P4 versus P1/P3 coverage makes this more than exporting one healthy model's complete skill set. The rejected P7 candidate additionally illustrates that correct programs may be recoverable from teacher behavior with low complete-answer accuracy.

The central supplied advantages are substantial: the right separable grammar, the digit alphabet, the complete common interface, and an exact task oracle. In this toy world the oracle already implements the desired functions, so simply calling it is a perfectly good solution. The experiment tests recovering those functions from neural behavior, not discovering previously unknown mathematics or outperforming a hand-written solver.

Neural-guided program extraction and reusable program libraries have close precedents: [VIPER](https://arxiv.org/abs/1805.08328), [PIRL](https://proceedings.mlr.press/v80/verma18a.html), and [DreamCoder](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf). The broad concept is not new. This study supplies evidence about the particular Atom teachers and this extraction/acquisition mechanism. It establishes no priority claim for the field; see [RELATED_WORK.md](RELATED_WORK.md).

The strongest next test of the same idea is to remove the answer-containing grammar advantage: use bounded programs with branching and cross-coordinate state, reserve entire program structures for evaluation, and compare extraction with direct program synthesis under equal query and compute budgets. The decisive outcome would be that imperfect neural teachers make verified discovery cheaper or more successful than synthesis without those teachers. Failure to beat that control would mean the grammar and search machinery are doing the useful work, and the proposed role for neural teachers needs revision. That extension has not been run here.

The current result warrants pursuing that falsifiable research direction. It does not warrant declaring that a new foundation for general AI has already been validated.
