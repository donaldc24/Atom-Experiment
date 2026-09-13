**Atom has produced useful evidence about the conditions under which learned computation becomes reusable. The strongest current opportunity is to study and engineer reliable interfaces between learned programs.**

The repository does not yet demonstrate an alternative to a large language model, an independently reusable library of discovered elementary operations, or an inference-memory breakthrough. It does contain a striking reproducible repair of failed composition, a substantial collection of misleading modularity metrics and their counterexamples, and a small learned subsystem whose behavior admits much stronger verification than its original evaluation used.

This assessment distinguishes existing experimental results, fresh post-hoc analysis, and proposed research. The new analysis does not amend any registered outcome. The review covered V1 and V2 specifications, decisions, implementation, diagnostics, and selected raw artifacts through E11. Validation included recomputing the unseen accuracy of all 54 V1 generation-v2 runs from 518,400 saved predictions, checking 5,184 sampled targets with independently implemented operations, replaying the strongest V2 handoff repair, and running 58 existing tests for model, split, leakage, E5, and E11 invariants. No models were trained for this review. A targeted check of related primary research informs the novelty assessment; it is not an exhaustive priority search.

1. **The most valuable established result is that a failed composition can be repaired using the model's own intermediate answer.**

   The D1 addendum describes executing the first learned token-program, decoding its intermediate state into six predicted digits, re-encoding those digits with the same model, and executing the next program. The weights remain frozen. Correct intermediate labels are not supplied to this inference path.

   A fresh replay of A14 seed 1, final step 30,000, produced:

   | Evaluation group | Raw latent handoff | Self decode/re-encode handoff |
   |---|---:|---:|
   | Trained pairs, fresh inputs | 98.7941% | 98.6985% |
   | L1: held-out surface pairings | 43.1563% | 98.6250% |
   | L2: held-out hidden-operation contexts | 39.4167% | 98.3333% |
   | L3: singleton-only P3 placed in compositions | 0.0333% | 99.3167% |

   L3 contains 15 tasks with 400 examples each: **2 of 6,000 correct becomes 5,959 of 6,000 correct**. Its first-boundary decode accuracy is 99.6833%. The original D1 table reports the same pattern across eight checkpoints; the five stronger checkpoints reach about 94.7–99.3% repaired L3, and weaker checkpoints reach about 54.7–77.2%. Reporting all eight matters: canonicalization can expose intermediate errors that a co-adapted continuation previously compensated for.

   This establishes a sufficient repair and strongly implicates the handoff representation. It does not isolate a single cause. The intervention simultaneously supplies an additional frozen decoder/encoder computation, converts the representation into an explicitly meaningful six-digit state, removes residual information, and returns the next program to the kind of input it encountered as a singleton. A co-trained transformer decoder participates in producing that intermediate answer. Therefore “the atom already performed the operation and only needed formatting” is stronger than the evidence.

   The scientific question suggested by this result is precise: **When does information that is recoverable from a learned state become usable by another computation?** Recoverability, usefulness to the original partner, and usefulness to a new partner are three different properties.

   Evidence: [original D1 addendum](../../AtomV2/results/d1/self_bottleneck.md), [checked-in replay mechanism](../../AtomV2/Harness/atomv2/e7_audit.py), [fresh replay script](d1_boundary_replay.py), and [fresh replay output](d1_boundary_replay.json). The original addendum's exact generation entrypoint was not found in the checked-in D1 script; the later harness mechanism reproduces its strongest result.

2. **The negative results are valuable because they expose convincing false positives.**

   In V1's 54-run generation-v2 battery, the privileged oracle averages 99.9988% unseen exact accuracy. The five non-oracle arms average 0.0243%, 0.0162%, 0.0579%, 0.6655%, and 0%. The saved predictions agree with these figures. All 45 non-oracle runs have closed-map coverage of only one of eight named operations.

   One V1 run, A3b seed 0/split 1234, nevertheless has a decoder-based alignment score of 90.19% and apparent coverage of all eight primitive labels. Its unseen accuracy is 0.0521%. The supposedly standalone, depth-matched diagnostic includes another learned atom before decoding. Its favorable result can belong to the surrounding computation rather than the component being credited. This is a concrete failure mode for component evaluation.

   V2 D2 makes a related distinction more directly. Several atoms score approximately 99–100% on particular surface operations from canonical inputs, while forcing those atoms to compose without the learned routing program gives approximately zero pair accuracy on covered tasks. The current evidence supports useful routed programs more strongly than isolated primitive atoms. The registered requirement of interchangeable, context-invariant atom behavior was not met.

   These counterexamples suggest a worthwhile benchmark contribution: supply systems that are genuinely compositional, constant or collapsed, correct only through their decoder, correct only with trained partners, or correct only through a multi-step program. Evaluate which proposed modularity metrics can tell those cases apart. The repository already has much of the material for that benchmark.

   Sources: [V1 results](../../AtomV1/results/v2/summary.md), [the illustrative V1 metrics](../../AtomV1/runs/v2/A3b_0_s1234_b017073/metrics.json), [V1 model](../../AtomV1/e1/model.py), and [D2 results](../../AtomV2/results/d2/summary.md).

3. **Fresh analysis finds a stronger positive result inside E11's correctly reported failure.**

   E11 failed its registered singleton gate. Both screened arms have a worst-token exact accuracy of 4.5%, far below 99%, and stop before pair training. That verdict stands. A33 learns P2 and P4–P8 perfectly on the sampled singleton evaluation, while P1 and P3 fail. Both failures require reversal. Its effective position map is approximately `[5,1,3,2,4,0]`, whereas reversal requires `[5,4,3,2,1,0]`: positions 1 and 4 were not exchanged.

   The architecture offers a useful opportunity for verification. For one token, input-independent transport T and a per-output-position reaction kernel K produce, before a positive row normalization,

   `score_j(y | x) = sum_i T[j,i] * K[j, x_i, y]`.

   The true answer at output position j depends on one source input digit, `x[pi(j)]`. Fix that source digit d and compare the true answer t against any competing digit y. The smallest possible score margin over all other input digits is

   `T[j,pi(j)] * (K[j,d,t] - K[j,d,y])`

   `+ sum_(i != pi(j)) min_v T[j,i] * (K[j,v,t] - K[j,v,y])`.

   Each other input position can be minimized separately. Checking every output position, source digit, and competitor therefore covers all one million six-digit inputs without enumerating them. A positive minimum implies the correct digit wins everywhere for the mathematical operator represented by those matrices.

   The fresh float64 evaluation of the saved A33 weights finds positive margins for **P2, P4, P5, P6, P7, and P8**. Their worst margins are all greater than 0.999998. A32 passes for P4–P8, with its smallest passing margin about 0.999982. P1 and P3 fail, and the audit constructs actual input examples demonstrating the mistakes.

   This is a numerical evaluation of an analytical sufficient condition, **not an interval-arithmetic or machine-checked proof of every floating-point execution**. The large positive margins make it a strong candidate for formal certification. Because the deployed interface is reset to one-hot digits after each token, universal correctness of individual operators implies correctness of arbitrary finite compositions drawn from the passing subset, under clean canonical execution and the stated arithmetic assumptions. This is an induction argument; testing length 16 is no longer the fundamental limit.

   This does not establish hidden-sub-operation discovery. E11 is strongly tailored to the task: it learns position transport and digit transformations over a supplied symbolic state. Nevertheless, it suggests a different research program: **train small operations, verify their behavior, and compile the verified subset into a compositional executable library.** Learning may remain heuristic while acceptance of a learned component becomes much more rigorous.

   Evidence: [E11 protocol](../../AtomV2/H1-Experiment11.md), [exchange implementation](../../AtomV2/Harness/atomv2/exchange_model.py), [new post-hoc margin audit](audit_e11_margins.py), and [audit output](e11_margin_audit.json). These new files are review artifacts, separate from the registered experiment.

4. **Several assumptions in the original framing deserve revision.**

   **The seven named sub-operations are a chosen vocabulary, not a uniquely identifiable atomic basis.** For example, multiplying by 3 twice modulo 10 is negation: `3 * 3 = 9 = -1 mod 10`. I verified `M then M == N` with the repository's exact canonical algebra. This does not invalidate its equivalence-aware split, and a two-call replacement has a different execution cost. It does show why recovering precisely seven named components cannot be treated as the only legitimate discovery. Reuse, replacement, generalization, and cost are better primary criteria than resemblance to the generator's recipe.

   **The difficult shift in V2 is especially revealing because the true data remain in distribution.** Every V2 operation is a bijection on the million-element digit space. Applying any such operation to a uniformly random input preserves the uniform distribution exactly. Thus the intended intermediate digit state is not inherently outside the singleton input distribution. The model can create a representation-level mismatch even when the underlying task state has the same distribution. This statement concerns the true states; model errors can of course distort the distribution of predicted intermediates.

   **The original application rent does not directly reward memory savings.** One complete surface operation executed in one call can be cheaper under per-call rent than two shared sub-operations executed in two calls. Once slots are available, the loss does not price the memory occupied by unused weights or the bytes transferred on a cache miss. It therefore does not establish an economic pressure toward the particular fine-grained reuse originally hoped for.

   **Three calls do not inherently require three models resident at once.** Repeated calls to the same atom can reuse resident weights. Calls to different atoms can be served sequentially, trading memory against transfer time. The one-step experiments are useful tests of learning and granularity, but a one-step success is not a logical prerequisite for paging. A systems objective should measure peak resident bytes, bytes transferred, cache behavior, and latency. The current hard-routing implementation still evaluates every atom, so its logical selection counts are not physical sparse execution.

   **Some robustness failures are required by the information available.** A uniformly random six-digit state contains independent digits. Erasing one digit without redundancy leaves ten equally plausible states, so expected exact recovery of a bijective task is at most 10%, even for an ideal estimator. An already correct bijective operator must change its output if given a different valid input digit. A corrupted-symbol test measures something different from resilience to small analog noise or quantized weights. Combine these only with an explicitly justified objective.

   Sources: [operation algebra](../../AtomV2/Harness/atomv2/ops.py), [original motivation and rent](../../AtomV2/H1Experiments.md), [hard execution implementation](../../AtomV2/Harness/atomv2/model.py), and [E11 perturbation implementation](../../AtomV2/Harness/atomv2/exchange_model.py).

5. **The plausible novelty is a specific mechanism and evaluation package; the broad ingredients have substantial prior art.**

   Structural modules failing to acquire independent functions is already studied by Béna and Goodman in [Dynamics of specialization in neural modules under resource constraints](https://www.nature.com/articles/s41467-024-55188-9). Their result makes the broad negative conclusion a point of connection to established work.

   Explicitly compatible representations and independently trained components are central to Castillo-Bolado and colleagues' [Design and independent training of composable and reusable neural modules](https://accedacris.ulpgc.es/bitstream/10553/106990/1/Design_independent_training.pdf). Calling the proposed direction a neural interface or neural data type therefore does not make it new.

   Representation choices improving composition also appear in Herzig and colleagues' [intermediate-representation study](https://arxiv.org/abs/2104.07478), Ren and colleagues' [iterated learning and simplicial embeddings](https://arxiv.org/abs/2310.18777), and Liang and colleagues' [forced rendering study](https://proceedings.mlr.press/v267/liang25n.html). The latter is particularly relevant to the gap between possessing a factored representation and actually using it compositionally.

   [Model stitching](https://arxiv.org/abs/2106.07682) already studies interchanging learned representations. [Functional Alignment Can Mislead](https://proceedings.mlr.press/v267/smith25a.html) shows why successful alignment itself can overstate representational similarity. A convincing new interface result should hold under multiple downstream uses and restricted adapters, not only one trained continuation.

   Kim's [Drawing with Strangers](https://arxiv.org/abs/2606.10582), already cited in the repo, studies communication across independently trained populations. Sampling strangers from a jointly trained atom library is a weaker separation than crossing independent training histories.

   Loading weights on demand to reduce resident memory is also established systems research; [LLM in a flash](https://aclanthology.org/2024.acl-long.678/) directly studies that goal. Atom would need to demonstrate a distinctive advantage in reusable computation, loading behavior, or both.

   What may be distinctive here is the combined case: strict held-out composition fails; semantic information remains recoverable; a frozen self-generated symbolic handoff repairs the failure; standalone and probe scores would have misclassified the learned components; and a different constrained architecture permits verification of an actually reusable subset. Establishing whether this precise package is publishably new requires a closer related-work comparison and replication beyond this task family. No first-in-literature claim is established by this review.

6. **The next experiment should explain the large repair, then test independence.**

   I would prioritize a small, controlled interface study over another broad architecture inspired by physical terminology. Preserve the existing checkpoints and split, and compare raw handoff, the full self decode/re-encode repair, an equal-compute continuous handoff, and a learned shared state projection. Keep correct intermediate labels out of the repair path. Report all preselected seeds, intermediate correctness, per-cell task success, runtime, and memory.

   The decisive comparison is whether a compact learned interface can recover the repair benefit while sharing its parameters across producer/consumer contexts. If an equal-compute continuous path performs equally well, the result points more toward extra computation or distribution alignment than discrete semantics. If explicit digit canonicalization alone works, that still supports a useful engineered symbolic interface, with a narrower claim.

   Then train libraries independently and swap complete token-programs across them. Start with explicit digit translation as the engineering control. The ambitious treatment is a shared learned interface whose training never sees the held-out partners or held-out compositions. Restrict translator capacity so it cannot simply relearn the tasks. Successful transfer must survive new inputs, new pairings, and independent model histories.

   In parallel, the E11 branch warrants a cheap control: compare its exchange routing against direct permutation selection plus digit mappings, keeping the canonical interface fixed. This separates optimization of multi-swap routing from interface quality. Add more seeds only under a declared protocol; do not retroactively extend the failed E11 screen. Pursue formal numerical bounds or exact compilation of the passing subset before claiming certification.

   The project also needs a second world before scaling. Include operations with cross-position content dependence, some lossy operations, and tasks requiring retained information beyond the visible output. The current digit state happens to be a complete, convenient interface. A useful learned interface must discover what information the continuation needs when that convenience disappears.

7. **The work inspires several separate hypotheses worth testing. None is established here as unique or successful in its new field.**

   | Direction | Specific hypothesis inspired by Atom | A discriminating first test |
   |---|---|---|
   | AI teams and tool workflows | Partner-specific conventions can make a member look competent only inside its original team; a task-state handoff may improve replacement. | Train or configure separate teams, swap one member, and compare free-form history, a fixed state schema, and a learned restricted handoff at matched task information and budget. Measure unseen-team success. |
   | Continual learning and software maintenance | Updating an encoder or decoder can break the meaning of frozen components even when their weights never change. | Freeze a library, update its surrounding codec on new tasks, and test whether a small collection of interface contracts predicts and prevents old-partner failures. Compare against ordinary end-to-end regression tests. |
   | Verified learned programs | A network can serve as a search procedure for operators that are later accepted by a verifier and replaced with exact executable operations. | Extract the E11 passing maps, formally bound numerical error or compile them, and compare exact execution with neural execution. Then attempt a less task-shaped operator family. |
   | Communication and distributed sensing | The best handoff may preserve task-relevant state while deliberately removing producer-specific information. | Train producers with different nuisance encodings; compare reconstruction-focused compression with a bottleneck evaluated on unseen receivers. Measure receiver performance and recoverability of producer identity separately. |
   | Human learning and transfer | Explicitly reconstructing an intermediate result may improve transfer to an unfamiliar procedure by making an internal convention shareable. | Compare matched-time worked examples with and without an external intermediate-state reconstruction, then test novel procedure combinations. This is an educational hypothesis, not a conclusion about human cognition. |

   The most ambitious common question is whether an interface can be learned from what future computations must agree on. Two hidden states would be treated as equivalent when relevant continuations cannot distinguish their meaning, while irrelevant training-history information is discarded. Such an approach must be related carefully to existing representation, abstraction, and information-bottleneck research. The concrete experimental challenge is generalization to continuations that were withheld when learning that equivalence.

8. **The strongest research judgment is to pursue transferable computation with explicit evidence of compatibility.**

   The repository's strength is the willingness to inspect apparent success and keep the failures. Its weakness is occasionally converting a plausible explanation into an exhaustive conclusion: the oracle changes several factors at once; probe readability does not locate the computation; failed screens do not prove broad architectural impossibility; and approximate conservation failing is not evidence against exact conservation. The E9–E11 records mostly preserve these distinctions and correctly stop when their gates fail.

   I would give the D1 repair and the modularity counterexamples the main scientific attention, with the E11 verification opportunity as a bounded second project. I would make memory-efficient deployment a separate measured engineering objective. A paper or benchmark about the gap between recoverable meaning and reusable computation is plausible after the controls and external replication; publication readiness is not established yet.

   A larger breakthrough would be an interface that lets independently learned skills be exchanged, verified, and executed under a bounded working memory budget across less artificial tasks. That outcome would substantiate the original ambition. The current repository provides concrete evidence and inexpensive tests for moving toward it.
