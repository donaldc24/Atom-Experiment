# Experiment Atom V2
**Motivation**
Frontier models are increasingly out of reach for not only individuals but small organizations, and while open weight models help this tremendously it still leaves the question of how does an individual or small org afford to run/host these models? This isnt an issue of total storage, but of peak resident memory. Mixture of Experts addresses compute, not VRAM. Mixtral must hold all 47B paramaters resident to serve arbitary queries despite activation ~13B per token.

Existing compression (4 bit quantization, structured pruning, distillation) aims to target this same broad problem but by squeezing weights while remaining apathetic to cross-task structure, the plausible fact that operations like "compare two quantities", "track an objecta across a paragraph", or "apply a formatting rule" might recur across thousands of different tasks with shared machinery.

Target: Reduce Peak Resident Memory at Inference
Fallback: Even if a query needs only a small working resident set but has a large unused atoms library that lives in RAM/NVMe this would still result in a much larger model becoming runnable on consumer hardware.

**Reference Papers**
- Béna & Goodman, "Dynamics of specialization in neural modules under resource constraints"
- Kim, "Drawing with Strangers: Population Scaling Drives Zero-Shot Mutual Intelligibility in Emergent Sketching"
- Redhardt, Akram & Schug, "Scaling can lead to compositional generalization"
- Lake & Baroni (2018): _Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks_
- Kottur, Moura, Lee & Batra (2017): _Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog_

**The Model**

Key Components:

- Encoder
	- Function: maps the raw length-6 digit list (0–9, all arithmetic mod 10) into a canonical, task-independent shared representation. **E0 amendment R8:** opaque task control moved to the composer after the pre-amendment oracle exposed pair-context shortcuts.
	- Shape: embedding layer (10 digits → 64) + 1 transformer layer with 4 heads (~50k params)
	- Output Dimension: 384 (width 64 × length 6, flattened into one state vector). Task identity is not present in the atom state.
	- Trained jointly with Atoms (representational co-design - allowed by design; the encoder and atoms are permitted to shape each other, atoms shaping *each other* is what we measure against)

- Atoms
	- Count (slots): 16 - deliberately more than the true answers (7 sub-ops, 8 primitives) so the architecture never votes on how many atoms "should" exist
	- Per atom shape: small MLP, 384 → 192 → 384, GELU, applied as a residual (atom adds its change onto the state instead of replacing it) - ~148k params each
	- Type: function-space (each atom is an invoked subnetwork, not a weight patch)
	- Signature: state in → state out (384 → 384). An atom's output must be a valid input for any other atom - that's the closed-map promise, and closed-map error measures whether it's kept
	- Weight sharing between atoms: none. Every slot is fully independent
	- Each atom owns its key: a 32-dim vector used for routing. Keys live in the atoms, not the composer

- Interface
	- The 384-dim state vector passed between composition steps. This is the *only* channel atoms can talk through - no side doors
	- Width 384 is written down as a named baseline number, because narrowing this channel is the first follow-up knob if pidgin behavior shows up again
	- Known confound: the old world was 512 wide, this one is 384. Narrower channels favor specialization, so this world is slightly friendlier to factorization as a side effect. Acknowledged here, isolated later with the registered bandwidth knob

- Composer
	- Function: at each step, looks at the current content state + the active opaque surface token + micro-step within that token, and picks one atom (compares its query against atom keys, Gumbel top-1). It never receives the partner token or absolute token position.
	- Routing options: 17 total - 16 atom keys + 1 dedicated pass key. Pass is a real routing choice with its own key, not an atom that learned to do nothing, so pass usage is inspectable and logged
	- The micro-step embedding resets to 1–3 for each token. Removing absolute token position is load-bearing for calling a singleton-learned P3 program in either L3 position.
	- Shape: small feedforward MLP (~28k params) - logged separately every run, must not grow when atom count grows
	- Memoryless on purpose: no GRU, no carried state. Last time the composer's memory became a side-channel to smuggle answers around the atoms. A memoryless composer means everything must travel through the interface we're measuring
	- Free routing only. No oracle, no forced assignments anywhere in training
	- Number of steps = number of task tokens (1 or 2), and each task token gets up to 3 micro steps (3 is overcomplete just like atom count is so to not tell the system the answer) with an explicit free pass action. No identity primitive, no padding, no learned halting

- Decoder
	- Function: turns the final state back into an output digit list
	- Shape: 1 transformer layer + a per-position head over digits 0–9 (~50k params)

- Loss
	- Task loss: per-position cross-entropy on the output digits
	- Usage cost: rent is charged per atom APPLICATION, not per slot - every time an atom gets picked at a micro-step it costs (coefficient λ_use, swept over 4 registered values). Pass costs nothing. Plenty of shelf space, but every shelf charges - scarcity is what makes reuse worth finding, and paying double rent per token to use smaller atoms is exactly the signal we want visible
	- Nothing else. No manifold supervision, no intermediate supervision, no round-trip loss (gamed twice before, permanently retired)

- Invariants (hard rules)
	- Task tokens are opaque IDs. Sub-op structure never appears in inputs, tokens, or any training signal
	- Probes read, never write - no gradient from any probe or diagnostic ever touches the model
	- The interface is the only inter-atom data channel. The active opaque token is exogenous routing control and carries no prior atom output.
	- "Atom in use" for the census: picked by hard routing more than ε% of the time on eval (ε fixed before first run)
	- Census logs two numbers, not one: atoms-in-use AND steps-per-token. The pair together is the granularity signal, neither alone carries the verdict
	- Routing evaluated with hard top-1; report soft and hard accuracy both — a big gap between them is itself weak evidence of co-adaptation

- Total size: ~2.6M params, CPU-sized. Do not scale up.
- λ_use: 0.0/0.001/0.01/0.1

**Designing the Data**
Fixed-length list of 6 digits (0–9, all arithmetic mod 10 by definition)
- Sampling: inputs drawn uniformly from the full 10^6 input space with a fixed seed
- Data volumes: 1,000 examples per training task, 400 per evaluation task
- Two evaluation sets, both required:
	- seen_heldout - FRESH examples of the training tasks (tests memorization vs generalization on familiar tasks)
	- unseen - examples of the held-out tasks per the split levels (tests composition)
	- Comparing unseen against training examples instead of seen_heldout would confound memorization with composition, so we keep both
- Sub-Operations (Hidden from the System):
	- Structural
		- R - reverse: [1, 3, 4, 2, 5, 9] -> [9, 5, 2, 4, 3, 1]
		- T - rotate_left: [1, 3, 4, 2, 5, 9] -> [3, 4, 2, 5, 9, 1]
		- W - swap_pairs: [1, 3, 4, 2, 5, 9] -> [3, 1, 2, 4, 9, 5]
	- Pointwise
		- I - increment: x -> x+1 mod 10
		- N - negate: x -> 10-x mod 10
		- M - multiply_3: x -> 3x mod 10
	- Mixed
		- A - add_index: x_i -> x_i + i mod 10, ZERO-BASED indexing (position 0 adds 0 so it passes through unchanged, position 5 adds 5). The generator, canonical forms, and probes must all agree on this convention
- Surface Operations:
	- P1 - R then I
	- P2 - T then A
	- P3 - A then R
	- P4 - M then T
	- P5 - N then W
	- P6 - I then M
	- P7 - W then A
	- P8 - T then N
		- P8 caveat: T and N commute, so P8's internal order is unrecoverable - no system and no probe can ever tell "T then N" from "N then T". The answer key for P8 is the SET {T, N}, not the sequence. Any probe touching P8 must be written set-wise. This is deliberate - its the reminder in our own data that forces the metrics to be built correctly
- Held in reserve (do not build now, on record for future dax variants and v2 of this world):
	- rotate_right (T inverse), multiply_7 (M inverse, since 3x7 = 1 mod 10), decrement (I inverse), add_reverse_index (x_i + (5-i))

**Degeneracies We Need to Keep in Mind for Building the Split**

*From the algebra (data-side):*
- Involutions (X then X = identity): R then R, N then N, W then W
	- Applying the op twice undoes it
	- For Splits: All 3 cells at whatever level they surface are one class, the identity class. Force it to train or exclude entirely
- Cross-Family Commutation: {I, N, M} x {R, T, W} - all 9 pairs
	- Pointwise ops dont care where digits sit, so value then shuffle = shuffle then value. Each of the 9 pairs collapses both orders into one function
	- For Splits: (X, Y) and (Y, X) merge into 1 class, train/held-out assignment applies to the class. Holding out (I,R) while training (R,I) is training on the test set
- Structural Commutation: R then W = W then R
	- At length 6, W acts inside pair-blocks and R permutes those blocks, they dont interfere. T commutes with neither R or W
	- For Splits: Same merge treatment as Cross-Family.
- Pointwise commutation: N then M = M then N (both = -3x)
	- two 0 shift affine maps, scaling order doesnt matter
	- For Splits: Merge into one class. Contrast I then M, M then I, I then N, N then I are all distinct functions...4 separate cells, splittable independently. Dont over merge
- Mixed Commutation: A then I = I then A
	- Both are additions (x+i+1) and additions commute. A commutes with nothing else...not N, not M, not any structural op.
	- For Splits: Merge that one pair. Everything else involving A stays distinct.
- Accidental Collisions (composites landing on the same function by coincidence)
	- N then M and M then N both equal x->7x, which is fine thats covered by Pointwise commutation, but also check whether any composite equals a different composite or a singleton through the affine arithmetic
	- For Splits: Dont go case by case. Reduce every one of the 49 sub-op pairs (and later all 64 task cells) to canonical form: permutation table + (a,b) + index-term. Cells with equal canonical forms = one class, whatever the recipe says.
- Singleton Absorption
	- Pair is same as some singleton
	- For Splits: any pair-class whose canonical form matches a singletons goes to train or exclusion never held-out.
- Higher-order cycles: T_6, I_10, M_4, A_10
	- Ops that return to identity only after x amount of steps
	- For Split: Orders 4+ are unreachable at depth 2 because no sub-op appears often enough in any task chain to cycle (verify per-op appearance counts off the table, dont trust this sentence). BUT order-2 cancellations DO fire inside depth-2 tasks - example: task (P8, P5) chains T, N, N, W and the N, N annihilates, collapsing it to T then W. Canonicalization catches these automatically, thats the point, but dont read this section as "nothing applies at depth 2"

*From the architecture (model-side):*
- Pass-abuse and step-smearing
	- Micro-steps + free pass gives gradient descent more room for creative nonsense: atoms doing half-jobs smeared across steps, pass used to dodge structure instead of express it. The pseudo-compositionality adversary gets a bigger stage
	- For the panel: this gets its own named row in the prereg. Tripwires already exist (ablation variance, standalone semantics, closed-map trajectory) but steps-per-token in the census is the first place it shows

**Split Math/Deck**
Math shown in SplitMath.md along with Split

**Split Levels**
Tiers of difficulty in how I hold out test tasks. Not all unseen is equally unseen, Lake & Baroni (2018): _Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks_ paper shows that a system can pass all easy hold-outs and fail a hard one. So instead of 1 train/test split I build 3 graded by how much geniune reuse it takes to pass:
- L1 - Unseen Surface Pairs:
	- System trains on tasks like [P2, P7] and [P4, P1] but never saw the ordered pair [P2, P4]. SO at test I ask for [P2, P4]
	- Every Primitive is familiar only the pairing is new
- L2 - Unseen Sub-Op Contexts:
	- P2 contains A, P4 contains M. If accross the entire training set no task ever causes the sub-op chain "...M then A..." to occur back to back then any test task that does create that hidden adjacency is L2
	- The surface primitives are all trained but a specific sub-op interaction is new
- L3 - The Dax Test
	- One surface operation appears in training only as a singleton, never inside any pair. At test it is composed in both positions, following the Lake & Baroni jump idea. **E0 amendment R8:** L3 tests transfer of a singleton-learned reusable program, but does not uniquely prove hidden-sub-op factorization—a context-independent P3 surface atom can also pass. Granularity is determined jointly by census, steps/token, standalone semantics, and surface-vs-sub-op probes.

**Metrics**
Split into the question they answer:
- Q1: Does it work?
	- Seen Accuracy
	- Unseen Accuracy, reported per split level (L1/L2/L3) and never averaged
	- Dissociation Gap (L1 minus L3), checking that we dont have pseudo-compositionality
- Q2: Is the interface healthy?
	- Closed-map Error (redefined against sub-op lattice), checks that when an atom hands off its result that the result is still a "legal state of the world" and its on the manifold. Checks that not only a co-adapted partner can read it
	- Time-resolved Closed-map error, same number but watched as a a curve over training instead of just a final snapshot. Idea comes from Béna & Goodman, "Dynamics of specialization in neural modules under resource constraints" paper that shows specialization can collapse at a moment when modules start talking
- Q3: What did it factorize into?
	- Sub-op Decodability, Train a small frozen probe  and see from the active atoms states can I read out which sub-ops are in play? High = it disovered the hidden layer
	- Surface Decodability, Same probe asked at surface level operations.
	- Atom Usage Census, under the overcomplete budget with usage cost how many atoms does it actually pay for?
- Q4: Is it co-adapted?
	- Ablation Consistency, Knock out one atom and measure the damage on every task that uses it. A real primitive hurts uniformly everywhere its used, co-adapted fragment would mean damage is wildly partner dependent. THe varience is the metric not the mean
	- Standalone Semantics, Run each atom alone (base + single atom) against your answer key, does it compute some coherent function?
- Q5: Are atoms speaking a shared language or pairwise pidgins
	- Cross-context Transfer Matrix, For each atom show accuracy when composed with each possible partner (seen and unseen pairings). A shared interface = mostly flat rows, pidgin = hot cells on training partners, cold elsewhere. Idea from Kim, "Drawing with Strangers: Population Scaling Drives Zero-Shot Mutual Intelligibility in Emergent Sketching" paper with there mutual-intelligibility test.

## H1 - Atom Factorization
**Hypothesis**
If Atoms can be factorized in such a way that is reusable and composable then a composer should be able to learn how to compose these atoms in new combinations to answer unseen queries while closed map error stays near zero on a data set of operations that allows for shared substructure such that the system can dynamically learn how many atoms to factorize into.

**Sub-Claims:**
- Recombination: Atoms compose into combinations unseen during training with performance comparable to seen pariings
- Ablation: Ablating an atom degrades performance consistently across all tasks that use it
- Standalone Semantics: Atoms retain measurable individual functions

**Falsifing Conditions**
- F1 - Unseen recombinations fail while seen succeed
- F2 - Ablation degredation is high variance
- F3 - Standalone Semantics show atoms dont have some coherent function
- F4 - Atoms of a given task have no or low sub op or surface decodability and/or Closed map error stays consistently high
- F5 - Cross context transfer matrix has hot cells on training partners and cold elsewhere

**Falsified If:**
- F1 & F2 - Atoms may have factorized but that factorization isnt meaningful
- F1 & (F3 &/|| F4) - Atoms are co-adapted
- F1 & F5 - Atoms only learned to talk to specific partners

**Ambigous Results**
- F4 - Ambigous result on its own, could jsut mean Atoms factorized into soemthing uninterprable

**Success Results**
- Unseen Recombinations succeed while seen also succeed, Ablation degredation has low variance, Standalone semantics show coherent function, closed map error moves near zero, and CC transfer matrix shows relativly flat rows.

Atoms are independent meaningful primitves, not co-adapted fragments. Joint training creates a strong attractor toward pseudo-compositionality: Atom A and Atom B each hold half the information for task AB, the the composer only needs to sum them. This reduces training loss and is therefore actively sought by gradient descent. With flexible atom count this only gives pseudo-compositionality more places to hide. Always report per cell accuracy, treat level means as summaries never as measurement, falsification threshold is k of n cells aboce/below X.

**Training Config**
All values registered here. Amendments allowed only during Experiment 0 calibration, frozen once E0 passes. No tuning against Experiment 1 results, ever.

- Optimizer: AdamW, learning rate 3e-4, betas default (0.9, 0.999), weight decay 0.01
- Warmup: linear warmup over first 500 steps, then constant LR. No decay schedule, keep it simple
- Batch size: 128, shuffled each epoch
- Task mixing: uniform over the 42 training tasks within each epoch, except P3 singleton oversampled to ~7,000 examples (frequency control per the Lake & Baroni rationale)
- Training length: FIXED budget of 20,000 steps (~55 epochs over ~48k examples). No early stopping. The final checkpoint is the result, no picking the best one
- Gradient clipping: clip global norm at 1.0
- Gumbel temperature: annealed linearly 2.0 -> 0.5 over the first 10,000 steps, held at 0.5 after
- Loss: task cross-entropy + λ_use × (L1 on soft atom-selection mass per micro-step, pass exempt). Raw magnitude of the rent term vs task loss checked at initialization before first run, per the registered calibration procedure for the λ grid
- Eval cadence: every 1,000 steps run seen_heldout + all three unseen levels + closed-map error (this cadence IS the time-resolved closed-map curve)
- Checkpoints: saved every 2,000 steps plus final. Full metric panel (probes, ablation, census, transfer matrix) runs on the FINAL checkpoint only, plus 3 intermediate checkpoints (steps 5k, 10k, 15k) for the trajectory view
- Seeds: 0, 1, 2. Everything seeded: data generation, split file, init, shuffling. One master seed per run controls all of it

### Experiment 0 - Calibration (Prove the Rig Works)
The new harness is built from scratch, so before any result from Experiment 1 can count I have to prove the instrument itself works. The old experiment gave me a known pattern: an oracle arm (forced routing + intermediate supervision) scores high on unseen while a free arm sits at the floor. If the new harness cant reproduce that pattern then a null result in Experiment 1 is uninterpretable, I wouldnt know if its a real finding or just a broken harness.

- Run 2 arms, 3 seeds each, 6 runs total:
	- A0-Oracle - forced routing + ground truth intermediate state supervision, using the quarantined calibration harness. This is the ONLY place oracle machinery is allowed to exist. None of it touches the Experiment 1 training path. Oracle should also force [atom, pass, pass] for the micro steps per token.
	- A0-Free - identical config but free routing and no supervision beyond task loss. λ_use = 0 for both arms so cost isnt a variable here
- Both arms run on the same data and same split as Experiment 1 will use

**Pass Criteria (qualitative pattern match, NOT number match)**
The old numbers cant be reproduced exactly because the world changed (length 6 not 8, 16 slots not 8, micro-steps added), so calibration is about the pattern not the values:
- A0-Oracle unseen accuracy is high (>70% on L1)
- A0-Free unseen accuracy is at or near the floor (<5% on L1)
- The gap between them is unmistakable, not marginal
- A0-Oracle closed-map error is low, A0-Free closed-map error is high, matching the old drift-tracks-accuracy relationship in direction

**If Calibration Fails**
- If A0-Oracle cant hit high unseen accuracy the rig is broken somewhere (data generator, split, model, or metrics) and nothing proceeds until its found. Debug the harness, not the hypothesis
- If A0-Free somehow scores HIGH on unseen, that is not a pass either, it means the new world leaks somewhere (task tokens not opaque, split contamination, degeneracy class missed) and the split/data pipeline gets audited before anything else runs

**What Experiment 0 is NOT**
- Its not evidence for or against H1. The oracle arm is scaffolding, it exists to validate the instrument and then it gets locked back in quarantine
- No predictions about factorization, census, or decodability are being tested here. The full metric panel still gets logged on all 6 runs though, because free calibration data on how the panel behaves under known-good and known-broken conditions is exactly what makes the panel trustworthy later

### Experiment 1 - Free Atoms vs Varying Costs
- Run 3 Seeds for each of these costs, for a total of 12 runs
	- A1 - λ_use = 0
	- A2 - λ_use = 0.001
	- A3 - λ_use = 0.01
	- A4 - λ_use = 0.1
- Prediction: With the current form of the cost function I expect to see heavy bias towards Atoms factorizing into operations not sub-ops due to the fact they will have to pay double rent to use sub-op. The results I think will almost be U shaped where when cost is too low or zero factorization quality will be weaker (census near 16 but unclear standalone semantics and no clear indicator on if they are sub-op or surface op functions), as where on the opposite end I would expect the opposite where the census is rather small (under 3) and task accuracy even degrading as the system sacrifices it to dodge rent while still havign questionable factorization quality (unclear standalone semantics). Now in the middle of the U is where I expect the positive result to live, cost is not too high or too low so I expect seen_accuracy to be high and unseen to be less than 50% but greater than 1%. 

|Metric| Prediction (A1/A2/A3/A4) |
|--|--|
| Seen Acc | >90% / >90% / >85% / 60-80% |
| Unseen Acc | <1% / 1-20% / 5-25% / 0-10% |
| Dissociation Gap | Low, expect equally fail L1 and L3 / Slightly higher / Slightly higher / Low - Slightly Elevated |
| Closed-map error | >0.90 / < 0.90 / < 0.90 / <0.95 |
| Decodibility? | None / Lean towards Surface / Lean toward Surface / None |
| Census | 16 / 5-12 / 3-8 / < 3 |
| Ablation Variance | High / Medium / Low / Low |
| Standalone Semantics | None / Slight / Slightly More / None |
| CC Transfer Matrix | More Pidgin like / More Flat / More FLat / Less Flat |


**Amendment 8/15/26** 
E0 A0-Free run was identitcal for what E1 A1 is, so below is the real results from that and the new predicted ones based on that:

|Metric| Prediction A0-Free/A1 |
|--|--|
| Seen Acc | 97.4% ± 0.6% |
| Unseen Acc | L1: 80.7% ± 15.0%, L2: 66.5% ± 14.0%, L3: 0.8% ± 0.2% |
| Dissociation Gap | 0.798 ± 0.152 |
| Closed-map Error | seen: 1.051, L1: 1.076, L3: 1.146 (oracle: 0.015). Rises through training 0.74 to 1.05, slow co-adaptive drift shape, not the step-jump |
| Task-identity Leakage (formerly decodability) | sub-op-set: 0.936 (floor 0.50), surface: 0.954 (floor 0.14). Task info highly present and linearly readable in deltas |
| Census (atoms in use) | 7 ± 1.7 |
| Steps per Token | 3.0 ± 0.0, pass rate 0.0. All micro-steps used every token, pass never chosen |
| Ablation Variance (CV median) | 0.437 ± 0.061 (oracle: 0.000) |
| Standalone Semantics | 0.096 ± 0.006 best-acc mean (oracle: 1.000). No atom computes a coherent function alone |
| CC Transfer Matrix | task row std 0.295, transplant row std 0.300 (oracle: 0.000). Legacy conflated metric, partner/input variance decomposition added per amendment |
| Soft-Hard Gap (seen) | -0.0001, none. Routing decisively hard, no mixture exploitation |
| Seed Stability | L1 swings 15 points across seeds (oracle: 1.000 ± 0.000 on all levels, all seeds) |

| Metric | A2 (0.001) | A3 (0.01) | A4 (0.1)
|--|--|--|--|
| Seen Acc | > 95% | > 95% | > 95% |
| Unseen Acc | L1 < 60%, L2 < 30%, L3 ~0 | L1 < 40%, L2 < 20%, L3 ~0 | L1 < 20%, L2 < 10%, L3 ~0 | 
| Dissociation Gap | > 0.70 | > 0.70 | > 0.70 |
| Closed-map Error | > 1 | > 1 | > 1 |
| Task-identity Leakage | No big change | No big change | No big change |
| Canonical Substitution (new) | route agree low, repair delta ~0 | same | same, rent does not make states legible |
| Census | 5-8 | 4-6 | 2-4 |
| Steps per Token | 2.0-2.8, pass appears | 1.2-2.0, pass common | ~1.0, pass rate 0.5+ |
| Ablation Variance | ~0.4, like A1 | 0.3-0.5, slightly lower | low but unstable, few atoms left | 
| Standalone Semantics | ~0.1, slight | 0.1-0.25, slightly more | ~0.1 or worse |
| CC Transfer Matrix | ~0.3 unchanged | ~0.25-0.3, marginally flatter | flat-cold, low variance low level |
| Soft-Hard Gap (seen) | ~0 | ~0 | small gap opens | 
| Seed Stability | volatile like A1 | volatile | most volatile |