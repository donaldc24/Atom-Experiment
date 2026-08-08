# Atoms: Compositional Primitives as a Standalone Replacement for Dense Models

**Preregistered research prospectus — v1.1**

> **Changes from v1.0.** (1) The weight-space/function-space fork is **closed by a rank argument** (§2.1) — weight-space atoms are incompatible with H4. (2) H3's decision procedure was written assuming weight-space atoms and **does not transfer**; replaced in §5. (3) E0 **downgraded** from go/no-go to a weak prior, for the same reason. (4) H2 now distinguishes *peak* from *expected* residency. (5) New §10 records the knowledge/skill separation as an **unvalidated proposal**, not a finding. H1, H2, H4, H5, H6 are otherwise unchanged — they were stated independently of atom representation.

---

## 1. Motivation

Frontier models are increasingly out of reach for individuals and small organizations, not because of total storage but because of **peak resident memory**. Mixture-of-Experts addresses compute, not VRAM: Mixtral must hold all 47B parameters resident to serve arbitrary queries despite activating ~13B per token.

Existing compression (4-bit quantization, structured pruning, distillation) squeezes weights while remaining agnostic to *cross-task* structure — the plausible fact that operations like "compare two quantities," "track an entity across a paragraph," or "apply a formatting rule" recur across thousands of tasks with shared machinery. That redundancy is the seam this project mines.

**Target resource:** peak resident memory at inference.
**Fallback result (valid even if compression fails):** if a query needs only a small resident working set and unused atoms live in RAM/NVMe, a large model becomes runnable on consumer hardware even with a library the same total size as the parent.

---

## 2. Core claim

A large teacher model's capabilities can be re-expressed as compositions over a library of reusable primitives (**Atoms**) plus a learned **composer**, such that inference requires no teacher or frozen parent model, and such that at equal peak-memory budget the system matches or exceeds a distilled dense model of the same footprint.

### 2.1 Atom representation — resolved by rank argument

Low-rank deltas composed by summation are **incompatible with H4**, and this is arithmetic rather than preference.

If each atom is rank *r* and composition is addition, then rank(Σᵢ Aᵢ) ≤ n·r. Reconstructing a full-rank d×d weight matrix with no parent to supply it requires n ≥ d/r atoms **for a single layer**. At d=4096, r=16 that is 256 atoms per layer with zero cross-layer reuse — at which point the system is a low-rank factorization of the model, a known compression technique that plateaus below 4-bit quantization.

Summation-composed atoms only work when something else supplies the substrate. That something is the frozen parent. **H4 forbids the parent, therefore weight-space atoms are foreclosed.**

Supporting evidence, independent of the rank argument: *LoRA Learns Less and Forgets Less* (Biderman et al., 2024) finds LoRA substantially underperforms full fine-tuning on knowledge-heavy domain adaptation while better preserving base capabilities — low-rank deltas steer behavior, they do not install knowledge.

**Consequence:** atoms are **function-space** — invoked subnetworks composed by routing and sequencing, not summation. This raises the salience of H5-depth (composer-as-interpreter) as the primary remaining architectural risk.

> **Empirical support from E1 (added after the fact; see `e1/DECISIONS.md` D4/D18).** The rank obstruction recurs one level down, inside a function-space atom, which makes E1 an unplanned test of this section's argument. An E1 atom is an MLP `512 → 256 → 512`, so its residual output is confined to the fixed 256-dimensional column space of `W₂`. Representing `rotate_left` as a block permutation `P` needs the delta `Ph − h` to span `64 × (8 − #cycles) = 448` dimensions. It does not fit, and no hand construction exists — which is why E1's oracle arm could not be hand-initialised as its spec described.
>
> The system nonetheless reaches a perfectly compositional solution: with intermediate states pinned to the encoder manifold, every atom becomes a verified closed map, `enc(z) → atomⱼ → enc(pⱼ(z))` at **0.999** for partners it never trained with. The obstruction dissolves because **the encoder learns a code in which each primitive's delta fits the subspace the operator can reach.**
>
> That is precisely the asymmetry between the two representations, and it is the reason function-space survives the argument that forecloses weight-space. A weight-space atom must hit a substrate fixed in advance by the parent, so the rank arithmetic in this section is binding. A function-space atom and its representation are **co-designed** — the code is free to move so the operators can reach what they must. The §2.1 decision was made on theoretical grounds alone; this is independent evidence for it, and it identifies *representational co-design* rather than "subnetworks not summation" as the load-bearing property.
>
> Caveat: E1 also shows the code being reachable is not sufficient. Composition still failed end-to-end (0.004 → 0.938 across oracle configurations) until the intermediate state was explicitly held on the manifold — residual composition does not preserve it for free. See `e1/DECISIONS.md` D12/D16.

---

## 3. Hypotheses

Hypotheses are ordered by dependency, not importance. **H6 is a precondition on interpreting H1 and H3** and must be tested first.

### H6 — Atom factorization *(precondition)*

Atoms are independently meaningful primitives, not co-adapted fragments.

Joint training creates a strong attractor toward **pseudo-compositionality**: atom A and atom B each hold half the information for task AB, so the composer need only sum them. This reduces training loss and is therefore actively sought by gradient descent, but neither atom is a reusable primitive. H1's marginal-cost curve would bend for a reason that does not generalize.

**Sub-claims:**
- **(a) Recombination.** Atoms compose into pairings unseen during training with performance comparable to seen pairings.
- **(b) Ablation consistency.** Ablating an atom degrades performance consistently across all tasks that use it; the *variance* of the degradation profile is the co-adaptation metric.
- **(c) Standalone semantics.** Atoms retain measurable individual function under independent probing (linear probe, or base + single atom).

**Falsified if:** unseen recombinations fail while seen pairings succeed; ablation degradation is high-variance and partner-specific; atoms have no coherent standalone function.

**Structural countermeasures during training:**
- Stochastic atom dropout — no atom can rely on a specific partner being present.
- Randomized co-occurrence — no stable pairing exists to co-adapt into.
- Sequential introduction with frozen library — new atoms cannot reshape old ones to meet halfway. Strongest guarantee, and already required by H1's incremental setup. Cost: no refactoring when a better decomposition emerges; mitigate with periodic consolidation passes.

---

### H1 — Marginal compositionality

The parameters required to reach fixed accuracy on capability *n+1* strictly decrease with library size *n*, converging toward the cost of a new *composition* rather than a new *atom*.

**Falsified if:** marginal cost is flat, or the curve fails to bend on a synthetic task family with guaranteed compositional structure.

**Threshold set in advance:** a curve that bends but never flattens (decreasing marginal cost without convergence) **counts as a positive result** and will be reported as such. This is committed to now to prevent post-hoc goalpost movement in either direction.

**Interpretation gate:** uninterpretable unless H6(a) passes. Run the recombination test before trusting any marginal-cost plot.

---

### H2 — Working-set sparsity

Any single query activates a small subset of the library, so peak resident memory ≪ total library size.

**Falsified if:** activation is dense, or the working set churns fast enough that streaming cost dominates.

**H2a — peak vs. expected residency (distinct claims).** *Peak* residency is set by the worst single query and is what the core claim's memory budget refers to. *Expected* residency is set by the typical query and is what determines paging traffic. Report both; do not let a good expected number stand in for a peak number.

**H2b — cross-query locality.** A given user's queries are not uniformly distributed over task space, so working sets are correlated *across* queries, not merely sparse within one. High locality permits deliberately undersizing the hot set for a user's distribution and absorbing a stall on rare queries — converting a hard memory constraint into a latency distribution. This is a favourable trade but is **not** a reduction in peak residency, and must not be reported as one.

*Mechanism is caching; import it rather than reinvent it (Denning working-set model, LRU/LFU, prefetching). The Mixtral-offloading work already does LRU plus speculative expert prefetch.*

*Novel element:* the always-resident core is **per-user**, not global. DeepSeek's shared experts are fixed for all users; a locality-adapted core is not.

*Interaction with the generative path:* caching and generation are **substitutes, not complements**. Memoizing generated atoms is storage again, so a hypernetwork only earns its keep on misses. Under high locality most requests are hits and the generator sits idle while consuming resident VRAM. Measure locality before committing to generation.

*Scope note:* H2b tests none of H1/H3–H6. It is a deployment-layer optimization and must not precede the H6 factorization gate.

**Critical sub-question, to be measured early:** does the working set change **per query** or **per token**?
- *Per query* → paging from RAM/NVMe is likely sufficient; the hypernetwork is a research question, not a necessity.
- *Per token* → PCIe bandwidth (~32 GB/s on 4.0 x16) becomes the bottleneck and generation is the only viable answer.

Note that MoE routes per token *because experts are cheap to keep resident* — the assumption this project removes.

---

### H3 — Atom-space geometry *(store vs. generate)*

> **Scope reduced in v1.1.** H3 no longer selects the atom representation — §2.1 did that. H3 now decides only **store-and-page vs. generate-on-demand**, and H2b may pre-empt even that.
>
> **Known measurement problem.** *r* as originally defined — hypernetwork reconstruction error on generated weight matrices — presumes atoms are weight matrices. Function-space atoms have structure and invocation semantics, and "reconstruction error of a generated atom" is not well-defined for them. **A function-space operationalization of *r* is an open problem** (see §9). Candidate substitutes: behavioural equivalence of generated vs. stored atoms on a probe set; edit distance over discrete atom structure; performance of the composed system under generated atoms. None is yet chosen. Treat the table below as describing the geometry *question*, not a procedure ready to run.

The library occupies a position on the orthogonality–smoothness axis characterized by **mutual predictability *r***, measured as the degree to which held-out atoms are predictable from task conditioning given the rest of the library.

| | **Low *r*** — smooth overcomplete manifold | **High *r*** — near-orthogonal basis |
|---|---|---|
| **Architecture** | Generative hypernetwork, \|H\| < \|library\| | Stored library, paged from RAM/NVMe |
| **Gains** | Interpolation to unseen task compositions; graceful degradation; continuous atom space; bandwidth-independent | Clean composition without destructive interference; interpretable, nameable atoms; simple routing; best stored compression |
| **Costs** | Cross-talk requires an interference-managing composer; less interpretable atoms; risk of imposing smoothness that isn't there | No graceful degradation — unseen directions require a genuinely new atom; marginal-cost curve only bends within the existing span |

**Not falsifiable — this is the measurement that selects the architecture.** Both outcomes are publishable results about the intrinsic dimension of task space.

**Prior expectation:** superposition results suggest models represent features in *near-orthogonal overcomplete* configurations rather than orthogonal bases, and that larger models yield *more* distinct features, not fewer. A true orthogonal basis over task space is likely unachievable at scale. Expect low-to-moderate *r*.

**Optionality asymmetry:** a minimal basis is derivable from a smooth overcomplete library (SVD, prune, deduplicate). A manifold is *not* recoverable from a minimal basis — deduplication destroys the information about how atoms relate. Default toward overcomplete.

**Confound with H6:** co-adaptation *creates* mutual predictability — paired atoms that split information are trivially predictable from each other. A suspiciously low *r* may indicate a manifold **or** widespread co-adaptation. Only the recombination test distinguishes them.

---

### H4 — Standalone viability

The system operates with no frozen parent at inference; atoms encode shared substrate as well as task-specific structure.

**Falsified if:** the only working configuration is one where some atoms are effectively "the base model, chunked" — a shared trunk in disguise.

**Threshold set in advance:** if the always-resident portion exceeds **[SET THRESHOLD — suggested 50%]** of parent size with a thin atom layer on top, H4 is falsified and the result is reported as a MoE variant, not a compositional system. This number must be fixed before the first scale point is run.

*This hypothesis is the one most likely to fail quietly rather than loudly.*

---

### H5 — Composer boundedness

The composer is fully resident, and:
- **Size:** parameter count grows at most as *o(library size)* — asymptotically negligible, not literally constant. Index-based routing requires a per-atom handle and is therefore *O(N·d)*. **Content-based routing** — composer emits a query vector into a shared embedding space, atoms carry their own keys — moves that cost into the library and achieves genuine *O(1)* in *N*. This is also the only routing scheme compatible with hypernet-generated atoms, which have no fixed index. **Design for content-based routing from the start.**
- **Depth:** sequential atom invocations per query are bounded and do not grow unboundedly with task complexity. A function-space composer can be small in parameters while becoming an interpreter whose latency eats all gains.

**Falsified if:** composer parameters approach a constant fraction of library size; composer + working set exceeds the dense baseline's footprint; per-query latency grows superlinearly in task complexity.

**Instrumentation — mandatory:** report composer size and library size as **separate line items** at every scale point; log the depth distribution per query. Never report a single combined "system size."

*H5 is primarily a failure detector.* The natural gradient of this architecture is for the composer to learn the task itself and demote atoms to a feature bank — the composer eating the library from the inside. That failure appears as flat atom growth with rising composer size, and is invisible under a combined total.

---

## 4. Baselines

**Primary:** a 4-bit quantized, pruned, distilled dense model at **equal peak resident memory**. Not the fp16 parent. Not *n* independently fine-tuned models. Not a joint model at unequal budget.

Quantization gives ~4× nearly free and composes with pruning (~2×). The honest bar is the stacked-compression baseline, and it is high.

**Secondary (H3-generative path only):** a stored library streamed from RAM/NVMe. The hypernetwork must beat paging on latency, memory, or unseen-task generalization — not merely match it. Paging already delivers "unused atoms don't occupy VRAM" for free, with no training difficulty and no generalization risk.

**Note on the generative path:** |H| is always resident, so peak VRAM is |H| + |working set|. Generation only wins if |H| < |library|, which *is* a compression claim regardless of framing — H must encode enough to produce any atom it can produce. The Game of Life analogy does not escape this: the 2-bit rule cannot be asked for a glider, because the information specifying *which* pattern appears lives in the initial state. In this architecture, the conditioning vector is that state. Either the conditioning is information-rich or H is.

---

## 5. Decision procedure

> **v1.0's procedure was in error and is withdrawn.** It specified hypernetwork-reconstructability as an auxiliary loss on atoms, with converged *r* read off as a free byproduct. That presumes atoms are weight matrices a hypernet can regress onto. Under §2.1 they are not, so the loss term has no well-defined form and the readout does not exist. Do not implement it.

**Revised ordering.** The store-vs-generate decision is deferred, not measured up front, because two cheaper results may pre-empt it:

1. **H6 gate first.** If atoms are not factorized, every downstream measurement is uninterpretable — including *r*, since co-adaptation manufactures mutual predictability. No geometry work before E1 passes.
2. **H2b locality second.** If cross-query locality is high, caching beats generation regardless of *r*, and H3 becomes moot for deployment (though still of scientific interest). Locality is far cheaper to measure than *r* under any function-space operationalization.
3. **H3 only if both clear.** Choose an *r* substitute (§9), then measure.

**Interpretation, once *r* is defined and measured:**

| | H6 passed | H6 failed |
|---|---|---|
| **Low *r*** | Manifold plausible → generative path viable *if* H2b locality is low | Predictability is co-adaptation artifact; fix factorization, remeasure |
| **High *r*** | Irreducible library → store and page | Uninterpretable |

**Anticipated endpoint — two-tier library.** A small deduplicated core of high-traffic atoms, always resident, plus a large periphery paged or generated for the long tail. Mirrors DeepSeek's shared-plus-routed split and matches Pareto-distributed task usage. Under H2b the core is per-user rather than global. Not designed up front — it falls out of measuring which atoms get used.

---

## 6. Experiments

### E1 — Factorization battery (H6) — **RUN FIRST, hard gate**

**Run this first.** Recombination splits (train AB, CD; test AD, CB); ablation degradation profiles with per-atom variance; standalone probing via differentiable weight masks (Csordás et al. method) and collapse/specialization metrics (Mittal et al.).

**Gate:** if atoms fail factorization, E2's marginal-cost curve and H3's *r* are both uninterpretable, and no systems work should proceed. Prior evidence says this is the most likely failure point — Mittal et al. found end-to-end-trained modular systems do not recover specialized solutions *even given explicit task context*.

### E2 — Marginal cost curve (H1)

Task family with known compositional structure (string transforms, SCAN/COGS-style splits, formatted arithmetic) at ~160M scale. Train atoms incrementally over tasks 1..*n*. Plot parameters-added-to-reach-fixed-performance for task *n+1*.

If it doesn't bend where composition is *guaranteed* to exist, it won't bend on natural language.

### E3 — Overlap diagnostic *(optional; ~1 week; weak prior, not a gate)*

> **Downgraded.** E0 measures overlap among LoRAs, which are weight-space objects. Under §2.1 atoms are function-space, so E0 characterizes the geometry of a representation the project has decided not to use. It retains value only as a **cheap prior on whether task space has shared structure at all** — if task LoRAs are near-orthogonal, that is bad news for compositionality in *any* representation. It cannot validate the project, and a positive result does not transfer to function-space atoms.

Take an existing library of task LoRAs (Ostapenko et al. released one) or train ~40 on a 160M model. Measure:
1. Pairwise subspace angles between adapters.
2. Reconstruction error of LoRA *k* from a linear combination of the other *k−1*, as a function of *k*.

**Read:** high reconstruction error across all *k* → weak evidence against compositional structure generally. Low error → weak evidence for it, *not* evidence for the generative path.

Output: rough estimate of the intrinsic dimension of task space. Compare against Aghajanyan et al. (d90 ≈ 200–1,600) and Task Vector Bases (~25–50% of task vectors retain ~97% performance).

*Sub-experiment (3) from v1.0 of this diagnostic — fitting a hypernet to generate held-out LoRAs — is withdrawn. It measured the weight-space *r* that §5 no longer uses.*

### E4 — Working-set characterization (H2)

Activation density per query; churn rate per token vs. per query; measured streaming cost against PCIe bandwidth. **Report peak and expected residency separately (H2a).**

### E4b — Locality characterization (H2b)

Hit rate of a fixed-size hot set under realistic per-user query sequences; sensitivity of hit rate to hot-set size; comparison of per-user vs. global core composition. Cheap, and may pre-empt H3 entirely.

### E5 — Head-to-head at equal peak memory

Against the primary baseline, with composer and library reported separately per H5.

---

## 7. Prior art

| Work | Relation |
|---|---|
| **Polytropon** (Ponti et al., 2022) | Inventory of skill modules + learned task–skill allocation. Library + composer, explicitly. Closest conceptual ancestor. |
| **Ostapenko et al., 2024** — Building and Reusing a Library of LoRAs | Multi-head routing over a learned library, zero-shot routing for unseen tasks. Released library usable for E0. |
| **Text-to-LoRA** (Sakana, 2025) | Hypernetwork generating adapters from task descriptions. Known weakness: generalization to task descriptions far from the training distribution. |
| **HyperFormer++** (Karimi Mahabadi et al.) | Hypernet-generated adapters conditioned on task and layer. |
| **LoRAHub, AdapterFusion, PHATGOOSE** | Composition and routing mechanisms. |
| **Task arithmetic / TIES-merging** | Weight-space algebra; TIES exists specifically to fight the interference that orthogonality would prevent. |
| **SAE / superposition work** (Anthropic, Claude 3 Sonnet features) | Empirical evidence on atom counts: larger models yield *more* distinct features, and remain capacity-limited. |
| **DeepSeek-MoE** | Shared + routed expert split; template for the two-tier endpoint. |

**Open territory:** none of these drop the frozen parent. H4 is the novel constraint.

> **Latent inconsistency, surfaced in v1.1.** Nearly every entry above assumes a frozen base and studies *deltas* against it. v1.0 asserted H4 (no parent) while importing methods, metrics, and intuitions from a literature where the parent silently supplies the substrate — and never confronted where the substrate comes from once it is removed. The rank argument in §2.1 is what made this visible; it was latent in the document from the start. **Prior-art results on adapter composition should be treated as suggestive analogies, not transferable findings.** The genuinely relevant lines are the substrate-carrying ones: Branch-Train-Merge / BTX, Progressive Neural Networks, Neural Module Networks, DreamCoder.

---

## 8. Claims explicitly *not* made

- **"Total system size grows sub-linearly."** Trivially true for any shared trunk; invites the shared-trunk objection; implied by H1 in stronger form. Dropped.
- **"Larger models need fewer atoms."** Contradicted by SAE scaling results. The defensible version — larger models are *relatively* more redundant and their features may be more cleanly separable, hence more composable — predicts atom count still grows with scale, with a better compression *ratio*.
- **Beating the fp16 parent.** The bar is the stacked-compression baseline at equal peak memory.

---

## 9. Open decisions

- [x] ~~**Weight-space vs. function-space atoms.**~~ **CLOSED (v1.1)** — foreclosed by the rank argument in §2.1. Function-space. Do not reopen without defeating that argument.
- [ ] **Function-space operationalization of *r*** — blocks H3 entirely. Candidates: behavioural equivalence on a probe set; edit distance over discrete atom structure; end-system performance under generated vs. stored atoms.
- [ ] **H4 resident-fraction threshold** — fix before first scale point. Suggested 50%.
- [ ] **Task family for E2** — must have guaranteed compositional structure.
- [ ] **Consolidation-pass schedule** if using frozen-library sequential training.
- [ ] **Training method for atoms/composer** — gradient descent may be the wrong tool for inducing modularity (Csordás 2021; Mittal 2022 both show specialization does not emerge from end-to-end training). Non-gradient options with an asymmetry in this project's favour: connection-cost evolution (Clune et al. 2013), modularly varying goals (Kashtan & Alon 2005), quality-diversity archives (MAP-Elites), non-learned routing (Hash Layers, BASE Layers), and DreamCoder's compression-based library abstraction. Likely resolution: gradient descent *within* atoms, non-gradient *between* them.
- [ ] **Whether to adopt §10's knowledge/skill separation** — changes the project from discovery to design.

---

## 10. Proposal under consideration — knowledge/skill separation

> **Status: NOT ESTABLISHED.** This is an attractive argument, not a finding. It is recorded here so the decision is made deliberately rather than drifted into. Adopting it materially changes the project's claim.

**The argument.** Knowledge and computation have opposite statistics. Facts are high-entropy, per-item, do not compress, do not generalize across tasks. Skills are low-entropy, reusable, compositional. A hypernetwork generating facts is a lossy compressed database — the known pathology of parametric knowledge. A hypernetwork generating *skills* is plausible precisely because skills may lie on a manifold. So do not spend a generator on the knowledge layer.

**Proposed split:**

| | Knowledge | Atoms (skills) |
|---|---|---|
| **Mechanism** | Explicit memory layers (Berges et al., Meta 2024) or retrieval | Pure computation, function-space |
| **Size** | Large | Small |
| **Access** | Genuinely sparse — a query touches a handful of facts | High hit rate |
| **Residency** | Paged | Always resident |
| **Locality (H2b)** | Strongest — factual needs are more predictable per-user than reasoning needs | Moderate |

**Note this inverts the current design.** §3 has atoms paging and the composer resident. Under the split, the large pageable thing is *knowledge*, and atoms + composer stay resident. That is a better fit to both access patterns.

**Costs, stated honestly:**

- **The split is architected, not discovered.** Transformers entangle knowledge and skill — Geva et al. (2021) show FFNs act as key-value memories while also computing. "Knows multiplication" and "knows 7×8=56" are not cleanly separable in a trained model. Some knowledge will leak into atoms regardless.
- **Method changes from distillation to design.** The teacher supervises a pre-specified architecture rather than having structure extracted from it. Weaker scientific claim, likelier engineering result.
- **H1 becomes easier and less impressive.** Once facts are externalized, sub-linear skill-library growth is close to tautological — the linearly-growing component was removed, not solved. If adopted, this must be stated plainly rather than reported as compositional compression.

**Decision required before Stage 1 architecture work.**

---

## 11. Staged plan

| Stage | Work | Gate to proceed |
|---|---|---|
| **0** | E3 overlap diagnostic *(optional)* | None — informational only |
| **1** | **E1 factorization battery** on a small atom library | **Hard gate.** Atoms must pass recombination, low ablation-variance, standalone probing. Most likely failure point in the whole program. |
| **2** | E2 marginal-cost curve; E4/E4b working set + locality | E2 curve must bend on a synthetic family with guaranteed compositional structure |
| **3** | H3 store-vs-generate, *only if* locality is low and an *r* substitute is defined | — |
| **4** | E5 head-to-head vs. 4-bit quantized/pruned/distilled dense **and** vs. index-paged offloading (PIPO/Mixtral-offloading) at equal peak VRAM | — |

**Scoop risk to monitor:** the 2025–2026 hypernetwork-generated-adapter cluster (Text-to-LoRA and successors: Doc-to-LoRA, Zhyper, HypeLoRA, LatentSkill) is moving fast on the generation stage. It does *not* address the no-parent constraint, which remains the distinctive claim.
