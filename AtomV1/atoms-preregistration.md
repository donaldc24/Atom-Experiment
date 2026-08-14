# Atoms: Compositional Primitives as a Standalone Replacement for Dense Models

**Preregistered research prospectus — v1.0**

---

## 1. Motivation

Frontier models are increasingly out of reach for individuals and small organizations, not because of total storage but because of **peak resident memory**. Mixture-of-Experts addresses compute, not VRAM: Mixtral must hold all 47B parameters resident to serve arbitrary queries despite activating ~13B per token.

Existing compression (4-bit quantization, structured pruning, distillation) squeezes weights while remaining agnostic to *cross-task* structure — the plausible fact that operations like "compare two quantities," "track an entity across a paragraph," or "apply a formatting rule" recur across thousands of tasks with shared machinery. That redundancy is the seam this project mines.

**Target resource:** peak resident memory at inference.
**Fallback result (valid even if compression fails):** if a query needs only a small resident working set and unused atoms live in RAM/NVMe, a large model becomes runnable on consumer hardware even with a library the same total size as the parent.

---

## 2. Core claim

A large teacher model's capabilities can be re-expressed as compositions over a library of reusable primitives (**Atoms**) plus a learned **composer**, such that inference requires no teacher or frozen parent model, and such that at equal peak-memory budget the system matches or exceeds a distilled dense model of the same footprint.

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

**Critical sub-question, to be measured early:** does the working set change **per query** or **per token**?
- *Per query* → paging from RAM/NVMe is likely sufficient; the hypernetwork is a research question, not a necessity.
- *Per token* → PCIe bandwidth (~32 GB/s on 4.0 x16) becomes the bottleneck and generation is the only viable answer.

Note that MoE routes per token *because experts are cheap to keep resident* — the assumption this project removes.

---

### H3 — Atom-space geometry *(the open fork)*

The library occupies a position on the orthogonality–smoothness axis characterized by **mutual predictability *r***, measured as the reconstruction error of a hypernetwork trained to generate held-out atoms from task conditioning.

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

Train atoms in stage 1 with **hypernetwork-reconstructability as an auxiliary loss term** — a small hypernet predicts each atom from its task conditioning; its reconstruction error is added to the atom loss.

Atoms are then pressured toward *as compact as possible subject to lying on a learnable manifold*. Converged *r* is a direct quantitative readout of H3, obtained free as a byproduct of stage 1:

- **Low *r*** (and H6 passes) → build the generative path.
- **Low *r*** (and H6 fails) → the predictability is co-adaptation; fix factorization first.
- **High *r*** → ship the stored library with paging.

Either way stage 1 is not wasted, and the fork resolves by measurement rather than prior commitment.

**Anticipated endpoint — two-tier library.** A small orthogonal deduplicated core of high-traffic atoms, always resident, carrying shared substrate; plus a large smooth periphery generated on demand for the long tail. Mirrors DeepSeek's shared-plus-routed expert split and matches the Pareto distribution of task usage. This is not designed up front — it falls out of measuring which atoms actually get used.

---

## 6. Experiments

### E0 — Overlap diagnostic *(do first; ~1 week; kills or validates the project)*

Take an existing library of task LoRAs (Ostapenko et al. released one) or train ~40 on a 160M model. Measure:
1. Pairwise subspace angles between adapters.
2. Reconstruction error of LoRA *k* from a linear combination of the other *k−1*, as a function of *k*.
3. Fit a small hypernet to generate held-out LoRAs from task embeddings; test whether it produces *working* adapters for unseen tasks.

**Reads:** (1)+(2) test whether shared atom structure exists at all — if reconstruction error stays high, adapters are near-orthogonal and the marginal-cost curve will never bend. (3) is a first read on H3: generalization to held-out tasks means the manifold exists and the generative path is live; reproducing only training atoms means a lookup table with extra steps, and paging wins.

Output: first estimate of the intrinsic dimension of task space → how many atoms are actually needed.

### E1 — Marginal cost curve

Task family with known compositional structure (string transforms, SCAN/COGS-style splits, formatted arithmetic) at ~160M scale. Train atoms incrementally over tasks 1..*n*. Plot parameters-added-to-reach-fixed-performance for task *n+1*.

If it doesn't bend where composition is *guaranteed* to exist, it won't bend on natural language.

### E2 — Factorization battery (H6)

Recombination splits (train AB, CD; test AD, CB); ablation degradation profiles with variance reported per atom; standalone probing. **Gates interpretation of E1.**

### E3 — Working-set characterization (H2)

Activation density per query; churn rate per token vs. per query; measured streaming cost against PCIe bandwidth.

### E4 — Head-to-head at equal peak memory

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

---

## 8. Claims explicitly *not* made

- **"Total system size grows sub-linearly."** Trivially true for any shared trunk; invites the shared-trunk objection; implied by H1 in stronger form. Dropped.
- **"Larger models need fewer atoms."** Contradicted by SAE scaling results. The defensible version — larger models are *relatively* more redundant and their features may be more cleanly separable, hence more composable — predicts atom count still grows with scale, with a better compression *ratio*.
- **Beating the fp16 parent.** The bar is the stacked-compression baseline at equal peak memory.

---

## 9. Open decisions

- [ ] **Weight-space vs. function-space atoms.** Weight-space (LoRA-like deltas, composed by summation) is cheap but composition is near-linear. Function-space (routed subroutines invoked in sequence) is more expressive but risks the composer becoming an interpreter, threatening H5-depth. *Everything downstream depends on this.*
- [ ] **H4 resident-fraction threshold** — fix before first scale point.
- [ ] **Task family for E1** — must have guaranteed compositional structure.
- [ ] **Consolidation-pass schedule** if using frozen-library sequential training.
