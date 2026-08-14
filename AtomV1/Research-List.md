# Prioritized Reading List for the "Atoms" Hypothesis

## TL;DR
- The **substrate-carrying, no-frozen-parent** framing (H4) is your most genuinely novel angle: nearly all modular-LLM work (LoRAHub, Ostapenko, PHATGOOSE, Text-to-LoRA) assumes a frozen base; the few from-scratch/independent-substrate lines (Branch-Train-Merge, Progressive Nets, Neural Module Networks) exist but hit exactly the failure modes your hypotheses must beat.
- **Two of your core preconditions have already been stress-tested and largely fail in prior work**: functional specialization does NOT reliably emerge from end-to-end training of modular architectures (Csordás 2021; Mittal, Bengio & Lajoie 2022), and neural-module-network-style function composition does not scale/generalize systematically without heavy inductive bias. Read these first — they are the strongest existing challenges to H6.
- For H2/H3 (working-set sparsity, paging-vs-generation), the memory-bound inference literature (DejaVu, LLM-in-a-flash, Mixtral-offloading, PowerInfer) already establishes contextual sparsity is per-token and PCIe-bound, and the hypernetwork-continual-learning line (von Oswald 2020) already demonstrates parameter growth that is roughly constant in the number of tasks — directly relevant to H5.

## Key Findings
- **Novelty verdict:** Your specific combination — a library of atoms that carry the substrate (not deltas), composed with no frozen parent, optimizing peak resident VRAM, with a later hypernetwork-generation stage — is not something I found published as a single unified program. But each pillar has close prior art, and the precondition H6 is where the literature is most discouraging.
- **Biggest risk flag:** H6 (atom factorization into independently meaningful primitives) is precisely the property that Csordás et al. and Mittal et al. show does NOT emerge from gradient training even when the architecture permits it. Design your H6 tests as the first go/no-go gate.
- **Most encouraging pillar:** H5 (bounded, ~O(1) composer). von Oswald et al.'s task-conditioned hypernetwork already keeps trainable weight count "comparable or smaller than target network size" regardless of task count — an existence proof for the composer-boundedness you need.

## Details — Tiered Reading List

### TIER 1 — READ FIRST (test your preconditions or define your baseline landscape)

**1. Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks** — Csordás, van Steenkiste & Schmidhuber, 2021 (ICLR). arXiv:2010.02066.
Introduces differentiable binary weight masks to identify which weights implement which function; finds that networks "specialize, but don't reuse" submodules. This is a ready-made methodology for your H6 standalone-probing and ablation tests, and direct negative evidence that co-adaptation is the default. Read for both method and cautionary result.

**2. Is a Modular Architecture Enough?** — Mittal, Bengio & Lajoie, 2022 (NeurIPS). arXiv:2206.02713.
On synthetic data with known ground-truth rules, end-to-end-trained modular systems suffer "collapse" and poor "specialization" and do not recover the perfectly-specialized oracle solution "even with explicit information about task context." The single most important refutation-in-advance of H6: architectural modularity ≠ functional factorization. Their conclusion — "additional inductive biases are required to learn adequately specialized solutions" — should shape your experimental design, and their collapse/specialization metrics are directly reusable for your ablation-degradation-variance test.

**3. Modular Deep Learning** — Pfeiffer, Ruder, Vulić & Ponti, 2023 (TMLR). arXiv:2302.11529.
The single best entry-point survey; unifies routing, aggregation, and module-training threads and taxonomizes exactly the weight-space-vs-function-space axis that is your open architectural decision. Use it to place your work and to mine citations rather than re-searching.

**4. Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time** — Liu et al., 2023 (ICML). arXiv:2310.17157.
Shows input-dependent sets of heads/MLP neurons — averaging "85% in attention, 95% in MLP" (a ~7× per-input parameter reduction) — reproduce dense outputs and can be predicted on the fly per-token, giving "over 2× reduction in token generation latency" on OPT-175B with no accuracy drop. The empirical backbone for H2: working sets are per-token, not per-query — the harder regime for your PCIe-bandwidth argument.

**5. LLM in a Flash: Efficient LLM Inference with Limited Memory** — Alizadeh et al. (Apple), 2023/2024 (ACL). arXiv:2312.11514.
Streams parameters from flash using windowing + row-column bundling and FFN sparsity, "running models up to twice the size of the available DRAM, with a 4-5x and 20-25x increase in inference speed compared to naive loading approaches in CPU and GPU, respectively" (tested on OPT-6.7B, Falcon-7B, Persimmon-8B across M1 Max, M2 Ultra, RTX 4090). The canonical statement of the paging-under-memory-constraint problem your project competes with and must beat.

### TIER 2 — READ IF YOU GO DOWN THE "DROP THE FROZEN PARENT" PATH (H4 novelty check)

**6. Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models** — Li et al., 2022. arXiv:2208.03306. And its successor **Branch-Train-MiX (BTX)** — Sukhbaatar et al., 2024. arXiv:2403.07816.
BTM trains independent expert LMs on data subsets with no shared trunk synchronization, then ensembles/averages; BTX folds them into MoE FFN layers with learned routing. Closest existing "modules carry substrate" work — read to see how far independent training gets you and why they still re-merge into a shared model (a warning for H4's "don't degenerate into shared trunk with adapters").

**7. Progressive Neural Networks** — Rusu et al., 2016. arXiv:1606.04671.
The canonical add-a-column-per-task architecture with frozen prior modules and lateral connections; explicitly documents linear parameter growth and that "only a fraction of the new capacity is actually utilized, and that this trend increases with more columns." Read as the cautionary baseline for H1 (marginal compositionality): it exhibits the failure mode (growth, not decreasing marginal cost) you must beat.

**8. Neural Module Networks** — Andreas et al., 2016 (CVPR), plus **How Modular Should NMNs Be for Systematic Generalization?** — D'Amario et al., 2021 (NeurIPS). arXiv:2106.08170.
NMNs are the classic function-space composition approach (routed subroutines invoked in sequence) — your function-space atom option. The follow-up shows systematic generalization is highly sensitive to the degree/placement of modularity and requires careful design. Read to understand why function-space composition historically needed hand-specified layouts and didn't scale.

### TIER 3 — READ FOR THE PAGING-VS-GENERATION AND ROUTING DECISIONS (H2, H3, H5)

**9. Continual Learning with Hypernetworks** — von Oswald, Henning, Sacramento & Grewe, 2020 (ICLR). arXiv:1906.00695.
A task-conditioned hypernetwork generates target weights from a compact embedding; long memory lifetimes are "achieved in a compressive regime, when the number of trainable hypernetwork weights is comparable or smaller than target network size," plus "chunked" generation for compression. Direct existence proof for H5 (composer grows o(library size), ideally O(1) in library size) and for your late-stage on-demand atom generation. Read closely. (Minor note: author ordering differs between arXiv metadata and the paper title page.)

**10. Fast Inference of Mixture-of-Experts Language Models with Offloading** — Eliseev & Mazur, 2023. arXiv:2312.17238. (Plus PowerInfer, Song et al. 2023; and PIPO for consumer devices, arXiv:2504.03664.)
Practical expert-offloading with LRU caching + mixed quantization to run Mixtral on consumer GPUs; PIPO reports ~12.5 tok/s on Mixtral-8×7B on a 6 GB-VRAM RTX 3060 laptop. Your H2 systems reference and effectively part of your baseline: it shows what index-based expert paging already achieves and where PCIe is the bottleneck — the paper notes that a 405B model over ~20 GB/s PCIe takes ~40 s to traverse every layer, illustrating why per-token working-set churn is fatal without generation or heavy caching.

**11. Editing Models with Task Arithmetic** — Ilharco et al., 2023 (ICLR). arXiv:2212.04089.
Establishes that task vectors from different tasks are "close to orthogonal, and speculate that this enables the combination of task vectors via addition with minimal interference." Central to your H3 geometry question and the weight-space-atoms (summation) option — evidence for a near-orthogonal basis. Skim the orthogonality figures and analogy results. (See also Task Vector Bases, arXiv:2502.01015, which shows ~25–50% of task vectors as a basis retains up to ~97% performance — a direct estimate for how few atoms you might need.)

**12. Learning to Route Among Specialized Experts for Zero-Shot Generalization (PHATGOOSE)** — Muqeeth et al., 2024. arXiv:2402.05859. (Plus the MoErging survey, arXiv:2408.07057, and GLIDER, arXiv:2410.07172.)
Post-hoc token- and module-level routing over independently trained experts; the MoErging survey taxonomizes embedding/retrieval-based routing where expert keys derive from training data — exactly your "content-based routing with atom keys living in the atoms" (H5). GLIDER shows token-level routing lacks global context on held-in tasks — a concrete design warning for your composer.

### TIER 4 — SKIM FOR PERSPECTIVE (theory, geometry, limits)

**13. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning** — Aghajanyan, Gupta & Zettlemoyer, 2021 (ACL). arXiv:2012.13255.
"By optimizing only 200 trainable parameters randomly projected back into the full space, we can tune a RoBERTa model to achieve 90% of the full parameter performance levels on MRPC" (d90 ranges ~200 for MRPC to ~1,600 for MNLI on RoBERTa-base), and larger models have lower intrinsic dimension. Your best quantitative anchor for "how many atoms / how many independent directions does task adaptation actually need."

**14. Breaking Neural Network Scaling Laws with Modularity** — Boopathy et al., 2025 (ICLR). arXiv:2409.05780.
Proves modular networks' sample complexity can be independent of task input dimensionality (vs. exponential for monolithic), and proposes a learning rule to align modules to task structure. Theoretical support for why H1's marginal-compositionality could hold — but note it requires the alignment that Csordás/Mittal show does not emerge for free.

**15. Generalization without Systematicity (SCAN)** — Lake & Baroni, 2018 (ICML), plus "Rearranging the Familiar" (Loula, Baroni & Lake, 2018, EMNLP BlackboxNLP).
The canonical negative result: seq2seq nets generalize to seen patterns but fail systematic recombination of primitives. You already know SCAN/COGS; read Loula for the sharper finding that even well-trained functional words don't recombine systematically — a direct warning for your recombination-into-unseen-pairings test (H6).

**16. Towards Monosemanticity / Toy Models of Superposition** — Bricken/Elhage et al. (Anthropic), 2022–2023.
You already know the SAE work; the crucial detail for H3 is Anthropic's own conclusion that architectural approaches to killing superposition were "insufficient to prevent polysemanticity" and that standard dictionary learning "had significant issues with overfitting" — i.e., features live on a smooth overcomplete manifold, not a clean orthogonal basis. This pushes your "orthogonal basis (store+page) vs overcomplete manifold (generate)" decision toward the generation side.

**17. Text-to-LoRA and the hypernetwork-generated-adapter wave (2025–2026)** — Charakorn et al. (Sakana), 2025 (ICML); follow-ups Doc-to-LoRA (arXiv:2602.15902), Zhyper (arXiv:2510.19733), HypeLoRA (arXiv:2603.19278), LatentSkill (arXiv:2606.06087).
You know T2L; the crucial newer detail is the fast-growing 2025–2026 cluster generating LoRAs on-demand from text with no per-task gradient updates. This is where you are most at risk of being scooped on the "atoms generated by a hypernetwork" stage — track it actively. Note reported hypernetwork expressivity limits: segment-wise MLP generators "fail to capture global dependencies across layers," and prior methods "can only generate a subset of LoRAs" or use bottlenecks that "severely limit expressivity" — direct evidence bearing on whether |H| < |library| is achievable.

## Recommendations
- **Stage 0 (go/no-go on H6 — do this before any systems work):** Replicate Csordás's differentiable weight-mask probing and Mittal et al.'s collapse/specialization metrics on your smallest atom library. Benchmark: atoms must pass standalone-probing and recombine into unseen pairings with low ablation-degradation variance. If they don't, the rest of the program is premature — this is the cheapest, highest-information experiment.
- **Stage 1 (H2/H3 measurement):** Instrument working-set activation per-token vs per-query using DejaVu-style contextual-sparsity prediction, and measure atom cosine similarity (Ilharco; Task Vector Bases) to decide store-and-page vs generate. Threshold that changes the plan: if working sets are strongly per-token with low overlap, PCIe (~32 GB/s) dominates and you must favor generation (hypernetwork) over paging.
- **Stage 2 (baseline bake-off):** Build BOTH the 4-bit quantized/pruned/distilled dense baseline AND a Mixtral-offloading/PIPO index-paging baseline at equal peak VRAM. You must beat both; the offloading systems are the stronger baseline for your VRAM objective.
- **Stage 3 (composer & generation):** Prototype content-based routing with keys in the atoms (MoErging-style) and a von-Oswald-style chunked hypernetwork for on-demand generation; benchmark against Text-to-LoRA. Change the plan if the hypernetwork's size must scale with atom diversity — that would refute the |H| < |library| premise (and the T2L-cluster expressivity limits suggest this is a real risk).

## Caveats
- Two of your core claims (H6 factorization; function-space composition scaling) already have substantial negative evidence; treat them as hypotheses to disprove, not assumptions.
- Much of the memory-bound inference literature is systems work with rapidly moving, hardware- and model-specific numbers (the DejaVu/flash/offloading speedups above); re-benchmark on your target consumer GPU rather than porting figures.
- The 2025–2026 hypernetwork-adapter cluster is moving fast; several entries are very recent preprints not yet peer-reviewed — verify claims before relying on them.
- Author ordering for the von Oswald hypernetwork paper differs between arXiv metadata and the paper title page; cite from the published ICLR version.