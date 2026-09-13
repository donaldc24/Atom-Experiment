# Related work and the scope of the Skill Forge experiment

The experiment asks whether a small neural teacher's behavior can be recovered as a compact executable operator, checked against an independent specification, and then composed with operators recovered from other teachers. The teacher is absent during the resulting execution. This is a local empirical hypothesis about the specified finite task family; a positive result would not establish general model compression, automatic discovery of a universal skill language, or a replacement for language models.

The main precedents are close and should be credited explicitly:

- **[VIPER: Verifiable Reinforcement Learning via Policy Extraction (Bastani, Pu, and Solar-Lezama, NeurIPS 2018)](https://arxiv.org/abs/1805.08328).** VIPER extracts decision-tree policies using a neural policy and its value estimates, allowing verification of the extracted policy. Neural behavior becoming a compact, verifiable executable object is therefore established prior art. Skill Forge instead tests finite input transformations, exact enumeration against an independent specification, and composition across separate teachers.
- **[Programmatically Interpretable Reinforcement Learning (Verma et al., ICML 2018)](https://proceedings.mlr.press/v80/verma18a.html).** PIRL searches a programmatic policy space with guidance from a neural policy. A supplied program grammar and a neural oracle are not new ingredients. The distinction tested here is whether a narrowly specified extraction procedure recovers operators that remain correct under previously untrained compositions.
- **[DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning (Ellis et al., PLDI 2021)](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf).** DreamCoder learns reusable program abstractions and a neural search policy together. A growing executable skill library predates this experiment. Skill Forge supplies its operator grammar instead of learning one, so it makes a substantially narrower claim about extraction, verification, and reuse.

## What this experiment could establish

For the frozen protocol's task distribution, calibration budget, teacher architectures, and extraction grammar, the experiment can measure whether:

1. Teacher predictions alone are sufficient to recover the executable operator, without the extractor receiving the task name, original implementation, or specification outputs.
2. An independent verifier rejects a recovered operator even when that operator faithfully reproduces a systematically incorrect teacher.
3. Certified operators from independently trained teachers compose correctly on held-out sequences while execution retains no neural parameters.
4. Executable payload size and measured execution cost are substantially below those of the particular teachers used.

The independent specification is an explicit resource. Exhaustive checking is practical because the domain is finite and small. A certificate applies only to that domain and specification; it is not a proof about arbitrary inputs, natural language, or an unbounded learned behavior.

## What is supplied rather than discovered

The grammar represents each output coordinate as a lookup table applied to one selected input coordinate. The source-position choices and lookup entries are recovered from observations. The grammar itself, coordinate structure, input alphabet, and verification specification are supplied by the experimenter. Arbitrary bijective tables and arbitrary position permutations are important tests because recovering only the repository's familiar affine recipes would be weaker evidence.

This grammar is closed under composition: composing source selections and their corresponding lookup tables produces another operator of the same form. That is a mathematical property of the supplied representation, not an empirical discovery. The experiment tests whether noisy learned behavior can be exported into that representation correctly and economically. Cross-coordinate nonlinear interactions generally lie outside the grammar and should be rejected, even if a larger neural network could represent them.

The most informative negative outcome is a capable teacher whose behavior cannot be exported at the frozen query budget or within this grammar. The most informative positive outcome is faithful extraction followed by independent certification and successful mixed-origin execution, accompanied by an honest accounting of calibration, verification, storage, and runtime costs.
