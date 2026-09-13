# Skill Forge: acquiring executable skills without importing neural weights

Status: prospective protocol recorded before any result-bearing extraction or validation in this experiment. Existing V1/V2 findings motivated the hypothesis; those observations are not new confirmatory evidence. Implementation may be debugged using synthetic smoke data before the main source fingerprint is frozen. Changes after main outcomes are observed must be documented and cannot silently replace this protocol.

## Idea and hypothesis

A neural network serves as an apprentice teacher. A behavior-only compiler queries its opaque singleton skills, proposes small executable operators, and exports them through a shared digit interface. A separate verifier may admit or reject a proposal. An archive computes a necessary algebraic boundary of its current capabilities, acquires a missing operation from an independently trained teacher, and executes new mixed-origin compositions after every neural weight has been removed from the runtime.

The ambitious direction is a growing intelligence whose durable knowledge is portable, verified executable skills, while neural networks are temporary learning machinery. This experiment tests only a constrained first step: whether the nonfactorized Atom teachers contain behavior that can be harvested into exact transferable operators. It cannot establish general program discovery, unrestricted AI, semantic interfaces learned without grounding, or a field-wide revolution.

The local primary hypothesis is that at least two of three independently trained A14 checkpoints each yield at least seven exact, accepted singleton operators, the union of real A14 and A33 teachers supplies all eight operations, and an A33 archive can acquire its missing reversal capability from A14 without weight transfer or joint training.

## Hardware and resource policy

Detected machine: AMD Ryzen 9 6900HX, eight cores/sixteen logical processors, about 16 GB physical RAM. At design time only about 1.7 GiB was available. Installed PyTorch is 2.9.0+cpu; CUDA is unavailable. Execute result-bearing workloads sequentially, set four PyTorch threads, use batches of at most 256 for harvesting the existing Atom teachers, and use bounded chunks for exhaustive integer verification. The existing descriptive baseline panel retains its native 400-example task batches, which were already successfully replayed during the earlier review. Do not modify or terminate unrelated applications. Record elapsed time, process memory where available, all source fingerprints, and checkpoint/data hashes.

## Teachers and controls

Real teachers are the existing final A14 checkpoints at model seeds 0, 1, and 2, and the existing final A33 checkpoint at seed 1. A14 seed 2 is retained regardless of its weaker prior performance. There is no post-selection of healthy teachers. No neural weights are updated in this experiment: fitting the program hypothesis from teacher examples is the learning step.

Three synthetic controls use the same compiler and query budget:

- Exact ground-truth teacher: compiler feasibility control, explicitly privileged and excluded from the experimental archive.
- Shuffled calibration labels: destroy input/output correspondence; validation uses the unshuffled exact teacher. This should fail candidate admission.
- Coherently wrong teacher: apply the exact operation and then exchange output columns 0 and 1. This is deliberately within the compiler's grammar. It should pass teacher-agreement admission and fail independent correctness verification.

The A33 teacher is an additional natural error control. Its original failed E11 verdict remains unchanged. The previously reviewed six passing A33 operations and missing-reversal permutation closure are prior observations to reproduce, not independently preregistered discoveries here.

## Data and compiler contract

For every token P1 through P8, use 2,048 distinct uniformly selected calibration input lists and 2,048 distinct validation lists, disjoint within the token. Inputs are drawn from the million-element six-digit domain using query seed 20260912, identically across teachers. No true intermediate labels enter fitting. Store inputs and teacher predictions so extraction is reproducible without querying again.

The compiler accepts only arrays of inputs and teacher predictions. It does not import the task generator, named operations, hidden recipes, affine coefficients, or correct outputs. Its declared prior is the following generic grammar:

`output[j] = lookup[j][input[source[j]]]`.

Each of six outputs can choose any of six source positions and any ten-entry categorical lookup table. For each candidate source, infer table entries by calibration majority vote. Select the source with highest calibration per-digit accuracy; break ties lexicographically. This is a domain-informed, separable grammar, not unconstrained discovery. It permits arbitrary digit permutations beyond the repository's affine transformations, but excludes operations depending jointly on multiple digits. Synthetic tests must verify that an out-of-class function does not masquerade as successful extraction.

A candidate is accepted for independent verification only if all of these hold:

- The six selected source positions form a permutation.
- Each digit lookup table is a permutation of 0 through 9.
- Every output's winning source exceeds its runner-up calibration predictability by at least 0.10.
- Exact six-digit agreement with the teacher on the separate validation inputs is at least 0.98.

Keep all rejected candidates and diagnostics. A candidate serializes as exactly 66 uint8 payload bytes: six source indices followed by six ten-entry lookup tables. Metadata, hashes, index storage, interpreter overhead, and filesystem allocation are additional costs and must be reported separately.

## Independent verification and archive admission

Freeze candidate files before verification. A separate module may use the true task generator to compare each compiled candidate with the true operation on all 1,000,000 possible inputs, in chunks of 8,192. This verifies the integer executable over the complete finite domain; it does not formally verify the Python interpreter or operating system.

Verify rejected candidates too as diagnostics, but admit a real-teacher skill only if it passed every compiler gate and has zero exhaustive mismatches. The verifier must never edit a candidate, repair it with truth, or feed counterexamples back into fitting in this experiment. Synthetic-control operators are never admitted to the experimental archive.

Initially construct the archive from accepted, verified A33 skills. Enumerate the complete closure of their position permutations. Absence of the reversal permutation is a sufficient impossibility certificate for implementing a reversal-bearing task using those skills, at any sequence length. Presence of a permutation alone is not a sufficient certificate for the full digit transformation.

Acquire missing verified tokens from A14 seed 0, then 1, then 2 in that fixed preference order. Keep every accepted origin variant for source-mixing tests. Use A33 operators preferentially for tokens it supplies. Recompute the permutation closure after acquisition and report the actual size; no target closure size is assumed.

## Execution and baselines

The deployment runtime must run in a fresh subprocess that imports neither PyTorch nor the Atom harness. It accepts only input digits and binary skill-file paths. Its operator cache holds at most one 66-byte decoded operator payload at a time. Record calls, loads, bytes read, peak loaded payload bytes, archive payload/metadata bytes, interpreter process memory, and time separately. Input/output arrays and plan/index memory are not counted as operator payload and must not be described as zero overhead.

Use execution seed 20260913. Test 100 unique token sequences at each length 2, 4, 8, 16, 64, and 256, with 32 fresh inputs per sequence. For length 2 there are only 64 possible sequences, so use all 64. Include every operation; cycle through accepted origin variants so compositions cross training origins. Targets are computed only by the independent evaluator. The runtime receives no reference outputs. Report exact accuracy separately by length and origin participation.

Compare with A14 seed 1's original latent handoff and self decode/re-encode handoff on the existing frozen V2 pair panel. This baseline checkpoint is chosen prospectively here based on the earlier review; it is descriptive context, not an unbiased selection among source models. Report exact counts per held-out level. Compilation's use of a known grammar and exhaustive verifier makes it a different method, not a matched-architecture neural training comparison. The resulting storage ratio is specific to this task family.

## Frozen verdict

`INVALID_RIG` if synthetic exact teacher fails to export and verify all eight operations, coherent-wrong control is admitted to the experimental archive, train/validation inputs overlap, or core executable/provenance checks fail. Fix a rig error transparently before interpreting scientific outcomes.

`PORTABLE_SKILL_ACQUISITION_SUPPORTED_IN_ATOM_WORLD` requires all of:

1. At least two of three A14 origins each yield at least seven accepted, exhaustively correct operators.
2. The real-teacher union supplies all eight tokens.
3. The initial A33 archive lacks the reversal permutation in its complete closure, and acquired A14 operators supply the missing reversal-bearing operations.
4. Every deployment panel has exactly 1.0 exact accuracy, including length 256 and mixed-origin sequences.
5. The fresh deployment process loads no neural library or weights and retains at most 66 decoded operator-payload bytes at a time.
6. The shuffled control yields no accepted operator and the coherently wrong control produces proposals that the exhaustive verifier rejects.

`PARTIAL_ACQUISITION_ONLY` if some valid skills are recovered but the full conjunction fails without a rig error. `EXTRACTION_FALSIFIED_AT_THIS_BUDGET` if no A14 origin yields any admitted skill. A failed screen does not disprove all compilers or all transferable computation. No extra query budget, altered gate, teacher selection, or grammar expansion is allowed after observing the main results.

## Predictions

Prediction 1: at least A14 seeds 0 and 1 pass the seven-skill threshold; weaker seed 2 may not.

Prediction 2: the accepted real-teacher union contains all eight exact operators, allowing acquisition of both reversal-bearing tokens absent from the initial A33 archive.

Prediction 3: coherent systematic teacher errors can pass behavioral agreement but will fail external verification, while shuffled labels fail earlier. Teacher confidence and correctness are distinct.

Prediction 4: compiled mixed-origin execution is perfect at every registered depth because admitted integer programs are correct on the entire domain and share an explicit complete state interface.

Prediction 5: decoded operator residency is constant with chain depth at 66 payload bytes, but total process memory is much larger and must be measured honestly.

## Novelty boundary

Program synthesis, extraction of verifiable programs from neural teachers, reusable learned libraries, and proof-carrying code are established directions. Relevant prior work includes VIPER, PIRL, and DreamCoder (see RELATED_WORK.md). The potential contribution here is the particular empirical recovery and acquisition of exact interoperable skills from Atom models that failed internal atom-factorization tests, together with a capability-gap check and actual weight-free execution. A positive result is a small-world mechanism result, not evidence that general intelligence can already be reduced to these 66-byte programs.
