# Atom-specific validation specification

Status: **locked before live runs** on 2026-08-06.

This stage follows the completed H1 roadmap. It tests whether the H1 storage result is
specifically attributable to composable learned atoms, whether a dictionary learned without a
target task transfers across every held-out task, and whether held-out LoRA updates lie in the
frozen learned atom span. It does not test a hypernetwork, routing, wake-dream consolidation,
plugins, language generation, or the 1T-to-10B system claim.

## Common contract

- Base and data: the locked H1 `prajjwal1/bert-tiny` and GLUE task contract.
- Tasks, in locked order: SST-2, MRPC, RTE, QNLI, QQP.
- Seeds: 17, 29, and 43 for every primary cell.
- Budget: 2,000 training rows per task, up to 500 validation rows, batch size 8,
  maximum length 128, three complete epochs, and the locked AdamW settings.
- Targets: query and value projections in both BERT-tiny attention layers; the base remains
  frozen.
- Scores: the H1 task-specific primary scores. Aggregate by first averaging each task over the
  three seeds and then taking the unweighted mean over tasks.
- Storage: count every persistent learned scalar and checkpoint byte. Report the common frozen
  base separately. Active adapter operations are reported separately from persistent storage.
- Valid reuse requires matching resolved configuration, model revision, task/seed, selected row
  IDs, fingerprints, tensor shapes, and component metadata. A filename alone is not evidence that
  a checkpoint is reusable.
- Every evaluation saves predictions, labels, task metrics, parameter categories, runtime,
  provenance, and compact component paths. No full BERT checkpoint is permitted.
- Runs are resumable by completed atomic records. `--force` is the only way to replace a complete
  cell.

All new artifacts live below `results/atom_validation/`. New outcomes may not change the rules in
this document; a revised rule requires a versioned specification and fresh affected runs.

## Experiment A: matched shared-LoRA/atom frontier

### Question

Does task-specific atom composition improve on ordinary shared multitask LoRA when persistent
storage and active adapter compute are closely matched?

### Systems

For each capacity `c in [1, 2, 4, 8]` and each seed:

1. **Shared multitask LoRA:** one rank-`c` LoRA bank is used by all five tasks, with five task
   heads. Use `alpha = rank`, so effective scaling is one.
2. **Shared atoms:** train `N = c` rank-1 atoms per target layer, task-specific coefficient rows,
   and five task heads. All `N` atoms are active during training and evaluation.

Both systems use the same balanced complete-pass multitask scheduler and 3,750 optimizer updates.
At matched `c`, both execute `1,024 * c` estimated adapter operations per token. Atom storage is
slightly larger only because its task-specific coefficients are honestly counted.

### Required report

Report every task/seed score; three-seed task means and standard deviations; overall mean and
worst task; exact parameters and bytes; runtime and peak RSS; matched-capacity score deltas; and
the exact and tolerance-aware Pareto frontiers over quality, worst-task quality, persistent
storage, and active operations.

The practical equality tolerance for mean primary score is `0.005`. An **atom-specific advantage**
requires at least one matched capacity where:

- atom mean exceeds shared-LoRA mean by at least `0.005`;
- atom worst-task score is no more than `0.01` below shared LoRA;
- atom persistent storage is no more than `1.02` times shared-LoRA storage; and
- active adapter operations are identical.

Otherwise this experiment does not support an atom-specific advantage, even if both shared
systems compress independent LoRA.

## Experiment B: crossed frozen-dictionary transfer

### Question

Can a dictionary learned on four tasks provide useful reusable capacity for each unseen fifth
task?

For each target task and seed, train one `N = 8` source dictionary jointly on the other four tasks.
Freeze every atom vector. For the held-out target, train only a new coefficient row and target
head, and evaluate both all-eight and deterministic magnitude top-4 inference.

Compare against:

1. a fresh independent rank-4 LoRA and target head;
2. a separately trained head-only model;
3. a deterministic random frozen `N = 8` dictionary with trained coefficients and head.

This creates 15 paired target/seed cells. Source tasks always preserve locked task order.

### Decision rules

The original primary transfer rule remains:

```text
aggregate learned-top4 mean / aggregate fresh-LoRA mean >= 0.95
```

and learned transfer must add fewer task-specific parameters than fresh LoRA.

For **strong reusable-basis support**, all of these diagnostics must also hold:

- every held-out task's three-seed mean retention is at least `0.90`;
- aggregate learned-top4 quality exceeds both head-only and random-frozen quality by at least
  `0.005`; and
- marginal new task state is no more than `10%` of fresh-LoRA task state.

Report marginal new state and total state including the reused dictionary. A result may pass the
original retention rule yet fail the stronger control-aware interpretation.

## Experiment C: oracle LoRA-update projection

### Question

When transfer fails, is the frozen atom span missing the held-out update, or did coefficient
learning fail to find an update that the span can already express?

For every Experiment B target/seed cell and every target module:

1. Load the fresh target rank-4 LoRA and form its exact effective matrix `B @ A`, including LoRA
   scaling.
2. Form one flattened effective matrix per frozen rank-1 atom,
   `atom_scaling * outer(atom_u[k], atom_v[k])`.
3. Solve the float64 least-squares coefficients that minimize Frobenius reconstruction error.
4. Repeat for the matched random frozen dictionary.
5. Evaluate the learned-span and random-span oracle updates with the **fresh LoRA task head**,
   using all eight atoms and magnitude top-4 coefficients.

Save coefficients and, per module, matrix dimensions, numerical rank, singular values or condition
diagnostics, absolute and relative Frobenius error, and explained energy. Aggregate errors are
weighted by target-update squared Frobenius norm; do not average layer percentages equally.

**Strong learned-span support** requires all of:

- learned-span all-eight aggregate quality retention versus fresh LoRA is at least `0.95`;
- every target's three-seed mean quality retention is at least `0.90`;
- learned-span aggregate relative reconstruction error is strictly below the random-span error;
  and
- learned-span oracle quality exceeds random-span oracle quality by at least `0.005`.

Top-4 oracle results are a matched-active-compute diagnostic. All-eight results are primary for
testing whether the learned span contains the target update at all.

## Interpretation boundary

- Experiment A isolates atoms from the simpler explanation of ordinary joint sharing.
- Experiment B tests frozen supervised reuse, not conditional generation: target labels still
  optimize coefficients.
- Experiment C is an oracle upper-bound diagnostic, not a deployable system.
- A failed oracle span test is evidence to revise the dictionary before building a hypernetwork.
- A successful oracle span test combined with failed learned coefficients motivates better
  coefficient inference or meta-training.
- None of these outcomes alone demonstrates semantic atom identity. That requires normalized
  functional contribution tests, causal ablations, and genuinely compositional held-out splits.
