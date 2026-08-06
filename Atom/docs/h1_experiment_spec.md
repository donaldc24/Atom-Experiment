# H1 experiment specification: shared atom dictionary vs. independent LoRA

Status: **locked experimental contract** for the first five-task H1 comparison.

This document defines the comparison before model implementation. A confirmatory run is valid only when its resolved configuration agrees with this document. Development runs may be smaller, but they do not contribute evidence to the H1 decision. Any change to a locked choice requires a documented contract revision and a fresh run of both systems for all confirmatory seeds.

## Scope and research question

The question is:

> Can one dictionary of learned rank-1 adapter atoms represent five task-specific adaptations more compactly than five independent rank-4 LoRA adapters while retaining most of their validation quality?

This experiment tests joint compression of known tasks. It does not test held-out transfer, natural-language routing, a hypernetwork, wake-dream consolidation, plugins, online learning, personalization, or a large language model. It must not be described as evidence for those claims.

The common frozen checkpoint is `prajjwal1/bert-tiny`. The dataset repository is
`nyu-mll/glue`; its five configurations are `sst2`, `mrpc`, `rte`, `qnli`, and `qqp`.

## Compared systems

Both systems use the same tokenizer, resolved checkpoint revision, frozen BERT weights, first-token representation, data rows, batching rules, task heads, optimizer family, and training hyperparameters. Each task has a separate two-label linear classification head. The BERT encoder is frozen throughout; any gradient or optimizer update to a base-model parameter invalidates the run.

### A. Independent LoRA baseline

Train one separate adapter and one separate classification head for each task. At every target linear layer, the correction is a rank-4 LoRA update

```text
delta(x) = B(A(x))
```

where `A` and `B` have no bias, the rank is 4, and the effective scale is 1 (`alpha = rank = 4`). LoRA dropout is 0. Initialize `A` from `Normal(0, 0.02)` and `B` to zeros so the initial correction is zero. No adapter parameter is shared between tasks, and no complete BERT copy is stored per task.

### B. Shared atom dictionary

Use one BERT instance. Each target linear layer owns a bank of 8 rank-1 atoms; that bank is shared by all five tasks. Atom parameters are shared across tasks but not across different target layers. For target layer `m`, atom `k` is an outer product of vectors `u[m,k]` and `v[m,k]`. Task `t` has a coefficient vector for that target layer:

```text
delta[m,t](x) = sum(k=0..7) coefficient[m,t,k] * u[m,k] * dot(v[m,k], x)
```

The scale is 1. All 8 atoms are active during training. Initialize both atom-vector sides from `Normal(0, 0.02)` and coefficients from `Normal(0, 0.001)`. This makes the initial correction near zero without cutting off gradient flow. Initialize every classification-head weight from `Normal(0, 0.02)` and every head bias to zero in both systems.

The only task-specific adaptation state is the per-layer coefficient row and the task head. There is no private residual adapter, hypernetwork, router, learned gate, or other task-specific capacity in this experiment.

The primary shared-atom result is obtained without retraining: in every task and target layer, keep the four coefficients with largest absolute value and set the other four to zero. Break exact ties by lower atom index. This is the **top-4** result. It matches the baseline's active rank of 4. The unpruned 8-atom score and an optional top-4 score after fixed-mask fine-tuning are diagnostics only and cannot be substituted for the primary result.

## Locked configuration

```yaml
base_model: prajjwal1/bert-tiny
device: cpu
tasks: [sst2, mrpc, rte, qnli, qqp]
train_examples_per_task: 2000
validation_examples_per_task: 500
max_length: 128
batch_size: 8
learning_rate: 0.0003
epochs: 3
weight_decay: 0.01
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 0.00000001
learning_rate_schedule: constant
target_modules: [query, value]
lora_rank: 4
lora_alpha: 4
lora_dropout: 0.0
atom_count: 8
active_atoms_during_training: 8
active_atoms_for_primary_evaluation: 4
atom_scaling: 1.0
sparsity_lambda: 0.00001
seeds: [17, 29, 43]
```

Use AdamW with the constants above, applying weight decay to every trainable tensor. Use a constant learning rate, no warmup, no gradient accumulation, no gradient clipping, and no early stopping. Train for all three epochs even though the best checkpoint is retained. Use mean cross-entropy classification loss. For an atom batch from task `t`, define the L1 term as the arithmetic mean of `abs(coefficient[m,t,k])` over all target modules `m` and all eight atoms `k`; the total loss is `classification_loss + 1e-5 * L1`. The roadmap describes this weight only as “small”; this contract resolves that ambiguity to `1e-5` before results are observed. No other regularizer is allowed in the initial H1 comparison.

Seed 17 is the development seed. Once the full pipeline, result schema, and parameter accounting are verified, rerun the locked pipeline from fresh initialization for all three seeds, including 17. The final decision requires the complete `17`, `29`, and `43` result grid; a one-seed result is preliminary.

Each seed controls Python `random`, NumPy, PyTorch initialization and data-loader order, deterministic PyTorch behavior where practical, dataset shuffling, and multitask batch ordering. Paired systems at a given seed use exactly the same selected examples. Record library versions, platform, resolved model revision, dataset fingerprints, resolved configuration, and any remaining nondeterministic backend operation.

## Exact adaptation targets

Adapt only the query and value projections in every self-attention layer. For the two-layer BERT-tiny encoder, the canonical encoder-relative targets are:

```text
encoder.layer.0.attention.self.query
encoder.layer.0.attention.self.value
encoder.layer.1.attention.self.query
encoder.layer.1.attention.self.value
```

A model wrapper may add a prefix such as `bert.`; that prefix is not semantically significant. Before training, resolve and save the full module paths from the actual module tree. A valid run must find exactly these four `nn.Linear` targets, with the same input and output dimensions in both systems. Match the full suffix `.attention.self.query` or `.attention.self.value`, not the bare words `query` and `value`.

Do not adapt keys, attention output projections, feed-forward layers, embeddings, poolers, or classification heads. Do not introduce adapter biases or learned scaling values. A missing, extra, duplicated, or differently shaped target invalidates the run.

## Data contract

All data comes from the named GLUE configuration. Use the official `train` and `validation` splits only; validation examples must never enter training.

| Task | Dataset/config | Input fields | Train subset | Validation subset | Labels |
|---|---|---|---:|---:|---|
| SST-2 | `nyu-mll/glue`, `sst2` | `sentence` | up to 2,000 | up to 500 | binary sentiment |
| MRPC | `nyu-mll/glue`, `mrpc` | `sentence1`, `sentence2` | up to 2,000 | up to 500 | binary paraphrase |
| RTE | `nyu-mll/glue`, `rte` | `sentence1`, `sentence2` | up to 2,000 | up to 500 | binary entailment |
| QNLI | `nyu-mll/glue`, `qnli` | `question`, `sentence` | up to 2,000 | up to 500 | binary entailment |
| QQP | `nyu-mll/glue`, `qqp` | `question1`, `question2` | up to 2,000 | up to 500 | binary duplicate-question status |

For each split and seed, select rows before tokenization as follows:

```python
n = min(requested_limit, len(split))
subset = split.shuffle(seed=seed).select(range(n))
```

Never duplicate rows to reach a limit. Thus a split with fewer than 500 examples, notably RTE validation and potentially another small GLUE validation split, uses every available row once. Save the selected original row identifiers or indices and the dataset fingerprint so both systems can prove that they used the same subset.

Tokenize sentence pairs in the field order shown above, truncate at 128 tokens, and use batch size 8. Do not augment examples or rebalance labels. All requested training subsets currently contain 2,000 examples; if an upstream revision does not, the same `min` rule applies and the resolved count must be reported.

## Training and checkpoint budget

For an independent task with 2,000 rows, one epoch is 250 batches and the fixed budget is 750 optimizer updates. Train each of the five independent adapters for three complete passes over its task subset, for 3,750 optimizer updates across the full baseline system. A smaller last batch is allowed only when the resolved subset size is not divisible by 8.

For the shared system, each multitask epoch contains one complete pass through each task loader. Interleave task batches using a seeded balanced schedule; each task contributes all of its batches exactly once per epoch. With 2,000 rows for every task, this is 1,250 shared optimizer updates per epoch and 3,750 updates total. This matches the sum of independent-LoRA task updates while preventing larger source datasets from dominating. Record realized batch and example counts per task.

Evaluate after every epoch. For independent LoRA, retain the checkpoint with the highest primary validation score for that task. For shared atoms, retain the checkpoint with the highest unweighted mean of the five unpruned primary task scores. Break checkpoint-selection ties in favor of the earlier epoch. Always report both best-checkpoint and final-epoch scores. Derive the primary top-4 result from the retained shared checkpoint, without additional updates.

Development plumbing runs may use two tasks, 500 training examples per task, fewer epochs, or seed 17 only. Their output directories must be marked `development`; they cannot be pooled with or replace confirmatory results.

## Predeclared evaluation metrics

Report accuracy for every task. Also report binary F1 for MRPC and QQP, treating GLUE label `1` as the positive class and using zero when F1 has a zero denominator. Report example-weighted mean training loss and validation loss, exact parameter counts, wall-clock runtime, and peak resident memory when practical.

The primary scalar for each task is fixed before results are observed:

| Task | Primary score |
|---|---|
| SST-2 | accuracy |
| MRPC | `(accuracy + F1) / 2` |
| RTE | accuracy |
| QNLI | accuracy |
| QQP | `(accuracy + F1) / 2` |

All quality scores are fractions in `[0, 1]`. Let `q[m,s,t]` be the primary score for model `m`, seed `s`, and task `t`. Aggregate in this order:

```text
task_score[m,t] = arithmetic mean over seeds 17, 29, 43 of q[m,s,t]
mean_score[m]   = unweighted arithmetic mean over the five task_score[m,t] values
```

Also report each seed, the across-seed standard deviation for every task, and the unpruned 8-atom result. Do not micro-average examples across tasks; doing so would overweight tasks with larger validation subsets.

For the preregistered comparison, define:

```text
quality_retention = mean_score[shared_top4] / mean_score[independent_lora]
task_gap[t]       = task_score[independent_lora,t] - task_score[shared_top4,t]
worst_task_gap    = max(0, max over tasks of task_gap[t])
relative_storage  = shared persistent adaptation parameters
                    / independent persistent adaptation parameters
```

The gap is an absolute score difference; `0.03` means three percentage points, not a three-percent relative change.

## Pass, fail, and validity rules

H1 receives a **PASS / Supported** decision only if every condition below is true for the valid three-seed primary comparison:

1. `quality_retention >= 0.97`.
2. `worst_task_gap <= 0.03`.
3. `relative_storage <= 0.50` under the accounting rules below.
4. The independent system has active rank 4 and the primary shared result has exactly 4 active atoms per task per target layer.
5. The shared system contains no uncounted task-specific residual, router, gate, alternate checkpoint, ensemble member, or other persistent learned state.

Failure of any one condition is a **FAIL to meet the preregistered H1 criteria**. A narrative label of “Partially supported” may be used when quality or storage is encouraging but at least one threshold fails; it remains a failed preregistered test and no threshold may be changed afterward. The result is “Not supported” when quality degrades beyond the limits, honest shared storage exceeds 50% of baseline storage, matched active capacity cannot be maintained, or apparent sharing depends on private task capacity. Atom partitioning into effectively private task groups, a need for large residuals, collapse under top-4 pruning, or parameter savings that disappear after full accounting must be reported as failure modes even when another metric looks favorable.

The following are protocol violations, not evidence for or against H1: different data rows between paired systems; a different base revision; trained base weights; missing or extra target modules; post-result metric selection; top-4 selection by validation score instead of coefficient magnitude; omitted learned storage; a changed optimizer or budget; missing required seeds; NaNs; corrupt checkpoints; or incomplete result files. Fix implementation defects and rerun the affected paired comparison from fresh initialization. Do not silently discard a bad seed. If the locked configuration repeatedly cannot produce a valid run, report an operational failure rather than tuning only the failing system until it passes.

## Honest parameter and storage accounting

The primary storage unit is the number of persistent learned scalar values needed to deploy all five tasks for one selected checkpoint. Count tensor elements with `numel()`, independent of dtype. Deduplicate only genuine tensor sharing by parameter identity or storage identity.

Let `M` be the four target modules, `T = 5`, `r = 4`, and `N = 8`. For target module `m` with dimensions `d_in[m]` and `d_out[m]`:

```text
independent adapter parameters
  = T * sum(m in M) r * (d_in[m] + d_out[m])

shared atom parameters
  = sum(m in M) N * (d_in[m] + d_out[m])

shared coefficient parameters
  = sum(m in M) T * N

head parameters
  = sum(t in tasks) (hidden_size * num_labels[t] + num_labels[t])
```

The independent total is all five LoRA adapters plus all five task heads. The shared total is every per-layer atom vector, every task coefficient (including coefficients zeroed by top-4), and all five task heads. Top-4 pruning changes active capacity, not the stored eight-atom dictionary, so it does not by itself reduce the primary parameter count.

Count all of the following if they exist, whether or not they have `requires_grad=True` at evaluation time:

- adapter matrices and adapter biases;
- all atom vectors and every task/module coefficient row;
- every task head, including bias;
- task-specific residuals or private atoms;
- learned scales, masks, gates, routers, hypernetworks, or regularizer parameters;
- any other learned tensor required to reproduce inference.

The initial contract prohibits most items in the last two bullets; if implementation introduces one, it must still be counted and the run is not the specified primary comparison.

Count a shared atom tensor once, not once per task. Do count separate atom banks in separate target modules. Count each independent task adapter once. Count structurally present zeros and pruned coefficients: compression through a sparse file format does not erase learned degrees of freedom. A fixed top-4 mask is not a learned scalar, but its indices or mask bytes must be included in checkpoint-byte reporting. If a gate or mask is learned, its scalar values are parameters.

Exclude the frozen base model from the relative-storage numerator and denominator only because the exact same base checkpoint is common to both systems. Report its total and trainable parameter counts separately, and require base trainable parameters to equal zero. Do not exclude task heads merely because both systems need them. Exclude optimizer state, gradients, activations, and transient training buffers from persistent parameter counts; report peak RAM separately. Constant hyperparameters and label names are metadata, not learned parameters.

Report, separately from scalar parameter counts:

- serialized checkpoint bytes, including masks/indices and tensor metadata;
- dtype and serialization format;
- training runtime and inference latency;
- peak resident RAM when practical;
- number of active atoms and estimated adapter operations.

Do not claim a storage win from lower precision, compression, or saving fewer metadata files unless the same representation is applied to both systems. Experimental replicates from different seeds are not summed as one deployable model, but disk usage for every retained result must still be reported.

For the expected BERT-tiny shape of four `128 x 128` targets and five two-label heads, the hand calculation is a useful assertion:

```text
independent LoRA adapters = 20,480
five task heads           =  1,290
independent total         = 21,770

shared atom vectors       =  8,192
task coefficients         =    160
five task heads           =  1,290
shared total              =  9,642
expected relative storage = 44.2903%
```

These expected values are not a substitute for measuring the loaded checkpoint. A dimension, target-count, head-design, bias, or sharing change must alter the measured count and be explained; code must never hard-code these totals as the result.

## Required run record

Every confirmatory result must save enough information to audit this contract:

- system name, task, seed, selected checkpoint epoch, and development/confirmatory status;
- resolved model revision, tokenizer revision, dataset fingerprints, and selected row identifiers;
- full resolved configuration and software/platform versions;
- resolved target module paths and dimensions;
- per-epoch losses and validation metrics;
- per-task accuracy, required F1, and primary score;
- exact categorized parameter counts and checkpoint bytes;
- realized examples, batches, optimizer updates, runtime, and peak memory;
- for atoms, coefficient matrices, top-4 masks, and unpruned/top-4 scores.

A fresh summary must be able to recompute every aggregate, storage ratio, threshold, and final decision from these saved records without manual edits.
