# H1 Follow-up Controls (Roadmap Chunk 25)

Status: **COMPLETE**

All controls use the locked seed 17 data selection, 2,000 training rows per task (or the upstream split minimum), up to 500 validation rows, batch size 8, and three epochs.

These diagnostics do not change the preregistered H1 decision.

## Primary scores

| Control | SST-2 | MRPC | RTE | QNLI | QQP | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Random frozen atoms + trained coefficients/heads | 0.6840 | 0.7655 | 0.5812 | 0.6340 | 0.6144 | 0.6558 |
| Average independent LoRA effective updates | 0.6900 | 0.7535 | 0.5487 | 0.6480 | 0.5554 | 0.6391 |
| Nearest other-task LoRA retrieval | 0.6000 | 0.7480 | 0.5812 | 0.6680 | 0.5760 | 0.6347 |
| One shared balanced multitask LoRA | 0.7240 | 0.7637 | 0.5632 | 0.6980 | 0.6792 | 0.6856 |
| Shared atoms with deterministically shuffled labels | 0.5280 | 0.7480 | 0.5162 | 0.5400 | 0.3250 | 0.5315 |
| Shared atoms without coefficient sparsity | 0.7080 | 0.7673 | 0.5668 | 0.6600 | 0.6277 | 0.6660 |

## Control definitions

### Random frozen atoms + trained coefficients/heads

State: **complete**. Intended to rule out random projection capacity.

- model: `shared_atom_dictionary`
- atom count: `8`
- active atoms for evaluation: `4`
- atoms frozen: `True`
- coefficients trainable: `True`
- heads trainable: `True`
- sparsity lambda: `1e-05`
- training labels shuffled: `False`
- multitask schedule: `seeded balanced complete pass`

### Average independent LoRA effective updates

State: **complete**. Intended to rule out simple adapter averaging.

- model: `one_shared_rank4_lora`
- average space: `effective_delta_weight`
- formula: `mean_t(B_t @ A_t)`
- rank projection: `deterministic truncated SVD`
- rank: `4`
- classification heads: `each target task's own selected core head`
- additional training updates: `0`

### Nearest other-task LoRA retrieval

State: **complete**. Intended to rule out memorized task lookup or nearest-adapter reuse.

- model: `one_retrieved_independent_rank4_lora_per_target`
- retrieval query: `target task's selected independent LoRA`
- similarity: `cosine of concatenated effective B @ A updates`
- candidate pool: `all four other tasks; target adapter excluded`
- tie break: `locked task order`
- classification heads: `target task's own selected core head`
- additional training updates: `0`

| Target task | Retrieved other task | Cosine similarity |
|---|---|---:|
| sst2 | qqp | 0.0311 |
| mrpc | qnli | 0.1309 |
| rte | qnli | 0.0410 |
| qnli | mrpc | 0.1309 |
| qqp | mrpc | 0.0972 |

### One shared balanced multitask LoRA

State: **complete**. Intended to rule out ordinary multitask sharing without atom composition.

- model: `one_shared_lora_bank_plus_five_task_heads`
- rank: `4`
- alpha: `4`
- multitask schedule: `seeded balanced complete pass`
- updates: `3750`
- checkpoint selection: `earliest epoch with highest unweighted mean task score`

### Shared atoms with deterministically shuffled labels

State: **complete**. Intended to rule out data-independent capacity or leakage.

- model: `shared_atom_dictionary`
- atom count: `8`
- active atoms for evaluation: `4`
- atoms frozen: `False`
- coefficients trainable: `True`
- heads trainable: `True`
- sparsity lambda: `1e-05`
- training labels shuffled: `True`
- multitask schedule: `seeded balanced complete pass`

### Shared atoms without coefficient sparsity

State: **complete**. Intended to rule out an overly strong sparsity penalty.

- model: `shared_atom_dictionary`
- atom count: `8`
- active atoms for evaluation: `4`
- atoms frozen: `False`
- coefficients trainable: `True`
- heads trainable: `True`
- sparsity lambda: `0.0`
- training labels shuffled: `False`
- multitask schedule: `seeded balanced complete pass`
