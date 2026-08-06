# CGMoE H1 Experiment Roadmap

## Shared Atom Dictionary vs. Independent LoRA

**Project:** Controlled Generative Mixture of Experts (CGMoE)  
**Hypothesis under test:** H1 — Reusable Operator Basis  
**Target machine:** Dell XPS 13, 16 GB RAM, CPU-first  
**Primary goal:** Determine whether several task-specific adaptations can reuse a compact shared dictionary of learned rank-1 operators while preserving most of the quality of independent LoRA adapters.

---

## 1. Research Question

This experiment tests the smallest useful form of the CGMoE dictionary hypothesis:

> Can a shared dictionary of learned adapter atoms represent several task-specific adaptations more compactly than storing an independent LoRA for every task?

The experiment deliberately does **not** include:

- A hypernetwork
- Dynamic routing from natural-language context
- Wake-dream consolidation
- Plugins
- Online learning
- User personalization
- Large language models

Those are separate research questions. This roadmap isolates H1 so a failure cannot hide behind unrelated system complexity.

---

## 2. Systems Being Compared

### System A: Independent LoRA Baseline

Each task receives its own rank-4 LoRA adapter.

```text
Frozen base BERT
├── SST-2 LoRA
├── MRPC LoRA
├── RTE LoRA
├── QNLI LoRA
└── QQP LoRA
```

Nothing is shared between task adapters.

### System B: Shared Atom Dictionary

All tasks share the same bank of rank-1 adapter atoms. Each task learns a small coefficient vector describing how to combine them.

```text
Frozen base BERT
├── Shared atoms
├── SST-2 coefficients
├── MRPC coefficients
├── RTE coefficients
├── QNLI coefficients
└── QQP coefficients
```

For a single adapted linear layer:

```text
Task adapter =
    coefficient[0] × atom[0]
  + coefficient[1] × atom[1]
  + ...
  + coefficient[N-1] × atom[N-1]
```

The first experiment uses stored coefficients indexed by task ID. A future H2 experiment may replace the lookup with a hypernetwork.

---

## 3. Initial Success Criteria

The first H1 experiment is considered promising when all of the following hold:

1. **Mean quality retention**

   ```text
   Shared-atom mean score >= 97% of independent-LoRA mean score
   ```

2. **Worst-task protection**

   ```text
   Worst individual task gap <= 3 percentage points
   ```

3. **Storage reduction**

   ```text
   Shared-atom adaptation parameters <= 50% of independent-LoRA adaptation parameters
   ```

4. **Matched active capacity**

   ```text
   Independent LoRA active rank = 4
   Shared atoms active per task = 4 after pruning
   ```

5. **No hidden storage**

   Count all of the following:

   - Shared atom parameters
   - Task coefficients
   - Task-specific residuals, if any
   - Task classification heads
   - Routing or gating parameters, if introduced

The frozen base model is excluded from the comparison because both systems use the same base.

---

## 4. Recommended Model and Tasks

### Base Model

Start with:

```text
prajjwal1/bert-tiny
```

This model is intentionally small. It is suitable for validating the training and architecture code on a 16 GB CPU laptop.

Later, repeat the experiment with a somewhat larger compact BERT model after the implementation is stable.

### Tasks

Build up gradually:

#### First integration tasks

- SST-2
- MRPC

#### Full initial H1 task set

- SST-2
- MRPC
- RTE
- QNLI
- QQP

Use small fixed training subsets so every experiment is repeatable and laptop-friendly.

Suggested initial limits:

```text
Training examples per task:   2,000
Validation examples per task: 500
Maximum sequence length:      128
Batch size:                   8
```

Smaller tasks such as RTE may have fewer than the requested number of examples. Use the available data without duplication unless the experiment configuration explicitly enables oversampling.

---

# Phase 0 — Freeze the Experimental Contract

## Chunk 0: Write the Experiment Specification

### Goal

Define exactly what will be compared before writing model code.

### Tasks

Create:

```text
docs/h1_experiment_spec.md
```

Record:

- Research question
- Baselines
- Model checkpoint
- Dataset names
- Dataset subset sizes
- Random seeds
- LoRA rank
- Atom count
- Target modules
- Training budget
- Evaluation metrics
- Success criteria
- Failure criteria
- Parameter-counting rules

### Initial configuration

```yaml
base_model: prajjwal1/bert-tiny
tasks:
  - sst2
  - mrpc
  - rte
  - qnli
  - qqp
train_examples_per_task: 2000
validation_examples_per_task: 500
max_length: 128
batch_size: 8
lora_rank: 4
atom_count: 8
target_modules:
  - query
  - value
seeds:
  - 17
  - 29
  - 43
```

Use one seed while developing. Run all three seeds only after the pipeline is stable.

### Done when

- The experiment has a written pass/fail rule.
- No architecture choice is being changed silently between runs.
- The parameter accounting rules are explicit.

### Commit

```bash
git add docs/h1_experiment_spec.md
git commit -m "Define H1 experiment contract"
```

---

# Phase A — Project and Environment

## Chunk 1: Create the Python Project

### Goal

Create a clean, reproducible project that imports correctly.

### Suggested structure

```text
cgmoe-h1/
├── README.md
├── pyproject.toml
├── requirements-lock.txt
├── configs/
│   ├── baseline.yaml
│   └── atoms.yaml
├── docs/
│   └── h1_experiment_spec.md
├── results/
├── scripts/
│   ├── check_environment.py
│   └── check_model.py
├── src/
│   └── cgmoe_h1/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── metrics.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── classifier.py
│       │   ├── lora.py
│       │   ├── atoms.py
│       │   └── injection.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── multitask.py
│       └── utils/
│           ├── __init__.py
│           ├── parameters.py
│           ├── reproducibility.py
│           └── serialization.py
└── tests/
    ├── test_lora.py
    ├── test_atoms.py
    ├── test_injection.py
    └── test_parameter_counts.py
```

### Create the environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Linux or WSL:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Install dependencies

```bash
python -m pip install torch transformers datasets evaluate scikit-learn tqdm pyyaml pytest
python -m pip freeze > requirements-lock.txt
```

### Environment smoke test

Create `scripts/check_environment.py` that prints:

- Python version
- Platform
- PyTorch version
- Transformers version
- Number of CPU threads
- CUDA availability
- Two SST-2 rows

### Model smoke test

Create `scripts/check_model.py` that:

1. Downloads the tokenizer.
2. Downloads `prajjwal1/bert-tiny`.
3. Tokenizes one sentence.
4. Runs one forward pass.
5. Prints tensor shapes and model parameter count.

### Done when

```text
[ ] Virtual environment activates
[ ] Project imports with `python -m`
[ ] PyTorch imports
[ ] Dataset rows load
[ ] BERT forward pass succeeds
[ ] pytest runs, even if no tests exist yet
```

### Commit

```bash
git add .
git commit -m "Set up H1 experiment project"
```

---

## Chunk 2: Add Configuration and Reproducibility

### Goal

Move experiment settings out of scattered Python constants.

### Files

```text
src/cgmoe_h1/config.py
src/cgmoe_h1/utils/reproducibility.py
configs/baseline.yaml
configs/atoms.yaml
```

### Configuration fields

Include:

```yaml
experiment_name: independent_lora
base_model: prajjwal1/bert-tiny
seed: 17
device: cpu
max_length: 128
batch_size: 8
learning_rate: 0.0003
epochs: 3
weight_decay: 0.01
train_examples_per_task: 2000
validation_examples_per_task: 500
tasks:
  - sst2
  - mrpc
target_modules:
  - query
  - value
lora_rank: 4
atom_count: 8
active_atoms: 4
```

### Reproducibility function

Implement one function that seeds:

- Python `random`
- NumPy
- PyTorch
- PyTorch deterministic settings where practical

Example API:

```python
def set_seed(seed: int) -> None:
    ...
```

### Tests

Verify that two runs with the same seed initialize identical tensors.

### Done when

- Both YAML files load into typed Python configuration objects.
- One command prints the resolved configuration.
- Repeated initialization with the same seed is identical.

### Commit

```bash
git add .
git commit -m "Add experiment configuration and seeding"
```

---

# Phase B — Understand and Prepare the Base Model

## Chunk 3: Inspect BERT Internals

### Goal

Locate the exact linear modules to adapt.

### Tasks

Write:

```text
scripts/inspect_model.py
```

Print:

- Every `nn.Linear` module name
- Input features
- Output features
- Whether its parameters require gradients

Find names corresponding to:

```text
encoder.layer.<n>.attention.self.query
encoder.layer.<n>.attention.self.value
```

The exact prefix may differ by model wrapper. Trust the printed module tree, not assumptions.

### Capture

Record:

- Number of transformer layers
- Hidden size
- Number of query matrices
- Number of value matrices
- Shape of each target matrix
- Total base-model parameters

### Done when

You can answer:

```text
How many modules will receive adapters?
What are their dimensions?
What exact module names must the injector match?
```

### Commit

```bash
git add scripts/inspect_model.py docs/
git commit -m "Inspect BERT adapter target modules"
```

---

## Chunk 4: Freeze the Base Model and Add a Classification Wrapper

### Goal

Build a reusable classifier that leaves BERT frozen.

### Files

```text
src/cgmoe_h1/models/classifier.py
tests/test_classifier.py
```

### Design

The wrapper should:

1. Hold a pretrained BERT encoder.
2. Freeze all encoder parameters.
3. Extract the first token representation.
4. Pass it through a task-specific classification head.
5. Return logits.

Initial API:

```python
class BertTaskClassifier(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

For multitask use later, support:

```python
forward(..., task_id: str)
```

### Tests

Verify:

- Output shape is `[batch_size, num_labels]`.
- Base-model parameters have `requires_grad=False`.
- Head parameters have `requires_grad=True`.
- Backward pass creates gradients for the head only.

### Done when

A dummy batch can complete forward and backward passes without modifying BERT.

### Commit

```bash
git add .
git commit -m "Add frozen BERT classification wrapper"
```

---

# Phase C — Data and Evaluation

## Chunk 5: Build the Dataset Layer

### Goal

Load all tasks through one consistent interface.

### Files

```text
src/cgmoe_h1/data.py
tests/test_data.py
```

### Task schemas

#### SST-2

```text
Input: sentence
Label: binary sentiment
```

#### MRPC

```text
Input: sentence1, sentence2
Label: paraphrase or not
```

#### RTE

```text
Input: sentence1, sentence2
Label: entailment or not
```

#### QNLI

```text
Input: question, sentence
Label: entailment or not
```

#### QQP

```text
Input: question1, question2
Label: duplicate or not
```

### Required behavior

Your dataset function should:

1. Load the named GLUE configuration.
2. Choose the correct text columns.
3. Tokenize single sentences or sentence pairs.
4. Truncate to `max_length`.
5. Select a deterministic subset.
6. Return PyTorch-compatible datasets.
7. Preserve task ID.

Suggested API:

```python
def load_task_data(
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    train_limit: int,
    validation_limit: int,
    max_length: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    ...
```

### Important rule

Select subsets deterministically:

```python
dataset.shuffle(seed=seed).select(range(limit))
```

Do not take arbitrary changing subsets across runs.

### Tests

For each task:

- Dataset loads.
- Expected columns exist.
- Tokenized tensors have valid shapes.
- Labels are in the expected range.
- The same seed selects the same examples.

### Done when

One batch from SST-2 and one batch from MRPC pass through the frozen classifier.

### Commit

```bash
git add .
git commit -m "Add deterministic GLUE data pipeline"
```

---

## Chunk 6: Add Metrics and Result Serialization

### Goal

Make every run produce comparable machine-readable output.

### Files

```text
src/cgmoe_h1/metrics.py
src/cgmoe_h1/utils/serialization.py
```

### Initial metrics

Use:

- Accuracy for all tasks
- F1 for MRPC and QQP
- Training loss
- Validation loss
- Parameter counts
- Runtime
- Peak resident memory if practical

For the first H1 summary, define one primary scalar per task:

```text
SST-2: accuracy
MRPC:  average of accuracy and F1, or report both and predeclare one
RTE:   accuracy
QNLI:  accuracy
QQP:   average of accuracy and F1, or report both and predeclare one
```

Do not choose the metric after seeing which one looks better.

### Result file example

```json
{
  "experiment": "independent_lora",
  "seed": 17,
  "task": "sst2",
  "model": "prajjwal1/bert-tiny",
  "metrics": {
    "accuracy": 0.78,
    "validation_loss": 0.51
  },
  "parameters": {
    "base_total": 4385920,
    "base_trainable": 0,
    "adapter_total": 8192,
    "head_total": 514
  }
}
```

### Done when

One evaluation call returns a Python dictionary and writes valid JSON under `results/`.

### Commit

```bash
git add .
git commit -m "Add metrics and result serialization"
```

---

# Phase D — Establish a Non-Adapter Baseline

## Chunk 7: Train Classification Heads Only

### Goal

Verify that the training loop works before adding LoRA.

### Why this matters

If loss does not decrease with a head-only model, the bug is probably in:

- Data
- Labels
- Tokenization
- Batching
- Loss calculation
- Optimizer
- Evaluation

It is better to find that before adapter code enters the room.

### Files

```text
src/cgmoe_h1/training/trainer.py
scripts/train_head_only.py
```

### Training loop requirements

Implement explicitly:

1. `model.train()`
2. Move batch to device.
3. Zero gradients.
4. Forward pass.
5. Cross-entropy loss.
6. Backward pass.
7. Optimizer step.
8. Track average loss.
9. Evaluate after each epoch.
10. Save best checkpoint by validation score.

### Start with

- SST-2 only
- 500 training examples
- 200 validation examples
- 1 to 3 epochs

### Done when

- Training loss decreases.
- Validation score is above random or majority-class behavior.
- Results are saved.
- Re-running with the same seed gives similar results.

### Commit

```bash
git add .
git commit -m "Add head-only training baseline"
```

---

# Phase E — Implement Independent LoRA

## Chunk 8: Implement a Standalone LoRA Linear Layer

### Goal

Understand LoRA mechanically before integrating it into BERT.

### Files

```text
src/cgmoe_h1/models/lora.py
tests/test_lora.py
```

### Behavior

Wrap an existing frozen `nn.Linear`.

For input `x`:

```text
output = frozen_linear(x) + scale × B(A(x))
```

Where:

- `A` projects from input dimension to rank.
- `B` projects from rank back to output dimension.
- Rank is 4.
- The original linear layer is frozen.
- The LoRA branch starts near zero.

Suggested class:

```python
class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        ...
```

### Initialization

A common safe pattern:

- Initialize `A` with a small random distribution.
- Initialize `B` to zeros.

This causes the adapter correction to begin at zero while preserving gradient flow.

### Tests

Verify:

1. Output shape matches the base layer.
2. Initial wrapped output matches the unwrapped base output.
3. Base parameters are frozen.
4. LoRA parameters receive gradients.
5. Parameter count equals:

   ```text
   rank × input_features + output_features × rank
   ```

6. Saving and loading preserves output.

### Done when

All LoRA unit tests pass without loading BERT.

### Commit

```bash
git add .
git commit -m "Implement and test LoRA linear layer"
```

---

## Chunk 9: Inject LoRA into BERT Query and Value Modules

### Goal

Replace selected BERT linear modules without changing the rest of the model.

### Files

```text
src/cgmoe_h1/models/injection.py
tests/test_injection.py
```

### Required functions

```python
def inject_lora(
    model: nn.Module,
    target_suffixes: tuple[str, ...],
    rank: int,
    alpha: float,
) -> list[str]:
    ...
```

Return the names of modified modules.

### Rules

Only replace modules whose names end in:

```text
attention.self.query
attention.self.value
```

Do not adapt:

- Keys
- Output projections
- Feed-forward layers
- Embeddings
- Classification heads

That keeps the first test controlled and small.

### Tests

Verify:

- Expected number of modules is replaced.
- Non-target modules are unchanged.
- Forward output shape remains valid.
- Only LoRA parameters and the task head are trainable.
- The number of trainable parameters matches a manual calculation.

### Done when

One SST-2 batch completes forward and backward passes through LoRA-injected BERT.

### Commit

```bash
git add .
git commit -m "Inject LoRA into BERT attention projections"
```

---

## Chunk 10: Train One Independent LoRA

### Goal

Establish a working rank-4 LoRA baseline on SST-2.

### Script

```text
scripts/train_independent_lora.py
```

### Configuration

```yaml
task: sst2
rank: 4
target_modules:
  - query
  - value
train_examples: 2000
validation_examples: 500
batch_size: 8
epochs: 3
```

### Record

- Best validation score
- Final validation score
- Adapter parameters
- Head parameters
- Runtime
- Peak memory if available
- Checkpoint path

### Debug expectations

If the loss does not move:

- Confirm LoRA parameters are in the optimizer.
- Confirm labels are correct.
- Confirm `model.train()` is called.
- Print gradient norms.
- Confirm the adapter output is not permanently zero.
- Reduce learning rate if loss explodes.

### Done when

The independent LoRA trains above the head-only baseline or at least demonstrates stable learning.

### Commit

```bash
git add .
git commit -m "Train first independent LoRA baseline"
```

---

## Chunk 11: Generalize Independent LoRA Training to Multiple Tasks

### Goal

Train one separate LoRA per task through the same code path.

### Initial step

Run:

- SST-2
- MRPC

Do not move to all five tasks until both complete.

### Final step

Run:

- SST-2
- MRPC
- RTE
- QNLI
- QQP

### Storage rule

Save only:

- LoRA parameters
- Task head
- Configuration
- Result JSON

Do not save five complete copies of BERT.

### Suggested layout

```text
results/
└── independent_lora/
    └── seed_17/
        ├── sst2/
        │   ├── adapter.pt
        │   ├── head.pt
        │   └── metrics.json
        ├── mrpc/
        ├── rte/
        ├── qnli/
        └── qqp/
```

### Done when

A summary script can print:

| Task | Primary score | Adapter params | Head params |
|---|---:|---:|---:|
| SST-2 | ... | ... | ... |
| MRPC | ... | ... | ... |
| RTE | ... | ... | ... |
| QNLI | ... | ... | ... |
| QQP | ... | ... | ... |

### Commit

```bash
git add .
git commit -m "Complete independent LoRA task baselines"
```

---

# Phase F — Implement Shared Atoms

## Chunk 12: Implement a Standalone Atom Linear Layer

### Goal

Implement the shared operator dictionary independently of BERT.

### Files

```text
src/cgmoe_h1/models/atoms.py
tests/test_atoms.py
```

### Initial design

For one adapted linear layer:

- `N` shared rank-1 atoms
- One coefficient vector per task
- All atoms active during early training

Each atom contains:

```text
u[k]: output_features
v[k]: input_features
```

For a batch input `x`, atom `k` contributes:

```text
u[k] × dot(v[k], x)
```

The task-specific correction is the weighted sum of those contributions.

### Suggested API

```python
class AtomLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        task_ids: list[str],
        atom_count: int,
        scaling: float = 1.0,
    ) -> None:
        ...

    def forward(
        self,
        x: torch.Tensor,
        task_id: str,
    ) -> torch.Tensor:
        ...
```

### Tensor shapes

For a base linear layer with:

```text
input_features = d_in
output_features = d_out
atom_count = N
```

Store:

```text
atom_v: [N, d_in]
atom_u: [N, d_out]
coefficients: [task_count, N]
```

### Initialization

Use near-zero initial adapter behavior.

One option:

- Initialize `atom_v` with small random values.
- Initialize `atom_u` with small random values.
- Initialize coefficients to zero plus tiny noise.

Another option:

- Initialize one side randomly and the other side near zero.

Record the chosen method in configuration.

### Tests

Verify:

1. Output shape matches the base layer.
2. All task IDs use the same atom tensors.
3. Different task IDs use different coefficient rows.
4. Atom parameters receive gradients from every task.
5. Only the selected task’s coefficient row receives gradients for one batch.
6. Initial output is close to the frozen base output.
7. Parameter count equals:

   ```text
   N × (input_features + output_features)
   + task_count × N
   ```

### Done when

A synthetic two-task training test can learn two different mappings while sharing atoms.

### Commit

```bash
git add .
git commit -m "Implement and test shared atom linear layer"
```

---

## Chunk 13: Add Atom Regularization

### Goal

Encourage reuse and discourage degenerate solutions.

### Main risks

#### Risk A: Private partitioning

Each task claims its own atoms.

```text
Task A uses atoms 1 and 2
Task B uses atoms 3 and 4
Task C uses atoms 5 and 6
```

This behaves like hidden independent adapters.

#### Risk B: Collapse

Every task learns nearly identical coefficients.

This behaves like one shared multitask adapter.

#### Risk C: Dead atoms

Some atoms receive almost no gradient and never become useful.

### Start simply

Do not add every regularizer at once.

Initial objective:

```text
total loss =
    task classification loss
  + lambda_sparse × coefficient L1 penalty
```

Use a very small `lambda_sparse`.

Later experiments may add:

- Atom diversity penalty
- Coefficient entropy control
- Usage balancing
- Orthogonality penalties
- Group sparsity

Each new term must be ablated. Otherwise the loss becomes a soup pot with no labels.

### Tests

- Regularization term is finite.
- Setting its weight to zero exactly removes its effect.
- Gradients reach coefficients.
- Training does not produce NaNs.

### Done when

The atom layer can train with and without regularization using configuration only.

### Commit

```bash
git add .
git commit -m "Add configurable atom coefficient regularization"
```

---

## Chunk 14: Inject Shared Atoms into BERT

### Goal

Replace BERT query and value modules with atom-enabled wrappers.

### Injection API

```python
def inject_atoms(
    model: nn.Module,
    task_ids: list[str],
    target_suffixes: tuple[str, ...],
    atom_count: int,
) -> list[str]:
    ...
```

### Important forwarding issue

BERT normally calls linear layers without a `task_id`.

You need a clean way to make the active task available.

### Recommended first solution: context object

Create a model-level task context:

```python
class TaskContext:
    current_task_id: str | None
```

Before a forward pass:

```python
model.set_task("sst2")
logits = model(...)
```

Each `AtomLinear` reads the active task from the shared context.

Avoid global module-level variables. Keep the context owned by the model wrapper.

### Alternative

Pass a task index through custom wrappers around attention modules. This is cleaner long-term but requires more invasive model surgery.

For the first experiment, prefer the smallest reliable implementation.

### Tests

Verify:

- Setting `task_id="sst2"` selects the SST-2 coefficient row.
- Setting `task_id="mrpc"` selects the MRPC row.
- Shared atom tensor identities remain identical across task calls.
- Unsupported task IDs fail clearly.
- Forward and backward passes succeed.

### Done when

One batch from SST-2 and one batch from MRPC can run through the same atom-injected BERT instance.

### Commit

```bash
git add .
git commit -m "Inject shared atom dictionary into BERT"
```

---

# Phase G — Multitask Training

## Chunk 15: Build a Balanced Multitask Batch Scheduler

### Goal

Train shared atoms from multiple tasks without allowing large datasets to dominate.

### Why not concatenate everything?

QNLI and QQP are much larger than RTE. Naive concatenation would make the shared atoms mostly optimize for the largest tasks.

### First scheduler

Use uniform task sampling:

```text
Choose a task uniformly.
Take the next batch from that task.
Reset its iterator when exhausted.
```

Suggested API:

```python
class UniformTaskBatchIterator:
    def __iter__(self) -> Iterator[TaskBatch]:
        ...
```

Where:

```python
@dataclass
class TaskBatch:
    task_id: str
    batch: dict[str, torch.Tensor]
```

### Development order

1. SST-2 and MRPC only
2. Add RTE
3. Add QNLI and QQP

### Tests

Over a long synthetic run:

- Each task is selected approximately equally.
- Every yielded batch includes a valid task ID.
- Exhausting one dataloader does not stop the others.
- Runs are repeatable with the same seed.

### Done when

A dry run prints 100 task selections with approximately balanced counts.

### Commit

```bash
git add .
git commit -m "Add balanced multitask batch scheduler"
```

---

## Chunk 16: Build the Shared-Atom Trainer

### Goal

Train:

- Shared atoms
- Per-task coefficients
- Per-task classification heads

while leaving BERT frozen.

### Training step

For each task batch:

1. Set the active task ID.
2. Select the correct classification head.
3. Run BERT with the shared atom dictionary.
4. Calculate task loss.
5. Add configured coefficient regularization.
6. Backpropagate.
7. Update atoms, coefficients, and heads.
8. Log gradient norms and coefficient statistics.

### Log at minimum

Per task:

- Training loss
- Validation score
- Coefficient norm
- Number of coefficients near zero
- Top-used atoms

Global:

- Atom gradient norm
- Atom parameter norm
- Training time
- Memory usage if available

### Gradient debugging

During early development, assert:

```text
At least one atom tensor has a nonzero gradient.
The active task coefficient row has a nonzero gradient.
Inactive task coefficient rows have zero or no gradient.
The frozen BERT weights have no gradient.
```

### Done when

A two-task atom model trains without NaNs and both task losses decrease.

### Commit

```bash
git add .
git commit -m "Train shared atoms across multiple tasks"
```

---

## Chunk 17: Train the Full Five-Task Atom Model

### Goal

Run the first full H1 shared-dictionary experiment.

### Initial configuration

```yaml
experiment_name: shared_atoms
atom_count: 8
active_atoms_during_training: 8
sparsity_lambda: small
tasks:
  - sst2
  - mrpc
  - rte
  - qnli
  - qqp
```

### Why all eight atoms initially?

Hard top-k selection early in training can create dead atoms because initial coefficient order is random.

First:

- Train with all eight atoms.
- Encourage small coefficients.
- Inspect learned usage.
- Prune later.

### Run order

1. Development run with 500 examples per task.
2. Full run with up to 2,000 examples per task.
3. Repeat only after the pipeline and result files are verified.
4. Run additional seeds last.

### Save

```text
results/
└── shared_atoms/
    └── seed_17/
        ├── atoms.pt
        ├── coefficients.pt
        ├── heads.pt
        ├── metrics_by_task.json
        ├── parameter_counts.json
        └── training_history.json
```

### Done when

- All five validation tasks evaluate.
- Results are saved.
- Parameter counts are exact.
- Coefficient matrices can be inspected.
- No full duplicate BERT checkpoint is stored.

### Commit

```bash
git add .
git commit -m "Run full five-task shared atom experiment"
```

---

# Phase H — Sparsity and Matched Active Capacity

## Chunk 18: Evaluate Top-k Atom Pruning

### Goal

Compare the atom system with the rank-4 LoRA baseline under matched active capacity.

### Procedure

For every task and adapted module:

1. Read the learned coefficient vector.
2. Rank atoms by absolute coefficient value.
3. Keep the largest four.
4. Set the remaining coefficients to zero.
5. Evaluate without retraining.
6. Optionally fine-tune briefly with the mask fixed.
7. Record both scores.

### Report

For every task:

| Task | All 8 atoms | Top-4 before tuning | Top-4 after tuning |
|---|---:|---:|---:|
| SST-2 | ... | ... | ... |
| MRPC | ... | ... | ... |
| RTE | ... | ... | ... |
| QNLI | ... | ... | ... |
| QQP | ... | ... | ... |

### Important interpretation

- If top-4 retains quality, the dictionary supports sparse use.
- If quality collapses, eight atoms may be needed actively.
- If each task selects a disjoint top-4 set, sharing may be weak.
- If all tasks select the same top-4 with identical coefficients, task specificity may be weak.

### Done when

The top-4 model has a saved score and fixed mask for every task.

### Commit

```bash
git add .
git commit -m "Evaluate sparse top-k atom activation"
```

---

# Phase I — Parameter and Resource Accounting

## Chunk 19: Implement Exact Parameter Counting

### Goal

Prevent accidental accounting wins.

### File

```text
src/cgmoe_h1/utils/parameters.py
tests/test_parameter_counts.py
```

### Count separately

#### Base model

- Total parameters
- Trainable parameters

#### Independent LoRA

- All LoRA matrices across all target modules
- One adapter per task
- All task heads

#### Shared atoms

- Atom vectors across all target modules
- All task coefficient vectors
- All task heads
- Any regularizer parameters
- Any masks or gates with learned values

### Required outputs

```json
{
  "base_model_parameters": 4385920,
  "independent_lora": {
    "adapter_parameters": 81920,
    "head_parameters": 2570,
    "total_persistent_task_parameters": 84490
  },
  "shared_atoms": {
    "atom_parameters": 32768,
    "coefficient_parameters": 320,
    "head_parameters": 2570,
    "total_persistent_task_parameters": 35658
  }
}
```

Numbers above are illustrative only. Use actual measured values.

### Also record

- Checkpoint bytes on disk
- Training runtime
- Inference latency
- Peak RAM if practical
- Number of active atoms
- Estimated active adapter operations

### Done when

Automated tests verify parameter counts against small hand-calculated modules.

### Commit

```bash
git add .
git commit -m "Add exact adaptation parameter accounting"
```

---

# Phase J — Analysis and H1 Decision

## Chunk 20: Generate the Comparison Report

### Goal

Produce one reproducible report that answers H1.

### Script

```text
scripts/summarize_h1.py
```

### Required table

| Model | Mean score | Worst task gap | Persistent adaptation params | Relative storage | Active rank/atoms |
|---|---:|---:|---:|---:|---:|
| Independent LoRA | ... | 0 | ... | 100% | rank 4 |
| Shared atoms, all 8 | ... | ... | ... | ... | 8 |
| Shared atoms, top 4 | ... | ... | ... | ... | 4 |

### Per-task table

| Task | Independent LoRA | Shared atoms top-4 | Absolute gap | Relative retention |
|---|---:|---:|---:|---:|
| SST-2 | ... | ... | ... | ... |
| MRPC | ... | ... | ... | ... |
| RTE | ... | ... | ... | ... |
| QNLI | ... | ... | ... | ... |
| QQP | ... | ... | ... | ... |

### Coefficient analysis

Generate:

1. Heatmap or table of task-by-atom usage.
2. Top atoms per task.
3. Pairwise coefficient similarity between tasks.
4. Atom utilization count.
5. Number of dead atoms.
6. Number of task-exclusive atoms.
7. Number of atoms reused by two or more tasks.

### Interpretations

#### Strong support for H1

- Quality passes the threshold.
- Storage is materially smaller.
- Multiple tasks reuse atoms.
- Top-4 performance remains stable.
- Residual private capacity is not required.

#### Partial support

- Quality is good but storage reduction is weak.
- Storage is good but one or more tasks regress strongly.
- Sharing exists only among closely related tasks.
- Top-4 pruning causes moderate degradation.

#### Failure

- Atom count must approach total independent LoRA rank.
- Tasks partition atoms into private groups.
- Large task-specific residuals are required.
- Mean performance falls substantially.
- Worst-task regression is unacceptable.
- Parameter savings vanish after honest counting.

### Final H1 decision template

```text
Decision: Supported / Partially supported / Not supported

Quality:
- Independent LoRA mean:
- Shared atom mean:
- Relative retention:
- Worst task gap:

Storage:
- Independent adaptation parameters:
- Shared atom adaptation parameters:
- Relative storage:

Reuse:
- Atoms used by multiple tasks:
- Task-exclusive atoms:
- Dead atoms:

Conclusion:
- What the result supports
- What it does not support
- Main observed failure mode
- Next experiment
```

### Done when

A fresh clone can regenerate the report from saved result files.

### Commit

```bash
git add .
git commit -m "Generate H1 comparison and research decision"
```

---

# Phase K — Stronger Follow-Up Tests

These are not required for the first H1 result. Complete them only after the basic comparison is trustworthy.

## Chunk 21: Frozen-Atom Transfer Test

### Goal

Test whether a new task can reuse an existing dictionary.

### Procedure

1. Train atoms using four tasks.
2. Freeze all atom parameters.
3. Introduce the fifth task.
4. Train only:
   - New task coefficients
   - New task classification head
5. Compare against:
   - A fresh rank-4 LoRA for the fifth task
   - Head-only tuning
   - Random frozen atoms plus coefficients

### Strong result

```text
Frozen atoms + new coefficients >= 95% of fresh-LoRA quality
```

while using substantially fewer new task-specific parameters.

### What this adds

The joint five-task experiment proves compression of known tasks.

The frozen-atom transfer test provides stronger evidence that the dictionary contains reusable structure rather than merely co-adapting to the training tasks.

---

## Chunk 22: Scaling Curve

### Goal

Test whether storage grows more slowly as task count increases.

### Runs

Compare systems with:

```text
1 task
2 tasks
3 tasks
4 tasks
5 tasks
```

### Plot or table

For each task count:

- Mean quality
- Worst-task score
- Independent LoRA storage
- Shared atom storage
- Relative storage
- Active capacity

### Desired pattern

```text
Capability count grows.
Independent storage grows roughly linearly.
Shared dictionary storage grows slowly.
Quality remains stable.
```

This scaling curve is more important than a single five-task number.

---

## Chunk 23: Atom Count Ablation

### Goal

Find the minimum dictionary size that retains quality.

### Test

```text
Atom counts: 2, 4, 6, 8, 12, 16
```

Keep all other settings fixed.

### Questions

- At what dictionary size does quality saturate?
- Does storage remain below the independent baseline?
- Does reuse increase or decrease with dictionary size?
- Do extra atoms become task-private?

---

## Chunk 24: Rank and Active-Capacity Ablation

### Goal

Ensure results are not an artifact of one rank choice.

### Independent LoRA

```text
Ranks: 1, 2, 4, 8
```

### Shared atoms

```text
Active atoms: 1, 2, 4, 8
```

Compare quality and active compute.

---

## Chunk 25: Random and Retrieval Controls

### Goal

Rule out easy alternative explanations.

### Controls

1. Random frozen atom dictionary plus trained coefficients
2. Average of independent LoRAs
3. Nearest task adapter retrieval
4. One shared multitask LoRA
5. Shared atoms with shuffled task labels
6. Shared atoms without sparsity regularization

These controls distinguish reusable composition from:

- Random projection capacity
- Simple multitask learning
- Adapter averaging
- Memorized task lookup
- Overly strong regularization

---

# 5. Suggested Command-Line Interface

By the end of the project, aim for commands like:

```bash
python -m cgmoe_h1.cli inspect-model \
  --config configs/baseline.yaml
```

```bash
python -m cgmoe_h1.cli train-independent \
  --config configs/baseline.yaml \
  --task sst2
```

```bash
python -m cgmoe_h1.cli train-atoms \
  --config configs/atoms.yaml
```

```bash
python -m cgmoe_h1.cli evaluate-top-k \
  --run-dir results/shared_atoms/seed_17 \
  --k 4
```

```bash
python -m cgmoe_h1.cli summarize \
  --baseline-dir results/independent_lora/seed_17 \
  --atoms-dir results/shared_atoms/seed_17
```

A CLI is not required at the beginning. Add it only after the scripts stabilize.

---

# 6. Laptop-Safe Training Rules

For the XPS 13 with 16 GB RAM:

- Use CPU first.
- Keep sequence length at 128.
- Use batch size 8 or 4 if memory pressure appears.
- Start with 500 examples per task.
- Use `num_workers=0` on Windows during early debugging.
- Avoid keeping all tokenized datasets duplicated in memory.
- Save adapter weights, not full BERT copies.
- Run one seed while developing.
- Disable expensive logging until correctness is established.
- Do not start with hyperparameter sweeps.
- Stop immediately on NaNs.
- Track elapsed time per epoch before scaling up.
- Close memory-heavy applications during full runs.

If training is still too slow:

1. Reduce training examples.
2. Reduce epochs.
3. Use two tasks.
4. Shorten sequence length to 64 for plumbing tests.
5. Profile before redesigning.

---

# 7. Debugging Checklist

## Training loss does not decrease

```text
[ ] Correct labels
[ ] Correct task head selected
[ ] Trainable parameters included in optimizer
[ ] model.train() called
[ ] Nonzero gradients
[ ] Learning rate not too low
[ ] Adapter branch not permanently zero
[ ] Inputs and attention masks valid
```

## Base BERT accidentally trains

```text
[ ] All base parameters have requires_grad=False
[ ] Optimizer receives only explicitly trainable parameters
[ ] No gradients appear on frozen weights
```

## Atom tasks interfere badly

```text
[ ] Task sampling is balanced
[ ] Each task uses the correct head
[ ] Each task selects its own coefficient row
[ ] Coefficient regularization is not too strong
[ ] Atom count is not too small
[ ] Learning rate is stable
```

## No atom sharing appears

```text
[ ] Dictionary is smaller than total independent rank
[ ] Tasks are trained jointly
[ ] Coefficients are not accidentally isolated by module copies
[ ] Atom tensors are truly shared
[ ] Regularization does not force task partitioning
```

## All tasks use identical coefficients

```text
[ ] Coefficients are actually trainable
[ ] Task IDs are reaching the atom layer
[ ] Task heads are not absorbing all adaptation
[ ] Coefficient initialization is not tied
[ ] Model has enough task-specific signal
```

## Memory usage is too high

```text
[ ] Batch size reduced
[ ] Sequence length reduced
[ ] No full model copy per task
[ ] Datasets not duplicated
[ ] Gradients cleared with set_to_none=True
[ ] Unneeded checkpoints removed from memory
```

---

# 8. Minimum Test Suite

Before trusting results, the following tests should pass:

```text
test_config_loads
test_seed_reproducibility
test_dataset_schema_for_each_task
test_classifier_output_shape
test_base_model_is_frozen
test_lora_initial_output_matches_base
test_lora_gradient_flow
test_lora_parameter_count
test_atom_initial_output_near_base
test_atom_shared_parameters
test_atom_task_specific_coefficients
test_atom_gradient_flow
test_inactive_coefficients_do_not_update
test_injection_replaces_expected_modules
test_non_target_modules_unchanged
test_multitask_sampler_is_balanced
test_result_json_round_trip
test_exact_parameter_accounting
test_checkpoint_round_trip
```

---

# 9. Milestone Summary

## Milestone 1: Environment Works

```text
Chunks 0–2
```

Result:

- Reproducible project
- Written experiment contract
- Working model and datasets

## Milestone 2: Base Pipeline Works

```text
Chunks 3–7
```

Result:

- Frozen classifier
- Deterministic data
- Metrics
- Head-only training

## Milestone 3: Independent LoRA Baseline Works

```text
Chunks 8–11
```

Result:

- Tested LoRA implementation
- Five independent task baselines
- Exact baseline storage

## Milestone 4: Shared Atom Model Works

```text
Chunks 12–17
```

Result:

- Tested atom implementation
- Shared dictionary injected into BERT
- Balanced multitask training
- Five-task atom checkpoint

## Milestone 5: H1 Is Evaluated

```text
Chunks 18–20
```

Result:

- Top-4 matched-capacity evaluation
- Honest parameter accounting
- Final H1 report and decision

## Milestone 6: Stronger Evidence

```text
Chunks 21–25
```

Result:

- Held-out reuse
- Scaling curves
- Ablations
- Control baselines

---

# 10. Definition of Done for the First H1 Experiment

The first H1 experiment is complete when the repository contains:

```text
[ ] Frozen base-model classifier
[ ] Deterministic five-task data pipeline
[ ] Independent rank-4 LoRA implementation
[ ] Independent LoRA results for every task
[ ] Shared rank-1 atom implementation
[ ] Shared atom results for every task
[ ] Top-4 atom evaluation
[ ] Exact parameter counts
[ ] Per-task quality comparison
[ ] Coefficient usage analysis
[ ] Reproducible configurations
[ ] Saved random seed
[ ] Final Supported / Partial / Failed decision
```

The experiment is not complete merely because the code trains.

It is complete when the result answers:

> Did the shared atom dictionary preserve enough task quality to justify its reduction in persistent adaptation storage?

---

# 11. What Comes After H1

Do not begin these until the reusable basis shows evidence of working.

## H2: Conditional Expert Generation

Replace direct task coefficient lookup:

```text
task_id -> stored coefficients
```

with:

```text
task context -> hypernetwork -> generated coefficients
```

Test held-out and compositional generalization.

## H3: Bounded Consolidation

Train temporary task adaptations and attempt to merge them into a fixed or slowly growing dictionary without catastrophic regression.

## Plugin Layer

Package proven capability representations into installable, removable, and reloadable units.

---

# 12. Research Discipline

Keep these rules visible throughout the project:

1. Change one major variable at a time.
2. Establish a simple baseline before adding complexity.
3. Save configuration with every result.
4. Count all learned storage.
5. Match active compute when claiming superiority.
6. Report average and worst-task behavior.
7. Treat residual growth as model growth.
8. Do not call task partitioning “sharing.”
9. Do not call joint memorization “held-out generation.”
10. A negative result is useful if the experiment is clean.

The first destination is not a giant model. It is one trustworthy table.
