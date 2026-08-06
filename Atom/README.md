# CGMoE H1 experiment

This repository tests whether five task-specific BERT adaptations can share a compact dictionary
of learned rank-1 operators while retaining the quality of independent rank-4 LoRA adapters. The
frozen base is [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny); the tasks are
SST-2, MRPC, RTE, QNLI, and QQP from GLUE.

The preregistered three-seed H1 comparison **passed**. Shared atoms with four active atoms retained
98.54% of independent-LoRA mean primary-score quality, kept the worst task gap to 0.0267, and used 9,642
persistent adaptation parameters instead of 21,770 (44.29%). All three locked thresholds passed.

That result supports compact storage for jointly trained, known tasks. It does **not** establish
held-out transfer: the seed-17 frozen-dictionary QQP follow-up retained only 90.85% of fresh-LoRA
quality, below its 95% criterion, and scored below both head-only and random-frozen-atom controls.
See [the full result and caveats](docs/h1_result.md) and the
[locked core H1 contract](docs/h1_experiment_spec.md).

The follow-on three-seed atom validation stage is now **complete**, and it separates the two
claims. Shared adaptation compresses: one shared multitask LoRA bank reached 0.6911 mean primary
score with 9,482 persistent parameters, beating independent LoRA's 0.6852 at 21,770. But the
compression is **not atom-specific** — rank-1 atom composition lost to shared multitask LoRA at all
four matched capacities and all three seeds — and the learned dictionary does **not** span held-out
tasks: an oracle least-squares projection of a held-out LoRA update onto the frozen learned basis
explains only 2.35% of its energy. See [the atom validation result](docs/atom_validation_result.md)
and its [locked specification](docs/atom_validation_spec.md).

## Result at a glance

| Experiment | Outcome | Main result |
|---|---|---|
| Core H1, seeds 17/29/43 | **PASS** | 0.6752 shared top-4 mean primary score vs 0.6852 LoRA; 44.29% relative storage |
| Frozen-atom QQP transfer, seed 17 | **FAIL** | 0.6091 vs 0.6704 fresh-LoRA primary score; 90.85% retention with 290 marginal new parameters |
| Task-count scaling, seed 17 | Exploratory | Shared storage falls below LoRA at three tasks and reaches 44.3% at five |
| Atom/rank/top-k ablations, seed 17 | Exploratory | N=2 was the smallest tested atom count within 0.005 of the best; rank 8 and k=8 had the best observed means |
| Chunk-25 controls, seed 17 | Complete diagnostic | Shared multitask LoRA had the best control mean, 0.685615 |
| Validation A: matched frontier, seeds 17/29/43 | **FAIL** | Atoms lost to shared multitask LoRA at every matched capacity (-0.009 to -0.012) |
| Validation B: crossed transfer, seeds 17/29/43 | Primary **PASS**, control-aware **FAIL** | 95.83% retention, but only +0.000016 over a random frozen dictionary |
| Validation C: oracle span projection, seeds 17/29/43 | **FAIL** | Learned span explains 2.35% of held-out update energy (random: 0.05%) |

The single-seed follow-ups are diagnostic and do not modify the preregistered core decision.
Control means were 0.655820 for random frozen atoms, 0.639124 for averaged LoRAs, 0.634651
for nearest-other-task retrieval, 0.685615 for shared multitask LoRA, 0.531454 with shuffled
labels, and 0.665958 without coefficient sparsity.

## What is implemented

The repository now contains the full H1 path: typed contract validation, deterministic GLUE
sampling and provenance, frozen BERT task heads, LoRA and shared-atom injection, single-task and
balanced multitask training, task-correct metrics, top-k inference, exact parameter/operation
accounting, compact checkpoint reloads, three-seed reporting, transfer/scaling studies, ablations,
and the six resumable control workflows. Unit tests use synthetic models and mock expensive
orchestration so they run offline; the commands below run the live models.

## Set up

Python 3.11 or newer is supported. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

To resolve dependencies from `pyproject.toml` instead of the lock file:

```powershell
python -m pip install -e ".[dev]"
```

The first live run downloads the model and GLUE data from Hugging Face; later runs reuse their
local caches.

## Reproduce the experiments

Run the locked core experiment, generate its decision report, then run the completed follow-ups:

```powershell
python scripts/run_h1.py
python scripts/summarize_h1.py
python scripts/run_transfer_scaling.py
python scripts/run_ablations.py
```

Every expensive runner is resumable by default. Use `--force` only when a deliberate rerun is
needed. A subset of core seeds is preliminary, for example
`python scripts/run_h1.py --seed 17`.

The completed six-control suite can be regenerated or resumed with:

```powershell
python scripts/run_controls.py
```

Its aggregate JSON and Markdown reports are
`results/controls/seed_17/{control_results.json,control_report.md}`.

Useful targeted commands include:

```powershell
python scripts/train_independent_lora.py --seed 17
python scripts/train_shared_atoms.py --seed 17
python scripts/evaluate_top_k.py --run-dir results/shared_atoms/seed_17 --seed 17 --k 4
python -m pytest
```

## Artifacts and compact checkpoints

Generated results are intentionally ignored by Git and can be regenerated from the commands
above. The main local artifacts are:

- `results/h1_report/{h1_summary.json,h1_report.md}`: preregistered three-seed decision.
- `results/independent_lora/seed_<seed>/`: one compact LoRA adapter and head per task.
- `results/shared_atoms/seed_<seed>/`: shared atoms, coefficients, heads, diagnostics, and masks.
- `results/followups/`: frozen transfer, task-prefix scaling, and their reports.
- `results/followup_ablations/`: atom-count, LoRA-rank, and active-capacity runs and report.
- `results/controls/`: completed chunk-25 controls and aggregate report.

Checkpoints never duplicate the frozen BERT base. Independent runs save `adapter.pt` and
`heads.pt`; shared runs save `atoms.pt`, `coefficients.pt`, and `heads.pt`. At each core seed the
five independent checkpoints occupy 118,070 serialized bytes, while the shared checkpoint occupies
50,229 bytes. Metrics records include the resolved configuration, data provenance, parameter
accounting, environment, runtime, and component paths needed to audit a run.

## Verify the environment

```powershell
python -m cgmoe_h1
python scripts/check_environment.py
python scripts/check_model.py
python -m pytest
```

`check_environment.py` loads two SST-2 rows. `check_model.py` performs an inference-only BERT
forward pass and reports shapes and the exact 4,385,920-parameter base-model count.

## Layout

- `configs/`: locked independent-LoRA and shared-atom configurations.
- `docs/`: preregistration, result interpretation, and caveats.
- `scripts/`: training, evaluation, reporting, ablation, and control entry points.
- `src/cgmoe_h1/`: models, data, training, accounting, reporting, and follow-up orchestration.
- `tests/`: offline unit and orchestration tests; expensive runners are mocked in follow-up tests.
- `results/`: ignored, resumable generated artifacts; only `.gitkeep` is tracked.
