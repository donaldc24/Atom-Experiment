# Atom V2 Harness — H1 (Atom Factorization)

Implementation of the experiment specified in [../H1Experiments.md](../H1Experiments.md)
and [../SplitMath.md](../SplitMath.md). Decisions the spec left open are
registered in [REGISTERED.md](REGISTERED.md) and frozen in
[atomv2/registered.py](atomv2/registered.py) (code is authoritative).

Design contract, inherited from V1's standing rules:

1. **Training computes no headline metrics.** `train.py` writes raw artifacts;
   `analyze.py` derives every reported number from disk. Any metric added later
   applies retroactively to completed runs without retraining.
2. **The split is derived, never transcribed.** `split.py` rebuilds all 72
   canonical triples from the op definitions, applies the registered held-out
   choices, and mechanically validates every enumerator duty on every load.
   The frozen file `splits/split_v2.json` is hash-pinned into each run and
   re-verified against the algebra at run start.
3. **One master seed per run** controls data, init, shuffling, Gumbel noise
   and probes through named independent streams. Two runs with the same seed
   are bit-identical (tested).
4. **Oracle machinery is quarantined** in `atomv2/oracle.py`, imported only on
   the E0 `A0-oracle` path. Free-arm loss is task CE + rent, nothing else.
5. **Probes read, never write.** The panel loads checkpoints with
   `requires_grad_(False)` and everything downstream reads artifacts only.
6. **Control/data separation (E0 amendment R8).** The 384-dim atom state is a
   digit-only canonical code. The memoryless composer receives only the active
   opaque surface token and micro-step 0..2; atoms never see task or partner
   tokens. See [REGISTERED.md](REGISTERED.md) for the failed-run evidence and
   amendment boundary.

## Setup

```
# from the repo root; CPU-only torch, do not install a CUDA wheel
python -m venv .venv
.venv/Scripts/pip install -r AtomV2/Harness/requirements.txt
```

The harness reuses the repo-root `.venv` (torch 2.9.0+cpu, numpy 2.2.4,
Python 3.11).

## Reproduction, start to finish

All commands run from `AtomV2/Harness/`.

```
# 0. Verify the rig: algebra, split duties, model invariants, determinism
python -m pytest tests/ -q

# 1. (Already done, committed) derive + freeze the split; re-runs verify only
python -m atomv2.split

# 2. Smoke the full pipeline end to end (minutes; writes runs/smoke_e0)
python -m atomv2.run_e0 --smoke

# 3. Experiment 0 — calibration (6 runs: A0-oracle + A0-free x seeds 0,1,2)
python -m atomv2.run_e0
#    -> runs/e0/<arm>_s<seed>_<gitsha>/..., results/e0/e0_verdict.json
#    The driver aborts if an oracle run misses the L1 ceiling (circuit breaker)
#    and audits the instruments against the oracle's ground truth.

# 4. Experiment 1 — the lambda_use sweep (12 runs: A1..A4 x seeds 0,1,2)
#    REFUSES to start unless results/e0/e0_verdict.json passed.
python -m atomv2.run_e1
#    -> runs/e1/..., results/e1/summary.md

# Interrupted batches: re-run the same command. Complete runs are skipped;
# a run that trained but died mid-panel resumes at the panel stage.
```

Single runs, if needed:

```
python -m atomv2.train --arm A2 --seed 1          # training + eval artifacts
python -m atomv2.panel --run-dir <run_dir>        # panel on 5k/10k/15k+final
python -m atomv2.analyze --run-dir <run_dir>      # metrics.json from artifacts
python -m atomv2.aggregate --experiment e1        # tables + summary.md
```

## What a run writes (everything needed to re-derive results)

```
runs/<exp>/<arm>_s<seed>_<gitsha>/
  config.json           full resolved config (every default explicit)
  env.json              versions, hostname, threads, git sha, timing, peak RSS
  split_ref.json        frozen split path + sha256
  data_manifest.json    per-task content hashes of the generated data
  init_calibration.json registered rent-vs-task-loss magnitude at init
  param_counts.json     encoder/decoder/composer/atoms/keys - composer and
                        library are separate line items, never summed
  train_log.jsonl       per-50-step losses (all terms), lr, tau, grad norm
  evals/stepNNNNNN.json every 1k steps: per-cell hard+soft accuracy for
                        seen_heldout and each unseen level (never averaged
                        across levels), trajectory closed-map error with
                        target-anchored companion + prefix-visit histogram,
                        census, pass rate, steps-per-token, soft-hard gap.
                        This cadence IS the time-resolved closed-map curve.
  traces/stepNNNNNN.npz per-example predictions, correctness, routing choices,
                        per-example closed-map minima (checkpoint cadence)
  checkpoints/          stepNNNNNN.pt every 2k and at 5k/10k/15k + final.pt
                        (weights + config)
  panel/{stepNNNNNN,final}/
    census.json, ablation.json + ablation.npz (raw damage/usage matrices +
    raw tuples incl. routing entropy during ablation; compensation probe on
    final only), standalone.json, closed_map_atom.json (+ npz tensors),
    decodability.json (probes + h0 floors + shuffled-label floors),
    transfer.json + transfer.npz (8x8 task matrix + program transplant),
    full_acc.json
  metrics.json          headline numbers, derived from the above only
  SHA256SUMS            checksums over every file
results/<exp>/          per_run.json, summary.json, summary.md,
                        e0_verdict.json (+ instrument audit) for E0
```

## Layout

```
atomv2/registered.py   every registered constant (the preregistration in code)
atomv2/ops.py          sub-op functions + (pi,a,b) triple algebra, self-verifying
atomv2/split.py        split enumerator + validator + freezer (python -m atomv2.split)
atomv2/data.py         seeded dataset generation, opaque tokens, P3 oversampling
atomv2/model.py        encoder / 16 atoms + keys / memoryless composer / decoder
atomv2/oracle.py       QUARANTINED E0 oracle (forced routing + state supervision)
atomv2/train.py        the loop; artifacts only, no headline metrics
atomv2/evaluate.py     periodic eval: per-cell accs, trajectory closed-map, census
atomv2/panel.py        checkpoint panel: ablation, standalone, decodability,
                       atom-centric closed map, transfer/transplant
atomv2/analyze.py      artifacts -> metrics.json
atomv2/aggregate.py    runs -> tables, E0 pattern verdict + instrument audit
atomv2/run_e0.py       E0 batch driver (circuit breaker)
atomv2/run_e1.py       E1 batch driver (gated on the E0 verdict)
tests/                 54 tests incl. the SplitMath.md diff duty
splits/split_v2.json   the frozen, derived, hash-pinned split
```

## Provenance rules

- Runs refuse to start from a dirty git source tree (`--allow-dirty` records
  a complete source-snapshot hash, including untracked files, and adds its
  short form to the run ID). Output dirs (`AtomV2/runs`, `AtomV2/results`) are
  exempt.
- Aggregation refuses to mix hostnames (determinism is a within-platform
  guarantee), source snapshots, protocol revisions, duplicate arm/seed keys,
  or smoke with real runs.
- The frozen split refuses regeneration if the derivation would change;
  `load_verified()` re-derives on every run start so a run can never train
  against a drifted split file.
