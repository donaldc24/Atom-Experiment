# E1 — Atom Factorization Battery (laptop / CPU edition)

Tests **H6** of the Atoms preregistration: are atoms independently meaningful
primitives, or co-adapted fragments? This is a hard gate — if atoms do not
factorize, H1's marginal-cost curve and H3's geometry are both uninterpretable.

**[`../GATES.md`](../GATES.md) is the experimental contract** — gates T1–T8, the frozen
§8 thresholds, the verdict rules, and every amendment that has ever occurred, in one
place. Read it before interpreting any number.

[DECISIONS.md](DECISIONS.md) is the long form: every deviation D1–D45 with its reason
and its effect on interpretation.

### Gates at a glance

| | asserts | kind | status |
|---|---|---|---|
| **T1** | oracle ceiling — a composing solution is reachable (**amended**, D18) | battery | pass |
| **T2** | determinism — two identical runs, byte-identical predictions | battery | pass |
| **T3** | no leakage — A4 stays at chance (≤ 0.02) | both | pass |
| **T4** | primitive independence — no primitive is a composition of ≤ 2 others | fast | pass |
| **T5** | split validity — sizes, equivalence classes, coverage | fast | pass |
| **T6** | metric reproducibility — `analyze` twice, identical output | battery | pass |
| **T7** | ablation sanity — all-ablated → chance, none-ablated → logged accuracy | fast | pass |
| **T8** | memory ceiling — peak RSS < 4 GB | fast | pass |

T1 is a hard stop: `run_all` aborts the batch if A0 misses it, because no other arm's
failure is interpretable without a demonstrated ceiling. It has fired once for real,
catching D44 at run 1 of 54.

## Layout

```
e1/primitives.py   8 unary transforms + the T4 independence check
e1/data.py         extensional equivalence classes, split construction, datasets
e1/make_split.py   generates and freezes splits/pairs_split.json (run once)
e1/model.py        Encoder / AtomBank / Composer / Decoder
e1/train.py        the 5 arms; writes artifacts, computes NO metrics
e1/evaluate.py     per-example predictions + raw ablation/alignment matrices
e1/analyze.py      M1-M6 from saved artifacts only -> metrics.json
e1/aggregate.py    summary.csv / summary.md / plots / verdict
e1/run_all.py      sequential 5x5 batch driver
tests/test_fast.py  T4, T5, T7, T8 + architecture invariants  (CI, ~1 min)
tests/test_gates.py T1, T2, T3, T6                             (need training runs)
```

## Generations (D40)

Both designs live in this one working tree. **Switching generations never requires
switching commits** — that is the point, because old results stop being reproducible
the moment the code moves on.

| | v1 | v2 |
|---|---|---|
| slot-3 primitive | `sort_asc` | `index_shift`, `x_i → (x_i + i) mod 10` |
| distinct pair functions | 39 / 64 | **42 / 64** |
| split seeds | 1234 | 1234, 5678, 9012 |
| optimisation seeds | 0–4 | 0–2 |
| **runs per arm** | 5 (5 × 1) | **9 (3 × 3)** |
| runs land in | `runs/v1/` | `runs/v2/` |
| results land in | `results/` | `results/v2/` |

v2 trades seeds for splits (D42): *more* runs per arm than v1, and the spread covers
split variance, which v1's five-seeds-one-split design could not see at all. The cost
is that per-split subgroup claims rest on n = 3 and must be labelled — the headline is
the pooled n = 9. If any arm's pooled `acc_unseen` std exceeds 0.10 it goes to 5 seeds
before being reported; that escalation is pre-registered, not discretionary.

**v1 is immutable.** Its split sha256 is pinned by a test and recorded in 30 committed
runs; `make_split --generation v1 --force` refuses. Generations are never pooled —
they measure different task families, so a combined mean describes neither.

Use v2 for new work. v1 exists to be re-run and reproduced, not extended.

### The archive (D41)

The battery as run on `Perro` (Intel) lives in `runs/archive_perro_v1/` and
`results/archive_perro_v1/`. `aggregate`, `backfill` and `status` **skip `archive*`
subtrees by default**, so a frozen battery is never pooled with new runs and
`backfill --reanalyze` cannot rewrite it. Read it by pointing at it:

```bash
python -m e1.aggregate --generation v1 --runs runs/archive_perro_v1
```

Aggregation also **refuses to span two hostnames** (D39) — determinism is a
within-platform guarantee, so a table mixing machines is not a comparison.

## Environment

CPU-only, and the pins matter — `numpy==2.2.4` has no cp314 wheels, so Python must be
≤ 3.13 (3.11 in use, see D39). The venv lives at the REPO root (one level above this
project folder):

```bash
py -3.11 -m venv .venv                    # from the repo root
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
.venv/Scripts/python -m pip install numpy==2.2.4 pandas==2.2.3 matplotlib==3.10.1 psutil==7.0.0
```

## Running

Every command below runs from `AtomV1/` (this project's root, where `e1/`, `runs/`
and `splits/` live), with the repo-root venv on PATH or invoked as
`../.venv/Scripts/python`.

```bash
python tests/test_fast.py        # must be green before anything else

python -m e1.make_split --generation v2   # frozen and committed; --force to regenerate

# Run in THIS order. A3 is the informative arm and D12 predicts A1 fails, so getting
# A3 early beats waiting for a single full block.
python -m e1.run_all --generation v2 --arms A0        # T1 gate: oracle must clear the bar
python -m e1.run_all --generation v2 --arms A3        # the informative arm
python -m e1.run_all --generation v2 --arms A1 A2 A4  # comparison arms, overnight
python -m e1.aggregate --generation v2                # results/v2/summary.md + verdict

# Reproduce the committed v1 battery instead:
python -m e1.run_all --generation v1
python -m e1.aggregate --generation v1                # -> results/

python tests/test_gates.py --t1 --t3
python tests/test_gates.py --t2                        # determinism, reduced scale
python tests/test_gates.py --t6 runs/v2/A0_0_s1234_<sha>
```

`run_all` sweeps every split seed the generation froze (v2: 3), with split seed as the
**outer** loop — a batch that dies partway still leaves complete arm × seed blocks for
the splits it finished, rather than a ragged fragment of every split. It stops the
batch if A0 misses the T1 gate; no other arm's failure would be interpretable.

Runs are discovered **recursively**, so `aggregate`, `backfill` and `status` see both
the nested layout and the flat pre-D40 runs.

## Arms

| Arm | What differs | Reads as |
|---|---|---|
| A0 | routing forced to ground truth | ceiling / harness check |
| A1 | joint, fixed co-occurrence order | the baseline pathology |
| A2 | + atom dropout 0.15, reshuffled order | do cheap countermeasures suffice |
| A3 | sequential growth, then frozen library | **the informative arm** |
| A4 | A2 with permuted training targets | leakage detector; must sit at chance |

The point of five arms is that a single-arm negative result is unusable: A3 passing
while A1/A2 fail is FAIL(optimizer) — H6 survives and the *training method* is at
fault. Everything failing including A3 is FAIL(representational). Do not drop arms.

A third outcome, **FAIL(architectural)**, was added after A0's diagnostics — see
DECISIONS.md D12/D16. If every failing arm shows correct closed-map atoms
(`M7_acc_teacher_forced` high) alongside failed composition, the composition
operator is at fault, not factorization, and it must not be reported as "H6 refuted".

## Reproducibility

`train.py` produces artifacts; `analyze.py` produces every headline number from those
artifacts alone. A training-loop bug cannot silently generate favourable metrics, and
every run directory carries `config.json`, `env.json`, `split_ref.json`,
`param_counts.json`, per-example `predictions_*.jsonl` and `SHA256SUMS`.

CPU only: no CUDA paths, 4 pinned threads, `torch.use_deterministic_algorithms(True)`,
`final.pt` only. Peak RSS is asserted below 4 GB and recorded in `env.json`.

**Thread count is a determinism parameter, not a performance knob** — it changes
reduction order. On the Ryzen 9 6900HX at 4 threads, steady-state cost is
**14.5 s/epoch**, measured as the marginal difference between an 8-epoch and a
2-epoch run (118.1 s vs 31.1 s) so that ~2 s of startup is excluded. Comparing thread
counts on 2-epoch runs: 8 threads buys ~5%, 16 is ~23% *worse* (SMT contention). It
stays at **4**. Sweep once with `--threads`, freeze one value, never vary it inside a
batch.

Each run also carries **~20 s** of non-training wall clock — process start, data
generation, `emit_artifacts`, checksums — which `train_seconds` excludes.

**This machine is at parity with Perro, not faster.** Perro's archived A1 runs
averaged 14.69 s/epoch against Perrito's 14.5. Budget accordingly; note that
`DECISIONS.md` D9's "38.9 s/epoch" does not match the runs that were actually made
(A0's 80 epochs would then be 52 min, and it took 23).

**Determinism is a within-platform guarantee.** The E1 battery ran on Intel; work
continues on AMD. No comparison may span the two — see D39.

Per H5, composer and library parameter counts are always reported as separate line
items — a composer quietly absorbing the atoms' work is invisible under a combined
total.
