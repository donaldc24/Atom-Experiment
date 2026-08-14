# The experimental contract — gates, thresholds and verdict rules

Every number that decides anything in this project, in one place. Values are frozen
in code, not in this file: `e1/config.py::THRESHOLDS`, `E1B_THRESHOLDS`, and the
constants at the top of `e1/aggregate.py`. If this document and the code disagree,
**the code is authoritative** and this document is the bug.

Nothing here may be changed after seeing results. Ambiguity is resolved by iterating
on the *training procedure*, never on the thresholds. Every deviation that has ever
occurred is recorded in [`e1/DECISIONS.md`](e1/DECISIONS.md) with its reason and its
effect on interpretation.

---

## 1. Acceptance gates T1–T8

`fast` gates run in CI on every commit (~1 min). `battery` gates need training runs.

| gate | asserts | where | kind | status |
|---|---|---|---|---|
| **T1** | **Oracle ceiling.** A composing solution is *reachable* in this architecture | `tests/test_gates.py --t1`, enforced live by `run_all` | battery | **amended — see §4** |
| **T2** | **Determinism.** Two identical runs produce byte-identical `predictions_unseen.jsonl` | `tests/test_gates.py --t2` | battery | pass |
| **T3** | **No leakage.** A4 (shuffled targets) stays at chance: `acc_seen` and `acc_unseen` ≤ **0.02** | static half in `test_fast.py`, live half `--t3` | both | pass |
| **T4** | **Primitive independence.** No primitive equals a composition of ≤ 2 *other* primitives, on 10,000 random inputs. `identity` exempt as a target (D3) | `test_fast.py` | fast | pass |
| **T5** | **Split validity.** Sizes are exactly 40/24; equivalence classes never straddle train/held-out; coverage constraints hold (D2, D14) | `test_fast.py` | fast | pass |
| **T6** | **Metric reproducibility.** `analyze.py` twice on one run directory yields identical `metrics.json` | `tests/test_gates.py --t6 <run>` | battery | pass |
| **T7** | **Ablation sanity.** Ablating all atoms drops to chance; ablating none reproduces the logged accuracy | `test_fast.py` | fast | pass |
| **T8** | **Memory ceiling.** Peak RSS stays below **4 GB** (`cfg.rss_fail_gb`), recorded in `env.json` | `test_fast.py` + asserted every run | fast | pass |

**T1 is a hard stop.** `run_all` aborts the batch if any A0 run misses it — no other
arm's failure is interpretable without a demonstrated ceiling. It has fired for real
once, catching D44 at run 1 of 54.

**T3 is the complement of T1.** A0 shows the task is solvable; A4 shows a score cannot
arise without a learnable relationship. Neither bound alone licenses reading the middle
arms.

---

## 2. Per-arm thresholds (spec §8) — frozen before the first run

PASS requires **all six**. A FAIL on any one is a FAIL. Judged on the mean across a
generation's runs.

| metric | PASS | FAIL | what it tests |
|---|---|---|---|
| `M1_acc_unseen` | ≥ 0.85 | ≤ 0.50 | H6(a) recombination into unseen pairings |
| `M1_gap` | ≤ 0.05 | ≥ 0.20 | seen-vs-unseen gap; memorisation vs composition |
| `M2_cv` | ≤ 0.35 | ≥ 0.75 | H6(b) ablation-degradation variance — the co-adaptation metric |
| `M3_align` | ≥ 0.85 | ≤ 0.50 | H6(c) standalone atom semantics |
| `M3_purity` | ≥ 0.50 | ≤ 0.20 | one atom per primitive, not many-to-one |
| `M5_dead` | ≤ 1 | ≥ 3 | library collapse |

Anything between PASS and FAIL is **AMBIGUOUS**.

**Which arms decide the verdict:** only **A1, A2, A3**. A0 is a ceiling, A4 a control,
A3b a diagnostic (D23) — all three are excluded from the program verdict.

### `M3_align` carries a standing caveat

It is pre-registered and still decides the §8 verdict, but it has now failed in three
independent ways and must never be read alone:

| failure mode | evidence |
|---|---|
| seed-noisy | v1 §2.6 — ~5× noisier than closed-map error; read **above** its 0.85 PASS threshold on 2 of 5 seeds whose atoms were inert |
| clean-on-dead | v1 §4 — perfect permutation on a system composing at 0.004 |
| split-noisy | v2 D45 — A1 varies 0.228–0.485 across splits while coverage is 1/8 in all 45 runs |

**`M3_closed_map_error_matched` + `M3_closed_map_coverage` adjudicate when probes
disagree** (D21). They are decoder-free and depth-free. The bare closed-map error is
*gameable* on its own — an untrained library scores 0.094, better than a converged
oracle — so it is only ever read **with coverage** (D24).

---

## 3. Program verdict rules

Decided mechanically in `e1/aggregate.py`, on values fixed in advance.

| verdict | condition |
|---|---|
| **PASS** | A3 passes, and A1 or A2 passes |
| **FAIL(optimizer)** | A3 passes while the joint arms fail |
| **FAIL(architectural)** (D16) | every failing non-A4 arm has teacher-forced ≥ **0.85**, `acc_unseen` ≤ **0.50**, `M3_align` ≥ **0.85** — atoms are correct, the *composition operator* is at fault |
| **FAIL(training-signal)** (D22) | oracle teacher-forced ≥ **0.99** and oracle closed-map ≤ **0.10**, while every failing arm has closed-map ≥ **0.30** |
| **FAIL(representational)** | all arms fail, drift does not explain it, and the oracle showed no reachable solution |

FAIL(architectural) and FAIL(training-signal) are mutually exclusive by construction.
FAIL(training-signal) was pre-registered (D22) **before A1, A2 or A4 had produced a
single run**, and is the outcome both generations returned.

---

## 4. Amendments — every threshold that has moved

### T1: threshold amended, conclusion invariant (D18, §9)

- **Original:** A0 reaches ≥ 0.99 exact-match on `unseen`.
- **Operational form now:** `M7_acc_teacher_forced ≥ 0.99`.

T1's stated rationale is whether a composing solution is *reachable*, which
teacher-forced composition measures directly; end-to-end accuracy also absorbs
manifold drift, which is the object of study rather than a property of the harness.

The conclusion is invariant: every comparison arm sits at `acc_unseen` 0.000–0.012
against A0's 0.951 (v1) / 1.000 (v2). **Any threshold between 0.02 and 0.95 gives the
same verdict.**

The substitution earned its keep in v2: D44's bug left A0 at `acc_unseen` **1.0000** —
the *original* T1 would have passed — while teacher-forced read 0.833 and stopped the
batch at run 1 of 54.

### Other recorded changes

| change | nature |
|---|---|
| Epoch budget 30 → 80, all arms (D17) | optimisation budget only; §12 explicitly authorises it |
| M3 probe literal → depth-matched (§2.5) | correction of an off-distribution measurement, not a threshold move |
| Split regenerated twice (D2, D14) | both **before** any factorization data; train-side coverage, not the evaluation criterion |
| E1b closed-map gate: bare error → matched + coverage (D24) | pre-registered before E1b ran; load-bearing |
| E1b seed counts reduced for 3 cells (D31) | scope reduction after a result — cells labelled `[REDUCED-n]` |
| v2 seeds 5 → 3, splits 1 → 3 (D42) | registered **before** the first v2 run; 9 runs/arm vs v1's 5 |

**No §8 threshold has ever been moved.**

---

## 5. E1b ladder thresholds (pre-registered, D24/D26)

Judged on the mean over seeds at the best rung.

| verdict | requires all of |
|---|---|
| **RECOVERS** | closed-map (matched) ≤ 0.15 **and** coverage ≥ 6/8 **and** `acc_unseen` ≥ 0.50 |
| **PARTIAL** | closed-map ≤ 0.30 **and** coverage ≥ 4/8 **and** `acc_unseen` ≥ 0.15 |
| **INCONCLUSIVE** | `code_residual` stays > 0.15 — the constraint never bound. Judged on **R2 only**: R3 projects through the code, so its residual is near-zero by construction and cannot distinguish "bound" from "learned nothing" (D25) |
| **DOES NOT RECOVER** | constraint bound, and nothing recovered |

The coverage floor is load-bearing: R2 w=40 reached closed-map **0.029**, inside the
RECOVERS band, and was correctly rejected on coverage (1/8) and `acc_unseen` (0.0001).
A bare-error gate would have registered the strongest possible wrong answer (D29).

**E1b is currently RETRACTED and mid-re-run** (D32). The `Sarb` rung is **blocked**
(D37): its target is input-independent and would return a confident wrong answer.

---

## 6. Standing rules

- **Generations are never pooled.** v1 and v2 measure different task families; a
  combined mean describes neither (D40).
- **No comparison spans machines.** Determinism is a within-platform guarantee;
  aggregation refuses to mix hostnames (D39, D41).
- **Archives are frozen.** `runs/archive_*` is skipped by `aggregate`, `backfill` and
  `status` by default, so a finished battery is never pooled with a running one and
  `backfill --reanalyze` cannot rewrite it (D41).
- **Training refuses to run from a dirty tree** unless `--allow-dirty`, which records
  the diff hash in `env.json`. Untracked files under `runs/`/`results/` are the
  batch's own output and are exempt (D33, D43).
- **Composer and library are always separate line items.** Never a combined "system
  size" — H5's failure mode is invisible under a total (spec §5). Note the detector
  has a known blind spot: the encoder/decoder can absorb library function while the
  composer stays small, so pair it with a functional check (closed-map error).
- **Every headline number is derived by `analyze.py` from saved artifacts alone.** A
  training-loop bug cannot silently produce favourable metrics, and any metric added
  mid-experiment is applied retroactively without retraining.

---

## 7. Current status

| | |
|---|---|
| **v1** (archived, `Perro`) | 30 runs · **FAIL(training-signal)** · T1 amended (§4), T2–T8 pass |
| **v2** (`Perrito`) | 54 runs · **FAIL(training-signal)** · all gates pass · 45/45 non-oracle runs at coverage 1/8 |
| **E1b** | **RETRACTED**, mid-re-run (D32). `Sarb` blocked (D37) |
| **H6** | **untested** after 84 runs — every failing arm fails *upstream* of H6's question |
| **E2** | gated, and stays gated until some arm builds closed maps without ground truth |
