# E1 — Atom Factorization Battery (laptop / CPU edition)

Tests **H6** of the Atoms preregistration: are atoms independently meaningful
primitives, or co-adapted fragments? This is a hard gate — if atoms do not
factorize, H1's marginal-cost curve and H3's geometry are both uninterpretable.

Read [DECISIONS.md](DECISIONS.md) before interpreting any number. It records the
frozen thresholds and every deviation from the spec, with reasons.

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

## Running

```bash
python -m e1.make_split          # already frozen and committed; --force to regenerate
python tests/test_fast.py        # must be green before anything else

# Run in THIS order. A3 is the informative arm and D12 predicts A1 fails, so getting
# A3 after ~3.5 h beats waiting ~10 h for a single 25-run block.
python -m e1.run_all --arms A0            # T1 gate: oracle must clear 99% on `unseen`
python -m e1.run_all --arms A3            # the informative arm
python -m e1.run_all --arms A1 A2 A4      # comparison arms, overnight
python -m e1.aggregate                    # results/summary.md + plots + verdict

python tests/test_gates.py --t1 --t3
python tests/test_gates.py --t2                        # determinism, reduced scale
python tests/test_gates.py --t6 runs/A0_0_<sha>
```

`run_all` stops the batch if A0 misses the T1 gate — no other arm's failure would be
interpretable.

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

Per H5, composer and library parameter counts are always reported as separate line
items — a composer quietly absorbing the atoms' work is invisible under a combined
total.
