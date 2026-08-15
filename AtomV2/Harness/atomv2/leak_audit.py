"""Formal leak audit over completed runs. Emits results/<exp>/leak_audit.json.

The prereg requires this ritual whenever the free arm scores high on unseen
(H1Experiments.md, "If Calibration Fails": a high free arm means the world may
leak and the split/data pipeline gets audited before anything else runs).

Every check reads what the RUNS actually recorded and re-derives the expected
value from the op algebra - nothing is compared against a transcription. The
audit reads only artifacts and the code; it never loads a model.

Checks:
  A  no P3 pair appears in any training or seen_heldout manifest; all 15 live
     in unseen at level L3; P3 is present as a singleton (the dax IS trained
     that way - its absence would be a different bug)
  B  every run's recorded split sha256 is the frozen file's, all runs agree,
     the split re-derives from the algebra, and the world block embedded in it
     (sub-op triples, surface recipes, surface triples, seq_len, vocab)
     matches the op definitions in code
  C  P3 oversampling is PRESENTATION frequency per amendment R4: 1,000 unique
     examples like every other task, appearing 7x in the epoch array as exact
     copies, totalling 48,000 presentations
  D  seen_heldout is genuinely fresh: same task set as train, zero byte-level
     input overlap per task, distinct manifest hashes, no duplicate rows
  E  each run's recorded data_manifest is reproducible from its saved config
  F  no held-out cell's FUNCTION equals any training cell's or singleton's

There is no world.json in this harness: the world definition lives as the
`world` block inside splits/split_v2.json, which is what check B audits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import ops
from . import registered as R
from . import split as split_mod
from .config import Config
from .utils import RESULTS_DIR, RUNS_DIR, read_json, sha256_of_file, write_json


def _runs(experiment: str) -> list[Path]:
    return sorted(p.parent for p in (RUNS_DIR / experiment).rglob("metrics.json"))


def _check(name: str, ok: bool, **detail) -> dict:
    return {"check": name, "ok": bool(ok), **detail}


def _is_pair_with_dax(task_id: str) -> bool:
    surface = ops.task_surface_ops(task_id)
    return len(surface) == 2 and R.DAX in surface


# --- A ----------------------------------------------------------------------

def check_a_no_dax_pairs(manifests: dict) -> dict:
    offenders, seen_offenders, singleton_present, l3_seen = {}, {}, {}, {}
    expected_l3 = sorted(t for t in split_mod.PAIR_IDS if _is_pair_with_dax(t))
    for run, m in manifests.items():
        offenders[run] = sorted(t for t in m["train"] if _is_pair_with_dax(t))
        seen_offenders[run] = sorted(t for t in m["seen_heldout"]
                                     if _is_pair_with_dax(t))
        singleton_present[run] = R.DAX in m["train"]
        l3_seen[run] = sorted(t for t, e in m["unseen"].items()
                              if _is_pair_with_dax(t) and e["level"] == "L3")
    ok = (all(not v for v in offenders.values())
          and all(not v for v in seen_offenders.values())
          and all(singleton_present.values())
          and all(v == expected_l3 for v in l3_seen.values()))
    return _check(
        "A_no_dax_pairs_in_training", ok,
        dax=R.DAX,
        dax_pairs_in_train={k: v for k, v in offenders.items()},
        dax_pairs_in_seen_heldout={k: v for k, v in seen_offenders.items()},
        dax_singleton_trained=singleton_present,
        n_dax_pairs_expected_in_unseen_L3=len(expected_l3),
        all_runs_carry_exactly_those=all(v == expected_l3 for v in l3_seen.values()),
        note="P3 as a SINGLETON is trained by design; only P3 PAIRS must be absent",
    )


# --- B ----------------------------------------------------------------------

def check_b_split_and_world(runs: list[Path]) -> dict:
    frozen = sha256_of_file(split_mod.SPLIT_PATH)
    recorded = {p.name: read_json(p / "split_ref.json")["sha256"] for p in runs}
    hashes_ok = all(v == frozen for v in recorded.values())

    rederives = True
    rederive_error = None
    try:
        split_mod.load_verified()
    except Exception as exc:                                  # noqa: BLE001
        rederives, rederive_error = False, str(exc)

    s = split_mod.load()
    w = s["world"]
    mismatches = []
    for name, tri in ops.SUBOP_TRIPLES.items():
        got = w["subops"].get(name)
        want = {"pi": list(tri[0]), "a": tri[1], "b": list(tri[2])}
        if got != want:
            mismatches.append({"subop": name, "in_split": got, "from_code": want})
    for p, tri in ops.SURFACE_TRIPLES.items():
        got = w["surface_triples"].get(p)
        want = {"pi": list(tri[0]), "a": tri[1], "b": list(tri[2])}
        if got != want:
            mismatches.append({"surface": p, "in_split": got, "from_code": want})
    for p, rec in R.SURFACE_RECIPES.items():
        if w["surface_recipes"].get(p) != list(rec):
            mismatches.append({"recipe": p, "in_split": w["surface_recipes"].get(p),
                               "from_code": list(rec)})
    if w["seq_len"] != ops.L or w["vocab"] != ops.MOD:
        mismatches.append({"dims": {"in_split": [w["seq_len"], w["vocab"]],
                                    "from_code": [ops.L, ops.MOD]}})

    return _check(
        "B_split_hash_and_world_block", hashes_ok and rederives and not mismatches,
        frozen_split_sha256=frozen,
        recorded_by_runs=recorded,
        all_runs_match_frozen=hashes_ok,
        split_rederives_from_algebra=rederives,
        rederive_error=rederive_error,
        world_block_mismatches=mismatches,
        note="no world.json exists; the world block lives inside split_v2.json",
    )


# --- C ----------------------------------------------------------------------

def check_c_dax_oversampling(bundles: dict, manifests: dict) -> dict:
    per_seed = {}
    for seed, (bundle, cfg) in bundles.items():
        arrays = data_mod.build_epoch_arrays(bundle, cfg)
        dax_task = data_mod.make_task(R.DAX, "train")
        is_dax = ((arrays["tokens"] == dax_task.tokens).all(axis=1)
                  & (arrays["n_tokens"] == 1))
        n_dax = int(is_dax.sum())
        unique_dax = len({r.tobytes() for r in arrays["x"][is_dax]})
        expected_presentations = (cfg.p3_oversample_factor
                                  * cfg.examples_per_train_task)
        # The k blocks must be exact copies of one examples_per_train_task set.
        # A wrong presentation count must REPORT a failure, never raise: an
        # audit that crashes is ambiguous precisely when something is wrong.
        if n_dax == expected_presentations and cfg.p3_oversample_factor > 0:
            block = arrays["x"][is_dax].reshape(cfg.p3_oversample_factor, -1,
                                                ops.L)
            copies_identical = bool(all(np.array_equal(block[0], block[i])
                                        for i in range(1, len(block))))
        else:
            copies_identical = False
        per_seed[str(seed)] = {
            "presentations": n_dax,
            "expected_presentations": expected_presentations,
            "unique_examples": unique_dax,
            "expected_unique": cfg.examples_per_train_task,
            "blocks_are_exact_copies": copies_identical,
            "total_presentations": int(len(arrays["x"])),
            "expected_total": (41 + cfg.p3_oversample_factor)
                              * cfg.examples_per_train_task,
        }
    manifest_n = {run: m["train"][R.DAX]["n"] for run, m in manifests.items()}
    others_n = {run: sorted({e["n"] for t, e in m["train"].items() if t != R.DAX})
                for run, m in manifests.items()}
    manifest_ok = all(manifest_n[r] == others_n[r][0] and len(others_n[r]) == 1
                      for r in manifest_n)
    ok = manifest_ok and all(
        v["presentations"] == v["expected_presentations"]
        and v["unique_examples"] == v["expected_unique"]
        and v["blocks_are_exact_copies"]
        and v["total_presentations"] == v["expected_total"]
        for v in per_seed.values())
    return _check(
        "C_dax_oversampling_is_presentation_frequency", ok,
        factor=R.P3_OVERSAMPLE_FACTOR,
        per_seed=per_seed,
        manifest_unique_count_for_dax=manifest_n,
        manifest_unique_counts_for_other_tasks=others_n,
        dax_unique_count_equals_every_other_task=manifest_ok,
        note="amendment R4: oversampling is presentation frequency; the "
             "manifest must show 1,000 unique examples for P3 like every "
             "other task, and the epoch array must repeat them 7x",
    )


# --- D ----------------------------------------------------------------------

def check_d_seen_heldout_fresh(bundles: dict, manifests: dict) -> dict:
    per_seed = {}
    for seed, (bundle, _cfg) in bundles.items():
        train_ids = [td.task.task_id for td in bundle.train]
        seen_ids = [td.task.task_id for td in bundle.seen_heldout]
        overlaps, dupes = {}, {}
        for tr, ev in zip(bundle.train, bundle.seen_heldout):
            tk = {r.tobytes() for r in tr.x}
            ek = {r.tobytes() for r in ev.x}
            inter = tk & ek
            if inter:
                overlaps[tr.task.task_id] = len(inter)
            if len(tk) != len(tr.x) or len(ek) != len(ev.x):
                dupes[tr.task.task_id] = {"train_unique": len(tk),
                                          "train_rows": len(tr.x),
                                          "eval_unique": len(ek),
                                          "eval_rows": len(ev.x)}
        per_seed[str(seed)] = {
            "task_sets_identical": train_ids == seen_ids,
            "n_tasks": len(train_ids),
            "tasks_with_input_overlap": overlaps,
            "tasks_with_duplicate_rows": dupes,
        }
    hash_distinct = {}
    for run, m in manifests.items():
        same = sorted(t for t in m["train"]
                      if m["train"][t]["x"] == m["seen_heldout"][t]["x"])
        hash_distinct[run] = same
    ok = (all(v["task_sets_identical"] and not v["tasks_with_input_overlap"]
              and not v["tasks_with_duplicate_rows"] for v in per_seed.values())
          and all(not v for v in hash_distinct.values()))
    return _check(
        "D_seen_heldout_genuinely_fresh", ok,
        per_seed=per_seed,
        tasks_whose_train_and_eval_hashes_collide=hash_distinct,
        note="per-task disjointness is the guarantee; inputs may recur ACROSS "
             "tasks by birthday collision, which carries no answer",
    )


# --- E ----------------------------------------------------------------------

def check_e_manifest_reproducible(runs: list[Path], manifests: dict,
                                  bundles: dict) -> dict:
    results = {}
    for p in runs:
        cfg = Config.from_dict(read_json(p / "config.json"))
        bundle, _ = bundles[cfg.seed]
        rebuilt = data_mod.data_manifest(bundle)
        results[p.name] = rebuilt == manifests[p.name]
    return _check("E_manifest_reproducible_from_config", all(results.values()),
                  per_run=results,
                  note="rebuilds each run's dataset from its saved config and "
                       "compares content hashes: proves the data on record is "
                       "what the current generator produces")


# --- F ----------------------------------------------------------------------

def check_f_no_function_level_leak() -> dict:
    s = split_mod.load()
    train_keys = {s["cells"][t]["triple_key"] for t in s["train_pairs"]}
    train_keys |= {ops.triple_key(ops.SURFACE_TRIPLES[p])
                   for p in s["singletons_train"]}
    leaks = {}
    for level in ("L1", "L2", "L3"):
        bad = [t for t in s["heldout"][level]
               if s["cells"][t]["triple_key"] in train_keys]
        if bad:
            leaks[level] = bad
    return _check("F_no_heldout_function_is_trained", not leaks,
                  leaked_cells=leaks,
                  n_train_functions=len(train_keys),
                  note="extensional check: a held-out cell computing the same "
                       "function as a trained task would be training on test")


def run_audit(experiment: str = "e0") -> dict:
    runs = _runs(experiment)
    if not runs:
        raise SystemExit(f"no completed runs under {RUNS_DIR / experiment}")
    manifests = {p.name: read_json(p / "data_manifest.json") for p in runs}
    bundles = {}
    for p in runs:
        cfg = Config.from_dict(read_json(p / "config.json"))
        if cfg.seed not in bundles:
            bundles[cfg.seed] = (data_mod.build_bundle(cfg), cfg)

    checks = [
        check_a_no_dax_pairs(manifests),
        check_b_split_and_world(runs),
        check_c_dax_oversampling(bundles, manifests),
        check_d_seen_heldout_fresh(bundles, manifests),
        check_e_manifest_reproducible(runs, manifests, bundles),
        check_f_no_function_level_leak(),
    ]
    report = {
        "experiment": experiment,
        "protocol_revision": R.PROTOCOL_REVISION,
        "n_runs_audited": len(runs),
        "runs": [p.name for p in runs],
        "all_passed": all(c["ok"] for c in checks),
        "checks": checks,
    }
    out = RESULTS_DIR / experiment
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "leak_audit.json", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Formal split/data leak audit")
    ap.add_argument("--experiment", default="e0", choices=["e0", "e1"])
    args = ap.parse_args()
    report = run_audit(args.experiment)
    print(f"leak audit: {args.experiment}  runs={report['n_runs_audited']}  "
          f"{'ALL PASSED' if report['all_passed'] else 'FAILURES PRESENT'}")
    for c in report["checks"]:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
