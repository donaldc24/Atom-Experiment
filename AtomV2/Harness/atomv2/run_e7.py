"""E7 batch driver (H1-Experiment7.md): simplify before translating.

Registered order (the run stages mirror the registered decision tree; the
driver never auto-advances across a human decision point):

  0. Implementation gates:
     a. Refactor equivalence, hard stop: the A6 config (micro_steps = 3,
        residual, 384) replayed under THIS refactored harness reproduces
        the completed E1b A6 seed-0 step-1/step-50 records - bit-identical
        same-env, or the registered cross-machine three-part substitution
        (E5-G1 lineage: step-1 non-grad records bit-identical, every
        mismatch <= E4_GATE_REL_TOL relative, environment difference
        documented).
     b. Structural gates (one route decision per token, no hidden atom
        calls, replacement mode free of residual addition, pass unchanged,
        consistent narrow-state dimensions, read-only diagnostics) are
        unit-enforced in tests/test_e7.py.
  1. --stage screen     : A18 seed 1 (the registered mechanism screen).
  2. --stage a18-seeds  : A18 seeds 0/2 (only after seed 1 is healthy).
  3. --stage stage2     : A19/A20/A21 seed 1 (matched control = A18 s1).
  4. --stage replicate --arms <winner> : seeds 0/2 of the selected arm
     (selection rules are registered; the driver takes the arm as input).

Health criterion: seen hard accuracy >= E7_HEALTH_SEEN_MIN. Poor
optimization is a result, not an implementation failure. Every run gets
the full panel, analyze, and the E7 audit (boundary panel, raw vs
canonicalized composition, Interface Closure Ratio, state content).
Reporting follows the registered order and never leads with L3.
"""
from __future__ import annotations

import argparse
import json

from . import registered as R
from .analyze import analyze
from .config import E7_ARMS, config_for_arm, run_dir_for
from .e7_audit import run_audit
from .panel import run_all_panels
from .run_e2 import _find_run_dir, _replay_first_steps
from .train import train_run
from .utils import RESULTS_DIR, read_json, write_json, write_sha256sums

STAGES = ("screen", "a18-seeds", "stage2", "replicate")


def _a6_refactor_gate(out_dir, filename: str = "e7_a6_equivalence.json"
                      ) -> None:
    """Gate 0a: micro_steps = 3 under the refactored implementation must
    reproduce the existing A6 execution path (H1-Experiment7.md gate 1)."""
    cfg = config_for_arm(R.E7_BASE_ARM, 0)
    assert cfg.micro_steps == 3 and cfg.atom_update == "residual"
    ref_dir = _find_run_dir("e1b", R.E7_BASE_ARM, 0)
    logged = {}
    with open(ref_dir / "train_log.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "step" and rec["step"] in (1, 50):
                logged[rec["step"]] = rec
    if set(logged) != {1, 50}:
        raise SystemExit(f"A6 refactor gate: steps 1/50 not found in "
                         f"{ref_dir}/train_log.jsonl")
    replayed = _replay_first_steps(cfg, n_steps=50)
    keys = ["loss", "loss_task", "grad_norm", "batch_acc"]
    mismatches = []
    for step in (1, 50):
        for key in keys:
            a, b = logged[step][key], replayed[step][key]
            if a != b:
                mismatches.append({"step": step, "key": key,
                                   "reference": a, "replayed": b})
    record = {"reference_run": str(ref_dir), "arm": R.E7_BASE_ARM,
              "compared_steps": [1, 50], "compared_keys": keys,
              "bit_identical": not mismatches, "mismatches": mismatches}
    if not mismatches:
        write_json(out_dir / filename, record)
        print("A6 refactor gate: bit-identical (steps 1 and 50)")
        return

    import platform as _platform
    import sys as _sys
    import numpy as _np
    import torch as _torch
    ref_env = read_json(ref_dir / "env.json")
    current_env = {"torch": _torch.__version__, "numpy": _np.__version__,
                   "python": _sys.version, "hostname": _platform.node(),
                   "processor": _platform.processor()}
    env_diffs = {k: {"reference": ref_env.get(k), "current": v}
                 for k, v in current_env.items() if ref_env.get(k) != v}
    cross_machine = any(k in env_diffs for k in ("hostname", "processor"))
    step1_forward_exact = all(
        m["step"] != 1 or m["key"] == "grad_norm" for m in mismatches)
    within_tol = all(
        abs(m["reference"] - m["replayed"])
        <= R.E4_GATE_REL_TOL * max(abs(m["reference"]), 1e-12)
        for m in mismatches)
    passed = cross_machine and step1_forward_exact and within_tol
    record["substitution"] = {
        "amendment": "E5-G1 lineage (cross-machine substituted form)",
        "environment_differences": env_diffs,
        "cross_machine": cross_machine,
        "step1_forward_bit_identical": step1_forward_exact,
        "within_rel_tol": within_tol, "rel_tol": R.E4_GATE_REL_TOL,
        "passed": passed,
    }
    write_json(out_dir / filename, record)
    if not passed:
        raise SystemExit(
            f"A6 REFACTOR GATE FAILED (strict AND substituted forms): "
            f"{mismatches[:2]}... The refactored micro_steps=3 path does "
            "not reproduce A6; nothing proceeds until it does.")
    print("A6 refactor gate: failed as written (environment differs: "
          f"{', '.join(sorted(env_diffs))}), SATISFIED IN PURPOSE under the "
          "cross-machine substituted form")


def _run_one(arm: str, seed: int, smoke: bool, allow_dirty: bool) -> dict:
    cfg = config_for_arm(arm, seed, smoke=smoke)
    run_dir = run_dir_for(cfg)
    if (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists():
        print(f"skip (complete): {run_dir}")
        m = read_json(run_dir / "metrics.json")
        if not (run_dir / "e7_audit.json").exists():
            run_audit(run_dir)
            write_sha256sums(run_dir)
        return m
    print(f"=== {arm} (micro_steps={cfg.micro_steps}, "
          f"state={cfg.state_dim}, update={cfg.atom_update}) seed={seed} "
          f"steps={cfg.total_steps} -> {run_dir}")
    if (run_dir / "checkpoints" / "final.pt").exists():
        print("    training already complete; resuming at panel")
    else:
        train_run(cfg, allow_dirty=allow_dirty)
    run_all_panels(run_dir)
    analyze(run_dir)                 # standard metrics (audit reads seen acc)
    audit = run_audit(run_dir)
    metrics = analyze(run_dir)       # re-derive to fold the audit headline in
    write_sha256sums(run_dir)
    v = audit["verdict"]

    def _f(x, spec=".4f"):
        return "-" if x is None else format(x, spec)
    # Registered reporting order: validity, competence, components, raw,
    # canonicalized, ICR, state content, routing. Never leads with L3.
    print(f"    valid={metrics.get('e1b_run_valid')} "
          f"seen={metrics['acc_seen_hard']:.4f} "
          f"healthy={audit['healthy']} | "
          f"mapped_atoms={v['n_mapped_atoms']} "
          f"raw={_f(v['raw_chain_acc'])} canon={_f(v['canon_chain_acc'])} "
          f"ICR={_f(v['interface_closure_ratio'], '.3f')} "
          f"({v['closure_band']}) | "
          f"ans_rec={audit['state_content']['answer_recoverability']:.3f} "
          f"orig_rec={audit['state_content']['orig_recoverability']:.3f} | "
          f"L1={metrics['acc_unseen_L1_hard']:.4f} "
          f"L3={metrics['acc_unseen_L3_hard']:.4f}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E7 battery "
                                             "(one-step interface)")
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline check: all four arms, smoke budget")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="replicate stage: the registered winner arm")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-equivalence", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    if args.plan:
        print("gate 0a: A6 refactor-equivalence replay vs e1b A6 s0")
        print("stage screen    : A18 seed 1 (mechanism screen; health = "
              f"seen >= {R.E7_HEALTH_SEEN_MIN})")
        print("stage a18-seeds : A18 seeds 0/2 (after healthy screen)")
        print("stage stage2    : A19/A20/A21 seed 1 (2x2: bandwidth x "
              "replacement)")
        print("stage replicate : --arms <winner> seeds 0/2 per the "
              "registered selection rules")
        return

    if args.smoke:
        batch = [(arm, 0) for arm in E7_ARMS]
    elif args.stage == "screen":
        batch = [("A18", R.E7_SCREEN_SEED)]
    elif args.stage == "a18-seeds":
        batch = [("A18", 0), ("A18", 2)]
    elif args.stage == "stage2":
        batch = [("A19", R.E7_SCREEN_SEED), ("A20", R.E7_SCREEN_SEED),
                 ("A21", R.E7_SCREEN_SEED)]
    elif args.stage == "replicate":
        if not args.arms:
            raise SystemExit("--stage replicate requires --arms <winner>")
        batch = [(arm, s) for arm in args.arms for s in (0, 2)]
    else:
        raise SystemExit("choose --stage {screen,a18-seeds,stage2,"
                         "replicate}, --smoke, or --plan; E7 runs are "
                         "staged by the registered decision tree and are "
                         "never launched all at once")
    if args.arms and args.stage != "replicate":
        batch = [(a, s) for a, s in batch if a in args.arms]
    if args.seeds is not None:
        batch = sorted({(a, s) for a, _ in batch for s in args.seeds})

    bad = [a for a, _ in batch if a not in E7_ARMS]
    if bad:
        raise SystemExit(f"unknown E7 arms {bad}; battery arms are "
                         f"{E7_ARMS}. A6 is the historical reference, "
                         "never re-run.")

    out = RESULTS_DIR / ("smoke_e7" if args.smoke else "e7")
    out.mkdir(parents=True, exist_ok=True)
    if not args.smoke and not args.skip_equivalence:
        _a6_refactor_gate(out)

    rows = [_run_one(arm, seed, args.smoke, args.allow_dirty)
            for arm, seed in batch]

    if args.stage == "screen" and rows:
        seen = rows[0].get("acc_seen_hard")
        healthy = seen is not None and seen >= R.E7_HEALTH_SEEN_MIN
        print(f"\nA18 screen verdict: seen={seen:.4f} -> "
              f"{'HEALTHY: proceed to --stage a18-seeds' if healthy else 'ONE-STEP CAPACITY / OPTIMIZATION FAILURE: stop E7, register the capacity-rescue experiment separately'}")


if __name__ == "__main__":
    main()
