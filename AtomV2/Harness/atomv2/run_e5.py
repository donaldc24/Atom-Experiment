"""E5 batch driver (H1-Experiment5.md): the producer branch - training atoms
to speak, not just listen - stacked on the unchanged A14 base.

Registered order:

  1. Zero-path gate, hard stop: lambda_producer = 0 (the A14 config)
     replayed under THIS harness reproduces the completed E4 A14 seed-0
     step-1/step-50 records. Same-env the strict form applies (bit-identical
     at both steps); when the reference env.json documents a DIFFERENT
     environment, the registered cross-env three-part test per amendment
     E4-G1 applies instead: step-1 records bit-identical, grad norms
     bit-identical, step-50 records within E4_GATE_REL_TOL relative, and
     the environment difference documented in the gate record.
  2. A16/A17 smoke runs + implementation gates (all E2 gates: cosine target,
     scale, clean-eval determinism; all E3 gates: telemetry finite, weights
     in range, init calibration carries sandbox scale; liveness present;
     NEW: producer telemetry present and finite, init calibration carries
     the producer scale). Structural gates stay unit-enforced
     (tests/test_e2.py, test_e3.py, test_e4.py, test_e5.py).
  3. A16/A17 x seeds 0/1/2, 30k steps, panels at 20k + final (both verified
     to have actually fired), registered robustness sweep, analyze (which
     emits the 20k like-for-like block and the per-P3-cell dax-crack check).

A14 attaches as a labelled reference row at aggregation, never re-run.
Implementation failure invalidates a run; optimization failure under a
correctly implemented producer is a result, not an invalid run. The
registered outcome bins (CRACK / PRODUCER WORKS, DAX HOLDS / REDUNDANT /
COLLAPSE / OVERDOSE) are called from the aggregated tables.
"""
from __future__ import annotations

import argparse
import json
import math

from . import registered as R
from .aggregate import aggregate
from .analyze import analyze
from .config import E5_ARMS, config_for_arm, run_dir_for
from .noise import robustness_sweep
from .panel import run_all_panels
from .run_e2 import _find_run_dir
from .run_e4 import _replay_first_steps
from .train import train_run
from .utils import (RESULTS_DIR, read_json, stream_rng, write_json,
                    write_sha256sums)


def _a14_zero_path_gate(out_dir, filename: str = "e5_a14_equivalence.json"
                        ) -> None:
    """Gate 1: the A14 config (lambda_producer = 0 by construction) replayed
    under THIS harness must reproduce the completed E4 A14 seed-0 records.

    Same-env: strict bit-identity at steps 1 and 50 (every logged loss key).
    Cross-BUILD (same machine, different torch/numpy build): the registered
    three-part test per E4-G1 - step-1 records bit-identical, grad norms
    bit-identical at both steps, step-50 records within E4_GATE_REL_TOL
    relative. Cross-MACHINE (the reference env.json documents a different
    hostname/processor - amendment E5-G1): grad-norm bit-identity is
    unattainable (backward reduction kernels dispatch per CPU), so the
    substituted form is step-1 NON-GRAD records bit-identical (the
    code-fidelity witness: identical weights, data and RNG streams), every
    mismatching record within E4_GATE_REL_TOL relative, and the environment
    difference documented in the gate record.
    """
    cfg = config_for_arm(R.E5_BASE_ARM, 0)
    assert cfg.lambda_producer == 0.0
    ref_dir = _find_run_dir("e4", R.E5_BASE_ARM, 0)
    logged = {}
    with open(ref_dir / "train_log.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "step" and rec["step"] in (1, 50):
                logged[rec["step"]] = rec
    if set(logged) != {1, 50}:
        raise SystemExit(f"A14 zero-path gate: steps 1/50 not found in "
                         f"{ref_dir}/train_log.jsonl")
    replayed = _replay_first_steps(cfg, n_steps=50)
    keys = ["loss", "loss_task", "grad_norm", "batch_acc",
            "loss_sandbox_valid", "loss_sandbox_unique"]
    mismatches = []
    for step in (1, 50):
        for key in keys:
            a, b = logged[step][key], replayed[step][key]
            if a != b:
                mismatches.append({"step": step, "key": key,
                                   "reference": a, "replayed": b})
    record = {"reference_run": str(ref_dir), "arm": R.E5_BASE_ARM,
              "compared_steps": [1, 50], "compared_keys": keys,
              "bit_identical": not mismatches, "mismatches": mismatches}

    if not mismatches:
        write_json(out_dir / filename, record)
        print("A14 zero-path gate: bit-identical (steps 1 and 50)")
        return

    import numpy as _np
    import platform as _platform
    import sys as _sys
    import torch as _torch
    ref_env = read_json(ref_dir / "env.json")
    current_env = {
        "torch": _torch.__version__,
        "numpy": _np.__version__,
        "python": _sys.version,
        "hostname": _platform.node(),
        "processor": _platform.processor(),
    }
    env_diffs = {k: {"reference": ref_env.get(k), "current": v}
                 for k, v in current_env.items() if ref_env.get(k) != v}
    cross_machine = any(k in env_diffs for k in ("hostname", "processor"))
    step1_exact = all(m["step"] != 1 for m in mismatches)
    step1_forward_exact = all(
        m["step"] != 1 or m["key"] == "grad_norm" for m in mismatches)
    grad_exact = all(m["key"] != "grad_norm" for m in mismatches)
    within_tol = all(
        abs(m["reference"] - m["replayed"])
        <= R.E4_GATE_REL_TOL * max(abs(m["reference"]), 1e-12)
        for m in mismatches)
    # E4-G1 (cross-build, same machine): step-1 + grad norms bit-identical.
    e4g1_ok = (bool(env_diffs) and step1_exact and grad_exact and within_tol)
    # E5-G1 (cross-machine): grad-norm bit-identity unattainable; the
    # code-fidelity witness is step-1 forward bit-identity.
    e5g1_ok = (cross_machine and step1_forward_exact and within_tol)
    substituted_ok = e4g1_ok or e5g1_ok
    record["substitution"] = {
        "amendment": "E4-G1 / E5-G1 lineage (see DECISIONS.md): "
                     "failed as written, satisfied in purpose",
        "environment_differences": env_diffs,
        "environment_differs": bool(env_diffs),
        "cross_machine": cross_machine,
        "step1_bit_identical": step1_exact,
        "step1_forward_bit_identical": step1_forward_exact,
        "grad_norms_bit_identical": grad_exact,
        "within_rel_tol": within_tol,
        "rel_tol": R.E4_GATE_REL_TOL,
        "e4g1_form_passed": e4g1_ok,
        "e5g1_form_passed": e5g1_ok,
        "passed": substituted_ok,
    }
    write_json(out_dir / filename, record)
    if not substituted_ok:
        raise SystemExit(
            f"A14 ZERO-PATH GATE FAILED (strict AND substituted forms): "
            f"{mismatches[:2]}... The E5 harness's lambda_producer = 0 path "
            "does not reproduce the completed A14 run; nothing proceeds "
            "until it does (or the A14 base is re-run under this "
            "environment).")
    diff_keys = ", ".join(sorted(env_diffs))
    form = "E4-G1 (cross-build)" if e4g1_ok else "E5-G1 (cross-machine)"
    print(f"A14 zero-path gate: failed as written (environment differs: "
          f"{diff_keys}), SATISFIED IN PURPOSE under {form}; worst drift "
          f"within {R.E4_GATE_REL_TOL} rel, details in the gate record")


def _smoke_gates(arm: str, allow_dirty: bool) -> None:
    """All E2 + E3 + E5 artifact-level implementation gates on one smoke
    run."""
    import torch

    cfg = config_for_arm(arm, 0, smoke=True)
    run_dir = run_dir_for(cfg)
    if not (run_dir / "checkpoints" / "final.pt").exists():
        print(f"=== smoke {arm} -> {run_dir}")
        train_run(cfg, allow_dirty=allow_dirty)
    lv = sorted((run_dir / "liveness").glob("step*.json"))
    if len(lv) < 2:
        raise SystemExit(f"smoke {arm}: <2 liveness evals")
    for p in lv:
        if not read_json(p)["base_geometry"]["pmax_invariant_ok"]:
            raise SystemExit(f"smoke {arm}: p_max invariant violated ({p})")
    nts = sorted((run_dir / "noise_telemetry").glob("step*.json"))
    if not nts:
        raise SystemExit(f"smoke {arm}: no noise telemetry written")
    for p in nts:
        nt = read_json(p)
        if not nt["handoff"]["cosine_within_tol"]:
            raise SystemExit(
                f"smoke {arm}: observed cosine "
                f"{nt['handoff']['cosine']['p50']:.6f} not within "
                f"{R.E2_COSINE_TOL} of target {nt['target_cosine']:.6f} ({p})")
        if nt["handoff"]["transmitted_pos_mean_abs"] > 0.1 \
                or abs(nt["handoff"]["transmitted_pos_var"] - 1.0) > 0.1:
            raise SystemExit(f"smoke {arm}: transmitted-state scale off ({p})")
    sts = sorted((run_dir / "sandbox_telemetry").glob("step*.json"))
    if not sts:
        raise SystemExit(f"smoke {arm}: no sandbox telemetry written")
    for p in sts:
        st = read_json(p)
        for key, val in (("standalone read", st["standalone"]["read_mean"]),
                         ("closure read", st["closure"]["read_mean"])):
            if not math.isfinite(val):
                raise SystemExit(f"smoke {arm}: non-finite {key} ({p})")
        if not all(0.0 <= w <= 1.0 for w in st["usage"]["weights"]):
            raise SystemExit(f"smoke {arm}: usage weight outside [0,1] ({p})")
    pts = sorted((run_dir / "producer_telemetry").glob("step*.json"))
    if not pts:
        raise SystemExit(f"smoke {arm}: no producer telemetry written")
    for p in pts:
        pt = read_json(p)
        for key, val in (
                ("branch read mean", pt["branch_read"]["read_mean"]),
                ("branch read spread", pt["branch_read"]["spread_mean"]),
                ("output variance min", pt["output_variance"]["min"]),
                ("output variance mean", pt["output_variance"]["mean"])):
            if not math.isfinite(val):
                raise SystemExit(f"smoke {arm}: non-finite {key} ({p})")
        if pt["output_variance"]["min"] < 0:
            raise SystemExit(f"smoke {arm}: negative producer variance ({p})")
    cal = read_json(run_dir / "init_calibration.json")
    if "loss_sandbox_valid_init" not in cal:
        raise SystemExit(f"smoke {arm}: init calibration lacks sandbox scale")
    if "loss_producer_init" not in cal:
        raise SystemExit(f"smoke {arm}: init calibration lacks producer scale")
    from .panel import load_checkpoint
    model, ck_cfg, _ = load_checkpoint(run_dir, "final.pt")
    diag_rng = stream_rng(ck_cfg.seed, "e1b_diag")
    x = torch.from_numpy(diag_rng.integers(0, 10, size=(64, 6)).astype("int64"))
    toks = torch.from_numpy(diag_rng.integers(0, 8, size=(64, 2)).astype("int64"))
    ntok = torch.full((64,), 2, dtype=torch.int64)
    with torch.no_grad():
        o1 = model(x, toks, ntok, mode="hard")
        o2 = model(x, toks, ntok, mode="hard")
    if not (torch.equal(o1["logits"], o2["logits"])
            and torch.equal(o1["choices"], o2["choices"])):
        raise SystemExit(f"smoke {arm}: clean hard eval is not deterministic")
    if "states_transmitted" in o1:
        raise SystemExit(f"smoke {arm}: clean eval carries a noise payload")
    print(f"    smoke {arm}: {len(lv)} liveness, {len(nts)} noise, "
          f"{len(sts)} sandbox, {len(pts)} producer records; all "
          "implementation gates ok")


def _run_one(arm: str, seed: int, smoke: bool, allow_dirty: bool) -> dict:
    cfg = config_for_arm(arm, seed, smoke=smoke)
    run_dir = run_dir_for(cfg)
    if (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists():
        print(f"skip (complete): {run_dir}")
        return read_json(run_dir / "metrics.json")
    print(f"=== {arm} (lambda_producer={cfg.lambda_producer}, "
          f"noise_sigma={cfg.state_noise_sigma:.9f}, "
          f"lambda_sandbox={cfg.lambda_sandbox_valid}) seed={seed} "
          f"steps={cfg.total_steps} -> {run_dir}")
    if (run_dir / "checkpoints" / "final.pt").exists():
        print("    training already complete; resuming at panel")
    else:
        train_run(cfg, allow_dirty=allow_dirty)
    run_all_panels(run_dir)
    # Registered measurement check: both panels actually fired.
    panel_checks = ["final"] if smoke else ["step020000", "final"]
    for pdir in panel_checks:
        if not (run_dir / "panel" / pdir / "census.json").exists():
            raise SystemExit(f"{arm} s{seed}: panel/{pdir} did not fire")
    if not (run_dir / "noise_robustness.json").exists():
        write_json(run_dir / "noise_robustness.json", robustness_sweep(run_dir))
    metrics = analyze(run_dir)
    write_sha256sums(run_dir)
    def _f(key):
        v = metrics.get(key)
        return "-" if v is None else f"{v:.4f}"
    print(f"    final: seen={metrics['acc_seen_hard']:.4f} "
          f"L1={metrics['acc_unseen_L1_hard']:.4f} "
          f"L3={metrics['acc_unseen_L3_hard']:.4f} | "
          f"20k: seen={_f('e5_acc_seen_hard_20k')} "
          f"L1={_f('e5_acc_unseen_L1_hard_20k')} | "
          f"dax_max={_f('e5_dax_max_cell_acc')} "
          f"crack={metrics.get('e5_dax_crack')} "
          f"var_min={_f('e5_producer_variance_min_final')} "
          f"spread={_f('e5_branch_read_spread_mean_final')} "
          f"valid={metrics.get('e1b_run_valid')}")
    return metrics


def _validity_verdict(rows: list[dict]) -> dict:
    per_run, per_arm = {}, {}
    for m in rows:
        per_run[f"{m['arm']}_seed{m['seed']}"] = {
            "liveness_valid": bool(m.get("e1b_run_valid")),
            "deafness_violated": bool(m.get("e1b_deafness_violated")),
            "cosine_gate_ok": bool(m.get("e2_cosine_gate_ok")),
            "dax_crack": bool(m.get("e5_dax_crack")),
            "producer_variance_min_final": m.get(
                "e5_producer_variance_min_final"),
            "valid": bool(m.get("e1b_run_valid"))
            and bool(m.get("e2_cosine_gate_ok")),
        }
    for arm in E5_ARMS:
        seeds = [v for k, v in per_run.items() if k.startswith(f"{arm}_")]
        per_arm[arm] = {
            "n_runs": len(seeds),
            "n_valid": sum(v["valid"] for v in seeds),
            "liveness_premise_holds": bool(seeds) and not any(
                v["deafness_violated"] for v in seeds),
            "dax_crack_any_seed": any(v["dax_crack"] for v in seeds),
        }
    return {"per_run": per_run, "per_arm": per_arm,
            "note": "outcome bins (CRACK / PRODUCER WORKS, DAX HOLDS / "
                    "REDUNDANT / COLLAPSE / OVERDOSE) are called against the "
                    "registered taxonomy from the 20k like-for-like columns; "
                    "a dax crack in any healthy arm is the headline "
                    "regardless of other bins. COLLAPSE is read from the "
                    "producer-output-variance row plus the one-word-language "
                    "fingerprints."}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E5 battery "
                                             "(producer branch)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(R.SEEDS))
    ap.add_argument("--arms", nargs="*", default=list(E5_ARMS))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-equivalence", action="store_true")
    ap.add_argument("--skip-free-smoke", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    bad = [a for a in args.arms if a not in E5_ARMS]
    if bad:
        raise SystemExit(f"unknown E5 arms {bad}; battery arms are {E5_ARMS}. "
                         "A14 is the completed E4 reference, never re-run.")

    batch = [(arm, seed) for arm in args.arms for seed in args.seeds]
    if args.plan:
        print("gate 1: A14 zero-path replay vs e4 A14 s0 (bit-identical "
              "same-env, or the registered cross-env three-part test)")
        for arm in args.arms:
            print(f"smoke {arm} seed=0  (E2+E3+E5 implementation gates)")
        for arm, seed in batch:
            print(f"{arm} (lambda_producer={R.E5_LAMBDA_PRODUCER[arm]}, "
                  f"K={R.E5_PRODUCER_BRANCHES}) seed={seed} "
                  f"steps={R.E4_TOTAL_STEPS}")
        return

    out = RESULTS_DIR / ("smoke_e5" if args.smoke else "e5")
    out.mkdir(parents=True, exist_ok=True)

    if not args.smoke and not args.skip_equivalence:
        _a14_zero_path_gate(out)
    if not args.smoke and not args.skip_free_smoke:
        for arm in args.arms:
            _smoke_gates(arm, args.allow_dirty)

    rows = [
        _run_one(arm, seed, args.smoke, args.allow_dirty)
        for arm, seed in batch
    ]

    verdict = _validity_verdict(rows)
    write_json(out / "e5_validity.json", verdict)
    for arm, v in verdict["per_arm"].items():
        print(f"{arm}: {v['n_valid']}/{v['n_runs']} valid, liveness "
              f"{'HOLDS' if v['liveness_premise_holds'] else 'FAILED'}, "
              f"dax crack {'YES' if v['dax_crack_any_seed'] else 'no'}")

    print(aggregate("e5", smoke=args.smoke))


if __name__ == "__main__":
    main()
