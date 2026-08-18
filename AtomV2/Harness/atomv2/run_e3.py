"""E3 batch driver (H1-Experiment3.md): atom-sandbox battery.

Registered order:
  1. A6 zero-sandbox equivalence gate - the completed E1b A6 runs may serve
     as the control ONLY if the current harness's lambda_sandbox_* = 0 path
     is bit-identical to the E1b implementation. Reuses run_e2's mechanical
     replay (the A6 config carries both state_noise_sigma = 0 and
     lambda_sandbox_* = 0, so one replay certifies both inert paths).
     Failure => rerun A6 under this harness (hard stop).
  2. A11/A12/A13 smoke runs + implementation gates (sandbox telemetry
     present and finite, usage weights in [0,1], liveness p_max invariant,
     clean-eval determinism with no sandbox involvement). Structural gates
     (gradient boundary, same-atom-update, stream isolation, detached usage
     weights) are enforced by tests/test_e3.py.
  3. A11/A12/A13 x seeds 0/1/2, 20k steps, full certified panel + analyze.

An implementation-gate failure invalidates the run. Optimization failure
under a correctly implemented sandbox is a result, not an invalid run.
"""
from __future__ import annotations

import argparse
import math

import torch

from . import registered as R
from .aggregate import aggregate
from .analyze import analyze
from .config import E3_ARMS, config_for_arm, run_dir_for
from .panel import run_all_panels
from .run_e2 import a6_equivalence_gate
from .train import train_run
from .utils import (RESULTS_DIR, read_json, stream_rng, write_json,
                    write_sha256sums)


def _smoke_gates(arm: str, allow_dirty: bool) -> None:
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
    sts = sorted((run_dir / "sandbox_telemetry").glob("step*.json"))
    if not sts:
        raise SystemExit(f"smoke {arm}: no sandbox telemetry written")
    for p in sts:
        st = read_json(p)
        for key, val in (("standalone read", st["standalone"]["read_mean"]),
                         ("standalone cycle", st["standalone"]["cycle_mean"]),
                         ("closure read", st["closure"]["read_mean"]),
                         ("closure cycle", st["closure"]["cycle_mean"])):
            if not math.isfinite(val):
                raise SystemExit(f"smoke {arm}: non-finite {key} ({p})")
        if not all(0.0 <= w <= 1.0 for w in st["usage"]["weights"]):
            raise SystemExit(f"smoke {arm}: usage weight outside [0,1] ({p})")
    cal = read_json(run_dir / "init_calibration.json")
    if "loss_sandbox_valid_init" not in cal:
        raise SystemExit(f"smoke {arm}: init calibration lacks sandbox scale")
    # Clean-eval determinism: repeated clean hard forward is bit-identical
    # (the sandbox exists only inside the training loss; nothing of it can
    # appear in an eval forward).
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
    print(f"    smoke {arm}: {len(lv)} liveness evals, {len(sts)} sandbox "
          "records, finiteness + weight-range + determinism gates ok")


def _run_one(arm: str, seed: int, smoke: bool, allow_dirty: bool) -> dict:
    cfg = config_for_arm(arm, seed, smoke=smoke)
    run_dir = run_dir_for(cfg)
    if (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists():
        print(f"skip (complete): {run_dir}")
        return read_json(run_dir / "metrics.json")
    print(f"=== {arm} (lambda_valid={cfg.lambda_sandbox_valid}, "
          f"lambda_unique={cfg.lambda_sandbox_unique}) seed={seed} "
          f"-> {run_dir}")
    if (run_dir / "checkpoints" / "final.pt").exists():
        print("    training already complete; resuming at panel")
    else:
        train_run(cfg, allow_dirty=allow_dirty)
    run_all_panels(run_dir)
    metrics = analyze(run_dir)
    write_sha256sums(run_dir)
    print(f"    seen={metrics['acc_seen_hard']:.4f} "
          f"L1={metrics['acc_unseen_L1_hard']:.4f} "
          f"L2={metrics['acc_unseen_L2_hard']:.4f} "
          f"L3={metrics['acc_unseen_L3_hard']:.4f} "
          f"repairL3={metrics.get('canon_repair_acc_L3')} "
          f"weighted_atoms={metrics.get('e3_usage_weighted_atoms_final')} "
          f"valid={metrics.get('e1b_run_valid')}")
    return metrics


def _validity_verdict(rows: list[dict]) -> dict:
    per_run, per_arm = {}, {}
    for m in rows:
        per_run[f"{m['arm']}_seed{m['seed']}"] = {
            "liveness_valid": bool(m.get("e1b_run_valid")),
            "deafness_violated": bool(m.get("e1b_deafness_violated")),
            "valid": bool(m.get("e1b_run_valid")),
        }
    for arm in E3_ARMS:
        seeds = [v for k, v in per_run.items() if k.startswith(f"{arm}_")]
        per_arm[arm] = {
            "n_runs": len(seeds),
            "n_valid": sum(v["valid"] for v in seeds),
            "liveness_premise_holds": bool(seeds) and not any(
                v["deafness_violated"] for v in seeds),
        }
    return {"per_run": per_run, "per_arm": per_arm,
            "note": "the sandbox carries no gate of its own; validity is the "
                    "inherited E1b liveness rule. Sandbox losses failing to "
                    "optimize is a RESULT, not an invalid run."}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E3 battery "
                                             "(atom sandbox)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(R.SEEDS))
    ap.add_argument("--arms", nargs="*", default=list(E3_ARMS))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-equivalence", action="store_true")
    ap.add_argument("--skip-free-smoke", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    bad = [a for a in args.arms if a not in E3_ARMS]
    if bad:
        raise SystemExit(f"unknown E3 arms {bad}; battery arms are {E3_ARMS}. "
                         "A6 is the reused E1b control, never re-run here "
                         "unless the equivalence gate fails.")

    batch = [(arm, seed) for arm in args.arms for seed in args.seeds]
    if args.plan:
        print("A6 equivalence gate (replay 50 steps vs e1b A6 s0 train_log; "
              "certifies the zero-sandbox path)")
        for arm in args.arms:
            print(f"smoke {arm} seed=0  (implementation gates)")
        for arm, seed in batch:
            print(f"{arm} (lambda_valid={R.E3_LAMBDA_VALID[arm]}, "
                  f"lambda_unique={R.E3_LAMBDA_UNIQUE[arm]}) seed={seed}")
        return

    out = RESULTS_DIR / ("smoke_e3" if args.smoke else "e3")
    out.mkdir(parents=True, exist_ok=True)

    if not args.smoke and not args.skip_equivalence:
        a6_equivalence_gate(out, filename="e3_a6_equivalence.json")
    if not args.smoke and not args.skip_free_smoke:
        for arm in args.arms:
            _smoke_gates(arm, args.allow_dirty)

    rows = [
        _run_one(arm, seed, args.smoke, args.allow_dirty)
        for arm, seed in batch
    ]

    verdict = _validity_verdict(rows)
    write_json(out / "e3_validity.json", verdict)
    for arm, v in verdict["per_arm"].items():
        print(f"{arm}: {v['n_valid']}/{v['n_runs']} valid, liveness "
              f"{'HOLDS' if v['liveness_premise_holds'] else 'FAILED'}")

    print(aggregate("e3", smoke=args.smoke))


if __name__ == "__main__":
    main()
