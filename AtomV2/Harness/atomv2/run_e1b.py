"""E1b batch driver (H1-E1bExperiment.md): anti-saturation router battery.

Registered run order:
  1. A5-oracle seed 0 - architectural regression check under the normalized
     routing stack. Circuit breaker: approximately perfect accuracy and
     closed-map error near the certified oracle's 0.015, or nothing proceeds.
     Forced routing bypasses Gumbel selection, so this does NOT certify
     free-router liveness - the smoke tests and telemetry do that.
  2. A5/A6/A7 free-routing smoke tests - short runs through at least two
     scheduled liveness evaluations. Implementation/liveness checks only;
     cannot be used to tune registered constants.
  3. A5, A6, A7 free routing x seeds 0/1/2, 20k steps, full panel + analyze.

Validity is judged per seed-run (implementation invariant + deafness rule,
evaluated by analyze.py from liveness artifacts). If any seed in an arm fails
the deafness rule, the arm has failed E1b's verified-liveness premise: its
telemetry and valid seed-runs are reported descriptively, and the verdict file
records that no arm-level compositionality claim may be made. Failed seeds are
never silently replaced.
"""
from __future__ import annotations

import argparse

from . import registered as R
from .aggregate import aggregate
from .analyze import analyze
from .config import E1B_ARMS, E1B_ORACLE_ARM, config_for_arm, run_dir_for
from .panel import run_all_panels
from .train import train_run
from .utils import RESULTS_DIR, read_json, write_json, write_sha256sums


def _run_one(arm: str, seed: int, smoke: bool, allow_dirty: bool) -> dict:
    cfg = config_for_arm(arm, seed, smoke=smoke)
    run_dir = run_dir_for(cfg)
    if (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists():
        print(f"skip (complete): {run_dir}")
        return read_json(run_dir / "metrics.json")
    print(f"=== {arm} (sigma={cfg.router_sigma:.6f}) seed={seed} -> {run_dir}")
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
          f"closed_map={metrics['closed_map_seen']:.4f} "
          f"valid={metrics.get('e1b_run_valid')} "
          f"deaf={metrics.get('e1b_deafness_violated')} "
          f"pmax_ok={metrics.get('e1b_pmax_invariant_ok')}")
    return metrics


def _oracle_regression_check(metrics: dict, out_dir) -> None:
    acc_ok = metrics["acc_seen_hard"] >= R.E1B_ORACLE_ACC_MIN \
        and metrics["acc_unseen_L1_hard"] >= R.E1B_ORACLE_ACC_MIN
    cmap_ok = metrics["closed_map_seen"] <= R.E1B_ORACLE_CLOSED_MAP_MAX
    record = {
        "arm": E1B_ORACLE_ARM,
        "acc_seen_hard": metrics["acc_seen_hard"],
        "acc_unseen_L1_hard": metrics["acc_unseen_L1_hard"],
        "closed_map_seen": metrics["closed_map_seen"],
        "thresholds": {"acc_min": R.E1B_ORACLE_ACC_MIN,
                       "closed_map_max": R.E1B_ORACLE_CLOSED_MAP_MAX},
        "passed": acc_ok and cmap_ok,
        "note": "broad architectural regression check only; forced routing "
                "bypasses Gumbel selection and does not certify free-router "
                "liveness",
    }
    write_json(out_dir / "e1b_oracle_check.json", record)
    if not record["passed"]:
        raise SystemExit(
            f"CIRCUIT BREAKER: A5-oracle regression check failed "
            f"(seen={metrics['acc_seen_hard']:.4f}, "
            f"L1={metrics['acc_unseen_L1_hard']:.4f}, "
            f"closed_map={metrics['closed_map_seen']:.4f}). The normalized "
            "routing stack broke certified oracle behavior; debug the "
            "harness, not the hypothesis.")


def _smoke_liveness_check(arm: str, allow_dirty: bool) -> None:
    """One short free run through >= 2 scheduled liveness evals; verify the
    telemetry exists and the implementation invariant holds in it."""
    cfg = config_for_arm(arm, 0, smoke=True)
    run_dir = run_dir_for(cfg)
    if not (run_dir / "checkpoints" / "final.pt").exists():
        print(f"=== smoke {arm} -> {run_dir}")
        train_run(cfg, allow_dirty=allow_dirty)
    lv_files = sorted((run_dir / "liveness").glob("step*.json"))
    if len(lv_files) < 2:
        raise SystemExit(f"smoke {arm}: fewer than 2 liveness evaluations "
                         f"({len(lv_files)}) - implementation broken")
    for p in lv_files:
        rec = read_json(p)
        if not rec["base_geometry"]["pmax_invariant_ok"]:
            raise SystemExit(
                f"smoke {arm}: implementation invariant violated at "
                f"step {rec['step']}: max base prob "
                f"{rec['base_geometry']['max_base_prob_observed']:.8f} > "
                f"p_max {rec['base_geometry']['p_max_registered']:.8f} + tol. "
                "The implementation is wrong; fix before any full run.")
    print(f"    smoke {arm}: {len(lv_files)} liveness evals, invariant ok")


def _validity_verdict(rows: list[dict]) -> dict:
    per_run, per_arm = {}, {}
    for m in rows:
        key = f"{m['arm']}_seed{m['seed']}"
        per_run[key] = {
            "valid": bool(m.get("e1b_run_valid")),
            "deafness_violated": bool(m.get("e1b_deafness_violated")),
            "deafness_step": m.get("e1b_deafness_step"),
            "pmax_invariant_ok": bool(m.get("e1b_pmax_invariant_ok")),
            "norm_growth_flagged": bool(m.get("e1b_norm_growth_flagged")),
        }
    for arm in E1B_ARMS:
        seeds = [v for k, v in per_run.items() if k.startswith(f"{arm}_")]
        deaf = any(v["deafness_violated"] for v in seeds)
        per_arm[arm] = {
            "n_runs": len(seeds),
            "n_valid": sum(v["valid"] for v in seeds),
            "liveness_premise_holds": bool(seeds) and not deaf,
            "note": (None if seeds and not deaf else
                     "a seed failed the deafness rule (or no runs): report "
                     "telemetry and valid seed-runs descriptively; no "
                     "arm-level compositionality claim; never silently "
                     "replace the failed seed"),
        }
    return {"per_run": per_run, "per_arm": per_arm,
            "rules": {"deafness_grad_norm": R.E1B_DEAF_GRAD_NORM,
                      "deafness_ratio": R.E1B_DEAF_RATIO,
                      "deafness_window": list(R.E1B_DEAF_WINDOW),
                      "pmax_tolerance": R.E1B_PMAX_TOL}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E1b battery "
                                             "(anti-saturation router)")
    ap.add_argument("--smoke", action="store_true",
                    help="run the whole battery at smoke scale (pipeline check)")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(R.SEEDS))
    ap.add_argument("--arms", nargs="*", default=list(E1B_ARMS))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true",
                    help="skip the A5-oracle regression check (already passed)")
    ap.add_argument("--skip-free-smoke", action="store_true",
                    help="skip the registered free-arm smoke/liveness checks")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    bad = [a for a in args.arms if a not in E1B_ARMS]
    if bad:
        raise SystemExit(f"unknown E1b arms {bad}; battery arms are {E1B_ARMS}. "
                         f"{E1B_ORACLE_ARM} runs automatically as the "
                         "regression check and is not a battery arm.")

    batch = [(arm, seed) for arm in args.arms for seed in args.seeds]
    if args.plan:
        print(f"{E1B_ORACLE_ARM} seed=0  (regression check)")
        for arm in args.arms:
            print(f"smoke {arm} seed=0  (liveness check)")
        for arm, seed in batch:
            print(f"{arm} (sigma={R.E1B_SIGMA[arm]:.6f}) seed={seed}")
        return

    out = RESULTS_DIR / ("smoke_e1b" if args.smoke else "e1b")
    out.mkdir(parents=True, exist_ok=True)

    # 1. Oracle regression check (full scale unless the battery is smoke).
    if not args.skip_oracle:
        metrics = _run_one(E1B_ORACLE_ARM, 0, args.smoke, args.allow_dirty)
        if not args.smoke:
            _oracle_regression_check(metrics, out)

    # 2. Free-arm smoke/liveness checks (skipped when the battery itself is
    #    smoke - it IS the smoke).
    if not args.smoke and not args.skip_free_smoke:
        for arm in args.arms:
            _smoke_liveness_check(arm, args.allow_dirty)

    # 3. The battery.
    rows = []
    for arm, seed in batch:
        rows.append(_run_one(arm, seed, args.smoke, args.allow_dirty))

    verdict = _validity_verdict(rows)
    write_json(out / "e1b_validity.json", verdict)
    for arm, v in verdict["per_arm"].items():
        print(f"{arm}: {v['n_valid']}/{v['n_runs']} valid, liveness premise "
              f"{'HOLDS' if v['liveness_premise_holds'] else 'FAILED'}")

    print(aggregate("e1b", smoke=args.smoke))


if __name__ == "__main__":
    main()
