"""Staged E11 driver and mechanical registered verdicts."""
from __future__ import annotations

import argparse

from . import registered as R
from .e11 import config_for_e11, run_dir_for_e11, train_e11
from .utils import RESULTS_DIR, read_json, write_json


def primary_score(metrics: dict) -> dict:
    values = {
        "singleton_min": metrics["acc_singleton_min_hard"],
        "seen_pair": metrics["acc_seen_pair_hard"],
        "L1": metrics["acc_unseen_L1_hard"],
        "L2": metrics["acc_unseen_L2_hard"],
        "L3": metrics["acc_unseen_L3_hard"],
        **{f"len{length}": metrics[f"acc_extrap_len{length}_hard"]
           for length in R.E11_EXTRAP_LENGTHS},
    }
    checks = {name: values[name] >= threshold
              for name, threshold in R.E11_BASE_GATES.items()}
    checks["phase1_gate"] = bool(metrics["phase1_pass"])
    checks["phase2_ran"] = bool(metrics["phase2_ran"])
    checks["conservation"] = (metrics["mass_conservation_relative_max"]
                              <= R.E11_CONSERVATION_MAX)
    checks["interface"] = metrics["interface_prediction_agreement"] == 1.0
    return {"values": values, "checks": checks,
            "passes": all(checks.values())}


def _long_mean(metrics: dict) -> float:
    return sum(metrics[f"acc_extrap_len{length}_hard"]
               for length in R.E11_EXTRAP_LENGTHS) / len(
                   R.E11_EXTRAP_LENGTHS)


def screen_verdict(by_arm: dict[str, dict]) -> dict:
    if not set(R.E11_SCREEN_ARMS).issubset(by_arm):
        raise ValueError(f"screen requires {list(R.E11_SCREEN_ARMS)}")
    primary = {arm: primary_score(metrics)
               for arm, metrics in by_arm.items()}
    direct_long = _long_mean(by_arm["A32"])
    reynolds_long = _long_mean(by_arm["A33"])
    reynolds_match = reynolds_long >= direct_long - R.E11_REYNOLDS_MATCH_TOL
    reynolds_pass = primary["A33"]["passes"] and reynolds_match

    viscosity = None
    if reynolds_pass and "A34" not in by_arm:
        outcome, winner = "VISCOSITY_REQUIRED", None
    elif reynolds_pass:
        inviscid = by_arm["A33"]
        viscous = by_arm["A34"]
        clean_long_ok = (_long_mean(viscous)
                         >= _long_mean(inviscid) - R.E11_VISCOSITY_CLEAN_TOL)
        robust_gain = (viscous["robustness_composite"]
                       - inviscid["robustness_composite"])
        viscosity_pass = (primary["A34"]["passes"] and clean_long_ok
                           and robust_gain >= R.E11_VISCOSITY_ROBUST_MARGIN)
        viscosity = {
            "A33_clean_long_mean": _long_mean(inviscid),
            "A34_clean_long_mean": _long_mean(viscous),
            "clean_tolerance": R.E11_VISCOSITY_CLEAN_TOL,
            "clean_tolerance_passes": clean_long_ok,
            "A33_robustness_composite": inviscid["robustness_composite"],
            "A34_robustness_composite": viscous["robustness_composite"],
            "robustness_gain": robust_gain,
            "required_gain": R.E11_VISCOSITY_ROBUST_MARGIN,
            "passes": viscosity_pass,
        }
        if viscosity_pass:
            outcome, winner = "VISCOUS_CANONICAL_REYNOLDS", "A34"
        else:
            outcome, winner = "INVISCID_CANONICAL_REYNOLDS", "A33"
    elif primary["A32"]["passes"]:
        outcome, winner = "DIRECT_CANONICAL_EXCHANGE_ONLY", None
    elif not by_arm["A32"]["phase1_pass"] and \
            not by_arm["A33"]["phase1_pass"]:
        outcome, winner = "SINGLETON_CRYSTALLIZATION_FAILED", None
    else:
        outcome, winner = "NO_REUSABLE_EXCHANGE", None

    return {
        "outcome": outcome,
        "registered_winner": winner,
        "viscosity_eligible": reynolds_pass,
        "primary": primary,
        "reynolds_contrast": {
            "A32_long_mean": direct_long,
            "A33_long_mean": reynolds_long,
            "tolerance": R.E11_REYNOLDS_MATCH_TOL,
            "matched": reynolds_match,
            "passes": reynolds_pass,
        },
        "viscosity_contrast": viscosity,
        "interpretation_guard": (
            "This is a discrete categorical neural operator, not a "
            "Navier-Stokes solver or theorem transfer."),
    }


def replication_verdict(arm: str, metrics: list[dict]) -> dict:
    if arm not in ("A33", "A34") or len(metrics) != 3:
        raise ValueError("replication requires A33/A34 at exactly three seeds")
    scored = {str(m["seed"]): primary_score(m) for m in metrics}
    conservation = {str(m["seed"]):
                    m["mass_conservation_relative_max"]
                    <= R.E11_CONSERVATION_MAX for m in metrics}
    interface = {str(m["seed"]):
                 m["interface_prediction_agreement"] == 1.0 for m in metrics}
    pass_count = sum(v["passes"] for v in scored.values())
    return {"arm": arm, "seeds": sorted(m["seed"] for m in metrics),
            "primary_pass_count": pass_count, "primary_pass_required": 2,
            "all_conservation_pass": all(conservation.values()),
            "all_interface_pass": all(interface.values()),
            "per_seed_primary": scored,
            "per_seed_conservation": conservation,
            "per_seed_interface": interface,
            "claim_holds": (pass_count >= 2 and all(conservation.values())
                            and all(interface.values()))}


def _summary_markdown(rows: list[dict]) -> str:
    head = ("| arm | seed | P1 min | singleton final | seen pair | L1 | L2 | "
            "L3 | len3 | len4 | len8 | len16 | noise | INT4 | mass |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|\n")
    body = ""
    for m in sorted(rows, key=lambda x: (x["arm"], x["seed"])):
        body += (f"| {m['arm']} | {m['seed']} | {m['phase1_singleton_min']:.4f} "
                 f"| {m['acc_singleton_min_hard']:.4f} | "
                 f"{m['acc_seen_pair_hard']:.4f} | "
                 f"{m['acc_unseen_L1_hard']:.4f} | "
                 f"{m['acc_unseen_L2_hard']:.4f} | "
                 f"{m['acc_unseen_L3_hard']:.4f} | "
                 f"{m['acc_extrap_len3_hard']:.4f} | "
                 f"{m['acc_extrap_len4_hard']:.4f} | "
                 f"{m['acc_extrap_len8_hard']:.4f} | "
                 f"{m['acc_extrap_len16_hard']:.4f} | "
                 f"{m['acc_pair_noisy_hard']:.4f} | "
                 f"{m['acc_pair_int4_hard']:.4f} | "
                 f"{m['mass_conservation_relative_max']:.2e} |\n")
    return "# E11 Canonical Conservative Exchange Network\n\n" + head + body


def _run_one(arm: str, seed: int, smoke: bool,
             allow_dirty: bool) -> tuple:
    cfg = config_for_e11(arm, seed, smoke=smoke)
    expected = run_dir_for_e11(cfg)
    if (expected / "metrics.json").exists() and \
            (expected / "SHA256SUMS").exists():
        print(f"=== {arm} seed={seed}: complete, reusing {expected}")
        run_dir = expected
    else:
        print(f"=== {arm} seed={seed} transport={cfg.transport} "
              f"nu={cfg.viscosity} phases={cfg.phase1_steps}+{cfg.phase2_steps} "
              f"-> {expected}")
        run_dir = train_e11(cfg, allow_dirty=allow_dirty)
    metrics = read_json(run_dir / "metrics.json")
    print(f"    P1min={metrics['phase1_singleton_min']:.4f} "
          f"Smin={metrics['acc_singleton_min_hard']:.4f} "
          f"pair={metrics['acc_seen_pair_hard']:.4f} "
          f"L1/L2/L3={metrics['acc_unseen_L1_hard']:.4f}/"
          f"{metrics['acc_unseen_L2_hard']:.4f}/"
          f"{metrics['acc_unseen_L3_hard']:.4f} "
          f"long={metrics['acc_extrap_len3_hard']:.4f}/"
          f"{metrics['acc_extrap_len4_hard']:.4f}/"
          f"{metrics['acc_extrap_len8_hard']:.4f}/"
          f"{metrics['acc_extrap_len16_hard']:.4f} "
          f"mass={metrics['mass_conservation_relative_max']:.2e}")
    return run_dir, metrics


def _merge_results(result_dir, current_rows: list[dict]) -> None:
    per_run_path = result_dir / "per_run.json"
    prior = read_json(per_run_path) if per_run_path.exists() else []
    merged = {(m["arm"], m["seed"]): m for m in prior}
    merged.update({(m["arm"], m["seed"]): m for m in current_rows})
    rows = list(merged.values())
    write_json(per_run_path, rows)
    (result_dir / "summary.md").write_text(
        _summary_markdown(rows), encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run E11 conservative exchange")
    ap.add_argument("--stage", choices=("smoke", "screen", "replicate"),
                    required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if args.stage == "smoke":
        rows = [_run_one(arm, 0, True, args.allow_dirty)[1]
                for arm in R.E11_ARMS]
        result_dir = RESULTS_DIR / "smoke_e11"
        result_dir.mkdir(parents=True, exist_ok=True)
        verdict = screen_verdict({m["arm"]: m for m in rows})
        write_json(result_dir / "e11_smoke_verdict.json", verdict)
        _merge_results(result_dir, rows)
        print(f"smoke verdict (non-evidence): {verdict['outcome']}")
        print(result_dir)
        return

    result_dir = RESULTS_DIR / "e11"
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "screen":
        rows = [_run_one(arm, R.E11_SCREEN_SEED, False,
                         args.allow_dirty)[1]
                for arm in R.E11_SCREEN_ARMS]
        by_arm = {m["arm"]: m for m in rows}
        verdict = screen_verdict(by_arm)
        if verdict["viscosity_eligible"]:
            viscous = _run_one("A34", R.E11_SCREEN_SEED, False,
                               args.allow_dirty)[1]
            rows.append(viscous)
            by_arm["A34"] = viscous
            verdict = screen_verdict(by_arm)
        write_json(result_dir / "e11_screen_verdict.json", verdict)
        _merge_results(result_dir, rows)
        print(f"screen verdict: {verdict['outcome']}"
              + (f"; replicate {verdict['registered_winner']}"
                 if verdict["registered_winner"] else "; stop"))
        print(result_dir)
        return

    verdict_path = result_dir / "e11_screen_verdict.json"
    if not verdict_path.exists():
        raise SystemExit(f"missing screen verdict {verdict_path}")
    screen = read_json(verdict_path)
    arm = screen.get("registered_winner")
    if arm not in ("A33", "A34"):
        raise SystemExit("screen has no structured winner to replicate")
    rows = [_run_one(arm, seed, False, args.allow_dirty)[1]
            for seed in R.E11_REPLICATE_SEEDS]
    screen_path = run_dir_for_e11(config_for_e11(
        arm, R.E11_SCREEN_SEED, smoke=False)) / "metrics.json"
    if not screen_path.exists():
        raise SystemExit(f"missing seed-1 winner run {screen_path}")
    all_metrics = [read_json(screen_path)] + rows
    verdict = replication_verdict(arm, all_metrics)
    write_json(result_dir / "e11_replication_verdict.json", verdict)
    _merge_results(result_dir, all_metrics)
    print("replication verdict: "
          + ("CLAIM HOLDS" if verdict["claim_holds"] else "CLAIM FAILS"))
    print(result_dir)


if __name__ == "__main__":
    main()
