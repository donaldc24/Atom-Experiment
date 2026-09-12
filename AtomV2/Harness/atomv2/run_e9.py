"""E9 batch driver: Navier-Stokes-inspired vortex crystallization.

The analogy is deliberately narrow. A completed, competent three-step model
is a fixed teacher. Students have one routing decision per token. A25 learns
from scratch, A26 copies the teacher then receives task loss only, and A27
adds a smoothly ramped boundary-state/logit force. The screen therefore
separates an easy one-step world, warm-start inheritance, and genuinely
teacher-forced concentration.
"""
from __future__ import annotations

import argparse

from . import registered as R
from .aggregate import aggregate
from .analyze import analyze
from .config import E9_ARMS, config_for_arm, run_dir_for
from .crystallization import find_teacher_run
from .e7_audit import run_audit
from .panel import run_all_panels
from .train import train_run
from .utils import RESULTS_DIR, read_json, write_json, write_sha256sums

STAGES = ("screen", "replicate")


def _teacher_metrics(seed: int) -> tuple[dict, str]:
    cfg = config_for_arm("A26", seed)
    run_dir = find_teacher_run(cfg)
    metrics = read_json(run_dir / "metrics.json")
    if metrics["acc_seen_hard"] < R.E9_TEACHER_SEEN_MIN:
        raise SystemExit(
            f"E9 teacher gate failed: {run_dir} seen="
            f"{metrics['acc_seen_hard']:.4f} < {R.E9_TEACHER_SEEN_MIN:.2f}")
    return metrics, str(run_dir)


def _run_one(arm: str, seed: int, smoke: bool, allow_dirty: bool) -> dict:
    cfg = config_for_arm(arm, seed, smoke=smoke)
    run_dir = run_dir_for(cfg)
    complete = ((run_dir / "metrics.json").exists()
                and (run_dir / "SHA256SUMS").exists())
    if complete:
        print(f"skip (complete): {run_dir}")
        return read_json(run_dir / "metrics.json")
    print(f"=== {arm} (mode="
          f"{'scratch' if not cfg.crystal_warm_start else 'forced' if cfg.lambda_crystal_state else 'warm-unforced'}, "
          f"micro_steps={cfg.micro_steps}, steps={cfg.total_steps}) "
          f"seed={seed} -> {run_dir}")
    if (run_dir / "checkpoints" / "final.pt").exists():
        print("    training already complete; resuming at panel")
    else:
        train_run(cfg, allow_dirty=allow_dirty)
    run_all_panels(run_dir)
    analyze(run_dir)
    if not smoke:
        run_audit(run_dir)
    metrics = analyze(run_dir)
    write_sha256sums(run_dir)
    print(
        f"    seen={metrics['acc_seen_hard']:.4f} "
        f"L1={metrics['acc_unseen_L1_hard']:.4f} "
        f"one-route={metrics.get('e9_one_route_per_token')} "
        f"state-E2-span={metrics.get('e9_state_energy_span_ratio'):.6f} "
        f"grad-core={metrics.get('e9_grad_atom_peak_fraction_within_atoms_final'):.4f}")
    return metrics


def screen_verdict(rows: list[dict], teacher: dict, teacher_run: str) -> dict:
    by_arm = {r["arm"]: r for r in rows}
    required = set(E9_ARMS)
    if set(by_arm) != required:
        raise ValueError(f"screen needs {sorted(required)}, got {sorted(by_arm)}")
    teacher_l1 = teacher["acc_unseen_L1_hard"]

    def score(arm: str) -> dict:
        m = by_arm[arm]
        l1_retention = (m["acc_unseen_L1_hard"] / teacher_l1
                        if teacher_l1 > 0 else None)
        checks = {
            "seen_healthy": m["acc_seen_hard"] >= R.E9_STUDENT_SEEN_MIN,
            "L1_retained": (l1_retention is not None
                            and l1_retention >= R.E9_L1_RETENTION_MIN),
            "one_route_per_token": bool(m.get("e9_one_route_per_token")),
        }
        return {
            "seen": m["acc_seen_hard"],
            "L1": m["acc_unseen_L1_hard"],
            "seen_retention": (m["acc_seen_hard"]
                               / teacher["acc_seen_hard"]),
            "L1_retention": l1_retention,
            "checks": checks,
            "passes": all(checks.values()),
        }

    scored = {arm: score(arm) for arm in E9_ARMS}
    seen_margin = scored["A27"]["seen"] - scored["A26"]["seen"]
    l1_margin = scored["A27"]["L1"] - scored["A26"]["L1"]
    material_help = (seen_margin >= R.E9_TREATMENT_MARGIN
                     or l1_margin >= R.E9_TREATMENT_MARGIN)
    if scored["A25"]["passes"]:
        outcome, winner = "ONE_STEP_BASE_LEARNS", "A25"
    elif scored["A26"]["passes"]:
        outcome, winner = "WARM_START_SUFFICIENT", "A26"
    elif scored["A27"]["passes"]:
        outcome, winner = "CONTROLLED_CRYSTALLIZATION", "A27"
    elif material_help:
        outcome, winner = "FORCING_HELPS_BUT_INCOMPLETE", None
    else:
        outcome, winner = "NO_ONE_STEP_RESCUE", None
    return {
        "outcome": outcome,
        "registered_winner": winner,
        "teacher": {
            "run_dir": teacher_run,
            "arm": teacher["arm"], "seed": teacher["seed"],
            "seen": teacher["acc_seen_hard"],
            "L1": teacher["acc_unseen_L1_hard"],
            "micro_steps": 3,
        },
        "arms": scored,
        "forced_vs_warm_unforced": {
            "seen_margin": seen_margin,
            "L1_margin": l1_margin,
            "registered_material_margin": R.E9_TREATMENT_MARGIN,
            "material_help": material_help,
        },
        "thresholds": {
            "teacher_seen_min": R.E9_TEACHER_SEEN_MIN,
            "student_seen_min": R.E9_STUDENT_SEEN_MIN,
            "L1_retention_min": R.E9_L1_RETENTION_MIN,
        },
        "interpretation_guard": (
            "A positive A27 result establishes controlled teacher-forced "
            "program compression, not spontaneous SGD concentration and not "
            "a theorem transfer from Navier-Stokes."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run E9 (vortex crystallization, 3 -> 1 routes/token)")
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    if args.plan:
        print("teacher gate : paired e0/A0-free final, seen >= "
              f"{R.E9_TEACHER_SEEN_MIN}")
        print("stage screen : A25 scratch, A26 warm/unforced, A27 "
              "warm/teacher-forced; seed 1")
        print("success      : one route/token, seen >= "
              f"{R.E9_STUDENT_SEEN_MIN}, L1 retention >= "
              f"{R.E9_L1_RETENTION_MIN}")
        print("replicate    : registered screen winner, seeds 0/2")
        return

    if args.smoke:
        batch = [(arm, 0) for arm in E9_ARMS]
    elif args.stage == "screen":
        batch = [(arm, R.E9_SCREEN_SEED) for arm in E9_ARMS]
    elif args.stage == "replicate":
        if not args.arms or len(args.arms) != 1 or args.arms[0] not in E9_ARMS:
            raise SystemExit(
                f"--stage replicate requires --arms <one of {E9_ARMS}>")
        seeds = args.seeds if args.seeds is not None else [0, 2]
        batch = [(args.arms[0], seed) for seed in seeds]
    else:
        raise SystemExit("choose --stage {screen,replicate}, --smoke, or --plan")
    if args.arms and args.stage != "replicate":
        raise SystemExit("--arms is only valid for --stage replicate")

    # Gate the external source before spending on any student, including the
    # scratch control, so the screen's retention denominator is legitimate.
    teacher_seed = batch[0][1]
    teacher, teacher_run = _teacher_metrics(teacher_seed)
    print(f"teacher gate: {teacher_run} seen={teacher['acc_seen_hard']:.4f} "
          f"L1={teacher['acc_unseen_L1_hard']:.4f}")

    rows = [_run_one(arm, seed, args.smoke, args.allow_dirty)
            for arm, seed in batch]
    out = RESULTS_DIR / ("smoke_e9" if args.smoke else "e9")
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke or args.stage == "screen":
        verdict = screen_verdict(rows, teacher, teacher_run)
        write_json(out / "e9_screen_verdict.json", verdict)
        print(f"screen verdict: {verdict['outcome']}"
              + (f"; replicate {verdict['registered_winner']} seeds 0/2"
                 if verdict["registered_winner"] else "; stop"))
    try:
        print(aggregate("e9", smoke=args.smoke))
    except SystemExit as exc:
        print(f"(aggregate deferred: {exc})")


if __name__ == "__main__":
    main()
