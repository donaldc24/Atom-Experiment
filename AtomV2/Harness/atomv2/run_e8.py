"""E8 batch driver (H1-Experiment8.md): capacity rescue - one route per
token, more computation per atom.

Registered order (stages mirror the registered run order; no stage runs
until its predecessor's verdict allows it):

  0. Zero-depth gate, hard stop: the A18 config (atom_layers = 1 by
     construction) replayed under THIS harness reproduces the completed E7
     A18 seed-1 step-1/step-50 records under the frozen gate ladder.
  1. --stage screen    : A23 s1 and A24 s1 (health = seen >= 0.80), then
                         A22 s1 - the 40k budget control, which runs its
                         full budget regardless and never stops early
                         (poor optimization is a result).
  2. --stage replicate --arms <winner> : seeds 0/2. If both depth arms pass
     health, the SHALLOWER (A23) is the registered winner - minimal
     sufficient capacity; the deeper seed-1 stays on record.

Every result-bearing run receives the full certified panel + the E7 audit
(same artifact, same ICR definitions - numbers compare directly across
E7/E8). Implementation failure invalidates a run; optimization failure is a
result.
"""
from __future__ import annotations

import argparse
import json

from . import registered as R
from .aggregate import aggregate
from .analyze import analyze
from .config import E8_ARMS, config_for_arm, run_dir_for
from .e7_audit import run_audit
from .panel import run_all_panels
from .run_e2 import _find_run_dir
from .run_e4 import _replay_first_steps
from .train import train_run
from .utils import RESULTS_DIR, read_json, write_json, write_sha256sums

STAGES = ("screen", "replicate")


def _a18_zero_depth_gate(out_dir,
                         filename: str = "e8_a18_equivalence.json") -> None:
    """Gate 0: atom_layers = 1 reproduces the completed E7 A18 s1 records."""
    cfg = config_for_arm(R.E8_BASE_ARM, R.E8_SCREEN_SEED)
    assert cfg.atom_layers == 1 and cfg.micro_steps == 1
    ref_dir = _find_run_dir("e7", R.E8_BASE_ARM, R.E8_SCREEN_SEED)
    logged = {}
    with open(ref_dir / "train_log.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "step" and rec["step"] in (1, 50):
                logged[rec["step"]] = rec
    if set(logged) != {1, 50}:
        raise SystemExit(f"A18 zero-depth gate: steps 1/50 not found in "
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
    record = {"reference_run": str(ref_dir), "arm": R.E8_BASE_ARM,
              "compared_steps": [1, 50], "compared_keys": keys,
              "bit_identical": not mismatches, "mismatches": mismatches}
    write_json(out_dir / filename, record)
    if mismatches:
        raise SystemExit(
            f"A18 ZERO-DEPTH GATE FAILED: {mismatches[:2]}... The depth-"
            "parameterized atom does not reproduce the certified A18 path; "
            "nothing proceeds until it does (same environment - the frozen "
            "ladder's strict rung applies).")
    print("A18 zero-depth gate: bit-identical (steps 1 and 50)")


def _run_one(arm: str, seed: int, smoke: bool, allow_dirty: bool) -> dict:
    cfg = config_for_arm(arm, seed, smoke=smoke)
    run_dir = run_dir_for(cfg)
    if (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists():
        print(f"skip (complete): {run_dir}")
        return read_json(run_dir / "metrics.json")
    print(f"=== {arm} (atom_layers={cfg.atom_layers}, "
          f"steps={cfg.total_steps}) seed={seed} -> {run_dir}")
    if (run_dir / "checkpoints" / "final.pt").exists():
        print("    training already complete; resuming at panel")
    else:
        train_run(cfg, allow_dirty=allow_dirty)
    run_all_panels(run_dir)
    analyze(run_dir)            # first pass: panel-derived metrics
    if not smoke:
        run_audit(run_dir)      # E7 audit (ICR etc.), then fold into metrics
    metrics = analyze(run_dir)
    write_sha256sums(run_dir)

    def _f(x, spec=".4f"):
        return "-" if x is None else f"{x:{spec}}"
    print(f"    valid={metrics.get('e1b_run_valid')} "
          f"seen={metrics['acc_seen_hard']:.4f} "
          f"healthy={metrics.get('e7_healthy')} | "
          f"mapped_atoms={metrics.get('e7_n_mapped_atoms')} "
          f"raw={_f(metrics.get('e7_raw_chain_acc'))} "
          f"canon={_f(metrics.get('e7_canon_chain_acc'))} "
          f"ICR={_f(metrics.get('e7_interface_closure_ratio'))} "
          f"({metrics.get('e7_closure_band')}) | "
          f"L1={metrics['acc_unseen_L1_hard']:.4f} "
          f"L3={metrics['acc_unseen_L3_hard']:.4f}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E8 battery "
                                             "(capacity rescue)")
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="replicate stage: the registered winner arm")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-equivalence", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    if args.plan:
        print("gate 0: A18 zero-depth replay vs e7 A18 s1 (bit-identical, "
              "same env)")
        print("stage screen    : A23 s1 (2-layer), A24 s1 (3-layer), then "
              "A22 s1 (1-layer, 40k budget control - full budget, no early "
              "stop)")
        print("stage replicate : --arms <winner> seeds 0/2; if both depth "
              "arms healthy, A23 (shallower) is the registered winner")
        return
    if args.stage is None:
        raise SystemExit("choose --stage {screen,replicate}; stages are "
                         "gated by the registered run order and are not "
                         "run in one shot")
    if args.arms and args.stage != "replicate":
        raise SystemExit("--arms is only for --stage replicate")

    out = RESULTS_DIR / ("smoke_e8" if args.smoke else "e8")
    out.mkdir(parents=True, exist_ok=True)

    if not args.smoke and not args.skip_equivalence:
        _a18_zero_depth_gate(out)

    if args.stage == "screen":
        batch = [("A23", R.E8_SCREEN_SEED), ("A24", R.E8_SCREEN_SEED),
                 ("A22", R.E8_SCREEN_SEED)]
    else:
        if not args.arms or len(args.arms) != 1 \
                or args.arms[0] not in E8_ARMS:
            raise SystemExit("--stage replicate requires --arms <one of "
                             f"{E8_ARMS}>")
        seeds = args.seeds if args.seeds is not None else [0, 2]
        batch = [(args.arms[0], s) for s in seeds]

    rows = []
    for arm, seed in batch:
        m = _run_one(arm, seed, args.smoke, args.allow_dirty)
        rows.append(m)
        if args.stage == "screen" and arm in ("A23", "A24"):
            verdict = ("HEALTHY" if m["acc_seen_hard"]
                       >= R.E8_HEALTH_SEEN_MIN else "UNHEALTHY")
            print(f"    {arm} screen: seen={m['acc_seen_hard']:.4f} -> "
                  f"{verdict} (gate {R.E8_HEALTH_SEEN_MIN})")
        if args.stage == "screen" and arm == "A22":
            inside = m["acc_seen_hard"] < R.E8_BUDGET_CONTROL_MAX
            verdict = ("CONFIRMED" if inside
                       else "REFUTED - E7 verdict needs a dated amendment")
            print(f"    A22 budget control: seen={m['acc_seen_hard']:.4f} "
                  f"at 40k -> registered prediction "
                  f"(<{R.E8_BUDGET_CONTROL_MAX}) {verdict}")

    if args.stage == "screen":
        healthy = {m["arm"]: m["acc_seen_hard"] >= R.E8_HEALTH_SEEN_MIN
                   for m in rows if m["arm"] in ("A23", "A24")}
        winner = ("A23" if healthy.get("A23")
                  else "A24" if healthy.get("A24") else None)
        write_json(out / "e8_screen_verdict.json", {
            "per_arm_seen": {m["arm"]: m["acc_seen_hard"] for m in rows},
            "healthy": healthy,
            "registered_winner": winner,
            "rule": "shallower wins if both healthy (minimal sufficient "
                    "capacity); replicate winner seeds 0/2",
        })
        if winner:
            print(f"screen verdict: registered winner = {winner}; next: "
                  f"--stage replicate --arms {winner}")
        else:
            print("screen verdict: STILL CAPACITY-LIMITED - no depth arm "
                  "reached health; E8 stops per the registered bins")
    try:
        print(aggregate("e8", smoke=args.smoke))
    except SystemExit as e:
        print(f"(aggregate deferred: {e})")


if __name__ == "__main__":
    main()
