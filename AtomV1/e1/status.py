"""Live progress of an E1 battery. Safe to run at any time; reads only, never writes.

    python -m e1.status          # one snapshot
    python -m e1.status --watch  # refresh every 30s until you Ctrl-C
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .config import ARMS, GENERATIONS, seeds_for
    from .train import RUNS_DIR
except ImportError:
    # Allows `python status.py` from inside e1/, or `python e1/status.py` from the
    # repo root, not just `python -m e1.status`. Status is the one module people
    # reach for casually, so it should not care where it is invoked from.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from e1.config import ARMS, GENERATIONS, seeds_for
    from e1.train import RUNS_DIR


def _read_log(path: Path) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def live_processes():
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for p in psutil.process_iter(["pid", "cmdline", "create_time"]):
        cl = p.info["cmdline"]
        if not cl:
            continue
        joined = " ".join(str(c) for c in cl)
        if "e1.train" in joined and "--arm" in joined and "bash" not in str(cl[0]).lower():
            parts = joined.split()
            try:
                arm = parts[parts.index("--arm") + 1]
                seed = parts[parts.index("--seed") + 1]
            except (ValueError, IndexError):
                continue
            out.append((arm, seed, (time.time() - p.info["create_time"]) / 60))
    return out


def snapshot(runs_dir: Path = RUNS_DIR, generation: str = "v2") -> None:
    # Recursive: runs/<generation>/<run_id> from D40 on, flat before it. Archived
    # batteries are excluded -- progress means progress on the CURRENT batch, and a
    # finished archive would otherwise report it as permanently complete (D41).
    from .aggregate import is_archived
    done = sorted(m for m in runs_dir.rglob("metrics.json")
                  if not is_archived(m.parent, runs_dir))
    # Per-arm denominator is seeds x splits, which differs by generation (D42): v1 is
    # 5 x 1, v2 is 3 x 3. A fixed len(SEEDS) would have reported v2 as 300% complete.
    spec = GENERATIONS[generation]
    per_arm = len(spec["seeds"]) * len(spec["split_seeds"])
    total_planned = len(ARMS) * per_arm
    print(f"=== E1 battery [{generation}]: {len(done)}/{total_planned} "
          f"runs complete ===\n")

    by_arm = {a: 0 for a in ARMS}
    for m in done:
        try:
            by_arm[json.loads(m.read_text())["arm"]] += 1
        except Exception:
            pass
    print("  " + "   ".join(f"{a}:{n}/{per_arm}" for a, n in by_arm.items()))

    running = live_processes()
    if running:
        print("\n--- in flight ---")
    for arm, seed, mins in running:
        print(f"  {arm} seed {seed}   elapsed {mins:.1f} min")
        # Find its run dir: has a train log but no metrics yet.
        for d in sorted(runs_dir.rglob(f"{arm}_{seed}_*")):
            if (d / "metrics.json").exists():
                continue
            rows = _read_log(d / "train_log.jsonl")
            epochs = [r for r in rows if r.get("event") == "epoch_end"]
            if not epochs:
                print("     (starting up)")
                continue
            last = epochs[-1]
            phases = {}
            for r in epochs:
                phases[r["phase"]] = phases.get(r["phase"], 0) + 1
            phase_str = "  ".join(f"{p}:{n}" for p, n in phases.items())
            print(f"     phase {last['phase']}  epoch {last['epoch']}  "
                  f"train_acc {last['epoch_train_acc']:.4f}")
            if len(phases) > 1:
                print(f"     phases so far: {phase_str}")
    if not running:
        print("\n  (no training process running)")

    if done:
        print("\n--- last 6 completed ---")
        rows = []
        for m in done:
            try:
                d = json.loads(m.read_text())
                rows.append((m.parent.stat().st_mtime, d))
            except Exception:
                pass
        for _, d in sorted(rows)[-6:]:
            print(f"  {d['arm']} seed{d['seed']}: "
                  f"unseen={d['M1_acc_unseen']:.4f} seen={d['M1_acc_seen']:.4f} "
                  f"closed_err={d.get('M3_closed_map_error', float('nan')):.3f} "
                  f"teacher={d['M7_acc_teacher_forced']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--generation", default="v2",
                    help="which generation's plan to count progress against")
    args = ap.parse_args()
    while True:
        snapshot(generation=args.generation)
        if not args.watch:
            return 0
        print(f"\n(refreshing every {args.interval}s -- Ctrl-C to stop)\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
