"""Experiment gates T1, T2, T6. These require training runs and are NOT part of CI.

    python tests/test_gates.py --t2          # determinism, reduced scale (~2 min)
    python tests/test_gates.py --t2 --full   # determinism at the real config (~40 min)
    python tests/test_gates.py --t6 runs/A0_0_abc1234
    python tests/test_gates.py --t1          # oracle ceiling, reads existing A0 runs
    python tests/test_gates.py --t3          # leakage, reads existing A4 runs
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e1.analyze import analyze                          # noqa: E402
from e1.config import config_for_arm                    # noqa: E402
from e1.train import train                              # noqa: E402
from e1.utils import read_json, sha256_file, write_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"

T1_THRESHOLD = 0.99
T3_CHANCE_PLUS = 0.02


def t1_oracle_ceiling() -> bool:
    """T1, substituted form (DECISIONS.md D18).

    The gate asks whether a fully composing solution is reachable in this
    architecture. That is measured directly by teacher-forced composition on the
    oracle; A0's end-to-end `acc_unseen` is reported alongside as an observation.
    """
    runs = sorted(RUNS_DIR.glob("A0_*"))
    if not runs:
        print("T1: no A0 runs found")
        return False
    ok = True
    for r in runs:
        m = read_json(r / "metrics.json")
        acc = m["M7_acc_teacher_forced"]
        status = "PASS" if acc >= T1_THRESHOLD else "FAIL"
        if acc < T1_THRESHOLD:
            ok = False
        print(f"T1 {status}  {r.name}  teacher_forced={acc:.4f}  "
              f"(end-to-end acc_unseen={m['M1_acc_unseen']:.4f}, drift="
              f"{m['M7_drift_step1']:.4f})")
    return ok


def t3_leakage() -> bool:
    """A4 (shuffled labels) must sit at chance. Exact-match chance on V^L is 1e-8."""
    runs = sorted(RUNS_DIR.glob("A4_*"))
    if not runs:
        print("T3: no A4 runs found")
        return False
    ok = True
    for r in runs:
        m = read_json(r / "metrics.json")
        for key in ("M1_acc_seen", "M1_acc_unseen"):
            acc = m[key]
            if acc > T3_CHANCE_PLUS:
                ok = False
                print(f"T3 FAIL  {r.name}  {key}={acc:.4f} > chance+{T3_CHANCE_PLUS}")
            else:
                print(f"T3 PASS  {r.name}  {key}={acc:.4f}")
    return ok


def t2_determinism(full: bool = False, arm: str = "A2", seed: int = 0) -> bool:
    """Two runs of the same run_id must produce byte-identical predictions."""
    tmp = Path(tempfile.mkdtemp(prefix="e1_t2_"))
    try:
        digests = []
        for rep in ("a", "b"):
            cfg = config_for_arm(arm, seed)
            if not full:
                cfg.examples_per_train_task = 150
                cfg.examples_per_eval_task = 80
                cfg.n_probe_examples = 80
                cfg.epochs = 2
            out = tmp / rep
            train(cfg, out)
            digests.append(sha256_file(out / "artifacts" / "predictions_unseen.jsonl"))
            print(f"T2 run {rep}: {digests[-1]}")
        ok = digests[0] == digests[1]
        print("T2", "PASS" if ok else "FAIL",
              f"({'full' if full else 'reduced'} scale, arm {arm})")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t6_metric_reproducibility(run_dir: Path) -> bool:
    """analyze.py twice on the same run directory must produce identical metrics.json."""
    tmp = Path(tempfile.mkdtemp(prefix="e1_t6_"))
    try:
        digests = []
        for rep in ("a", "b"):
            m = analyze(run_dir)
            p = tmp / f"metrics_{rep}.json"
            write_json(p, m)
            digests.append(sha256_file(p))
        ok = digests[0] == digests[1]
        print("T6", "PASS" if ok else "FAIL", f"{run_dir.name} {digests[0]}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", action="store_true")
    ap.add_argument("--t2", action="store_true")
    ap.add_argument("--t3", action="store_true")
    ap.add_argument("--t6", default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--arm", default="A2")
    args = ap.parse_args()

    results = {}
    if args.t1:
        results["T1"] = t1_oracle_ceiling()
    if args.t2:
        results["T2"] = t2_determinism(full=args.full, arm=args.arm)
    if args.t3:
        results["T3"] = t3_leakage()
    if args.t6:
        results["T6"] = t6_metric_reproducibility(Path(args.t6))
    if not results:
        ap.print_help()
        return 0
    print("\n" + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
