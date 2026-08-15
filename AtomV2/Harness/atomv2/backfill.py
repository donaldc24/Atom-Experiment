"""Record the harness-source identity of runs that predate amendment R10.

Runs written before the `harness_source_sha256` key existed recorded only the
older provenance values (git sha, and for dirty trees a snapshot fingerprint
that also covered incidental files like editor config). Those cannot be
compared against the new content fingerprint, so aggregation refuses to mix the
two schemes rather than guessing.

This tool writes the missing key, computed from the harness source AS COMMITTED
at an explicit revision, and marks it as backfilled together with the reason -
a backfilled value is never silently indistinguishable from one a run recorded
about itself. It touches no artifact that any metric is derived from: env.json
provenance fields only, then SHA256SUMS is regenerated.

Only run this when you can state why that revision is the code that produced
the runs. The intended evidence is a deterministic replay: re-run one of the
seeds from the revision and diff its train_log against the original.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .utils import (RUNS_DIR, harness_source_sha256_at, read_json, write_json,
                    write_sha256sums)


def backfill_run(run_dir: Path, source_sha: str, rev: str, reason: str,
                 force: bool = False) -> str:
    env_path = Path(run_dir) / "env.json"
    env = read_json(env_path)
    existing = env.get("harness_source_sha256")
    if existing is not None and not force:
        return "already recorded" if existing == source_sha else (
            f"CONFLICT: has {existing[:16]}, refusing to overwrite")
    env["harness_source_sha256"] = source_sha
    env["harness_source_provenance"] = {
        "recorded_by": "backfill",
        "revision": rev,
        "reason": reason,
    }
    write_json(env_path, env)
    write_sha256sums(Path(run_dir))
    return "backfilled"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill harness_source_sha256 into completed runs (R10)")
    ap.add_argument("--rev", required=True,
                    help="git revision whose harness source produced the runs")
    ap.add_argument("--experiment", default="e0", choices=["e0", "e1"])
    ap.add_argument("--reason", required=True,
                    help="why this revision is the code that produced them")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing, differing value")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source_sha = harness_source_sha256_at(args.rev)
    print(f"harness source @ {args.rev}: {source_sha}")
    for mpath in sorted((RUNS_DIR / args.experiment).rglob("env.json")):
        run_dir = mpath.parent
        if args.dry_run:
            env = read_json(mpath)
            state = env.get("harness_source_sha256", "(missing)")
            print(f"  {run_dir.name}: {state if state == '(missing)' else state[:16]}")
            continue
        status = backfill_run(run_dir, source_sha, args.rev, args.reason,
                              force=args.force)
        print(f"  {run_dir.name}: {status}")


if __name__ == "__main__":
    main()
