"""Regenerate added artifacts for runs that completed before the artifact existed.

Loads each run's final.pt and re-emits only the missing matrices, then re-runs
analyze.py. No retraining, so the model is bit-identical to the one that produced
the run's predictions -- the backfilled artifact is exactly what the run would have
written had the code existed at the time.

    python -m e1.backfill              # all runs missing an artifact
    python -m e1.backfill --force      # recompute for every run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .analyze import analyze
from .config import Config
from .data import build_bundle, load_split
from .evaluate import (atom_residual_norms, compute_alignment_tensor,
                       compute_code_diagnostics, compute_code_spread,
                       compute_state_alignment)
from .model import AtomNet
from .train import RUNS_DIR
from .utils import set_threads, write_json, write_sha256sums

ARTIFACTS = ("alignment_tensor.npy", "atom_residual_norms.npy",
             "state_alignment_err.npy", "state_alignment_acc.npy")


def backfill_run(run_dir: Path, force: bool = False) -> bool:
    art = run_dir / "artifacts"
    ckpt = run_dir / "checkpoints" / "final.pt"
    if not ckpt.exists():
        print(f"  {run_dir.name}: no final.pt, skipping")
        return False
    if not force and all((art / a).exists() for a in ARTIFACTS):
        return False

    ck = torch.load(ckpt, weights_only=False)
    cfg = Config.from_dict(ck["config"])
    set_threads(cfg.num_threads, cfg.num_interop_threads)
    model = AtomNet(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    bundle = build_bundle(cfg, load_split())
    np.save(art / "alignment_tensor.npy",
            compute_alignment_tensor(model, bundle.probe_inputs, cfg))
    np.save(art / "atom_residual_norms.npy",
            atom_residual_norms(model, bundle.probe_inputs))
    err, sacc = compute_state_alignment(model, bundle.probe_inputs, cfg)
    np.save(art / "state_alignment_err.npy", err)
    np.save(art / "state_alignment_acc.npy", sacc)
    for name, tasks in (("seen", bundle.seen_heldout), ("unseen", bundle.unseen)):
        cd = compute_code_diagnostics(model, tasks, cfg)
        np.save(art / f"code_residual_{name}.npy", cd["residual_hard"])
        np.save(art / f"code_residual_soft_{name}.npy", cd["residual_soft"])
        np.save(art / f"code_entropy_{name}.npy", cd["entropy"])
        write_json(art / f"code_diversity_{name}.json", cd["per_task_diversity"])
    write_json(art / "code_spread.json",
               compute_code_spread(model, bundle.probe_inputs))

    metrics = analyze(run_dir)
    write_json(run_dir / "metrics.json", metrics)
    write_sha256sums(run_dir)
    print(f"  {run_dir.name}: backfilled  "
          f"M3_align={metrics['M3_align']:.3f} -> "
          f"M3_align_best_s={metrics['M3_align_best_s']:.3f} "
          f"closed_err={metrics['M3_closed_map_error']:.3f} "
          f"state_align={metrics['M3_state_align']:.3f}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS_DIR))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reanalyze", action="store_true",
                    help="re-run analyze.py on every complete run without recomputing "
                         "artifacts; needed when a long-lived run_all process held a "
                         "stale analyze module in memory")
    args = ap.parse_args()

    n = 0
    for run_dir in sorted(Path(args.runs).glob("*")):
        if not (run_dir / "metrics.json").exists():
            continue
        n += int(backfill_run(run_dir, args.force))
        if args.reanalyze:
            metrics = analyze(run_dir)
            write_json(run_dir / "metrics.json", metrics)
            write_sha256sums(run_dir)
    print(f"backfilled {n} run(s)"
          + (f"; re-analyzed all complete runs" if args.reanalyze else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
