"""Generate and freeze a generation's splits. Run once per generation, then commit.

    python -m e1.make_split                        # v1: reports the frozen split
    python -m e1.make_split --generation v2        # v2: writes one file per split seed

v1's split is **immutable**. Its sha256 is recorded in the `split_ref.json` of every
committed run, so regenerating it would orphan the whole E1 battery from the report
that cites it. `--force` is refused for v1 (D40).
"""

from __future__ import annotations

import argparse
import sys

from .config import GENERATIONS
from .data import build_split, split_hash, split_path_for, verify_split
from .primitives import primitive_names
from .utils import write_json


def describe(split: dict, path, pset: str) -> None:
    names = primitive_names(pset)
    print(f"wrote {path}")
    print(f"sha256: {split_hash(path)}")
    print(f"primitive set: {pset}   split seed: {split['split_seed']}")
    print(f"train pairs:   {split['n_train_pairs']}")
    print(f"heldout pairs: {split['n_heldout_pairs']}")
    print(f"distinct pair functions: {split['n_distinct_pair_functions']} / 64")
    print(f"distinct functions among held-out: {split['n_distinct_functions_heldout']}")
    print(f"classes forced into train (== a length-1 task): "
          f"{split['n_classes_forced_train']}")
    print("\ninformative coverage in train (D14 requires >=2 for non-identity):")
    for p, name in enumerate(names):
        print(f"  {name:12s} pos1={split['informative_position1_counts_train'][p]:2d} "
              f"pos2={split['informative_position2_counts_train'][p]:2d}")
    print("\nheld-out pairs:")
    for a, b in split["heldout_pairs"]:
        print(f"  {names[a]:12s} -> {names[b]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", default="v1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.generation not in GENERATIONS:
        print(f"unknown generation {args.generation!r}; known: {sorted(GENERATIONS)}",
              file=sys.stderr)
        return 2

    spec = GENERATIONS[args.generation]
    pset = spec["primitive_set"]
    written = 0

    for split_seed in spec["split_seeds"]:
        path = split_path_for(args.generation, split_seed)
        if path.exists():
            if args.generation == "v1":
                # Immutable by policy, not by convention. Every committed run's
                # split_ref.json pins this file's sha256.
                print(f"{path} is FROZEN (v1); refusing to regenerate.")
                print(f"sha256: {split_hash(path)}")
                continue
            if not args.force:
                print(f"{path} already exists. Pass --force to regenerate.")
                print(f"sha256: {split_hash(path)}")
                continue

        split = build_split(seed=split_seed, pset=pset)
        problems = verify_split(split, pset)
        if problems:
            print(f"SPLIT CONSTRAINTS VIOLATED for seed {split_seed}:")
            for p in problems:
                print("  -", p)
            return 1

        write_json(path, split)
        describe(split, path, pset)
        print()
        written += 1

    print(f"generation {args.generation}: {written} split(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
