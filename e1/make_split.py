"""Generate and freeze splits/pairs_split.json. Run once, then commit the result."""

from __future__ import annotations

import sys

from .data import SPLIT_PATH, build_split, split_hash, verify_split
from .primitives import PRIMITIVE_NAMES
from .utils import write_json


def main(force: bool = False) -> int:
    if SPLIT_PATH.exists() and not force:
        print(f"{SPLIT_PATH} already exists (frozen). Pass --force to regenerate.")
        print(f"sha256: {split_hash()}")
        return 0

    split = build_split()
    problems = verify_split(split)
    if problems:
        print("SPLIT CONSTRAINTS VIOLATED:")
        for p in problems:
            print("  -", p)
        return 1

    write_json(SPLIT_PATH, split)
    print(f"wrote {SPLIT_PATH}")
    print(f"sha256: {split_hash()}")
    print(f"train pairs:   {split['n_train_pairs']}")
    print(f"heldout pairs: {split['n_heldout_pairs']}")
    print(f"distinct pair functions: {split['n_distinct_pair_functions']} / 64")
    print(f"classes forced into train (== a length-1 task): {split['n_classes_forced_train']}")
    print("\nposition-1 counts in train:")
    for p, name in enumerate(PRIMITIVE_NAMES):
        print(f"  {name:12s} pos1={split['position1_counts_train'][p]:2d} "
              f"pos2={split['position2_counts_train'][p]:2d}")
    print("\nheld-out pairs:")
    for a, b in split["heldout_pairs"]:
        print(f"  {PRIMITIVE_NAMES[a]:12s} -> {PRIMITIVE_NAMES[b]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
