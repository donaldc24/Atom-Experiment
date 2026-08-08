"""Figures for results/E1_REPORT.md.

    python -m e1.report_figures
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from .primitives import PRIMITIVE_NAMES   # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "figures"

# A0 supervision ladder (DECISIONS.md D18). These configurations were run
# sequentially during harness validation and superseded one another, so the values
# are transcribed from D18 rather than read from runs/.
A0_LADDER = [
    ("forced routing\nonly", 0.004, None),
    ("+ intermediate\ndecode", 0.236, None),
    ("+ state consist.\n(w=1)", 0.428, 0.512),
    ("+ normalised\n(w=10)", 0.685, 0.108),
    ("+ 80 epochs\n(converged)", 0.899, 0.044),
    ("+ w=40\n(final)", 0.938, 0.034),
]

ARM_ORDER = ["A0", "A1", "A2", "A3", "A3b", "A4"]
ARM_LABEL = {
    "A0": "A0\noracle", "A1": "A1\nnaive joint", "A2": "A2\nprotected",
    "A3": "A3\nseq. frozen", "A3b": "A3b\nassigned+frozen", "A4": "A4\nshuffled",
}


def _load_arms():
    out = {}
    for f in glob.glob(os.path.join(REPO_ROOT, "runs", "*", "metrics.json")):
        m = json.load(open(f))
        if m.get("rung"):
            continue
        out.setdefault(m["arm"], []).append(m)
    return out


def fig_a0_ladder():
    fig, ax1 = plt.subplots(figsize=(8.4, 4.2))
    xs = np.arange(len(A0_LADDER))
    acc = [a for _, a, _ in A0_LADDER]
    ax1.plot(xs, acc, "o-", color="#2b6cb0", lw=2, ms=7, label="acc_unseen (end-to-end)")
    ax1.axhline(0.9989, ls=":", c="#2b6cb0", lw=1.4,
                label="teacher-forced (never below 0.9947)")
    ax1.set_ylabel("exact-match accuracy on unseen", color="#2b6cb0")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")

    ax2 = ax1.twinx()
    dx = [i for i, (_, _, d) in enumerate(A0_LADDER) if d is not None]
    dy = [d for _, _, d in A0_LADDER if d is not None]
    ax2.plot(dx, dy, "s--", color="#c05621", lw=2, ms=7, label="drift ||h1-enc(y1)||/||enc(y1)||")
    ax2.set_ylabel("relative drift", color="#c05621")
    ax2.set_ylim(0, 0.6)
    ax2.tick_params(axis="y", labelcolor="#c05621")

    ax1.set_xticks(xs)
    ax1.set_xticklabels([n for n, _, _ in A0_LADDER], fontsize=8)
    ax1.set_title("A0 supervision ladder: architecture, optimizer, data and split fixed;\n"
                  "only the supervision signal changes", fontsize=11)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "a0_ladder.png", dpi=160)
    plt.close(fig)


def fig_closed_map_by_arm(arms):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    present = [a for a in ARM_ORDER if a in arms]

    for i, a in enumerate(present):
        vals = [m["M3_closed_map_error_matched"] for m in arms[a]]
        ax.bar(i, np.mean(vals), 0.6, color="#4c72b0", alpha=0.75)
        ax.scatter(np.full(len(vals), i), vals, s=22, c="k", zorder=3)
    ax.axhline(0.15, ls="--", c="green", lw=1.2, label="E1b RECOVERS <= 0.15")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([ARM_LABEL[a] for a in present], fontsize=8)
    ax.set_ylabel("closed-map error (matched assignment)")
    ax.set_title("Are atoms closed maps on the code?\n(lower is better; per-seed points)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    for i, a in enumerate(present):
        vals = [m["M3_closed_map_coverage"] for m in arms[a]]
        ax2.bar(i, np.mean(vals), 0.6, color="#dd8452", alpha=0.8)
        ax2.scatter(np.full(len(vals), i), vals, s=22, c="k", zorder=3)
    ax2.axhline(8, ls="--", c="green", lw=1.2, label="full coverage 8/8")
    ax2.set_xticks(range(len(present)))
    ax2.set_xticklabels([ARM_LABEL[a] for a in present], fontsize=8)
    ax2.set_ylabel("distinct primitives covered")
    ax2.set_ylim(0, 8.6)
    ax2.set_title("Coverage: a dead library collapses to 1/8\n(D24 -- must be read with the error)",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "closed_map_by_arm.png", dpi=160)
    plt.close(fig)


def fig_alignment_a0_vs_a1(arms):
    """The single figure that carries the diagnostic-failure point (report section 4)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
    for ax, (arm, sub) in zip(axes, [("A0", "oracle"), ("A1", "naive joint")]):
        run = sorted(arms[arm], key=lambda m: m["seed"])[0]
        rid = [d for d in glob.glob(os.path.join(REPO_ROOT, "runs", f"{arm}_*"))
               if f"_{run['seed']}_" in os.path.basename(d)][0]
        A = np.load(os.path.join(rid, "artifacts", "alignment_matrix.npy"))
        im = ax.imshow(A, vmin=0, vmax=1, cmap="magma", aspect="auto")
        ax.set_title(f"{arm} {sub} (seed {run['seed']})\n"
                     f"M3_align={run['M3_align']:.3f}  "
                     f"closed-map={run['M3_closed_map_error_matched']:.2f}  "
                     f"unseen={run['M1_acc_unseen']:.3f}", fontsize=9)
        ax.set_xticks(range(len(PRIMITIVE_NAMES)))
        ax.set_xticklabels(PRIMITIVE_NAMES, rotation=90, fontsize=7)
        ax.set_yticks(range(A.shape[0]))
        ax.set_yticklabels([f"atom {i}" for i in range(A.shape[0])] if ax is axes[0] else [],
                           fontsize=7)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.03, label="standalone exact-match")
    fig.suptitle("M3 alignment: a clean permutation (A0) vs a scattered library (A1)",
                 fontsize=11)
    fig.savefig(OUT / "alignment_a0_vs_a1.png", dpi=160)
    plt.close(fig)


def fig_e1b_ladder():
    """E1b ladder. Skipped silently until E1b runs exist."""
    cells = {}
    for f in glob.glob(os.path.join(REPO_ROOT, "runs", "e1b_*", "metrics.json")):
        m = json.load(open(f))
        cells.setdefault((m.get("rung"), m.get("code_consistency_weight", 0.0)), []).append(m)
    if not cells:
        print("  (no E1b runs yet -- e1b_ladder.png skipped)")
        return
    order = [("R0", 0.0), ("R1", 0.0), ("R2", 1.0), ("R2", 10.0), ("R2", 40.0), ("R3", 0.0)]
    order = [c for c in order if c in cells]
    labels = [f"{r}\nw={w:g}" if r in ("R2", "R3") else r for r, w in order]

    fig, ax1 = plt.subplots(figsize=(8.6, 4.2))
    xs = np.arange(len(order))
    err = [np.mean([m["M3_closed_map_error_matched"] for m in cells[c]]) for c in order]
    ax1.plot(xs, err, "o-", color="#4c72b0", lw=2, ms=7, label="closed-map error (matched)")
    ax1.axhline(0.15, ls="--", c="green", lw=1.2, label="RECOVERS <= 0.15")
    ax1.set_ylabel("closed-map error / code spread")


    spread = [np.mean([m.get("code_spread", np.nan) for m in cells[c]]) for c in order]
    ax1.plot(xs, spread, "^-", color="#6b46c1", lw=1.8, ms=7, alpha=0.8,
             label="code spread (representation retained)")

    ax2 = ax1.twinx()
    acc = [np.mean([m["M1_acc_unseen"] for m in cells[c]]) for c in order]
    ax2.plot(xs, acc, "s--", color="#c05621", lw=2, ms=7, label="acc_unseen")
    ax2.axhline(0.50, ls=":", c="#c05621", lw=1.2, label="RECOVERS >= 0.50")
    ax2.set_ylabel("acc_unseen", color="#c05621")
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(axis="y", labelcolor="#c05621")

    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{l}\n(n={len(cells[c])})" for l, c in zip(labels, order)], fontsize=8)
    ax1.set_title("E1b manifold ladder: does a self-supervised on-manifold constraint\n"
                  "substitute for the oracle's ground truth?", fontsize=11)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "e1b_ladder.png", dpi=160)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = _load_arms()
    fig_a0_ladder()
    fig_closed_map_by_arm(arms)
    fig_alignment_a0_vs_a1(arms)
    fig_e1b_ladder()
    print("figures written to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
