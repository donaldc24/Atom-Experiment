"""Aggregate per-run metrics.json into the E1 deliverables.

Emits:
    results/summary.csv    one row per arm, mean +/- std of every metric
    results/summary.md     the same as a readable table, plus the verdict
    results/per_run.csv    one row per (arm, seed)
    results/plots/         alignment heatmaps, ablation CV distributions,
                           recombination gap chart with per-seed points overlaid

Usage:
    python -m e1.aggregate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARMS, THRESHOLDS
from .primitives import PRIMITIVE_NAMES
from .utils import read_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"

HEADLINE = [
    "M1_acc_seen", "M1_acc_unseen", "M1_gap",
    "M2_cv", "M2_dead", "M2_cv_multitask_only",
    "M3_align", "M3_purity", "M3_align_1step",
    "M3_align_best_s", "M3_closed_map_error",
    "M3_closed_map_coverage", "M3_closed_map_error_matched",
    "M4_routing_acc_seen", "M4_routing_acc_unseen",
    "M5_entropy", "M5_dead",
    "M6_soft_hard_gap",
    "M7_drift_step1", "M7_acc_teacher_forced", "M7_recovery",
    "acc_singleton", "acc_ablate_all",
]

ARM_LABELS = {
    "A0": "A0 oracle",
    "A1": "A1 naive joint",
    "A2": "A2 protected joint",
    "A3": "A3 sequential frozen",
    "A3b": "A3b sequential frozen (assigned)",
    "A4": "A4 shuffled-label",
}


ARCHIVE_PREFIX = "archive"


def is_archived(run_dir: Path, runs_dir: Path) -> bool:
    """True when `run_dir` sits under an `archive*` subtree of `runs_dir`.

    Archived runs are a frozen record of a finished battery on a specific machine.
    They are skipped by default so they cannot be silently pooled with new runs --
    D39 forbids a comparison spanning two hosts, and archives are the most likely way
    that would happen by accident. Pointing `--runs` AT an archive still reads it,
    because then the relative path contains no archive component: opting in is
    explicit.
    """
    try:
        parts = run_dir.relative_to(runs_dir).parts
    except ValueError:
        return False
    return any(p.startswith(ARCHIVE_PREFIX) for p in parts)


def collect(runs_dir: Path = RUNS_DIR, generation: str = "v1",
            include_archived: bool = False) -> pd.DataFrame:
    """Every run of ONE generation, found at any depth under `runs_dir`.

    Discovery is recursive because runs are laid out `runs/<generation>/<run_id>/`
    from D40 onward, while the runs made before it sit flat at `runs/<run_id>/`.

    Generations are never mixed. v1 and v2 have different primitive sets, so their
    accuracies are not measurements of the same task family -- averaging them would
    produce a number that describes no experiment. A run whose config predates the
    generation field is v1 by definition, since v1 is what existed then.
    """
    rows = []
    hosts: dict[str, str] = {}
    for mpath in sorted(runs_dir.rglob("metrics.json")):
        run_dir = mpath.parent
        if is_archived(run_dir, runs_dir) and not include_archived:
            continue
        m = read_json(mpath)
        # E1b cells are built from config_for_arm("A1", ...) and keep arm="A1", so
        # without this they would be folded into the A1 arm and silently corrupt the
        # E1 battery table. E1b is aggregated separately. See D32.
        if m.get("rung"):
            continue
        cfg_path = run_dir / "config.json"
        run_gen = read_json(cfg_path).get("generation", "v1") if cfg_path.exists() else "v1"
        if run_gen != generation:
            continue
        # `run_path` is carried alongside `run_id` because a run's artifacts can no
        # longer be located by joining runs_dir with its name: runs sit at
        # runs/<generation>/<run_id> from D40 on. Reassembling the path flatly is the
        # same defect that broke test_gates' discovery -- see D45.
        row = {"run_id": run_dir.name, "run_path": str(run_dir),
               "generation": run_gen, "arm": m["arm"], "seed": m["seed"]}
        for k in HEADLINE:
            row[k] = m.get(k, np.nan)
        row["params_composer"] = m["params"]["composer"]
        row["params_atoms_total"] = m["params"]["atoms_total"]
        row["params_encoder"] = m["params"]["encoder"]
        row["params_decoder"] = m["params"]["decoder"]
        row["composer_over_atoms"] = m["params"]["composer_over_atoms"]
        env_path = run_dir / "env.json"
        if env_path.exists():
            hosts[run_dir.name] = read_json(env_path).get("hostname", "?")
        rows.append(row)
    if not rows:
        raise SystemExit(
            f"no generation-{generation} metrics.json found under {runs_dir}"
        )
    # D39: determinism is a within-platform guarantee. Two hosts in one table is a
    # comparison spanning machines, which no arm-vs-arm claim may rest on.
    distinct = sorted(set(hosts.values()))
    if len(distinct) > 1:
        byhost = {h: sorted(r for r, v in hosts.items() if v == h) for h in distinct}
        raise SystemExit(
            f"refusing to aggregate runs from {len(distinct)} hosts {distinct}: "
            f"determinism is a within-platform guarantee and D39 forbids a "
            f"comparison spanning machines. Runs per host: "
            + "; ".join(f"{h}={len(v)}" for h, v in byhost.items())
        )
    return pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in HEADLINE if c in df.columns]
    g = df.groupby("arm")[cols]
    out = pd.concat({"mean": g.mean(), "std": g.std(ddof=1)}, axis=1)
    out.columns = [f"{m}_{s}" for s, m in out.columns]
    out.insert(0, "n_seeds", df.groupby("arm").size())
    return out.reset_index()


def _cmp(op: str, value, threshold) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return value >= threshold if op == "ge" else value <= threshold


def verdict_for_arm(means: dict) -> tuple[str, dict]:
    """PASS requires all six pre-registered metrics. FAIL on any one is a fail."""
    per_metric = {}
    for name, rule in THRESHOLDS.items():
        v = means.get(name)
        if _cmp(rule["pass"][0], v, rule["pass"][1]):
            per_metric[name] = "PASS"
        elif _cmp(rule["fail"][0], v, rule["fail"][1]):
            per_metric[name] = "FAIL"
        else:
            per_metric[name] = "AMBIGUOUS"
    states = set(per_metric.values())
    if states == {"PASS"}:
        return "PASS", per_metric
    if "FAIL" in states:
        return "FAIL", per_metric
    return "AMBIGUOUS", per_metric


# FAIL(architectural) signature: the library is factorized (atoms are correct closed
# maps, alignment is a permutation) but residual composition loses the encoder
# manifold, so atoms are never asked to be closed maps on a stable code.
ARCH_TEACHER_MIN = 0.85    # teacher-forced composition still works this well ...
ARCH_ACTUAL_MAX = 0.50     # ... while actual composition is this bad
ARCH_ALIGN_MIN = 0.85      # ... and the atoms individually are correct


def architectural_failure(means: dict) -> bool:
    return (means.get("M7_acc_teacher_forced", 0) >= ARCH_TEACHER_MIN
            and means.get("M1_acc_unseen", 1) <= ARCH_ACTUAL_MAX
            and means.get("M3_align", 0) >= ARCH_ALIGN_MIN)


# FAIL(training-signal) signature (D22, pre-registered before A1/A2/A4 ran): the
# architecture and optimizer are both adequate -- the oracle proves it -- but no
# unsupervised arm makes its atoms closed maps, because nothing in the task loss
# asks them to. Mutually exclusive with FAIL(architectural) by construction.
SIGNAL_ORACLE_TEACHER_MIN = 0.99
SIGNAL_ORACLE_CLOSED_MAX = 0.10
SIGNAL_ARM_CLOSED_MIN = 0.30


def training_signal_failure(arm_means: dict, failing: list) -> bool:
    a0 = arm_means.get("A0", {})
    if a0.get("M7_acc_teacher_forced", 0) < SIGNAL_ORACLE_TEACHER_MIN:
        return False
    if a0.get("M3_closed_map_error", 1.0) > SIGNAL_ORACLE_CLOSED_MAX:
        return False
    if not failing:
        return False
    return all(
        arm_means.get(a, {}).get("M3_closed_map_error", 0) >= SIGNAL_ARM_CLOSED_MIN
        for a in failing
    )


def program_verdict(arm_verdicts: dict, oracle_ok: bool, arm_means: dict) -> str:
    a3 = arm_verdicts.get("A3")
    a1 = arm_verdicts.get("A1")
    a2 = arm_verdicts.get("A2")
    if not oracle_ok:
        return ("HARNESS NOT VALIDATED -- A0 did not reach >=99% on unseen (T1). "
                "No other arm's result is interpretable.")
    if a3 == "PASS":
        if a1 == "PASS" or a2 == "PASS":
            return "PASS -- H6 survives; atoms factorize under joint training as well."
        return ("FAIL(optimizer) -- H6 survives structurally (A3 passes) but joint "
                "gradient descent does not produce factorized atoms. Redirects the "
                "program toward non-gradient outer loops.")

    failing = [a for a, v in arm_verdicts.items()
               if v == "FAIL" and a not in ("A4", "A3b")]
    arch = [a for a in failing if architectural_failure(arm_means.get(a, {}))]
    if failing and len(arch) == len(failing):
        return (
            "FAIL(architectural) -- every failing arm shows the D12 signature: atoms are "
            "correct closed maps on the encoder manifold (teacher-forced composition "
            f">={ARCH_TEACHER_MIN:.2f}, alignment >={ARCH_ALIGN_MIN:.2f}) while actual "
            "composition fails, because residual composition does not preserve the "
            "manifold. This is a property of the composition operator, NOT a refutation "
            "of H6 -- do not report it as such. Untried fixes: LayerNorm after each atom "
            "application, drift augmentation during atom training, or a decode/re-encode "
            "bottleneck between steps.")

    if training_signal_failure(arm_means, failing):
        return (
            "FAIL(training-signal) -- a factorized, composing solution is REACHABLE "
            "here: the oracle reaches one over the SAME architecture and optimizer "
            f"(teacher-forced >={SIGNAL_ORACLE_TEACHER_MIN:.2f}, closed-map error "
            f"<={SIGNAL_ORACLE_CLOSED_MAX:.2f}). Every unsupervised arm fails by never "
            f"making its atoms closed maps (closed-map error >={SIGNAL_ARM_CLOSED_MIN:.2f}). "
            "The best-supported reading is that a training SIGNAL is missing -- nothing "
            "in the task loss requires intermediate states to stay on the encoder "
            "manifold. This is NOT FAIL(representational): H6 cannot be refuted by arms "
            "that fail while an oracle over the same architecture succeeds. NOTE: the "
            "oracle varies routing supervision, intermediate targets, state consistency, "
            "objective AND effective budget at once, so it establishes feasibility under "
            "privileged supervision, NOT that the optimizer and learned-routing "
            "architecture are adequate for DISCOVERY (E1_REPORT 6b). Next step is to "
            "supply the missing signal, then re-run.")

    if all(v == "FAIL" for a, v in arm_verdicts.items()
           if v is not None and a not in ("A4", "A3b")):
        return ("FAIL(representational) -- no arm passes, including the structurally "
                "protected one, the failure is not explained by manifold drift, and the "
                "oracle did not demonstrate a reachable factorized solution. "
                "H6 is refuted for this architecture. Do not proceed to E2.")
    if a1 == "PASS" and a3 != "PASS":
        return ("SUSPICIOUS -- only the naive joint arm passes. Check T3 and inspect the "
                "alignment matrix by hand.")
    return ("AMBIGUOUS -- iterate on the training procedure, not on the thresholds.")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, runs_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a for a in ARMS if a in set(df["arm"])]

    # -- alignment heatmaps, one representative seed (lowest available) per arm
    fig, axes = plt.subplots(1, len(arms), figsize=(2.9 * len(arms) + 1.2, 4.2),
                             squeeze=False, constrained_layout=True)
    for col, (ax, arm) in enumerate(zip(axes[0], arms)):
        sub = df[df["arm"] == arm].sort_values("seed")
        A = np.load(Path(sub.iloc[0]["run_path"]) / "artifacts" / "alignment_matrix.npy")
        im = ax.imshow(A, vmin=0, vmax=1, cmap="magma", aspect="auto")
        ax.set_title(f"{ARM_LABELS[arm]}\nseed {int(sub.iloc[0]['seed'])}", fontsize=9)
        ax.set_xticks(range(len(PRIMITIVE_NAMES)))
        ax.set_xticklabels(PRIMITIVE_NAMES, rotation=90, fontsize=7)
        # Row labels on the leftmost panel only -- repeating them collides with the
        # neighbouring panel and clips to unreadable stubs.
        ax.set_yticks(range(A.shape[0]))
        if col == 0:
            ax.set_yticklabels([f"atom {i}" for i in range(A.shape[0])], fontsize=7)
        else:
            ax.set_yticklabels([])
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, label="standalone exact-match")
    fig.suptitle("M3 standalone alignment A[atom, primitive]  (depth-matched probe)",
                 fontsize=12)
    fig.savefig(out_dir / "alignment_heatmaps.png", dpi=160)
    plt.close(fig)

    # -- ablation CV distribution per arm (per-atom CVs pooled across seeds)
    fig, ax = plt.subplots(figsize=(1.7 * len(arms) + 2, 3.6))
    data, labels = [], []
    for arm in arms:
        vals = []
        for run_path in df[df["arm"] == arm]["run_path"]:
            d = np.load(Path(run_path) / "artifacts" / "ablation_matrix.npy")
            for i in range(d.shape[0]):
                row = d[i][~np.isnan(d[i])]
                if row.size and row.mean() > 0.05:
                    vals.append(row.std(ddof=0) / row.mean())
        data.append(vals if vals else [np.nan])
        labels.append(ARM_LABELS[arm])
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)),
                   vals, s=12, alpha=0.7, zorder=3)
    ax.axhline(THRESHOLDS["M2_cv"]["pass"][1], ls="--", c="green", lw=1, label="PASS <= 0.35")
    ax.axhline(THRESHOLDS["M2_cv"]["fail"][1], ls="--", c="red", lw=1, label="FAIL >= 0.75")
    ax.set_ylabel("per-atom ablation CV")
    ax.set_title("M2 ablation degradation variance")
    ax.legend(fontsize=8)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    fig.savefig(out_dir / "ablation_cv.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # -- recombination gap, per-seed points overlaid (n=5: spread matters)
    fig, ax = plt.subplots(figsize=(1.7 * len(arms) + 2, 3.8))
    width = 0.38
    xs = np.arange(len(arms))
    for off, key, colour, lab in [(-width / 2, "M1_acc_seen", "#4C72B0", "seen_heldout"),
                                  (width / 2, "M1_acc_unseen", "#DD8452", "unseen")]:
        means = [df[df["arm"] == a][key].mean() for a in arms]
        ax.bar(xs + off, means, width, color=colour, alpha=0.75, label=lab)
        for i, a in enumerate(arms):
            pts = df[df["arm"] == a][key].values
            ax.scatter(np.full(len(pts), xs[i] + off), pts, s=16, c="k", zorder=3)
    ax.axhline(THRESHOLDS["M1_acc_unseen"]["pass"][1], ls="--", c="green", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("M1 recombination (per-seed points overlaid)")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "recombination_gap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_markdown(df, summary, arm_verdicts, per_metric, overall, path: Path) -> None:
    lines = ["# E1 -- Atom Factorization Battery: results", ""]
    lines.append(f"Runs aggregated: {len(df)} across {df['arm'].nunique()} arms.")
    lines.append("")
    lines.append("## Headline metrics (mean +/- std over seeds)")
    lines.append("")
    lines.append("Pre-registered gate metrics (spec 8): "
                 "`M1_acc_unseen`, `M1_gap`, `M2_cv`, `M3_align`, `M3_purity`, "
                 "`M5_dead`. Everything else is diagnostic.")
    lines.append("")
    lines.append("`M3_closed_map_error` is decoder-free and depth-free and **adjudicates "
                 "when the M3 probes disagree** (DECISIONS.md D21): the pre-registered "
                 "`M3_align` proved ~4.5x noisier across seeds and read above its PASS "
                 "threshold on two A3 seeds whose atoms were inert. `M3_align` still "
                 "decides the verdict as pre-registered; no threshold was moved.")
    lines.append("")
    header = ["arm", "n"] + HEADLINE
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for _, r in summary.iterrows():
        cells = [ARM_LABELS.get(r["arm"], r["arm"]), str(int(r["n_seeds"]))]
        for k in HEADLINE:
            mu, sd = r[f"{k}_mean"], r[f"{k}_std"]
            cells.append("n/a" if pd.isna(mu) else
                         (f"{mu:.3f} ± {sd:.3f}" if not pd.isna(sd) else f"{mu:.3f}"))
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Pre-registered threshold judgement (spec 8)", "",
              "PASS requires all six. FAIL on any one is a fail.", "",
              "**Read the A0 row with care.** A0 is the oracle: it is trained with "
              "ground-truth routing and intermediate-state supervision that no other "
              "arm receives, so its PASS is a *ceiling*, not evidence that atoms "
              "factorize under training (DECISIONS.md D4). A0 and the diagnostic arms "
              "(A4 leakage control, A3b) are excluded from the program verdict; only "
              "A1/A2/A3 decide it.", ""]
    metric_names = list(THRESHOLDS)
    lines.append("| arm | " + " | ".join(metric_names) + " | verdict |")
    lines.append("|" + "---|" * (len(metric_names) + 2))
    for arm in sorted(per_metric):
        cells = [ARM_LABELS.get(arm, arm)] + [per_metric[arm][m] for m in metric_names]
        cells.append(f"**{arm_verdicts[arm]}**")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Parameter accounting (H5)", "",
              "Composer and library are separate line items; never combine them.", "",
              "| arm | composer | atoms total | encoder | decoder | composer/atoms |",
              "|---|---|---|---|---|---|"]
    for arm in sorted(df["arm"].unique()):
        s = df[df["arm"] == arm].iloc[0]
        lines.append(
            f"| {ARM_LABELS.get(arm, arm)} | {int(s['params_composer']):,} | "
            f"{int(s['params_atoms_total']):,} | {int(s['params_encoder']):,} | "
            f"{int(s['params_decoder']):,} | {s['composer_over_atoms']:.4f} |")

    lines += ["", "## Verdict", "", overall, "",
              "![alignment](plots/alignment_heatmaps.png)",
              "![ablation cv](plots/ablation_cv.png)",
              "![recombination](plots/recombination_gap.png)", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS_DIR))
    ap.add_argument("--out", default=None,
                    help="default: results/ for v1, results/<generation>/ otherwise")
    ap.add_argument("--generation", default="v1",
                    help="which generation to aggregate; never mixed (D40)")
    ap.add_argument("--include-archived", action="store_true",
                    help="also read runs under archive* subtrees (D41); off by "
                         "default so a frozen battery is never pooled with new runs")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    if args.out is not None:
        out_dir = Path(args.out)
    else:
        # v1 keeps writing to results/ so the committed report's paths stay valid.
        out_dir = RESULTS_DIR if args.generation == "v1" else RESULTS_DIR / args.generation
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect(runs_dir, generation=args.generation,
                 include_archived=args.include_archived)
    summary = summarise(df)

    arm_verdicts, per_metric, arm_means = {}, {}, {}
    for _, r in summary.iterrows():
        means = {k: r[f"{k}_mean"] for k in THRESHOLDS}
        v, pm = verdict_for_arm(means)
        arm_verdicts[r["arm"]] = v
        per_metric[r["arm"]] = pm
        arm_means[r["arm"]] = {
            k: r[f"{k}_mean"] for k in HEADLINE if f"{k}_mean" in r
        }

    # T1 substituted form (DECISIONS.md D18): the gate is whether a fully composing
    # solution is reachable in this architecture, measured by teacher-forced
    # composition on the oracle -- not by A0's end-to-end accuracy.
    a0 = df[df["arm"] == "A0"]["M7_acc_teacher_forced"]
    a0_end2end = df[df["arm"] == "A0"]["M1_acc_unseen"]
    oracle_ok = bool(len(a0)) and float(a0.mean()) >= 0.99
    overall = program_verdict(arm_verdicts, oracle_ok, arm_means)

    df.to_csv(out_dir / "per_run.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_json(out_dir / "verdict.json", {
        "arm_verdicts": arm_verdicts,
        "architectural_signature": {a: bool(architectural_failure(m))
                                    for a, m in arm_means.items()},
        "training_signal_signature": bool(training_signal_failure(
            arm_means, [a for a, v in arm_verdicts.items()
                        if v == "FAIL" and a not in ("A4", "A3b")])),
        "per_metric": per_metric,
        "oracle_t1_passed": oracle_ok,
        "t1_form": "substituted: A0 M7_acc_teacher_forced >= 0.99 (DECISIONS.md D18)",
        "a0_teacher_forced_mean": float(a0.mean()) if len(a0) else None,
        "a0_acc_unseen_mean": float(a0_end2end.mean()) if len(a0_end2end) else None,
        "program_verdict": overall,
    })
    # Deliverable 6: parameter accounting as its own artifact. Per H5, composer and
    # library are never combined into a single total -- a composer quietly absorbing
    # the atoms' work is invisible under one number.
    pc_rollup = {}
    for arm in sorted(df["arm"].unique()):
        s = df[df["arm"] == arm].iloc[0]
        pc_rollup[arm] = {
            "composer": int(s["params_composer"]),
            "atoms_total": int(s["params_atoms_total"]),
            "encoder": int(s["params_encoder"]),
            "decoder": int(s["params_decoder"]),
            "composer_over_atoms": float(s["composer_over_atoms"]),
        }
    write_json(out_dir / "param_counts_rollup.json", {
        "note": "H5: composer and library are separate line items by design. "
                "Composer size is independent of N (asserted in tests/test_fast.py).",
        "per_arm": pc_rollup,
    })

    if not args.no_plots:
        make_plots(df, runs_dir, out_dir / "plots")
    write_markdown(df, summary, arm_verdicts, per_metric, overall,
                   out_dir / "summary.md")

    print(summary.to_string(index=False))
    print("\nVERDICT:", overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
