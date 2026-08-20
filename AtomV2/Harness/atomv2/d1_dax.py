"""D1 - Dax Diagnosis: is the wall a conditioning problem? (read-only)

PREMISE CORRECTION, recorded before any number: the D1 brief's suspect is
"task tokens are baked into h0" (enc(x, [P3, Pb]) vs enc(x, [P3])). Under
amendment R8 this architecture HAS no such object: the encoder is digit-only
(model.code(x) takes no tokens; the h0 leakage floors sit at chance in every
panel), and the composer receives the active token exogenously - the partner
token is never visible during token 1. Consequently:

  - For P3-FIRST cells (P3_Pb), token 1's entire execution (routing AND
    produced state) is bit-identical to the singleton P3 execution by
    construction. P-A verifies this mechanically instead of assuming it.
  - The only places "a region that never existed in training" CAN exist are
    (a) the composer's content input at token 2 - a program running on
    another program's product - and (b) the token-boundary states themselves.

The three probes attach there, keeping the brief's questions and bins:

  P-A routing audit: token-1 program vs that op's singleton program
      (bit-identity witness) and token-2 program vs the second op's
      singleton program (content-driven routing deviation). Trained pairs
      provide the in-distribution baseline for the same flip metric.
  P-B content audit: frozen-decoder read of the token-1 boundary state vs
      the ground-truth intermediate pa(x). Decode WRONG = emission
      corrupted; decode RIGHT while composition fails = correct-but-
      unreadable (the closure story). Trained pairs baseline again.
  P-C geometry audit: nearest-neighbor distance from each cell's boundary
      states to the pool of boundary states trained token-2 programs
      actually consumed (the real "map"), plus relative distance to the
      canonical encoding of the true intermediate. Reported for trained
      (leave-one-task-out) / L1 / L2 / L3 - the difficulty-ordering
      geometry question, asked in the space where a region can be missing.

Everything is measurement: @torch.no_grad, final checkpoints from disk, no
training, no new losses. Ground-truth generator use (ops.SURFACE_FNS) is
legal here - the quarantine covers the training path only.

Run:  python -m atomv2.d1_dax          -> results/d1/
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

from . import data as data_mod
from . import ops
from . import registered as R
from .panel import load_checkpoint
from .utils import RESULTS_DIR, RUNS_DIR, write_json

N_EXAMPLES = 400          # per cell: the full fixed eval set
POOL_PER_TASK = 100       # boundary states per trained task in the P-C pool


@torch.no_grad()
def _forward(model, task, x: np.ndarray) -> dict:
    n = len(x)
    toks = torch.from_numpy(np.tile(task.tokens, (n, 1)))
    ntok = torch.full((n,), task.n_tokens, dtype=torch.int64)
    out = model(torch.from_numpy(x), toks, ntok, mode="hard")
    return {
        "choices": out["choices"].numpy(),                    # [n,6]
        "boundary": out["states"][R.MICRO_STEPS],             # [n,384] torch
        "preds": out["logits"].argmax(-1).numpy(),            # [n,6]
    }


def _flip(a: np.ndarray, b: np.ndarray) -> dict:
    per = [float((a[:, k] != b[:, k]).mean()) for k in range(a.shape[1])]
    return {"per_micro_step": per, "overall": float(np.mean(per))}


@torch.no_grad()
def _cell_probes(model, td) -> dict:
    pa, pb = td.task.surface_ops
    x = td.x[:N_EXAMPLES]
    y = td.y[:N_EXAMPLES]
    cell = _forward(model, td.task, x)
    sa = _forward(model, data_mod.make_task(pa, "probe"), x)
    sb = _forward(model, data_mod.make_task(pb, "probe"), x)

    tok1 = _flip(cell["choices"][:, :R.MICRO_STEPS],
                 sa["choices"][:, :R.MICRO_STEPS])
    tok2 = _flip(cell["choices"][:, R.MICRO_STEPS:],
                 sb["choices"][:, :R.MICRO_STEPS])
    truth_mid = ops.SURFACE_FNS[pa](x)
    mid_digits = model.decoder(cell["boundary"]).argmax(-1).numpy()
    return {
        "cell": td.task.task_id,
        "level": td.task.level,
        "p3_position": ("first" if pa == R.DAX else
                        "second" if pb == R.DAX else "none"),
        "n": len(x),
        "pa_routing": {
            "token1_flip_vs_singleton": tok1,
            "token1_boundary_bit_identical": bool(
                torch.equal(cell["boundary"], sa["boundary"])),
            "token2_flip_vs_singleton": tok2,
        },
        "pb_content": {
            "token1_decode_acc": float((mid_digits == truth_mid).all(1).mean()),
            "singleton_pa_acc": float((sa["preds"] == truth_mid).all(1).mean()),
            "raw_cell_acc": float((cell["preds"] == y).all(1).mean()),
        },
        "_x": x[:POOL_PER_TASK],
        "_boundary": cell["boundary"].numpy().astype(np.float32)[
            :POOL_PER_TASK],
    }


def _nn_dist(queries: np.ndarray, pool: np.ndarray, chunk: int = 64) -> np.ndarray:
    out = np.empty(len(queries))
    for lo in range(0, len(queries), chunk):
        q = queries[lo:lo + chunk]
        d = np.linalg.norm(q[:, None, :] - pool[None, :, :], axis=-1)
        out[lo:lo + chunk] = d.min(axis=1)
    return out


@torch.no_grad()
def run_d1(run_dir, label: str) -> dict:
    model, cfg, step = load_checkpoint(run_dir, "final.pt")
    model.eval()
    bundle = data_mod.build_bundle(cfg)

    cells = []
    groups = {"trained": [t for t in bundle.seen_heldout
                          if t.task.n_tokens == 2],
              "L1": bundle.unseen["L1"], "L2": bundle.unseen["L2"],
              "L3": bundle.unseen["L3"]}
    for group, tds in groups.items():
        for td in tds:
            entry = _cell_probes(model, td)
            entry["group"] = group
            cells.append(entry)

    pool = np.concatenate([c["_boundary"] for c in cells
                           if c["group"] == "trained"])
    pool_task = np.concatenate([
        np.full(len(c["_boundary"]), i) for i, c in enumerate(cells)
        if c["group"] == "trained"])
    for i, c in enumerate(cells):
        q = c["_boundary"]
        d = _nn_dist(q, pool[pool_task != i] if c["group"] == "trained"
                     else pool)
        pa = c["cell"].split("_")[0]
        canon = model.code(torch.from_numpy(
            ops.SURFACE_FNS[pa](c["_x"]))).numpy()
        rel = np.linalg.norm(q - canon, axis=-1) \
            / np.maximum(np.linalg.norm(canon, axis=-1), 1e-6)
        c["pc_geometry"] = {
            "nn_dist_to_trained_boundary_pool": {
                "median": float(np.median(d)),
                "p90": float(np.quantile(d, .9)), "max": float(d.max())},
            "rel_dist_to_canonical_intermediate": {
                "median": float(np.median(rel)),
                "p90": float(np.quantile(rel, .9))},
        }
        del c["_boundary"], c["_x"]

    return {"label": label, "run_dir": str(run_dir), "checkpoint_step": step,
            "arm": cfg.arm, "seed": cfg.seed, "n_cells": len(cells),
            "premise_note": (
                "R8 encoder is digit-only: h0 carries no task conditioning; "
                "token-1 execution of any pair is bit-identical to its "
                "singleton by construction, verified per cell in "
                "token1_boundary_bit_identical."),
            "cells": cells}


def _group_stat(cells, group, path, sub=None):
    vals = []
    for c in cells:
        if c["group"] != group:
            continue
        v = c
        for k in path:
            v = v[k]
        vals.append(v)
    return float(np.mean(vals)) if vals else None


def main() -> None:
    run_specs = []
    for exp, arm_glob, label in (("e1b", "A6_s*", "A6"),
                                 ("e4", "A14_s*", "A14"),
                                 ("e5", "A16_s*", "A16")):
        for rd in sorted(glob.glob(str(RUNS_DIR / exp / arm_glob))):
            if os.path.exists(os.path.join(rd, "checkpoints", "final.pt")):
                seed = os.path.basename(rd).split("_")[1]
                run_specs.append((rd, f"{label}_{seed}"))

    out = RESULTS_DIR / "d1"
    results = []
    for rd, label in run_specs:
        print(f"=== D1 on {label} ({rd})")
        res = run_d1(rd, label)
        write_json(out / f"{label}.json", res)
        results.append(res)
        bad_bit = [c["cell"] for c in res["cells"]
                   if not c["pa_routing"]["token1_boundary_bit_identical"]]
        print(f"    token1 bit-identity violations: {bad_bit or 'none'}")

    # ---- the three cross-run tables + summary ----------------------------
    lines = ["# D1 - Dax Diagnosis (read-only)", "",
             "Premise correction: the R8 encoder is digit-only; h0 carries "
             "no task conditioning and token-1 execution of any pair is "
             "bit-identical to its singleton (verified mechanically below). "
             "The probes therefore attach to the composer's token-2 content "
             "input and the token-boundary states - the only places 'a "
             "region that never existed in training' can exist.", ""]

    lines += ["## P-A routing audit (token-2 program flips vs singleton)", "",
              "| run | trained | L1 | L2 | L3 (P3 first) | L3 (P3 second) | "
              "token1 bit-identical everywhere |",
              "|---|---|---|---|---|---|---|"]
    for res in results:
        cs = res["cells"]
        def flip(group, pos=None):
            vals = [c["pa_routing"]["token2_flip_vs_singleton"]["overall"]
                    for c in cs if c["group"] == group
                    and (pos is None or c["p3_position"] == pos)]
            return f"{np.mean(vals):.3f}" if vals else "-"
        bit = all(c["pa_routing"]["token1_boundary_bit_identical"]
                  for c in cs)
        lines.append(f"| {res['label']} | {flip('trained')} | {flip('L1')} | "
                     f"{flip('L2')} | {flip('L3', 'first')} | "
                     f"{flip('L3', 'second')} | {bit} |")

    lines += ["", "## P-B content audit (token-1 boundary decode vs truth)",
              "",
              "| run | trained decode | L1 decode | L2 decode | L3 decode "
              "(P3 first) | L3 decode (P3 second) | L3 raw acc | singleton "
              "P3 acc |", "|---|---|---|---|---|---|---|---|"]
    for res in results:
        cs = res["cells"]
        def dec(group, pos=None):
            vals = [c["pb_content"]["token1_decode_acc"] for c in cs
                    if c["group"] == group
                    and (pos is None or c["p3_position"] == pos)]
            return f"{np.mean(vals):.3f}" if vals else "-"
        l3raw = np.mean([c["pb_content"]["raw_cell_acc"] for c in cs
                         if c["group"] == "L3"])
        p3acc = np.mean([c["pb_content"]["singleton_pa_acc"] for c in cs
                         if c["p3_position"] == "first"])
        lines.append(f"| {res['label']} | {dec('trained')} | {dec('L1')} | "
                     f"{dec('L2')} | {dec('L3', 'first')} | "
                     f"{dec('L3', 'second')} | {l3raw:.4f} | {p3acc:.3f} |")

    lines += ["", "## P-C geometry audit (NN distance of boundary states to "
              "the trained-consumption pool; canonical distance)", "",
              "| run | trained NN (LOO) | L1 NN | L2 NN | L3 NN | trained "
              "canon | L1 canon | L2 canon | L3 canon |",
              "|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        cs = res["cells"]
        def geo(group, key):
            vals = [c["pc_geometry"][key]["median"] for c in cs
                    if c["group"] == group]
            return f"{np.mean(vals):.3f}" if vals else "-"
        lines.append(
            f"| {res['label']} | "
            f"{geo('trained', 'nn_dist_to_trained_boundary_pool')} | "
            f"{geo('L1', 'nn_dist_to_trained_boundary_pool')} | "
            f"{geo('L2', 'nn_dist_to_trained_boundary_pool')} | "
            f"{geo('L3', 'nn_dist_to_trained_boundary_pool')} | "
            f"{geo('trained', 'rel_dist_to_canonical_intermediate')} | "
            f"{geo('L1', 'rel_dist_to_canonical_intermediate')} | "
            f"{geo('L2', 'rel_dist_to_canonical_intermediate')} | "
            f"{geo('L3', 'rel_dist_to_canonical_intermediate')} |")

    lines += ["", "Per-cell, per-seed detail lives in the per-run JSON files "
              "beside this summary. Outcome-bin reading belongs to the "
              "operator; the registered bins are in the D1 brief.", ""]
    with open(out / "summary.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
