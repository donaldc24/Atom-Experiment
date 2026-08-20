"""D3 - boundary position probe (read-only). Registered in DECISIONS.md
(2026-08-20) BEFORE running.

Finished-answer-in-strange-code versus rough-draft-plus-editor: fit
per-position linear probes on state slices and ask whether position j
linearly holds the moved content Pa(x)[j] or the original content x[j].
A linear probe cannot compute a cross-position permutation, so wherever
the digits physically sit is the answer. Value maps are per-position
bijections and therefore probe-invisible - the positional axis is the
whole question; P6 (identity permutation) is the machinery control.

Token-1 execution of any pair is bit-identical to its singleton (D1), so
singleton states stand in for all pair boundary states.

Run:  python -m atomv2.d3_position_probe     -> results/d3/
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

D3_RIDGE_LAMBDA = 1.0
D3_TRAIN_FRACTION = 0.7      # first 70% train, rest held-out
D3_N = 400                   # examples per singleton
D3_MARGIN = 0.20             # IN-STATE / EDITOR verdict margin
D3_LOCALIZED_MIN = 0.30      # below this for both targets = DELOCALIZED

# Net token-1 permutation family per surface op (from the recipes).
FAMILY = {"P1": "reverse", "P3": "reverse",
          "P2": "rotate", "P4": "rotate", "P8": "rotate",
          "P5": "swap", "P7": "swap",
          "P6": "identity"}


def _ridge_probe_acc(H: np.ndarray, y: np.ndarray) -> float:
    """Closed-form multiclass ridge on one 64-dim slice; held-out accuracy."""
    n = len(H)
    n_tr = int(n * D3_TRAIN_FRACTION)
    Hb = np.concatenate([H, np.ones((n, 1), dtype=np.float64)], axis=1)
    Y = np.eye(R.VOCAB)[y]
    A = Hb[:n_tr]
    W = np.linalg.solve(A.T @ A + D3_RIDGE_LAMBDA * np.eye(A.shape[1]),
                        A.T @ Y[:n_tr])
    pred = (Hb[n_tr:] @ W).argmax(axis=1)
    return float((pred == y[n_tr:]).mean())


def _probe_state(state: torch.Tensor, x: np.ndarray,
                 moved: np.ndarray) -> dict:
    h = state.numpy().astype(np.float64).reshape(len(x), R.SEQ_LEN, R.D_MODEL)
    acc_moved = [_ridge_probe_acc(h[:, j], moved[:, j])
                 for j in range(R.SEQ_LEN)]
    acc_orig = [_ridge_probe_acc(h[:, j], x[:, j])
                for j in range(R.SEQ_LEN)]
    return {"moved_per_pos": acc_moved, "orig_per_pos": acc_orig,
            "moved": float(np.mean(acc_moved)),
            "orig": float(np.mean(acc_orig))}


@torch.no_grad()
def _head_only_acc(model, state: torch.Tensor, target: np.ndarray) -> float:
    """D3-A1 control (b): decoder head per slice, transformer layer (and
    its cross-position attention) bypassed."""
    h = state.view(-1, R.SEQ_LEN, R.D_MODEL)
    digits = model.decoder.head(h).argmax(-1).numpy()
    return float((digits == target).all(axis=1).mean())


@torch.no_grad()
def run_d3(run_dir, label: str) -> dict:
    model, cfg, step = load_checkpoint(run_dir, "final.pt")
    model.eval()
    bundle = data_mod.build_bundle(cfg)
    singles = {td.task.task_id: td for td in bundle.seen_heldout
               if td.task.n_tokens == 1}

    # Encoder-localization baseline on pooled inputs.
    x_pool = np.concatenate([td.x[:D3_N // len(singles)]
                             for td in singles.values()])
    z0 = model.code(torch.from_numpy(x_pool))
    base = _probe_state(z0, x_pool, x_pool)   # moved == orig == x here
    z0_loc = base["orig"]

    # D3-A1 control (a): off-position probes on z0 - x[k] from slice j.
    h0 = z0.numpy().astype(np.float64).reshape(-1, R.SEQ_LEN, R.D_MODEL)
    off = [_ridge_probe_acc(h0[:, j], x_pool[:, k])
           for j in range(R.SEQ_LEN) for k in range(R.SEQ_LEN) if k != j]
    z0_offpos = float(np.mean(off))

    # D3-A1 control (b): head-only readout vs full decoder, on z0.
    z0_head_only = _head_only_acc(model, z0, x_pool)
    z0_full_dec = float(
        (model.decoder(z0).argmax(-1).numpy() == x_pool).all(1).mean())

    per_task = {}
    for tid, td in sorted(singles.items()):
        x = td.x[:D3_N]
        n = len(x)
        toks = torch.from_numpy(np.tile(td.task.tokens, (n, 1)))
        ntok = torch.full((n,), td.task.n_tokens, dtype=torch.int64)
        states = model(torch.from_numpy(x), toks, ntok,
                       mode="hard")["states"]
        moved = ops.SURFACE_FNS[tid](x)
        per_task[tid] = {
            "family": FAMILY[tid],
            "steps": {f"step{k}": _probe_state(states[k], x, moved)
                      for k in (1, 2, 3)},
            # D3-A1 (b): does readout of the finished answer need the
            # decoder's cross-position attention?
            "boundary_head_only_acc": _head_only_acc(model, states[3], moved),
            "boundary_full_decoder_acc": float(
                (model.decoder(states[3]).argmax(-1).numpy()
                 == moved).all(1).mean()),
        }

    families: dict[str, dict] = {}
    for tid, entry in per_task.items():
        fam = entry["family"]
        b = entry["steps"]["step3"]
        families.setdefault(fam, {"moved": [], "orig": []})
        families[fam]["moved"].append(b["moved"])
        families[fam]["orig"].append(b["orig"])
    verdicts = {}
    for fam, v in sorted(families.items()):
        m, o = float(np.mean(v["moved"])), float(np.mean(v["orig"]))
        if max(m, o) < D3_LOCALIZED_MIN:
            call = "DELOCALIZED"
        elif fam == "identity":
            call = "control"
        elif m - o >= D3_MARGIN:
            call = "IN-STATE"
        elif o - m >= D3_MARGIN:
            call = "EDITOR"
        else:
            call = "MIXED"
        verdicts[fam] = {"moved": m, "orig": o, "call": call}

    return {"label": label, "run_dir": str(run_dir),
            "checkpoint_step": step, "arm": cfg.arm, "seed": cfg.seed,
            "thresholds": {"margin": D3_MARGIN,
                           "localized_min": D3_LOCALIZED_MIN,
                           "ridge_lambda": D3_RIDGE_LAMBDA,
                           "train_fraction": D3_TRAIN_FRACTION},
            "z0_localization_acc": z0_loc,
            "z0_per_pos": base["orig_per_pos"],
            "z0_offposition_acc": z0_offpos,
            "z0_head_only_acc": z0_head_only,
            "z0_full_decoder_acc": z0_full_dec,
            "per_task": per_task,
            "family_verdicts_step3": verdicts}


def main() -> None:
    run_specs = []
    for exp, arm_glob, lab in (("e1b", "A6_s*", "A6"),
                               ("e4", "A14_s*", "A14"),
                               ("e4", "A15_s*", "A15"),
                               ("e5", "A16_s*", "A16")):
        for rd in sorted(glob.glob(str(RUNS_DIR / exp / arm_glob))):
            if os.path.exists(os.path.join(rd, "checkpoints", "final.pt")):
                seed = os.path.basename(rd).split("_")[1]
                run_specs.append((rd, f"{lab}_{seed}"))

    out = RESULTS_DIR / "d3"
    results = []
    for rd, label in run_specs:
        print(f"=== D3 on {label}")
        res = run_d3(rd, label)
        write_json(out / f"{label}.json", res)
        results.append(res)
        v = res["family_verdicts_step3"]
        heads = [t["boundary_head_only_acc"] for t in res["per_task"].values()]
        fulls = [t["boundary_full_decoder_acc"]
                 for t in res["per_task"].values()]
        print(f"    z0_loc={res['z0_localization_acc']:.3f} "
              f"offpos={res['z0_offposition_acc']:.3f} "
              f"z0_head={res['z0_head_only_acc']:.3f} "
              f"bnd_head={np.mean(heads):.3f} bnd_full={np.mean(fulls):.3f}  "
              + "  ".join(f"{f}:{d['call']}" for f, d in v.items()
                          if f != 'identity'))

    lines = ["# D3 - boundary position probe (read-only)", "",
             "Registered in DECISIONS.md (2026-08-20). Per-position linear "
             "probes on boundary (= singleton final) states: moved = "
             "Pa(x)[j] at slice j, orig = x[j] at slice j. Linear probes "
             "cannot permute across positions; value maps are "
             "probe-invisible bijections. P6 = identity-permutation "
             "control.", "",
             "| run | z0 loc | z0 off-pos | z0 head-only | bnd head-only | "
             "bnd full dec | reverse (m/o) | rotate (m/o) | swap (m/o) | "
             "calls |", "|---|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        v = res["family_verdicts_step3"]
        def mo(fam):
            d = v.get(fam)
            return "-" if d is None else f"{d['moved']:.2f}/{d['orig']:.2f}"
        calls = ", ".join(f"{f}:{d['call']}" for f, d in v.items()
                          if f != "identity")
        heads = np.mean([t["boundary_head_only_acc"]
                         for t in res["per_task"].values()])
        fulls = np.mean([t["boundary_full_decoder_acc"]
                         for t in res["per_task"].values()])
        lines.append(f"| {res['label']} | "
                     f"{res['z0_localization_acc']:.2f} | "
                     f"{res['z0_offposition_acc']:.2f} | "
                     f"{res['z0_head_only_acc']:.2f} | "
                     f"{heads:.2f} | {fulls:.2f} | {mo('reverse')} | "
                     f"{mo('rotate')} | {mo('swap')} | {calls} |")
    lines += ["", "Registered calls: IN-STATE (moved - orig >= 0.20: atoms "
              "finished the positional work in-state), EDITOR (orig - moved "
              ">= 0.20: draft in place, decoder permutes at readout), "
              "DELOCALIZED (both < 0.30: content not positionally "
              "organized), MIXED otherwise. Mid-token steps are descriptive "
              "and live in the per-run JSONs.", ""]
    with open(out / "summary.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
