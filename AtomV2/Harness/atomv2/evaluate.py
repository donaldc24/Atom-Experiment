"""Periodic evaluation (every cfg.eval_every steps) + eval-trace emission.

Everything here is measurement: @torch.no_grad, model.eval(), probes read and
never write. Per-cell numbers are always recorded; level means are summaries,
never the measurement (H1Experiments.md: "Always report per cell accuracy").

Outputs per eval:
  - summary dict -> evals/step{N}.json: per-task hard AND soft exact-match
    accuracy for seen_heldout and every unseen level (never averaged across
    levels), per-position accuracy, trajectory closed-map error per task with
    its target-anchored companion and prefix-visit histogram, routing stats
    (census under the registered epsilon, pass rate, steps-per-token,
    per-step choice histogram), soft-hard gap.
  - traces dict (saved as npz at checkpoint cadence): per-example predictions,
    correctness bits, routing choices, per-example closed-map minima - enough
    to recompute or extend any metric without touching the model.
"""
from __future__ import annotations

import numpy as np
import torch

from . import ops
from . import registered as R

EVAL_BATCH = 200


def _batches(n: int, size: int = EVAL_BATCH):
    for lo in range(0, n, size):
        yield lo, min(lo + size, n)


@torch.no_grad()
def _forward_task(model, td, mode: str, tau: float):
    """Run one task's eval set; returns dict of numpy arrays."""
    n = len(td.x)
    toks = np.tile(td.task.tokens, (n, 1))
    ntok = np.full(n, td.task.n_tokens, dtype=np.int64)
    preds, choices, states_per_step = [], [], None
    for lo, hi in _batches(n):
        out = model(torch.from_numpy(td.x[lo:hi]),
                    torch.from_numpy(toks[lo:hi]),
                    torch.from_numpy(ntok[lo:hi]), mode=mode, tau=tau)
        preds.append(out["logits"].argmax(dim=-1).numpy())
        choices.append(out["choices"].numpy())
        st = np.stack([s.numpy() for s in out["states"]], axis=1)  # [b,7,384]
        states_per_step = st if states_per_step is None else np.concatenate(
            [states_per_step, st])
    preds = np.concatenate(preds)
    choices = np.concatenate(choices)
    exact = (preds == td.y).all(axis=1)
    return {"preds": preds, "choices": choices, "exact": exact,
            "per_pos": (preds == td.y).mean(axis=0),
            "states": states_per_step}


@torch.no_grad()
def _closed_map_task(model, td, states: np.ndarray) -> dict:
    """Trajectory closed-map error for one task (registered definition).

    After every live micro-step, relative L2 from the current state to the
    nearest encoding of the task's sub-op-lattice prefix values (all prefixes
    of the canonical chain incl. depth 0, under the current canonical encoder).
    Companions: distance-to-target at the final live step (an all-pass dead
    system parks at prefix 0 and scores near-zero on the headline; the target
    distance exposes it) and the prefix-visit histogram.
    """
    n = len(td.x)
    prefix_values = ops.lattice_prefix_values(td.task.task_id, td.x)
    prefix_codes = []
    for pv in prefix_values:
        codes = []
        for lo, hi in _batches(n):
            codes.append(model.code(torch.from_numpy(pv[lo:hi])).numpy())
        prefix_codes.append(np.concatenate(codes))
    enc = np.stack(prefix_codes, axis=1)                    # [n, P, 384]
    enc_norm = np.linalg.norm(enc, axis=-1).clip(1e-6)      # [n, P]

    n_live = td.task.n_tokens * R.MICRO_STEPS
    step_states = states[:, 1: n_live + 1]                  # [n, S, 384]
    dist = np.linalg.norm(step_states[:, :, None, :] - enc[:, None, :, :],
                          axis=-1)                          # [n, S, P]
    rel = dist / enc_norm[:, None, :]
    per_step_min = rel.min(axis=2)                          # [n, S]
    nearest_prefix = rel.argmin(axis=2)                     # [n, S]

    visits = np.zeros((n_live, len(prefix_values)), dtype=np.int64)
    for s in range(n_live):
        visits[s] = np.bincount(nearest_prefix[:, s], minlength=len(prefix_values))
    return {
        "error": float(per_step_min.mean()),
        "error_per_step": per_step_min.mean(axis=0).tolist(),
        "final_dist_to_target": float(rel[:, -1, -1].mean()),
        "prefix_visit_hist": visits.tolist(),
        "n_prefixes": len(prefix_values),
        "per_example_min": per_step_min.astype(np.float16),  # trace only
    }


def _routing_stats(task_results: dict) -> dict:
    """Census + routing numbers over seen_heldout (registered denominators).

    Census: atom in use if its share of hard ATOM-selections (pass excluded
    from the denominator) exceeds CENSUS_EPS. Pass usage is its own number.
    """
    all_choices, all_ntok = [], []
    for tid, res in task_results.items():
        all_choices.append(res["choices"])
        all_ntok.append(np.full(len(res["choices"]), res["n_tokens"]))
    choices = np.concatenate(all_choices)                   # [n, 6], -1 dead
    ntok = np.concatenate(all_ntok)

    live = choices >= 0
    picks = choices[live]
    atom_picks = picks[picks < R.N_ATOMS]
    n_atom_picks = len(atom_picks)
    counts = np.bincount(atom_picks, minlength=R.N_ATOMS)
    share = counts / max(n_atom_picks, 1)
    in_use = share > R.CENSUS_EPS
    pass_rate = float((picks == R.PASS_INDEX).mean()) if len(picks) else 0.0
    atom_steps_per_example = ((choices >= 0) & (choices < R.N_ATOMS)).sum(axis=1)
    steps_per_token = (float((atom_steps_per_example / ntok).mean())
                       if len(ntok) else 0.0)

    per_step_hist = np.stack([
        np.bincount(choices[:, s][choices[:, s] >= 0],
                    minlength=R.N_ATOMS + 1)
        for s in range(choices.shape[1])])
    return {
        "census_eps": R.CENSUS_EPS,
        "census_denominator": "hard atom-selections on seen_heldout, pass excluded",
        "atoms_in_use": int(in_use.sum()),
        "in_use_mask": in_use.tolist(),
        "atom_selection_share": share.tolist(),
        "n_atom_picks": int(n_atom_picks),
        "pass_rate": pass_rate,
        "steps_per_token": steps_per_token,
        "per_step_choice_hist": per_step_hist.tolist(),
    }


def task_usage_matrix(task_results: dict, n_atoms: int = R.N_ATOMS):
    """usage[i, j]: fraction of task j's examples picking atom i at any live
    step under hard routing (V1 lineage; gates ablation reporting at 0.5)."""
    tids = list(task_results)
    usage = np.zeros((n_atoms, len(tids)))
    for j, tid in enumerate(tids):
        ch = task_results[tid]["choices"]
        for i in range(n_atoms):
            usage[i, j] = (ch == i).any(axis=1).mean()
    return usage, tids


@torch.no_grad()
def run_eval(model, bundle, cfg, step: int) -> tuple[dict, dict]:
    """Full periodic eval. Returns (summary_json, traces_npz_dict)."""
    model.eval()
    tau_soft = cfg.tau_end
    summary = {"step": step, "sets": {}}
    traces = {}
    hard_results_seen = {}

    eval_sets = [("seen_heldout", bundle.seen_heldout)]
    for level in ("L1", "L2", "L3"):
        eval_sets.append((f"unseen_{level}", bundle.unseen[level]))

    for set_name, tds in eval_sets:
        set_block = {"tasks": {}}
        hard_accs, soft_accs, cmap_errs, cmap_target_dists = [], [], [], []
        for td in tds:
            hard = _forward_task(model, td, "hard", tau_soft)
            soft = _forward_task(model, td, "soft", tau_soft)
            cmap = _closed_map_task(model, td, hard["states"])
            per_example_min = cmap.pop("per_example_min")
            tid = td.task.task_id
            set_block["tasks"][tid] = {
                "level": td.task.level,
                "n": len(td.x),
                "acc_hard": float(hard["exact"].mean()),
                "acc_soft": float(soft["exact"].mean()),
                "acc_per_pos_hard": hard["per_pos"].tolist(),
                "closed_map": cmap,
            }
            hard_accs.append(hard["exact"].mean())
            soft_accs.append(soft["exact"].mean())
            cmap_errs.append(cmap["error"])
            cmap_target_dists.append(cmap["final_dist_to_target"])
            traces[f"{set_name}/{tid}/preds"] = hard["preds"].astype(np.int8)
            traces[f"{set_name}/{tid}/exact"] = hard["exact"]
            traces[f"{set_name}/{tid}/choices"] = hard["choices"].astype(np.int8)
            traces[f"{set_name}/{tid}/closed_map_min"] = per_example_min
            if set_name == "seen_heldout":
                hard_results_seen[tid] = {
                    "choices": hard["choices"], "n_tokens": td.task.n_tokens}
        set_block["mean_acc_hard"] = float(np.mean(hard_accs))
        set_block["mean_acc_soft"] = float(np.mean(soft_accs))
        set_block["mean_closed_map_error"] = float(np.mean(cmap_errs))
        set_block["mean_closed_map_final_dist_to_target"] = float(
            np.mean(cmap_target_dists))
        set_block["soft_hard_gap"] = (set_block["mean_acc_soft"]
                                      - set_block["mean_acc_hard"])
        summary["sets"][set_name] = set_block

    # Dissociation gap: L1 minus L3, per registered definition (means of
    # per-cell accuracies within each level; levels never averaged together).
    summary["dissociation_gap_hard"] = (
        summary["sets"]["unseen_L1"]["mean_acc_hard"]
        - summary["sets"]["unseen_L3"]["mean_acc_hard"])
    summary["routing"] = _routing_stats(hard_results_seen)
    return summary, traces
