"""Full metric panel, run on checkpoints FROM DISK (final + 5k/10k/15k).

The panel never sees a live training process: it loads a checkpoint, rebuilds
the bit-identical model and data from the config stored inside it, and writes
raw matrices + JSON under <run_dir>/panel/step{N}/. Everything downstream
(analyze.py, aggregate.py) reads only these artifacts - the panel can be
re-run or extended at any time without retraining.

Instruments (Q2-Q5 of H1Experiments.md, definitions registered 2026-08-14):
  census        atoms-in-use under eps=2% of non-pass hard picks + steps/token
  ablation      zero-delta intercept, one atom at a time, damage vs same-run
                unablated accuracy, gated at task-usage >= 0.5; F2 = per-atom
                damage variance. Raw tuples (atom, task, level, acc_delta,
                routing_entropy) always saved. FINAL checkpoint only: routing-
                mask compensation probe, observational, non-gating.
  standalone    each atom alone (code -> one application -> decode) against
                the full candidate answer key (7 sub-ops + 8 surface ops +
                identity) in the shared, task-independent content code.
  closed-map    V1-style atom-centric probe vs the same candidates: relative
                L2 in state space, matched assignment + coverage companions.
  decodability  linear probes on detached states/deltas for sub-ops in play
                (SET-wise: P8 = {T,N}) and surface ops, each with an h0 floor
                and a shuffled-label chance floor. Probes read, never write.
  transfer      task-level 8x8 accuracy matrix + the program-transplant test:
                singleton routing programs concatenated and force-run on pair
                tasks (mutual intelligibility, Kim-style).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as data_mod
from . import ops
from . import registered as R
from .config import Config
from .evaluate import task_usage_matrix
from .model import AtomModel, N_STEPS
from .utils import stream_rng, write_json

# Candidate answer key: 7 sub-ops + 8 surface ops + identity = 16 candidates.
CANDIDATES = (list(ops.CANDIDATE_OPS) + ["identity"])


def _candidate_apply(name: str, x: np.ndarray) -> np.ndarray:
    if name == "identity":
        return x.copy()
    return ops.CANDIDATE_OPS[name](x)


def load_checkpoint(run_dir: Path, which: str):
    path = Path(run_dir) / "checkpoints" / which
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ck["config"])
    model = AtomModel(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():        # probes read, never write
        p.requires_grad_(False)
    return model, cfg, ck["step"]


@torch.no_grad()
def _run_task(model, td, mode="hard", tau=None, ablate=None, atom_mask=None,
              forced=None):
    n = len(td.x)
    toks = np.tile(td.task.tokens, (n, 1))
    ntok = np.full(n, td.task.n_tokens, dtype=np.int64)
    preds, choices = [], []
    for lo in range(0, n, 200):
        hi = min(lo + 200, n)
        out = model(torch.from_numpy(td.x[lo:hi]),
                    torch.from_numpy(toks[lo:hi]),
                    torch.from_numpy(ntok[lo:hi]), mode=mode,
                    tau=tau if tau is not None else 0.5,
                    ablate=ablate, atom_mask=atom_mask,
                    forced=(torch.from_numpy(forced[lo:hi])
                            if forced is not None else None))
        preds.append(out["logits"].argmax(-1).numpy())
        choices.append(out["choices"].numpy())
    preds = np.concatenate(preds)
    choices = np.concatenate(choices)
    return {"acc": float((preds == td.y).all(1).mean()), "preds": preds,
            "choices": choices}


def _routing_entropy(choices: np.ndarray) -> float:
    live = choices[choices >= 0]
    if len(live) == 0:
        return 0.0
    p = np.bincount(live, minlength=R.N_ATOMS + 1).astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------

def census(seen_results: dict) -> dict:
    all_choices = np.concatenate([r["choices"] for r in seen_results.values()])
    live = all_choices[all_choices >= 0]
    atom_picks = live[live < R.N_ATOMS]
    counts = np.bincount(atom_picks, minlength=R.N_ATOMS)
    share = counts / max(len(atom_picks), 1)
    n_tokens = np.concatenate([
        np.full(len(r["choices"]), r["n_tokens"]) for r in seen_results.values()])
    atom_steps = ((all_choices >= 0) & (all_choices < R.N_ATOMS)).sum(axis=1)
    return {
        "eps": R.CENSUS_EPS,
        "denominator": "hard atom-selections on seen_heldout, pass excluded",
        "atoms_in_use": int((share > R.CENSUS_EPS).sum()),
        "in_use_mask": (share > R.CENSUS_EPS).tolist(),
        "atom_selection_share": share.tolist(),
        "pass_rate": float((live == R.PASS_INDEX).mean()),
        "steps_per_token": float((atom_steps / n_tokens).mean()),
    }


# ---------------------------------------------------------------------------
# Ablation (F2) + compensation probe
# ---------------------------------------------------------------------------

def ablation(model, all_tasks: list, full_acc: dict, usage, tids,
             compensation: bool) -> dict:
    n_atoms = R.N_ATOMS
    tid_index = {t: j for j, t in enumerate(tids)}
    damage = np.full((n_atoms, len(tids)), np.nan)
    mask_damage = np.full((n_atoms, len(tids)), np.nan)
    raw_tuples = []
    for i in range(n_atoms):
        ab = torch.zeros(n_atoms, dtype=torch.bool)
        ab[i] = True
        for td in all_tasks:
            j = tid_index[td.task.task_id]
            if usage[i, j] < R.USAGE_TASK_THRESHOLD:
                continue
            res = _run_task(model, td, ablate=ab)
            d = full_acc[td.task.task_id] - res["acc"]
            damage[i, j] = d
            raw_tuples.append({
                "atom": i, "task": td.task.task_id, "level": td.task.level,
                "usage": float(usage[i, j]), "acc_full": full_acc[td.task.task_id],
                "acc_ablated": res["acc"], "acc_delta": d,
                "routing_entropy_during_ablation": _routing_entropy(res["choices"]),
            })
            if compensation:
                res_m = _run_task(model, td, atom_mask=ab)
                mask_damage[i, j] = full_acc[td.task.task_id] - res_m["acc"]

    per_atom = {}
    for i in range(n_atoms):
        row = damage[i][~np.isnan(damage[i])]
        entry = {"n_tasks_used": int(len(row))}
        if len(row) >= 2:
            mean = float(row.mean())
            entry.update({
                "damage_mean": mean,
                "damage_std": float(row.std(ddof=1)),
                # F2: THE VARIANCE IS THE METRIC, NOT THE MEAN
                "damage_cv": (float(row.std(ddof=1) / abs(mean))
                              if abs(mean) > 1e-9 else None),
            })
        elif len(row) == 1:
            entry.update({"damage_mean": float(row[0]), "damage_std": None,
                          "damage_cv": None})
        if compensation:
            mrow = mask_damage[i][~np.isnan(mask_damage[i])]
            if len(mrow):
                entry["compensation_damage_mean"] = float(mrow.mean())
                entry["compensation_gap_mean"] = float(
                    (damage[i][~np.isnan(mask_damage[i])] - mrow).mean())
        per_atom[str(i)] = entry

    return {"mechanism": "zero_delta_intercept",
            "usage_task_threshold": R.USAGE_TASK_THRESHOLD,
            "per_atom": per_atom, "raw_tuples": raw_tuples,
            "_damage_matrix": damage, "_mask_damage_matrix": mask_damage,
            "_tids": tids}


# ---------------------------------------------------------------------------
# Standalone semantics + atom-centric closed map
# ---------------------------------------------------------------------------

@torch.no_grad()
def standalone_and_closed_map(model, probe_x: np.ndarray) -> dict:
    """Both atom-centric instruments share the probe pass structure.

    R8 makes the content code task-independent, so the old eight singleton
    encoding contexts collapse to one canonical context. Partner dependence
    is measured by the task and transplant transfer matrices instead.
    """
    n = len(probe_x)
    n_atoms, n_cand = R.N_ATOMS, len(CANDIDATES)
    ctx_names = ["content_only"]
    # candidate ground-truth outputs (shared across contexts)
    cand_y = {c: _candidate_apply(c, probe_x) for c in CANDIDATES}

    acc = np.zeros((n_atoms, n_cand, len(ctx_names)))      # decode-and-match
    err = np.zeros((n_atoms, n_cand, len(ctx_names)))      # state-space relL2
    for ci, _ctx in enumerate(ctx_names):
        h0 = model.code(torch.from_numpy(probe_x))
        cand_codes = {}
        for c in CANDIDATES:
            cand_codes[c] = model.code(torch.from_numpy(cand_y[c]))
        for i in range(n_atoms):
            hi = model.step_once(h0, i)
            digits = model.decoder(hi).argmax(-1).numpy()
            for cj, c in enumerate(CANDIDATES):
                acc[i, cj, ci] = (digits == cand_y[c]).all(1).mean()
                target = cand_codes[c]
                rel = (torch.linalg.norm(hi - target, dim=-1)
                       / torch.linalg.norm(target, dim=-1).clamp_min(1e-6))
                err[i, cj, ci] = rel.mean().item()

    acc_mean = acc.mean(axis=2)                            # [atoms, cand]
    err_mean = err.mean(axis=2)

    standalone = {}
    for i in range(n_atoms):
        best = int(acc_mean[i].argmax())
        standalone[str(i)] = {
            "best_candidate": CANDIDATES[best],
            "best_acc": float(acc_mean[i, best]),
            "best_acc_per_context": acc[i, best].tolist(),
            "context_spread": float(acc[i, best].max() - acc[i, best].min()),
            "identity_acc": float(acc_mean[i, CANDIDATES.index("identity")]),
        }

    # matched one-to-one assignment (greedy by ascending row-min), V1 lineage
    order = np.argsort(err_mean.min(axis=1))
    taken: set[int] = set()
    matched = {}
    for i in order:
        row = err_mean[i].copy()
        row[list(taken)] = np.inf
        c = int(row.argmin())
        taken.add(c)
        matched[str(int(i))] = {"candidate": CANDIDATES[c], "err": float(err_mean[i, c])}
        if len(taken) == len(CANDIDATES):
            break
    argmin = err_mean.argmin(axis=1)
    closed_map = {
        "candidates": CANDIDATES,
        "contexts": ctx_names,
        "error_min_mean": float(err_mean.min(axis=1).mean()),
        "argmin_candidate": [CANDIDATES[a] for a in argmin],
        "coverage": int(len(set(argmin.tolist()))),
        "matched": matched,
        "matched_error_mean": float(np.mean([m["err"] for m in matched.values()])),
    }
    return {"standalone": standalone, "closed_map_atom": closed_map,
            "_acc_tensor": acc, "_err_tensor": err}


# ---------------------------------------------------------------------------
# Decodability probes
# ---------------------------------------------------------------------------

def _collect_probe_material(model, seen_tds, max_examples_per_task=100):
    """Detached (h0, delta, state_after) samples from hard forwards, with
    per-token labels. Labels are SET-wise for sub-ops (P8 -> {T, N})."""
    sub_index = {s: i for i, s in enumerate(ops.SUBOP_NAMES)}
    surf_index = {p: i for i, p in enumerate(ops.SURFACE_NAMES)}
    h0s, h0_sub, h0_surf, h0_example = [], [], [], []
    deltas, states, sub_labels, surf_labels, example_ids = [], [], [], [], []
    eid = 0
    for td in seen_tds:
        m = min(max_examples_per_task, len(td.x))
        toks = np.tile(td.task.tokens, (m, 1))
        ntok = np.full(m, td.task.n_tokens, dtype=np.int64)
        with torch.no_grad():
            out = model(torch.from_numpy(td.x[:m]), torch.from_numpy(toks),
                        torch.from_numpy(ntok), mode="hard")
        st = [s.numpy() for s in out["states"]]
        choices = out["choices"].numpy()
        subsets = ops.task_subop_sets(td.task.task_id)
        for b in range(m):
            sub0 = np.zeros(len(sub_index))
            for s in subsets[0]:
                sub0[sub_index[s]] = 1.0
            h0s.append(st[0][b])
            h0_sub.append(sub0)
            h0_surf.append(surf_index[td.task.surface_ops[0]])
            h0_example.append(eid)
            for k in range(td.task.n_tokens * R.MICRO_STEPS):
                if choices[b, k] < 0 or choices[b, k] == R.PASS_INDEX:
                    continue  # pass applies no atom; there is no delta to probe
                tok_idx = k // R.MICRO_STEPS
                subl = np.zeros(len(sub_index))
                for s in subsets[tok_idx]:
                    subl[sub_index[s]] = 1.0
                deltas.append(st[k + 1][b] - st[k][b])
                states.append(st[k + 1][b])
                sub_labels.append(subl)
                surf_labels.append(surf_index[td.task.surface_ops[tok_idx]])
                example_ids.append(eid)
            eid += 1
    return {
        "h0": (np.array(h0s), np.array(h0_sub), np.array(h0_surf), np.array(h0_example)),
        "step": (np.array(deltas), np.array(states), np.array(sub_labels),
                 np.array(surf_labels), np.array(example_ids)),
    }


def _train_linear_probe(x, y, kind, seed, train_frac=R.PROBE_TRAIN_FRACTION,
                        example_ids=None, epochs=300):
    """Linear probe on detached features. Split by EXAMPLE so steps of one
    example never straddle train/test. kind: 'multilabel' | 'categorical'."""
    if len(x) == 0:
        return {"n_train": 0, "n_test": 0, "score": None,
                "note": "no probe material (e.g. router picked only pass)"}
    rng = np.random.default_rng(seed)
    if example_ids is None:
        example_ids = np.arange(len(x))
    uniq = np.unique(example_ids)
    perm = rng.permutation(len(uniq))
    n_train = int(len(uniq) * train_frac)
    train_ex = set(uniq[perm[:n_train]].tolist())
    tr = np.array([e in train_ex for e in example_ids], dtype=bool)
    te = ~tr
    if tr.sum() == 0 or te.sum() == 0:
        return {"n_train": int(tr.sum()), "n_test": int(te.sum()), "score": None}

    xt = torch.from_numpy(x[tr]).float()
    xv = torch.from_numpy(x[te]).float()
    g = torch.Generator().manual_seed(int(seed))
    if kind == "multilabel":
        w = torch.zeros(x.shape[1], y.shape[1], requires_grad=True)
        b = torch.zeros(y.shape[1], requires_grad=True)
        yt = torch.from_numpy(y[tr]).float()
        yv = torch.from_numpy(y[te]).float()
    else:
        n_cls = int(y.max()) + 1
        w = torch.zeros(x.shape[1], n_cls, requires_grad=True)
        b = torch.zeros(n_cls, requires_grad=True)
        yt = torch.from_numpy(y[tr]).long()
        yv = torch.from_numpy(y[te]).long()
    with torch.no_grad():
        w.normal_(0, 0.01, generator=g)
    opt = torch.optim.Adam([w, b], lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        logits = xt @ w + b
        loss = (F.binary_cross_entropy_with_logits(logits, yt)
                if kind == "multilabel" else F.cross_entropy(logits, yt))
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = xv @ w + b
        if kind == "multilabel":
            pred = (logits > 0).float()
            per_label = []
            for j in range(pred.shape[1]):
                pos = yv[:, j] == 1
                neg = ~pos
                tpr = pred[pos, j].mean().item() if pos.any() else float("nan")
                tnr = (1 - pred[neg, j]).mean().item() if neg.any() else float("nan")
                per_label.append(float(np.nanmean([tpr, tnr])))
            return {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                    "score": float(np.nanmean(per_label)),
                    "per_label_balanced_acc": per_label}
        pred = logits.argmax(-1)
        return {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                "score": float((pred == yv).float().mean().item())}


def decodability(model, seen_tds, master_seed: int) -> dict:
    import zlib  # stable per-probe seeds; hash() is process-randomized
    mat = _collect_probe_material(model, seen_tds)
    seed_base = int(stream_rng(master_seed, "probe_train").integers(2**31))
    h0_x, h0_sub, h0_surf, h0_eid = mat["h0"]
    d_x, s_x, sub_y, surf_y, eid = mat["step"]

    results = {"labels_subops": list(ops.SUBOP_NAMES),
               "labels_surface": list(ops.SURFACE_NAMES),
               "p8_note": "sub-op labels are SETS; P8 contributes {T,N}",
               "n_step_samples": int(len(d_x))}
    specs = [
        ("subop_from_delta", d_x, sub_y, "multilabel", eid),
        ("subop_from_state", s_x, sub_y, "multilabel", eid),
        ("subop_h0_floor", h0_x, h0_sub, "multilabel", h0_eid),
        ("surface_from_delta", d_x, surf_y, "categorical", eid),
        ("surface_from_state", s_x, surf_y, "categorical", eid),
        ("surface_h0_floor", h0_x, h0_surf, "categorical", h0_eid),
    ]
    for name, x, y, kind, ids in specs:
        results[name] = _train_linear_probe(x, y, kind, seed_base + zlib.crc32(name.encode()) % 10000,
                                            example_ids=ids)
        if len(x):  # shuffled-label chance floor, permuted at the example level
            rng = np.random.default_rng(seed_base + 77)
            uniq, inv = np.unique(ids, return_inverse=True)
            y_arr = np.asarray(y)
            # permute whole-example label blocks
            ex_labels = {}
            for u in uniq:
                ex_labels[u] = y_arr[ids == u][0]
            shuffled_uniq = rng.permutation(uniq)
            y_shuf = np.stack([ex_labels[shuffled_uniq[i]] for i in inv])
            results[name + "_shuffled"] = _train_linear_probe(
                x, y_shuf, kind, seed_base + zlib.crc32(name.encode()) % 10000, example_ids=ids)
    return results


# ---------------------------------------------------------------------------
# Transfer matrix + program transplant
# ---------------------------------------------------------------------------

def transfer(model, full_acc: dict, seen_results: dict, split: dict,
             probe_x: np.ndarray) -> dict:
    names = list(ops.SURFACE_NAMES)
    n = len(names)
    task_matrix = np.full((n, n), np.nan)
    level_matrix = [["" for _ in range(n)] for _ in range(n)]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            tid = f"{a}_{b}"
            cell = split["cells"][tid]
            level_matrix[i][j] = cell["split"]
            if tid in full_acc:
                task_matrix[i, j] = full_acc[tid]

    # program transplant: modal singleton routing program per surface op,
    # concatenated and force-run on every pair cell over fresh probe inputs.
    programs = {}
    for p in names:
        ch = seen_results[p]["choices"][:, : R.MICRO_STEPS]
        modal = Counter(map(tuple, ch.tolist())).most_common(1)[0][0]
        programs[p] = list(modal)
    transplant = np.full((n, n), np.nan)
    npx = len(probe_x)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            tid = f"{a}_{b}"
            sched = np.tile(np.array(programs[a] + programs[b], dtype=np.int64),
                            (npx, 1))
            td = data_mod.TaskData(data_mod.make_task(tid, "probe"),
                                   probe_x, ops.apply_task(tid, probe_x))
            res = _run_task(model, td, mode="forced", forced=sched)
            transplant[i, j] = res["acc"]

    def row_stats(m):
        return {"row_means": np.nanmean(m, axis=1).tolist(),
                "row_stds": [float(np.nanstd(m[i])) for i in range(n)]}

    return {"surface_names": names,
            "task_matrix": task_matrix.tolist(),
            "level_matrix": level_matrix,
            "task_row_stats": row_stats(task_matrix),
            "singleton_programs": {p: programs[p] for p in names},
            "transplant_matrix": transplant.tolist(),
            "transplant_row_stats": row_stats(transplant),
            "_task_matrix": task_matrix, "_transplant_matrix": transplant}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_panel(run_dir: Path, which: str, is_final: bool) -> Path:
    model, cfg, step = load_checkpoint(run_dir, which)
    bundle = data_mod.build_bundle(cfg)
    out_dir = Path(run_dir) / "panel" / (f"step{step:06d}" if not is_final
                                         else "final")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hard eval over every eval task once: accuracies + routing for everything.
    all_tasks = bundle.all_eval_tasks
    full_acc, task_results = {}, {}
    for td in all_tasks:
        res = _run_task(model, td)
        full_acc[td.task.task_id] = res["acc"]
        task_results[td.task.task_id] = {
            "choices": res["choices"], "n_tokens": td.task.n_tokens}
    seen_ids = {td.task.task_id for td in bundle.seen_heldout}
    seen_results = {t: r for t, r in task_results.items() if t in seen_ids}

    cen = census(seen_results)
    write_json(out_dir / "census.json", cen)

    usage, tids = task_usage_matrix(task_results)
    abl = ablation(model, all_tasks, full_acc, usage, tids,
                   compensation=is_final)
    np.savez_compressed(out_dir / "ablation.npz",
                        damage=abl.pop("_damage_matrix"),
                        mask_damage=abl.pop("_mask_damage_matrix"),
                        usage=usage, task_ids=np.array(abl.pop("_tids")))
    write_json(out_dir / "ablation.json", abl)

    sac = standalone_and_closed_map(model, bundle.probe_inputs)
    np.savez_compressed(out_dir / "standalone_closed_map.npz",
                        acc=sac.pop("_acc_tensor"), err=sac.pop("_err_tensor"))
    write_json(out_dir / "standalone.json", sac["standalone"])
    write_json(out_dir / "closed_map_atom.json", sac["closed_map_atom"])

    dec = decodability(model, bundle.seen_heldout, cfg.seed)
    write_json(out_dir / "decodability.json", dec)

    tr = transfer(model, full_acc, seen_results, bundle.split, bundle.probe_inputs)
    np.savez_compressed(out_dir / "transfer.npz",
                        task_matrix=tr.pop("_task_matrix"),
                        transplant_matrix=tr.pop("_transplant_matrix"))
    write_json(out_dir / "transfer.json", tr)

    write_json(out_dir / "full_acc.json", full_acc)
    write_json(out_dir / "panel_meta.json",
               {"checkpoint": which, "step": step, "is_final": is_final})
    return out_dir


def run_all_panels(run_dir: Path) -> list:
    run_dir = Path(run_dir)
    import json
    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg_d = json.load(f)
    done = []
    for s in cfg_d["panel_steps"]:
        ck = f"step{s:06d}.pt"
        if not (run_dir / "checkpoints" / ck).exists():
            raise FileNotFoundError(
                f"registered panel checkpoint missing: {run_dir / 'checkpoints' / ck}")
        done.append(run_panel(run_dir, ck, is_final=False))
    done.append(run_panel(run_dir, "final.pt", is_final=True))
    return done


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the metric panel on a run's checkpoints")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--only-final", action="store_true")
    a = ap.parse_args()
    if a.only_final:
        print(run_panel(Path(a.run_dir), "final.pt", is_final=True))
    else:
        for p in run_all_panels(Path(a.run_dir)):
            print(p)
