"""Artifact generation. Produces raw per-example predictions and raw matrices only.

No headline metric is computed here -- that is analyze.py's job, working from these
files alone (spec 6). A bug in the training loop therefore cannot silently produce
favourable metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .primitives import K
from .utils import write_json

USAGE_THRESHOLD = 0.5      # M2 step 1: atom counts as "used" by a task at >=50% of examples
EVAL_BATCH = 200


def _to_tensor(arr) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).long()


def _instruction_tensor(task, n: int) -> torch.Tensor:
    return torch.tensor(task.instruction, dtype=torch.long).unsqueeze(0).expand(n, -1)


@torch.no_grad()
def _run_task(model, td, mode: str, ablate: torch.Tensor | None = None):
    """Return (preds [n, L], routing_hard [n, T], routing_soft [n, T, N])."""
    xs = _to_tensor(td.inputs)
    instr = _instruction_tensor(td.task, xs.shape[0])
    preds, hard, soft = [], [], []
    for i in range(0, xs.shape[0], EVAL_BATCH):
        out = model(
            xs[i:i + EVAL_BATCH],
            instr[i:i + EVAL_BATCH],
            mode=mode,
            ablate=ablate,
        )
        preds.append(out["logits"].argmax(dim=-1))
        hard.append(out["routing_hard"])
        soft.append(out["routing_soft"])
    return torch.cat(preds), torch.cat(hard), torch.cat(soft)


def _exact_match(preds: torch.Tensor, targets: np.ndarray) -> np.ndarray:
    tgt = _to_tensor(targets)
    return (preds == tgt).all(dim=-1).numpy()


@torch.no_grad()
def write_predictions(model, tasks, path: Path, cfg) -> dict:
    """Per-example rows under hard routing, plus soft-routing correctness for M6."""
    path.parent.mkdir(parents=True, exist_ok=True)
    per_task = {}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for td in tasks:
            hard_preds, routing_hard, routing_soft = _run_task(model, td, "hard")
            soft_preds, _, _ = _run_task(model, td, "soft")
            correct = _exact_match(hard_preds, td.targets)
            correct_soft = _exact_match(soft_preds, td.targets)
            per_task[td.task.task_id] = {
                "acc_hard": float(correct.mean()),
                "acc_soft": float(correct_soft.mean()),
                "n": int(correct.shape[0]),
                "primitives": list(td.task.primitives),
                "kind": td.task.kind,
            }
            hp = hard_preds.numpy()
            rh = routing_hard.numpy()
            rs = np.round(routing_soft.numpy(), 4)
            for i in range(hp.shape[0]):
                row = {
                    "task_id": td.task.task_id,
                    "primitives": list(td.task.primitives),
                    "instruction": list(td.task.instruction),
                    "input": td.inputs[i].tolist(),
                    "target": td.targets[i].tolist(),
                    "pred": hp[i].tolist(),
                    "correct": bool(correct[i]),
                    "correct_soft": bool(correct_soft[i]),
                    "routing_hard": rh[i].tolist(),
                    "routing_soft": rs[i].tolist(),
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")
    return per_task


@torch.no_grad()
def compute_alignment(model, probe_inputs: np.ndarray, cfg):
    """M3 raw matrices A[i, p]: standalone accuracy of atom i against primitive p.

    Returns (A_depth_matched, A_one_step, identity_atom).

    The depth-matched probe applies atom_i then the model's own identity atom -- i.e.
    exactly the length-1 task (p, identity) that spec 1 puts in training to make
    standalone probing in-distribution. The one-step probe is the literal reading of
    spec 5 and is retained as a diagnostic; it is off-distribution because T is always
    2 and the decoder never sees a state with a single residual add. See DECISIONS.md D11.
    """
    from .primitives import apply_primitive

    xs = _to_tensor(probe_inputs)
    targets = {p: _to_tensor(apply_primitive(p, probe_inputs)) for p in range(K)}
    id_atom = model.identity_atom(xs)

    A = np.zeros((cfg.n_atoms, K), dtype=np.float64)
    A1 = np.zeros((cfg.n_atoms, K), dtype=np.float64)
    for i in range(cfg.n_atoms):
        dm, one = [], []
        for s in range(0, xs.shape[0], EVAL_BATCH):
            chunk = xs[s:s + EVAL_BATCH]
            dm.append(model.probe_atom_depth_matched(chunk, i, id_atom).argmax(dim=-1))
            one.append(model.probe_atom(chunk, i).argmax(dim=-1))
        dm, one = torch.cat(dm), torch.cat(one)
        for p in range(K):
            A[i, p] = (dm == targets[p]).all(dim=-1).float().mean().item()
            A1[i, p] = (one == targets[p]).all(dim=-1).float().mean().item()
    return A, A1, id_atom


@torch.no_grad()
def compute_ablation(model, eval_tasks, cfg):
    """M2 raw matrices.

    Returns dict with:
      usage    [N, n_tasks] fraction of a task's examples in which atom i is selected
      full_acc [n_tasks]
      abl_acc  [N, n_tasks] accuracy with atom i ablated (NaN where not evaluated)
      d        [N, n_tasks] full_acc - abl_acc  (NaN outside T_i)
      routing_counts [T, N]
      ablate_all_acc [n_tasks]
    """
    n_tasks = len(eval_tasks)
    N, T = cfg.n_atoms, cfg.depth

    usage = np.zeros((N, n_tasks))
    full_acc = np.zeros(n_tasks)
    routing_counts = np.zeros((T, N), dtype=np.int64)

    for j, td in enumerate(eval_tasks):
        preds, hard, _ = _run_task(model, td, "hard")
        full_acc[j] = _exact_match(preds, td.targets).mean()
        h = hard.numpy()
        for t in range(T):
            routing_counts[t] += np.bincount(h[:, t], minlength=N)
        for i in range(N):
            usage[i, j] = (h == i).any(axis=1).mean()

    abl = np.full((N, n_tasks), np.nan)
    d = np.full((N, n_tasks), np.nan)
    for i in range(N):
        ablate = torch.zeros(N, dtype=torch.bool)
        ablate[i] = True
        for j, td in enumerate(eval_tasks):
            if usage[i, j] < USAGE_THRESHOLD:
                continue
            preds, _, _ = _run_task(model, td, "hard", ablate=ablate)
            abl[i, j] = _exact_match(preds, td.targets).mean()
            d[i, j] = full_acc[j] - abl[i, j]

    ablate_all = torch.ones(N, dtype=torch.bool)
    ablate_all_acc = np.zeros(n_tasks)
    for j, td in enumerate(eval_tasks):
        preds, _, _ = _run_task(model, td, "hard", ablate=ablate_all)
        ablate_all_acc[j] = _exact_match(preds, td.targets).mean()

    return {
        "usage": usage,
        "full_acc": full_acc,
        "abl_acc": abl,
        "d": d,
        "routing_counts": routing_counts,
        "ablate_all_acc": ablate_all_acc,
    }


@torch.no_grad()
def compute_alignment_tensor(model, probe_inputs: np.ndarray, cfg) -> np.ndarray:
    """Assumption-free standalone probe: T[i, s, p] for every trailing atom s.

    The depth-matched probe (D11) fixes s to the model's identity atom. That is valid
    only if an identity slot exists -- and in free-routing arms it does not: the
    composer routes a length-1 task (p, identity) to (atom_p, atom_p), applying the
    same atom twice. Fixing s would then measure the wrong composition and depress
    M3_align for a reason unrelated to factorization.

    Recording the full tensor lets every variant be derived after the fact, and lets
    a human audit which trailing atom carried a given reading. See DECISIONS.md D21.
    """
    from .primitives import apply_primitive

    xs = _to_tensor(probe_inputs)
    n = xs.shape[0]
    targets = {p: _to_tensor(apply_primitive(p, probe_inputs)) for p in range(K)}
    T = np.zeros((cfg.n_atoms, cfg.n_atoms, K), dtype=np.float64)

    instr = torch.zeros(n, cfg.depth, dtype=torch.long)
    for i in range(cfg.n_atoms):
        for s in range(cfg.n_atoms):
            forced = torch.empty(n, cfg.depth, dtype=torch.long)
            forced[:, 0] = i
            forced[:, 1:] = s
            preds = []
            for b in range(0, n, EVAL_BATCH):
                preds.append(model(xs[b:b + EVAL_BATCH], instr[b:b + EVAL_BATCH],
                                   mode="forced",
                                   forced=forced[b:b + EVAL_BATCH])["logits"].argmax(-1))
            preds = torch.cat(preds)
            for p in range(K):
                T[i, s, p] = (preds == targets[p]).all(dim=-1).float().mean().item()
    return T


@torch.no_grad()
def compute_state_alignment(model, probe_inputs: np.ndarray, cfg):
    """Decoder-free, depth-free standalone probe. Needs no trailing atom at all.

    Both decoder-based probes need a second residual add, and are therefore only as
    trustworthy as the atom supplying it: assuming an identity slot understates an
    arm that has none, and taking the best trailing atom overstates one whose best
    partner does real work (audited via atom_residual_norms). This probe sidesteps
    the issue by asking the closed-map question directly in state space:

        does  h0 + atom_i(h0)  land where  enc(p(x))  is?

    Returns (err, acc) where
      err[i, p]  mean relative L2 distance from atom i's output state to enc(p(x))
      acc[i, p]  fraction of inputs whose nearest primitive-encoding is p

    err ~ 0 for the matched primitive means atom i is a genuine closed map on the
    encoder manifold -- the property A0 established is achievable. See D21.
    """
    from .primitives import apply_primitive

    xs = _to_tensor(probe_inputs)
    h0 = model.code(xs)
    # Canonical targets, and the model's own transition -- never a hand-built
    # h0 + atom(h0), which diverges from the real path once LayerNorm or the R3
    # bottleneck is in it (D26). `project=False`: the closed-map question is whether
    # the ATOM maps code to code, and R3's projection would snap any output onto a
    # valid code and hide exactly what is being measured.
    encs = torch.stack([model.code(_to_tensor(apply_primitive(p, probe_inputs)))
                        for p in range(K)], dim=1)          # [B, K, S]

    err = np.zeros((cfg.n_atoms, K))
    acc = np.zeros((cfg.n_atoms, K))
    denom = encs.norm(dim=2).clamp_min(1e-6)                # [B, K]
    for i in range(cfg.n_atoms):
        hi = model.step(h0, i, project=False).unsqueeze(1)  # [B, 1, S]
        dist = (hi - encs).norm(dim=2)                      # [B, K]
        rel = dist / denom
        err[i] = rel.mean(dim=0).numpy()
        nearest = dist.argmin(dim=1)
        for p in range(K):
            acc[i, p] = (nearest == p).float().mean().item()
    return err, acc


@torch.no_grad()
def atom_residual_norms(model, probe_inputs: np.ndarray) -> np.ndarray:
    """Mean ||atom_s(h0)|| / ||h0||, so a 'best trailing atom' reading is auditable.

    A trailing atom with a near-zero residual is acting as a no-op and the reading is
    clean; a large residual means atom s did real work and the reading is inflated.
    """
    xs = _to_tensor(probe_inputs)
    # model.code, not model.encoder: under R1+ the atoms see normed states, so norms
    # taken against an un-normed h0 would not be comparable across rungs (D26).
    h0 = model.code(xs)
    outs = model.atoms.outputs(h0)
    return (outs.norm(dim=2) / h0.norm(dim=1, keepdim=True)).mean(dim=0).numpy()


@torch.no_grad()
def compute_code_spread(model, probe_inputs: np.ndarray) -> dict:
    """Collapse detector: does the code still distinguish different inputs?

    A code-consistency constraint is satisfiable by DESTROYING the representation --
    map every input to one point and enc(dec(h)) matches h exactly, drift vanishes
    because every encoding is the same encoding, and closed-map error goes to zero
    with coverage 1/8. Accuracy goes to zero too, so the pre-registered gate rejects
    it, but every manifold metric looks excellent in isolation. Measured on R2 w=10:
    spread 0.944 -> 0.025, a 25x collapse. See D29.
    """
    xs = _to_tensor(probe_inputs)
    h = model.code(xs)
    spread = float((h - h.mean(0, keepdim=True)).norm(dim=1).mean()
                   / h.norm(dim=1).mean().clamp_min(1e-6))
    k = min(64, h.shape[0])
    d = torch.cdist(h[:k], h[:k])
    off = d[~torch.eye(k, dtype=torch.bool)]
    return {
        "code_spread": spread,
        "code_pairwise_mean": float(off.mean()),
        "code_pairwise_min": float(off.min()),
    }


@torch.no_grad()
def compute_code_diagnostics(model, tasks, cfg) -> dict:
    """E1b: is the intermediate state a valid code, and is the constraint being gamed?

    `residual_hard` alone says whether the R2/R3 constraint BOUND, separately from
    whether it helped -- a rung whose residual stays high was not trained hard enough,
    and its negative result is uninformative rather than a finding.

    Three companions make the known gaming modes visible at a glance instead of by
    hand-investigation after the fact (D28):

    - `residual_soft` -- the same round trip through softmax rather than argmax. A
      large soft/hard GAP is the signature of D27's failure: the model satisfies a
      soft constraint by emitting an uninformative decode that re-encodes to one
      fixed point, without h_t ever being a valid code.
    - `entropy` -- decoder entropy at the intermediate step, in nats. D27 was caught
      by this number (2.203 of a maximum 2.303) and it was computed by hand.
    - `distinct_frac` -- distinct argmax sequences per task, as a fraction of that
      task's examples. **This is the one that matters for R3.** R3 projects through a
      hard Gumbel decode, so every decode is confident by construction and entropy
      will look perfect no matter what. R3's collapse mode is mapping every input to
      the SAME token sequence, which only a diversity count reveals.
    """
    res_hard, res_soft, ents, per_task = [], [], [], []
    for td in tasks:
        xs = _to_tensor(td.inputs)
        instr = _instruction_tensor(td.task, xs.shape[0])
        seqs, n_seen = set(), 0
        for b in range(0, xs.shape[0], EVAL_BATCH):
            out = model(xs[b:b + EVAL_BATCH], instr[b:b + EVAL_BATCH], mode="hard")
            for t in range(cfg.depth - 1):
                h_t = out["states"][:, t]
                res_hard.append(model.code_residual(h_t).numpy())
                soft = model.recode(h_t, hard=False)
                res_soft.append(
                    ((h_t - soft).norm(dim=1)
                     / h_t.norm(dim=1).clamp_min(1e-6)).numpy())
                logits = model.decoder(h_t)
                ents.append((-(F.softmax(logits, -1) * F.log_softmax(logits, -1))
                             .sum(-1).mean(-1)).numpy())
                if t == 0:
                    seqs.update(map(tuple, logits.argmax(-1).tolist()))
            n_seen += xs[b:b + EVAL_BATCH].shape[0]
        per_task.append({"task_id": td.task.task_id,
                         "distinct": len(seqs), "n": n_seen,
                         "distinct_frac": len(seqs) / max(n_seen, 1)})
    cat = lambda v: np.concatenate(v) if v else np.zeros(0)
    return {
        "residual_hard": cat(res_hard),
        "residual_soft": cat(res_soft),
        "entropy": cat(ents),
        "per_task_diversity": per_task,
    }


@torch.no_grad()
def compute_code_residual(model, tasks, cfg) -> np.ndarray:
    """Backwards-compatible view of compute_code_diagnostics' hard residual."""
    return compute_code_diagnostics(model, tasks, cfg)["residual_hard"]


@torch.no_grad()
def compute_manifold_diagnostic(model, unseen_tasks, cfg):
    """M7 -- does residual composition preserve the encoder manifold?

    This is the diagnostic from DECISIONS.md D12, promoted to a standard metric
    because it is what separates FAIL(architectural) from FAIL(representational)
    and FAIL(optimizer). For each unseen task it measures:

      drift[t]        relative L2 distance from h_t to enc(y_t), the encoding of the
                      partial composition -- how far off-manifold the state has gone
      acc_teacher     accuracy when h_1 is REPLACED by the true enc(y_1) and the
                      model's own routing choice is then applied

    High `acc_teacher` alongside low actual accuracy means every atom is already a
    correct closed map on the encoder manifold and only the composition operator is
    at fault. That is an architectural finding, not a refutation of H6.
    """
    from .primitives import apply_composition

    n = len(unseen_tasks)
    drift = np.zeros((n, cfg.depth))
    acc_teacher = np.zeros(n)
    acc_actual = np.zeros(n)

    for j, td in enumerate(unseen_tasks):
        xs = _to_tensor(td.inputs)
        instr = _instruction_tensor(td.task, xs.shape[0])
        d_acc = np.zeros(cfg.depth)
        t_hits = a_hits = total = 0

        for s in range(0, xs.shape[0], EVAL_BATCH):
            xb, ib = xs[s:s + EVAL_BATCH], instr[s:s + EVAL_BATCH]
            out = model(xb, ib, mode="hard")
            chosen = out["routing_hard"]           # what the model actually does

            for t in range(cfg.depth):
                y_t = apply_composition(td.task.instruction[:t + 1],
                                        td.inputs[s:s + EVAL_BATCH])
                enc_t = model.code(_to_tensor(y_t))
                h_t = out["states"][:, t]
                rel = (h_t - enc_t).norm(dim=1) / enc_t.norm(dim=1).clamp_min(1e-6)
                d_acc[t] += float(rel.sum())

            # Teacher-forced: start step 2 from the true encoding of the intermediate.
            y1 = apply_composition(td.task.instruction[:1], td.inputs[s:s + EVAL_BATCH])
            enc1 = model.code(_to_tensor(y1))
            # Reproduce the model's own final step. forward() does not project after
            # the last step, so project=False here mirrors it exactly (D26).
            h2 = model.step(enc1, chosen[:, 1], project=False)
            tgt = _to_tensor(td.targets[s:s + EVAL_BATCH])
            t_hits += int((model.decoder(h2).argmax(-1) == tgt).all(dim=-1).sum())
            a_hits += int((out["logits"].argmax(-1) == tgt).all(dim=-1).sum())
            total += xb.shape[0]

        drift[j] = d_acc / max(total, 1)
        acc_teacher[j] = t_hits / max(total, 1)
        acc_actual[j] = a_hits / max(total, 1)

    return {"drift": drift, "acc_teacher": acc_teacher, "acc_actual": acc_actual}


def emit_artifacts(model, bundle, cfg, run_dir: Path, param_counts: dict) -> None:
    art = run_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    model.eval()

    per_task = {}
    per_task["seen_heldout"] = write_predictions(
        model, bundle.seen_heldout, art / "predictions_seen_heldout.jsonl", cfg)
    per_task["unseen"] = write_predictions(
        model, bundle.unseen, art / "predictions_unseen.jsonl", cfg)
    per_task["singleton"] = write_predictions(
        model, bundle.singleton, art / "predictions_singleton.jsonl", cfg)

    A, A1, id_atom = compute_alignment(model, bundle.probe_inputs, cfg)
    np.save(art / "alignment_matrix.npy", A)
    np.save(art / "alignment_matrix_1step.npy", A1)

    eval_tasks = bundle.all_eval_tasks
    abl = compute_ablation(model, eval_tasks, cfg)
    np.save(art / "ablation_matrix.npy", abl["d"])
    np.save(art / "ablation_usage.npy", abl["usage"])
    np.save(art / "ablation_full_acc.npy", abl["full_acc"])
    np.save(art / "ablation_abl_acc.npy", abl["abl_acc"])
    np.save(art / "routing_counts.npy", abl["routing_counts"])
    np.save(art / "ablate_all_acc.npy", abl["ablate_all_acc"])

    np.save(art / "alignment_tensor.npy",
            compute_alignment_tensor(model, bundle.probe_inputs, cfg))
    np.save(art / "atom_residual_norms.npy",
            atom_residual_norms(model, bundle.probe_inputs))
    _err, _acc = compute_state_alignment(model, bundle.probe_inputs, cfg)
    np.save(art / "state_alignment_err.npy", _err)
    np.save(art / "state_alignment_acc.npy", _acc)

    for name, tasks in (("seen", bundle.seen_heldout), ("unseen", bundle.unseen)):
        cd = compute_code_diagnostics(model, tasks, cfg)
        np.save(art / f"code_residual_{name}.npy", cd["residual_hard"])
        np.save(art / f"code_residual_soft_{name}.npy", cd["residual_soft"])
        np.save(art / f"code_entropy_{name}.npy", cd["entropy"])
        write_json(art / f"code_diversity_{name}.json", cd["per_task_diversity"])
    write_json(art / "code_spread.json",
               compute_code_spread(model, bundle.probe_inputs))

    man = compute_manifold_diagnostic(model, bundle.unseen, cfg)
    np.save(art / "manifold_drift.npy", man["drift"])
    np.save(art / "manifold_acc_teacher.npy", man["acc_teacher"])
    np.save(art / "manifold_acc_actual.npy", man["acc_actual"])

    write_json(art / "param_counts.json", param_counts)
    write_json(art / "task_index.json", {
        "ablation_columns": [td.task.task_id for td in eval_tasks],
        "ablation_column_split": (
            ["seen_heldout"] * len(bundle.seen_heldout) + ["unseen"] * len(bundle.unseen)
        ),
        "ablation_column_primitives": [list(td.task.primitives) for td in eval_tasks],
    })
    write_json(art / "raw_accuracies.json", {
        "per_task": per_task,
        "identity_atom": id_atom,
        "ablate_all_mean_acc": float(abl["ablate_all_acc"].mean()),
        "full_mean_acc_eval_tasks": float(abl["full_acc"].mean()),
        "usage_threshold": USAGE_THRESHOLD,
    })
