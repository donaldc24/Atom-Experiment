"""E9 teacher-forced program crystallization and concentration telemetry.

The completed A0-free model is a fixed three-routing-decisions-per-token
teacher.  E9 students execute one decision per token.  A26 tests whether
copying the learned parameters is sufficient; A27 adds a smoothly ramped,
bounded auxiliary loss that matches the teacher at token boundaries.

The concentration measurements intentionally mirror only the *norm shape* of
the Navier-Stokes result: L2-like total energy, L-infinity-like peak, and an
inverse-participation effective support.  They do not identify neural-network
dynamics with fluid dynamics.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import registered as R
from .config import Config
from .model import AtomModel
from .utils import RUNS_DIR, read_json, sha256_of_file


def find_teacher_run(cfg: Config) -> Path:
    """Resolve the one registered, completed, seed-paired teacher run."""
    root = RUNS_DIR / cfg.crystal_teacher_experiment
    hits = []
    for p in sorted(root.glob(
            f"{cfg.crystal_teacher_arm}_s{cfg.seed}_*")):
        if ((p / "checkpoints" / "final.pt").exists()
                and (p / "metrics.json").exists()):
            metrics = read_json(p / "metrics.json")
            if (metrics.get("protocol_revision")
                    == R.PROTOCOL_REVISION):
                hits.append(p)
    if len(hits) != 1:
        raise SystemExit(
            f"E9 requires exactly one completed seed-paired teacher "
            f"{cfg.crystal_teacher_experiment}/{cfg.crystal_teacher_arm} "
            f"seed {cfg.seed}; found {len(hits)} under {root}: {hits}")
    return hits[0]


def _teacher_metrics(run_dir: Path) -> dict:
    m = read_json(run_dir / "metrics.json")
    return {
        "acc_seen_hard": m["acc_seen_hard"],
        "acc_unseen_L1_hard": m["acc_unseen_L1_hard"],
        "acc_unseen_L2_hard": m["acc_unseen_L2_hard"],
        "acc_unseen_L3_hard": m["acc_unseen_L3_hard"],
        "final_step": m["final_step"],
        "protocol_revision": m["protocol_revision"],
    }


def copy_teacher_to_student(student: AtomModel,
                            teacher_state: dict[str, torch.Tensor]) -> dict:
    """Copy a 3-step teacher into a 1-step student without inventing weights.

    All tensors have identical shapes except ``composer.micro_emb.weight``;
    the student receives row zero, the only micro-step it will execute.
    Any other mismatch is a hard implementation failure.
    """
    if student.cfg.micro_steps != 1:
        raise ValueError("E9 student must execute exactly one step per token")
    target = student.state_dict()
    copied, sliced, bad = [], [], []
    for key, dst in target.items():
        src = teacher_state.get(key)
        if src is None:
            bad.append({"key": key, "reason": "missing from teacher"})
        elif src.shape == dst.shape:
            target[key] = src.detach().clone()
            copied.append(key)
        elif (key == "composer.micro_emb.weight" and src.ndim == 2
              and dst.shape == src[:1].shape):
            target[key] = src[:1].detach().clone()
            sliced.append({"key": key, "teacher_rows": int(src.shape[0]),
                           "student_rows": 1, "copied_row": 0})
        else:
            bad.append({"key": key, "teacher_shape": list(src.shape),
                        "student_shape": list(dst.shape)})
    extra = sorted(set(teacher_state) - set(target))
    if bad or extra:
        raise RuntimeError(
            f"E9 teacher/student copy mismatch: bad={bad}, extra={extra}")
    student.load_state_dict(target, strict=True)
    return {
        "same_shape_tensors": copied,
        "sliced_tensors": sliced,
        "n_same_shape_tensors": len(copied),
        "n_student_tensors": len(target),
        "n_copied_elements": int(sum(target[k].numel() for k in copied)
                                 + sum(target[x["key"]].numel()
                                       for x in sliced)),
        "n_student_elements": int(sum(v.numel() for v in target.values())),
    }


def prepare_student(student: AtomModel, cfg: Config,
                    teacher_checkpoint: Path | None = None,
                    require_teacher_metrics: bool = True
                    ) -> tuple[AtomModel | None, dict]:
    """Optionally warm-start the student and return its frozen teacher.

    The scratch arm records its intended reference but never reads or copies a
    teacher.  Warm-start arms pin the exact checkpoint by SHA-256.  A teacher
    is returned only when a nonzero forcing loss needs it during training.
    """
    base = {
        "student_arm": cfg.arm,
        "student_seed": cfg.seed,
        "student_micro_steps": cfg.micro_steps,
        "teacher_experiment": cfg.crystal_teacher_experiment,
        "teacher_arm": cfg.crystal_teacher_arm,
        "lambda_crystal_state": cfg.lambda_crystal_state,
        "lambda_crystal_logits": cfg.lambda_crystal_logits,
        "force_ramp_steps": cfg.crystal_force_ramp_steps,
    }
    if not cfg.crystal_warm_start:
        return None, {**base, "mode": "scratch", "teacher_read": False,
                      "warm_started": False}

    if teacher_checkpoint is None:
        teacher_run = find_teacher_run(cfg)
        teacher_checkpoint = teacher_run / "checkpoints" / "final.pt"
    else:
        teacher_checkpoint = Path(teacher_checkpoint)
        teacher_run = teacher_checkpoint.parent.parent
    ck = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher_cfg = Config.from_dict(ck["config"])
    if (teacher_cfg.seed != cfg.seed
            or teacher_cfg.arm != cfg.crystal_teacher_arm
            or teacher_cfg.experiment != cfg.crystal_teacher_experiment
            or teacher_cfg.micro_steps != 3):
        raise RuntimeError(
            "E9 teacher identity mismatch: expected "
            f"{cfg.crystal_teacher_experiment}/{cfg.crystal_teacher_arm} "
            f"seed={cfg.seed}, micro_steps=3; got "
            f"{teacher_cfg.experiment}/{teacher_cfg.arm} "
            f"seed={teacher_cfg.seed}, micro_steps={teacher_cfg.micro_steps}")
    metrics = None
    if require_teacher_metrics:
        metrics = _teacher_metrics(teacher_run)
        if metrics["acc_seen_hard"] < R.E9_TEACHER_SEEN_MIN:
            raise SystemExit(
                f"E9 teacher gate failed: {teacher_run} seen="
                f"{metrics['acc_seen_hard']:.4f} < "
                f"{R.E9_TEACHER_SEEN_MIN:.2f}")

    teacher = AtomModel(teacher_cfg)
    teacher.load_state_dict(ck["model"], strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    copy = copy_teacher_to_student(student, ck["model"])
    forced = cfg.lambda_crystal_state > 0 or cfg.lambda_crystal_logits > 0
    receipt = {
        **base,
        "mode": "teacher_forced" if forced else "warm_start_unforced",
        "teacher_read": True,
        "warm_started": True,
        "teacher_run": str(teacher_run.resolve()),
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256_of_file(teacher_checkpoint),
        "teacher_config": {
            "arm": teacher_cfg.arm, "seed": teacher_cfg.seed,
            "experiment": teacher_cfg.experiment,
            "protocol_revision": teacher_cfg.protocol_revision,
            "micro_steps": teacher_cfg.micro_steps,
        },
        "teacher_metrics": metrics,
        "copy": copy,
    }
    return (teacher if forced else None), receipt


def force_scale(cfg: Config, zero_based_step: int) -> float:
    """Linear, continuous-at-origin ramp to the registered forcing dose."""
    if cfg.crystal_force_ramp_steps <= 0:
        return 1.0
    return min((zero_based_step + 1) / cfg.crystal_force_ramp_steps, 1.0)


def distillation_losses(student_out: dict, teacher_out: dict,
                        n_tokens: torch.Tensor, student_micro_steps: int,
                        teacher_micro_steps: int) -> dict[str, torch.Tensor]:
    """Match private token-boundary states and the final output distribution."""
    if student_micro_steps != 1 or teacher_micro_steps != 3:
        raise ValueError("E9 loss is registered for a 3 -> 1 contraction")
    boundary = []
    for pos in range(2):
        live = n_tokens > pos
        if bool(live.any()):
            s = student_out["states"][(pos + 1) * student_micro_steps][live]
            t = teacher_out["states"][(pos + 1) * teacher_micro_steps][live]
            boundary.append(F.mse_loss(s, t.detach()))
    state = torch.stack(boundary).mean()
    logits = F.kl_div(
        F.log_softmax(student_out["logits"], dim=-1),
        F.softmax(teacher_out["logits"].detach(), dim=-1),
        reduction="batchmean") / student_out["logits"].shape[1]
    with torch.no_grad():
        agree = (student_out["logits"].argmax(-1)
                 == teacher_out["logits"].argmax(-1)).all(-1).float().mean()
    return {"loss_crystal_state": state, "loss_crystal_logits": logits,
            "crystal_teacher_agreement": agree}


def diagnostic_batch(bundle, cfg: Config) -> dict[str, np.ndarray]:
    """Deterministic, task-balanced training batch; consumes no RNG stream."""
    each = max(1, cfg.batch_size // len(bundle.train))
    xs, ys, tokens, n_tokens = [], [], [], []
    for td in bundle.train:
        n = min(each, len(td.x))
        xs.append(td.x[:n])
        ys.append(td.y[:n])
        tokens.append(np.tile(td.task.tokens, (n, 1)))
        n_tokens.append(np.full(n, td.task.n_tokens, dtype=np.int64))
    return {"x": np.concatenate(xs), "y": np.concatenate(ys),
            "tokens": np.concatenate(tokens),
            "n_tokens": np.concatenate(n_tokens)}


def tensor_concentration(values: torch.Tensor) -> dict:
    """L2 energy, L-infinity peak, and inverse-participation support."""
    x = values.detach().float().reshape(-1, values.shape[-1])
    sq = x.square()
    energy = sq.sum(-1)
    fourth = sq.square().sum(-1)
    nonzero = energy > 0
    peak_fraction = torch.zeros_like(energy)
    effective = torch.zeros_like(energy)
    peak_fraction[nonzero] = (sq.max(-1).values[nonzero]
                              / energy[nonzero])
    effective[nonzero] = energy[nonzero].square() / fourth[nonzero]
    return {
        "n": int(len(x)),
        "dimension": int(x.shape[-1]),
        "energy_l2_sq_mean": float(energy.mean()),
        "energy_l2_sq_max": float(energy.max()),
        "peak_linf_mean": float(x.abs().max(-1).values.mean()),
        "peak_linf_max": float(x.abs().max()),
        "peak_energy_fraction_mean": float(peak_fraction.mean()),
        "effective_coordinates_mean": float(effective.mean()),
    }


@torch.no_grad()
def measure(model: AtomModel, diag: dict[str, np.ndarray], step: int) -> dict:
    """Read-only concentration record on token-boundary states and updates."""
    was_training = model.training
    model.eval()
    x = torch.from_numpy(diag["x"])
    tokens = torch.from_numpy(diag["tokens"])
    n_tokens = torch.from_numpy(diag["n_tokens"])
    out = model(x, tokens, n_tokens, mode="hard")
    boundaries, updates = [], []
    ms = model.cfg.micro_steps
    for pos in range(2):
        live = n_tokens > pos
        if bool(live.any()):
            start = out["states"][pos * ms][live]
            end = out["states"][(pos + 1) * ms][live]
            boundaries.append(end)
            updates.append(end - start)
    states = torch.cat(boundaries)
    deltas = torch.cat(updates)
    choices = out["choices"][out["live"]]
    counts = torch.bincount(choices, minlength=model.cfg.n_atoms + 1).float()
    probs = counts / counts.sum()
    route_eff = (1.0 / probs[probs > 0].square().sum()
                 if bool((probs > 0).any()) else torch.tensor(0.0))
    atom_counts = counts[:model.cfg.n_atoms]
    atom_probs = (atom_counts / atom_counts.sum()
                  if float(atom_counts.sum()) else atom_counts)
    atom_eff = (1.0 / atom_probs[atom_probs > 0].square().sum()
                if bool((atom_probs > 0).any()) else torch.tensor(0.0))
    expected_live = int(n_tokens.sum())
    record = {
        "step": step,
        "arm": model.cfg.arm,
        "micro_steps": ms,
        "state": tensor_concentration(states),
        "token_update": tensor_concentration(deltas),
        "routing": {
            "n_live_decisions": int(len(choices)),
            "expected_one_per_token": expected_live,
            "one_route_per_token": bool(len(choices) == expected_live),
            "counts": counts.to(torch.int64).tolist(),
            "effective_routes": float(route_eff),
            "effective_atoms_excluding_pass": float(atom_eff),
            "top_route_fraction": float(probs.max()),
            "pass_fraction": float(probs[-1]),
        },
    }
    if was_training:
        model.train()
    return record


def gradient_concentration(model: AtomModel, grad_clip: float) -> dict:
    """Concentration of the current pre-clip gradient across atom transforms."""
    global_energy = 0.0
    for p in model.parameters():
        if p.grad is not None:
            global_energy += float(p.grad.detach().square().sum())

    energies, sizes = [], []
    direct = (model.atoms.w1, model.atoms.b1,
              model.atoms.w2, model.atoms.b2)
    for i in range(model.cfg.n_atoms):
        energy, size = 0.0, 0
        for p in direct:
            size += p[i].numel()
            if p.grad is not None:
                energy += float(p.grad[i].detach().square().sum())
        if hasattr(model.atoms, "wm"):
            for p in (model.atoms.wm, model.atoms.bm):
                size += p[:, i].numel()
                if p.grad is not None:
                    energy += float(p.grad[:, i].detach().square().sum())
        energies.append(energy)
        sizes.append(size)

    e = np.asarray(energies, dtype=np.float64)
    atom_total = float(e.sum())
    effective = (atom_total * atom_total / float(np.square(e).sum())
                 if atom_total > 0 and float(np.square(e).sum()) > 0 else 0.0)
    global_norm = math.sqrt(global_energy)
    return {
        "grad_global_norm_preclip": global_norm,
        "grad_global_norm_postclip": min(global_norm, grad_clip),
        "grad_global_energy_preclip": global_energy,
        "grad_atom_transform_energy": atom_total,
        "grad_atom_peak_fraction_within_atoms": (
            float(e.max() / atom_total) if atom_total > 0 else 0.0),
        "grad_atom_peak_fraction_global": (
            float(e.max() / global_energy) if global_energy > 0 else 0.0),
        "grad_atom_effective_support": effective,
        "grad_atom_peak_rms": max(
            (math.sqrt(x / n) if n else 0.0)
            for x, n in zip(energies, sizes)),
        "grad_atom_energy_per_atom": energies,
    }
