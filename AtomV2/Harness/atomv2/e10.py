"""E10 training, evaluation, extrapolation, and flow audits."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as data_mod
from . import ops
from . import registered as R
from . import split as split_mod
from .flow_model import ReynoldsFlowModel
from .utils import (JsonlLogger, RUNS_DIR, check_rss, env_info, git_info,
                    require_clean_tree, seed_everything, set_threads,
                    stream_rng, write_json, write_sha256sums)


@dataclass
class E10Config:
    arm: str
    seed: int
    smoke: bool = False
    experiment: str = "e10"
    protocol_revision: str = R.E10_PROTOCOL_REVISION
    transport: str = ""
    incompressible: bool = False
    viscosity: float = 0.0
    width: int = R.E10_WIDTH
    token_dim: int = R.E10_TOKEN_DIM
    pos_dim: int = R.E10_POS_DIM
    force_hidden: int = R.E10_FORCE_HIDDEN
    pulse_rank: int = R.E10_PULSE_RANK
    sinkhorn_iters: int = R.E10_SINKHORN_ITERS
    transport_temp: float = R.E10_TRANSPORT_TEMP
    identity_bias: float = R.E10_IDENTITY_BIAS
    examples_per_train_task: int = R.EXAMPLES_PER_TRAIN_TASK
    examples_per_eval_task: int = R.EXAMPLES_PER_EVAL_TASK
    n_probe_examples: int = R.N_PROBE_EXAMPLES
    p3_oversample_factor: int = R.P3_OVERSAMPLE_FACTOR
    lr: float = R.E10_LR
    betas: tuple = R.E10_BETAS
    weight_decay: float = R.E10_WEIGHT_DECAY
    warmup_steps: int = R.E10_WARMUP_STEPS
    batch_size: int = R.E10_BATCH_SIZE
    total_steps: int = R.E10_TOTAL_STEPS
    eval_every: int = R.E10_EVAL_EVERY
    log_every: int = R.E10_LOG_EVERY
    grad_clip: float = R.E10_GRAD_CLIP
    extrap_tasks_per_length: int = R.E10_EXTRAP_TASKS_PER_LENGTH
    extrap_examples_per_task: int = R.E10_EXTRAP_EXAMPLES_PER_TASK
    noise_sigma: float = R.E10_NOISE_SIGMA

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["betas"] = list(self.betas)
        return out


def config_for_e10(arm: str, seed: int, smoke: bool = False) -> E10Config:
    if arm not in R.E10_ARMS:
        raise ValueError(f"unknown E10 arm {arm!r}")
    cfg = E10Config(
        arm=arm, seed=seed, smoke=smoke,
        transport=R.E10_TRANSPORT[arm],
        incompressible=R.E10_INCOMPRESSIBLE[arm],
        viscosity=R.E10_VISCOSITY[arm])
    if smoke:
        cfg.examples_per_train_task = 48
        cfg.examples_per_eval_task = 24
        cfg.n_probe_examples = 24
        cfg.warmup_steps = 20
        cfg.total_steps = 300
        cfg.eval_every = 100
        cfg.log_every = 20
        cfg.extrap_tasks_per_length = 8
        cfg.extrap_examples_per_task = 12
    return cfg


def _run_id(cfg: E10Config) -> str:
    git = git_info()
    source = git["git_sha_short"]
    if git.get("dirty_source_sha256"):
        source += f"-dirty{git['dirty_source_sha256'][:8]}"
    return f"{cfg.arm}_s{cfg.seed}_{cfg.protocol_revision}_{source}"


def run_dir_for_e10(cfg: E10Config, out: str | None = None) -> Path:
    base = Path(out) if out else RUNS_DIR
    sub = "smoke_e10" if cfg.smoke else "e10"
    return base / sub / _run_id(cfg)


def _stable_rng(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"e10|{seed}|{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _unique_inputs(rng: np.random.Generator, n: int) -> np.ndarray:
    seen: set[bytes] = set()
    rows = []
    while len(rows) < n:
        for row in rng.integers(0, R.VOCAB,
                                size=(max(64, n), R.SEQ_LEN),
                                dtype=np.int64):
            key = row.tobytes()
            if key not in seen:
                seen.add(key)
                rows.append(row.copy())
                if len(rows) == n:
                    break
    return np.stack(rows)


def _apply_sequence(names: tuple[str, ...], x: np.ndarray) -> np.ndarray:
    y = x
    for name in names:
        y = ops.SURFACE_FNS[name](y)
    return y


def _array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def build_extrapolation(cfg: E10Config) -> tuple[dict[int, list[dict]], dict]:
    """Fixed selected sequences; fresh deterministic inputs per run seed."""
    panels: dict[int, list[dict]] = {}
    manifest = {
        "selection_seed": R.E10_EXTRAP_SELECTION_SEED,
        "run_seed": cfg.seed,
        "lengths": {},
    }
    for length in (3, 4):
        all_seq = list(itertools.product(ops.SURFACE_NAMES, repeat=length))
        select_rng = np.random.default_rng(
            R.E10_EXTRAP_SELECTION_SEED + length)
        picked = np.sort(select_rng.choice(
            len(all_seq), size=cfg.extrap_tasks_per_length, replace=False))
        panels[length] = []
        entries = []
        for index in picked:
            names = tuple(all_seq[int(index)])
            task_id = "_".join(names)
            x = _unique_inputs(
                _stable_rng(cfg.seed, f"extrap|{length}|{task_id}"),
                cfg.extrap_examples_per_task)
            y = _apply_sequence(names, x)
            tokens = np.asarray(
                [data_mod.SURFACE_INDEX[name] for name in names],
                dtype=np.int64)
            panels[length].append({"task_id": task_id, "x": x, "y": y,
                                   "tokens": tokens})
            entries.append({"task_id": task_id, "n": len(x),
                            "x_sha256": _array_hash(x),
                            "y_sha256": _array_hash(y)})
        manifest["lengths"][str(length)] = entries
    return panels, manifest


@torch.no_grad()
def _task_accuracy(model: ReynoldsFlowModel, x: np.ndarray, y: np.ndarray,
                   tokens: np.ndarray, noise: np.ndarray | None = None,
                   batch_size: int = 256) -> float:
    model.eval()
    correct = []
    tiled = np.tile(tokens[None, :], (len(x), 1))
    for lo in range(0, len(x), batch_size):
        hi = min(lo + batch_size, len(x))
        nz = None if noise is None else torch.from_numpy(noise[lo:hi])
        out = model(torch.from_numpy(x[lo:hi]),
                    torch.from_numpy(tiled[lo:hi]),
                    handoff_noise=nz)
        pred = out["logits"].argmax(dim=-1).cpu().numpy()
        correct.append((pred == y[lo:hi]).all(axis=1))
    return float(np.concatenate(correct).mean())


@torch.no_grad()
def evaluate_standard(model: ReynoldsFlowModel, bundle: data_mod.Bundle) -> dict:
    sets = [("seen", bundle.seen_heldout),
            ("L1", bundle.unseen["L1"]),
            ("L2", bundle.unseen["L2"]),
            ("L3", bundle.unseen["L3"])]
    out = {}
    for name, tasks in sets:
        per_task = {}
        for td in tasks:
            per_task[td.task.task_id] = _task_accuracy(
                model, td.x, td.y, td.task.tokens[:td.task.n_tokens])
        out[name] = {"mean": float(np.mean(list(per_task.values()))),
                     "tasks": per_task}
    return out


@torch.no_grad()
def evaluate_extrapolation(model: ReynoldsFlowModel,
                           panels: dict[int, list[dict]]) -> dict:
    out = {}
    for length, tasks in panels.items():
        per_task = {td["task_id"]: _task_accuracy(
            model, td["x"], td["y"], td["tokens"]) for td in tasks}
        out[str(length)] = {"mean": float(np.mean(list(per_task.values()))),
                            "tasks": per_task}
    return out


def _all_pair_tasks(bundle: data_mod.Bundle) -> list[data_mod.TaskData]:
    return ([td for td in bundle.seen_heldout if td.task.n_tokens == 2]
            + bundle.unseen["L1"] + bundle.unseen["L2"]
            + bundle.unseen["L3"])


@torch.no_grad()
def evaluate_hidden_noise(model: ReynoldsFlowModel, bundle: data_mod.Bundle,
                          cfg: E10Config) -> dict:
    per_clean, per_noisy = {}, {}
    for td in _all_pair_tasks(bundle):
        tokens = td.task.tokens[:2]
        clean = _task_accuracy(model, td.x, td.y, tokens)
        rng = _stable_rng(cfg.seed, f"hidden-noise|{td.task.task_id}")
        noise = (cfg.noise_sigma * rng.standard_normal(
            (len(td.x), 1, R.SEQ_LEN, R.E10_WIDTH))).astype(np.float32)
        noisy = _task_accuracy(model, td.x, td.y, tokens, noise=noise)
        per_clean[td.task.task_id] = clean
        per_noisy[td.task.task_id] = noisy
    return {
        "sigma": cfg.noise_sigma,
        "clean_mean": float(np.mean(list(per_clean.values()))),
        "noisy_mean": float(np.mean(list(per_noisy.values()))),
        "drop": float(np.mean(list(per_clean.values()))
                - np.mean(list(per_noisy.values()))),
        "clean_tasks": per_clean,
        "noisy_tasks": per_noisy,
    }


@torch.no_grad()
def flow_audit(model: ReynoldsFlowModel, seed: int) -> dict:
    model.eval()
    rng = _stable_rng(seed, "flow-audit")
    test_field = torch.from_numpy(rng.standard_normal(
        (R.N_SURFACE, R.SEQ_LEN, R.E10_WIDTH)).astype(np.float32))
    tokens = torch.arange(R.N_SURFACE, dtype=torch.int64)
    p = model.transport_matrix(tokens)
    transported = torch.einsum("bij,bjd->bid", p, test_field)
    before_mass = test_field.sum(dim=1)
    after_mass = transported.sum(dim=1)
    denom = before_mass.norm(dim=1).clamp_min(1e-8)
    mass_rel = (after_mass - before_mass).norm(dim=1) / denom
    row_err = (p.sum(dim=-1) - 1.0).abs().amax(dim=-1)
    col_err = (p.sum(dim=-2) - 1.0).abs().amax(dim=-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=-1)
    peak = p.amax(dim=(-1, -2))

    z = torch.from_numpy(rng.standard_normal(
        (64, R.SEQ_LEN, R.E10_WIDTH)).astype(np.float32))
    hf_before = ((z - torch.roll(z, 1, dims=1)) ** 2).mean()
    lap = torch.roll(z, 1, dims=1) + torch.roll(z, -1, dims=1) - 2 * z
    z_visc = z + model.viscosity * lap
    hf_after = ((z_visc - torch.roll(z_visc, 1, dims=1)) ** 2).mean()

    digits = torch.from_numpy(rng.integers(
        0, R.VOCAB, size=(64, R.SEQ_LEN), dtype=np.int64))
    rollout_tokens = torch.from_numpy(rng.integers(
        0, R.N_SURFACE, size=(64, 16), dtype=np.int64))
    rollout = model(digits, rollout_tokens)
    energies = [float((state ** 2).sum(dim=(1, 2)).mean())
                for state in rollout["states"]]

    pulse_center_error = None
    signed_pulse_mean_error = None
    if model.transport_kind == "reynolds":
        a, b = model.pulse_profiles(tokens)
        pulse_center_error = float(max(a.mean(-1).abs().max(),
                                       b.mean(-1).abs().max()))
        signed = torch.stack([a, -a], dim=2)
        signed_pulse_mean_error = float(signed.mean(dim=2).abs().max())

    matrices = {}
    for i, name in enumerate(ops.SURFACE_NAMES):
        matrices[name] = p[i].cpu().tolist()
    return {
        "transport": model.transport_kind,
        "incompressible": model.incompressible,
        "viscosity": model.viscosity,
        "transport_parameter_count": model.transport_parameter_count,
        "row_sum_error_max": float(row_err.max()),
        "column_sum_error_max": float(col_err.max()),
        "mass_conservation_relative_max": float(mass_rel.max()),
        "pulse_center_error_max": pulse_center_error,
        "signed_pulse_mean_error_max": signed_pulse_mean_error,
        "transport_entropy_mean": float(entropy.mean()),
        "transport_peak_max": float(peak.max()),
        "per_token_entropy": {ops.SURFACE_NAMES[i]: float(entropy[i])
                              for i in range(R.N_SURFACE)},
        "per_token_peak": {ops.SURFACE_NAMES[i]: float(peak[i])
                           for i in range(R.N_SURFACE)},
        "transport_matrices": matrices,
        "viscous_high_frequency_energy_before": float(hf_before),
        "viscous_high_frequency_energy_after": float(hf_after),
        "viscous_high_frequency_ratio": float(hf_after / hf_before),
        "rollout_state_energy": energies,
        "rollout_state_energy_span_ratio": max(energies) / min(energies),
    }


def _parameter_counts(model: ReynoldsFlowModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"total": total,
            "transport": model.transport_parameter_count,
            "non_transport": total - model.transport_parameter_count}


def _lr(cfg: E10Config, step: int) -> float:
    return cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))


def train_e10(cfg: E10Config, out: str | None = None,
              allow_dirty: bool = False) -> Path:
    set_threads()
    seed_everything(cfg.seed)
    git = require_clean_tree(allow_dirty)
    run_dir = run_dir_for_e10(cfg, out)
    if run_dir.exists():
        if (run_dir / "metrics.json").exists() and \
                (run_dir / "SHA256SUMS").exists():
            return run_dir
        raise SystemExit(f"refusing to overwrite incomplete E10 run {run_dir}")
    for name in ("evals", "checkpoints"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)

    bundle = data_mod.build_bundle(cfg)
    arrays = data_mod.build_epoch_arrays(bundle, cfg)
    panels, extrap_manifest = build_extrapolation(cfg)
    model = ReynoldsFlowModel(cfg.arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  betas=cfg.betas,
                                  weight_decay=cfg.weight_decay)

    write_json(run_dir / "config.json", cfg.to_dict())
    write_json(run_dir / "split_ref.json", split_mod.split_ref())
    write_json(run_dir / "data_manifest.json", data_mod.data_manifest(bundle))
    write_json(run_dir / "extrapolation_manifest.json", extrap_manifest)
    write_json(run_dir / "param_counts.json", _parameter_counts(model))
    write_json(run_dir / "flow_audit_init.json", flow_audit(model, cfg.seed))
    log = JsonlLogger(run_dir / "train_log.jsonl")
    n_examples = len(arrays["x"])
    steps_per_epoch = math.ceil(n_examples / cfg.batch_size)
    log.log(event="start", run_dir=str(run_dir), arm=cfg.arm, seed=cfg.seed,
            n_train_presentations_per_epoch=n_examples,
            steps_per_epoch=steps_per_epoch, **git)

    t0 = time.time()
    peak_rss = 0.0
    step = 0
    epoch = 0
    while step < cfg.total_steps:
        order = stream_rng(cfg.seed, "shuffle", epoch).permutation(n_examples)
        for b in range(steps_per_epoch):
            if step >= cfg.total_steps:
                break
            idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            xb = torch.from_numpy(arrays["x"][idx])
            yb = torch.from_numpy(arrays["y"][idx])
            tb = torch.from_numpy(arrays["tokens"][idx])
            nb = torch.from_numpy(arrays["n_tokens"][idx])
            lr = _lr(cfg, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            pred = model(xb, tb, nb)["logits"]
            loss = F.cross_entropy(pred.reshape(-1, R.VOCAB), yb.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_pre = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                      cfg.grad_clip)
            optimizer.step()
            step += 1

            if step == 1 or step % cfg.log_every == 0:
                exact = float((pred.argmax(-1) == yb).all(-1).float().mean())
                log.log(event="step", step=step, epoch=epoch, lr=lr,
                        loss=float(loss.detach()), batch_acc=exact,
                        grad_norm_preclip=float(grad_pre.detach()),
                        grad_norm_postclip=float(min(float(grad_pre.detach()),
                                                     cfg.grad_clip)))
            if step % cfg.eval_every == 0 or step == cfg.total_steps:
                evaluation = evaluate_standard(model, bundle)
                write_json(run_dir / "evals" / f"step{step:06d}.json",
                           evaluation)
                log.log(event="eval", step=step,
                        seen=evaluation["seen"]["mean"],
                        L1=evaluation["L1"]["mean"],
                        L2=evaluation["L2"]["mean"],
                        L3=evaluation["L3"]["mean"])
                peak_rss = max(peak_rss, check_rss())
        epoch += 1

    standard = evaluate_standard(model, bundle)
    extrap = evaluate_extrapolation(model, panels)
    noise = evaluate_hidden_noise(model, bundle, cfg)
    audit = flow_audit(model, cfg.seed)
    metrics = {
        "arm": cfg.arm,
        "seed": cfg.seed,
        "smoke": cfg.smoke,
        "experiment": cfg.experiment,
        "protocol_revision": cfg.protocol_revision,
        "final_step": step,
        "acc_seen_hard": standard["seen"]["mean"],
        "acc_unseen_L1_hard": standard["L1"]["mean"],
        "acc_unseen_L2_hard": standard["L2"]["mean"],
        "acc_unseen_L3_hard": standard["L3"]["mean"],
        "acc_extrap_triple_hard": extrap["3"]["mean"],
        "acc_extrap_quad_hard": extrap["4"]["mean"],
        "acc_pair_clean_hard": noise["clean_mean"],
        "acc_pair_noisy_hard": noise["noisy_mean"],
        "hidden_noise_drop": noise["drop"],
        "mass_conservation_relative_max":
            audit["mass_conservation_relative_max"],
        "row_sum_error_max": audit["row_sum_error_max"],
        "column_sum_error_max": audit["column_sum_error_max"],
        "pulse_center_error_max": audit["pulse_center_error_max"],
        "signed_pulse_mean_error_max":
            audit["signed_pulse_mean_error_max"],
        "transport_entropy_mean": audit["transport_entropy_mean"],
        "transport_peak_max": audit["transport_peak_max"],
        "viscous_high_frequency_ratio":
            audit["viscous_high_frequency_ratio"],
        "rollout_state_energy_span_ratio":
            audit["rollout_state_energy_span_ratio"],
        "param_counts": _parameter_counts(model),
        "standard": standard,
        "extrapolation": extrap,
        "hidden_noise": noise,
        "flow_audit": audit,
    }
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "step": step}, run_dir / "checkpoints" / "final.pt")
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "env.json", {
        **env_info(), **git, "train_seconds": time.time() - t0,
        "peak_rss_gb": peak_rss})
    log.log(event="done", step=step, train_seconds=time.time() - t0,
            peak_rss_gb=peak_rss)
    log.close()
    write_sha256sums(run_dir)
    return run_dir
