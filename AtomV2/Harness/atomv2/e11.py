"""E11 curriculum, evaluation, robustness panels, and conservation audits."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
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
from .exchange_model import ConservativeExchangeModel, MATCHINGS
from .utils import (JsonlLogger, RUNS_DIR, check_rss, env_info, git_info,
                    require_clean_tree, seed_everything, set_threads,
                    write_json, write_sha256sums)


@dataclass
class E11Config:
    arm: str
    seed: int
    smoke: bool = False
    experiment: str = "e11"
    protocol_revision: str = R.E11_PROTOCOL_REVISION
    transport: str = ""
    viscosity: float = 0.0
    exchange_stages: int = R.E11_EXCHANGE_STAGES
    gate_bias: float = R.E11_GATE_BIAS
    stress_scale: float = R.E11_STRESS_SCALE
    reaction_identity_bias: float = R.E11_REACTION_IDENTITY_BIAS
    examples_per_train_task: int = R.EXAMPLES_PER_TRAIN_TASK
    examples_per_eval_task: int = R.EXAMPLES_PER_EVAL_TASK
    n_probe_examples: int = R.N_PROBE_EXAMPLES
    p3_oversample_factor: int = 1
    lr: float = R.E11_LR
    betas: tuple = R.E11_BETAS
    weight_decay: float = R.E11_WEIGHT_DECAY
    warmup_steps: int = R.E11_WARMUP_STEPS
    batch_size: int = R.E11_BATCH_SIZE
    phase1_steps: int = R.E11_PHASE1_STEPS
    phase2_steps: int = R.E11_PHASE2_STEPS
    eval_every: int = R.E11_EVAL_EVERY
    log_every: int = R.E11_LOG_EVERY
    grad_clip: float = R.E11_GRAD_CLIP
    phase1_singleton_min: float = R.E11_PHASE1_SINGLETON_MIN
    extrap_tasks_per_length: int = R.E11_EXTRAP_TASKS_PER_LENGTH
    extrap_examples_per_task: int = R.E11_EXTRAP_EXAMPLES_PER_TASK
    noise_sigma: float = R.E11_NOISE_SIGMA

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["betas"] = list(self.betas)
        return out


def config_for_e11(arm: str, seed: int, smoke: bool = False) -> E11Config:
    if arm not in R.E11_ARMS:
        raise ValueError(f"unknown E11 arm {arm!r}")
    cfg = E11Config(arm=arm, seed=seed, smoke=smoke,
                    transport=R.E11_TRANSPORT[arm],
                    viscosity=R.E11_VISCOSITY[arm])
    if smoke:
        cfg.examples_per_train_task = 48
        cfg.examples_per_eval_task = 24
        cfg.n_probe_examples = 24
        cfg.warmup_steps = 20
        cfg.phase1_steps = 150
        cfg.phase2_steps = 150
        cfg.eval_every = 75
        cfg.log_every = 25
        cfg.extrap_tasks_per_length = 8
        cfg.extrap_examples_per_task = 12
    return cfg


def _run_id(cfg: E11Config) -> str:
    git = git_info()
    source = git["git_sha_short"]
    if git.get("dirty_source_sha256"):
        source += f"-dirty{git['dirty_source_sha256'][:8]}"
    return f"{cfg.arm}_s{cfg.seed}_{cfg.protocol_revision}_{source}"


def run_dir_for_e11(cfg: E11Config, out: str | None = None) -> Path:
    base = Path(out) if out else RUNS_DIR
    return base / ("smoke_e11" if cfg.smoke else "e11") / _run_id(cfg)


def _stable_rng(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"e11|{seed}|{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


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


def build_extrapolation(cfg: E11Config) -> tuple[dict[int, list[dict]], dict]:
    """Arm-paired unique token sequences and seed-indexed fresh inputs."""
    panels: dict[int, list[dict]] = {}
    manifest = {"selection_seed": R.E11_EXTRAP_SELECTION_SEED,
                "run_seed": cfg.seed, "lengths": {}}
    for length in R.E11_EXTRAP_LENGTHS:
        select_rng = np.random.default_rng(
            R.E11_EXTRAP_SELECTION_SEED + length)
        selected: set[tuple[int, ...]] = set()
        while len(selected) < cfg.extrap_tasks_per_length:
            selected.add(tuple(int(v) for v in select_rng.integers(
                0, R.N_SURFACE, size=length)))
        panels[length] = []
        entries = []
        for token_tuple in sorted(selected):
            names = tuple(ops.SURFACE_NAMES[i] for i in token_tuple)
            task_id = "_".join(names)
            x = _unique_inputs(
                _stable_rng(cfg.seed, f"extrap|{length}|{task_id}"),
                cfg.extrap_examples_per_task)
            y = _apply_sequence(names, x)
            tokens = np.asarray(token_tuple, dtype=np.int64)
            panels[length].append({"task_id": task_id, "x": x, "y": y,
                                   "tokens": tokens})
            entries.append({"task_id": task_id, "n": len(x),
                            "x_sha256": _array_hash(x),
                            "y_sha256": _array_hash(y)})
        manifest["lengths"][str(length)] = entries
    return panels, manifest


def _flat_tasks(tasks: list[data_mod.TaskData]) -> dict[str, np.ndarray]:
    xs, ys, tokens, counts = [], [], [], []
    for td in tasks:
        n = len(td.x)
        xs.append(td.x)
        ys.append(td.y)
        tokens.append(np.tile(td.task.tokens, (n, 1)))
        counts.append(np.full(n, td.task.n_tokens, dtype=np.int64))
    return {"x": np.concatenate(xs), "y": np.concatenate(ys),
            "tokens": np.concatenate(tokens),
            "n_tokens": np.concatenate(counts)}


def build_curriculum(bundle: data_mod.Bundle, cfg: E11Config) -> tuple[dict,
                                                                         dict]:
    singleton = _flat_tasks([td for td in bundle.train
                             if td.task.n_tokens == 1])
    pair = _flat_tasks([td for td in bundle.train if td.task.n_tokens == 2])
    half = cfg.batch_size // 2
    if cfg.batch_size % 2:
        raise ValueError("E11 batch size must be even")
    p1 = _stable_rng(cfg.seed, "phase1-singleton").integers(
        0, len(singleton["x"]), size=(cfg.phase1_steps, cfg.batch_size),
        dtype=np.int64)
    p2s = _stable_rng(cfg.seed, "phase2-singleton").integers(
        0, len(singleton["x"]), size=(cfg.phase2_steps, half),
        dtype=np.int64)
    p2p = _stable_rng(cfg.seed, "phase2-pair").integers(
        0, len(pair["x"]), size=(cfg.phase2_steps, half), dtype=np.int64)
    arrays = {"singleton": singleton, "pair": pair,
              "phase1_indices": p1, "phase2_single_indices": p2s,
              "phase2_pair_indices": p2p}
    manifest = {
        "task_balanced": True,
        "p3_oversample_factor": 1,
        "phase1": {"steps": cfg.phase1_steps,
                   "batch_singleton": cfg.batch_size,
                   "indices_sha256": _array_hash(p1)},
        "phase2": {"steps": cfg.phase2_steps,
                   "batch_singleton": half, "batch_pair": half,
                   "singleton_indices_sha256": _array_hash(p2s),
                   "pair_indices_sha256": _array_hash(p2p)},
        "singleton_pool_size": len(singleton["x"]),
        "pair_pool_size": len(pair["x"]),
    }
    return arrays, manifest


@torch.no_grad()
def _task_accuracy(model: ConservativeExchangeModel, x: np.ndarray,
                   y: np.ndarray, tokens: np.ndarray,
                   batch_size: int = 512, **perturb) -> dict[str, float]:
    model.eval()
    exact, cell = [], []
    tiled = np.tile(tokens[None, :], (len(x), 1))
    for lo in range(0, len(x), batch_size):
        hi = min(lo + batch_size, len(x))
        kwargs = {}
        for name, value in perturb.items():
            kwargs[name] = None if value is None else torch.from_numpy(
                value[lo:hi])
        result = model(torch.from_numpy(x[lo:hi]),
                       torch.from_numpy(tiled[lo:hi]), **kwargs)
        pred = result["probs"].argmax(dim=-1).cpu().numpy()
        matches = pred == y[lo:hi]
        exact.append(matches.all(axis=1))
        cell.append(matches.reshape(-1))
    return {"exact": float(np.concatenate(exact).mean()),
            "cell": float(np.concatenate(cell).mean())}


def _evaluate_tasks(model: ConservativeExchangeModel,
                    tasks: list[data_mod.TaskData]) -> dict:
    per_task = {}
    for td in tasks:
        per_task[td.task.task_id] = _task_accuracy(
            model, td.x, td.y, td.task.tokens[:td.task.n_tokens])
    return {
        "mean": float(np.mean([v["exact"] for v in per_task.values()])),
        "cell_mean": float(np.mean([v["cell"] for v in per_task.values()])),
        "min": float(min(v["exact"] for v in per_task.values())),
        "tasks": per_task,
    }


@torch.no_grad()
def evaluate_standard(model: ConservativeExchangeModel,
                      bundle: data_mod.Bundle) -> dict:
    singletons = [td for td in bundle.seen_heldout if td.task.n_tokens == 1]
    seen_pairs = [td for td in bundle.seen_heldout if td.task.n_tokens == 2]
    return {"singleton": _evaluate_tasks(model, singletons),
            "seen_pair": _evaluate_tasks(model, seen_pairs),
            **{level: _evaluate_tasks(model, bundle.unseen[level])
               for level in ("L1", "L2", "L3")}}


@torch.no_grad()
def evaluate_extrapolation(model: ConservativeExchangeModel,
                           panels: dict[int, list[dict]]) -> dict:
    out = {}
    for length, tasks in panels.items():
        per_task = {td["task_id"]: _task_accuracy(
            model, td["x"], td["y"], td["tokens"])
                    for td in tasks}
        out[str(length)] = {
            "mean": float(np.mean([v["exact"] for v in per_task.values()])),
            "cell_mean": float(np.mean([v["cell"] for v in per_task.values()])),
            "tasks": per_task,
        }
    return out


def _all_pair_tasks(bundle: data_mod.Bundle) -> list[data_mod.TaskData]:
    return ([td for td in bundle.seen_heldout if td.task.n_tokens == 2]
            + bundle.unseen["L1"] + bundle.unseen["L2"]
            + bundle.unseen["L3"])


def quantized_copy(model: ConservativeExchangeModel,
                   bits: int) -> ConservativeExchangeModel:
    """Symmetric per-tensor quantization of learned weights only."""
    if bits not in R.E11_QUANT_BITS:
        raise ValueError(f"unregistered quantization width {bits}")
    result = copy.deepcopy(model)
    qmax = (1 << (bits - 1)) - 1
    with torch.no_grad():
        for parameter in result.parameters():
            scale = parameter.abs().max() / qmax
            if float(scale) > 0:
                parameter.copy_(torch.round(parameter / scale) * scale)
    return result


@torch.no_grad()
def evaluate_interface(model: ConservativeExchangeModel,
                       bundle: data_mod.Bundle) -> dict:
    agreements, probability_error = [], []
    for td in _all_pair_tasks(bundle):
        tokens = td.task.tokens[:2]
        for lo in range(0, len(td.x), 512):
            x = torch.from_numpy(td.x[lo:lo + 512])
            t1 = torch.full((len(x), 1), int(tokens[0]), dtype=torch.long)
            t2 = torch.full((len(x), 1), int(tokens[1]), dtype=torch.long)
            both = torch.cat([t1, t2], dim=1)
            ordinary = model(x, both)["probs"]
            produced = model(x, t1)["probs"].argmax(dim=-1)
            restarted = model(produced, t2)["probs"]
            agreements.append((ordinary.argmax(-1) == restarted.argmax(-1))
                              .all(-1).cpu().numpy())
            probability_error.append(float((ordinary - restarted).abs().max()))
    return {"prediction_agreement": float(np.concatenate(agreements).mean()),
            "probability_error_max": max(probability_error),
            "contract": "ordinary pair equals explicit decode/restart pair"}


@torch.no_grad()
def flow_audit(model: ConservativeExchangeModel, seed: int) -> dict:
    model.eval()
    rng = _stable_rng(seed, "flow-audit")
    raw = rng.random((R.N_SURFACE, R.SEQ_LEN, R.VOCAB), dtype=np.float32)
    field = torch.from_numpy(raw)
    field = field / field.sum(dim=-1, keepdim=True)
    tokens = torch.arange(R.N_SURFACE, dtype=torch.long)
    transported, gates, stages = model.transport(field, tokens)
    before_mass = field.sum(dim=1)
    denom = before_mass.norm(dim=1).clamp_min(1e-12)
    residuals = [((stage.sum(dim=1) - before_mass).norm(dim=1) / denom)
                 for stage in stages]

    identity = torch.eye(R.SEQ_LEN).expand(R.N_SURFACE, -1, -1)
    matrix, _, _ = model.transport(identity, tokens)
    row_error = (matrix.sum(-1) - 1.0).abs().max()
    col_error = (matrix.sum(-2) - 1.0).abs().max()
    predicted_pi = matrix.argmax(dim=-1).cpu().numpy()
    true_pi = np.asarray([ops.SURFACE_TRIPLES[name][0]
                          for name in ops.SURFACE_NAMES])
    permutation_match = float((predicted_pi == true_pi).mean())
    ideal_mass = float(np.mean([
        matrix[i, np.arange(R.SEQ_LEN), true_pi[i]].mean().item()
        for i in range(R.N_SURFACE)]))

    kernel = model.reaction_kernel(tokens)
    entropy = -(kernel * kernel.clamp_min(1e-12).log()).sum(-1).mean()
    reaction_row_error = (kernel.sum(-1) - 1.0).abs().max()

    z = torch.from_numpy(rng.random(
        (64, R.SEQ_LEN, R.VOCAB), dtype=np.float32))
    z = z / z.sum(-1, keepdim=True)
    hf_before = ((z - torch.roll(z, 1, dims=1)) ** 2).mean()
    z_visc = ((1.0 - 2.0 * model.viscosity) * z
              + model.viscosity * torch.roll(z, 1, dims=1)
              + model.viscosity * torch.roll(z, -1, dims=1))
    hf_after = ((z_visc - torch.roll(z_visc, 1, dims=1)) ** 2).mean()

    pulse_center = None
    signed_mean = None
    if model.transport_kind == "reynolds":
        a, b = model.pulse_profiles(tokens)
        pulse_center = float(max(a.mean(-1).abs().max(),
                                 b.mean(-1).abs().max()))
        signed_mean = float(max(torch.stack([a, -a], 2).mean(2).abs().max(),
                                torch.stack([b, -b], 2).mean(2).abs().max()))

    return {
        "transport": model.transport_kind,
        "viscosity": model.viscosity,
        "exchange_stages": len(MATCHINGS),
        "binary_schedule_reachable_permutations": 720,
        "stage_mass_conservation_relative": [float(x.max()) for x in residuals],
        "mass_conservation_relative_max": float(max(x.max() for x in residuals)),
        "composite_row_sum_error_max": float(row_error),
        "composite_column_sum_error_max": float(col_error),
        "gate_min": float(gates.min()),
        "gate_max": float(gates.max()),
        "gate_mean_distance_to_binary": float(torch.minimum(
            gates, 1.0 - gates).mean()),
        "gate_values": {ops.SURFACE_NAMES[i]: gates[i].cpu().tolist()
                        for i in range(R.N_SURFACE)},
        "effective_transport": {ops.SURFACE_NAMES[i]: matrix[i].cpu().tolist()
                                for i in range(R.N_SURFACE)},
        "permutation_argmax_match": permutation_match,
        "ideal_permutation_probability_mass": ideal_mass,
        "reaction_row_sum_error_max": float(reaction_row_error),
        "reaction_entropy_mean": float(entropy),
        "pulse_center_error_max": pulse_center,
        "signed_pulse_mean_error_max": signed_mean,
        "viscous_high_frequency_ratio": float(hf_after / hf_before),
        "transport_output_row_mass_error_max": float(
            (transported.sum(-1) - field.sum(-1)).abs().max()),
    }


@torch.no_grad()
def evaluate_robustness(model: ConservativeExchangeModel,
                        bundle: data_mod.Bundle, cfg: E11Config,
                        extrap: dict) -> dict:
    quantized = {bits: quantized_copy(model, bits) for bits in R.E11_QUANT_BITS}
    task_rows = {}
    for td in _all_pair_tasks(bundle):
        n = len(td.x)
        tokens = td.task.tokens[:2]
        noise_rng = _stable_rng(cfg.seed, f"noise|{td.task.task_id}")
        noise = (cfg.noise_sigma * noise_rng.standard_normal(
            (n, 1, R.SEQ_LEN, R.VOCAB))).astype(np.float32)
        damage_rng = _stable_rng(cfg.seed, f"cell-damage|{td.task.task_id}")
        positions = damage_rng.integers(0, R.SEQ_LEN, size=(n, 1),
                                        dtype=np.int64)
        first_name = td.task.surface_ops[0]
        intermediate = ops.SURFACE_FNS[first_name](td.x)
        wrong = np.empty((n, 1), dtype=np.int64)
        wrong[:, 0] = ((intermediate[np.arange(n), positions[:, 0]] + 1)
                       % R.VOCAB)
        row = {
            "clean": _task_accuracy(model, td.x, td.y, tokens),
            "gaussian": _task_accuracy(model, td.x, td.y, tokens,
                                       handoff_noise=noise),
            "dropped": _task_accuracy(model, td.x, td.y, tokens,
                                      drop_positions=positions),
            "corrupted": _task_accuracy(
                model, td.x, td.y, tokens, corrupt_positions=positions,
                corrupt_digits=wrong),
        }
        for bits, qmodel in quantized.items():
            row[f"int{bits}"] = _task_accuracy(qmodel, td.x, td.y, tokens)
        task_rows[td.task.task_id] = row

    names = ("clean", "gaussian", "int8", "int4", "dropped", "corrupted")
    means = {}
    for name in names:
        means[name] = {
            "exact": float(np.mean([v[name]["exact"]
                                    for v in task_rows.values()])),
            "cell": float(np.mean([v[name]["cell"]
                                   for v in task_rows.values()])),
        }
    composite_parts = {
        "gaussian_pair": means["gaussian"]["exact"],
        "int4_pair": means["int4"]["exact"],
        "dropped_pair": means["dropped"]["exact"],
        "corrupted_pair": means["corrupted"]["exact"],
        "length8": extrap["8"]["mean"],
        "length16": extrap["16"]["mean"],
    }
    return {"noise_sigma": cfg.noise_sigma, "means": means,
            "composite_parts": composite_parts,
            "composite": float(np.mean(list(composite_parts.values()))),
            "tasks": task_rows}


def _parameter_counts(model: ConservativeExchangeModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "transport": model.transport_parameter_count,
            "reaction": model.reaction_logits.numel()}


def _lr(cfg: E11Config, step: int) -> float:
    return cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))


def _batch(pool: dict[str, np.ndarray], indices: np.ndarray) -> tuple:
    return tuple(torch.from_numpy(pool[name][indices])
                 for name in ("x", "y", "tokens", "n_tokens"))


def _mixed_batch(arrays: dict, local_step: int) -> tuple:
    si = arrays["phase2_single_indices"][local_step]
    pi = arrays["phase2_pair_indices"][local_step]
    singleton = arrays["singleton"]
    pair = arrays["pair"]
    outputs = []
    for name in ("x", "y", "tokens", "n_tokens"):
        value = np.concatenate([singleton[name][si], pair[name][pi]], axis=0)
        outputs.append(torch.from_numpy(value))
    return tuple(outputs)


def train_e11(cfg: E11Config, out: str | None = None,
              allow_dirty: bool = False) -> Path:
    set_threads()
    seed_everything(cfg.seed)
    git = require_clean_tree(allow_dirty)
    run_dir = run_dir_for_e11(cfg, out)
    if run_dir.exists():
        if (run_dir / "metrics.json").exists() and \
                (run_dir / "SHA256SUMS").exists():
            return run_dir
        raise SystemExit(f"refusing to overwrite incomplete E11 run {run_dir}")
    for name in ("evals", "checkpoints"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)

    bundle = data_mod.build_bundle(cfg)
    curriculum, curriculum_manifest = build_curriculum(bundle, cfg)
    panels, extrap_manifest = build_extrapolation(cfg)
    model = ConservativeExchangeModel(cfg.arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  betas=cfg.betas,
                                  weight_decay=cfg.weight_decay)

    write_json(run_dir / "config.json", cfg.to_dict())
    write_json(run_dir / "split_ref.json", split_mod.split_ref())
    write_json(run_dir / "data_manifest.json", data_mod.data_manifest(bundle))
    write_json(run_dir / "curriculum_manifest.json", curriculum_manifest)
    write_json(run_dir / "extrapolation_manifest.json", extrap_manifest)
    write_json(run_dir / "param_counts.json", _parameter_counts(model))
    write_json(run_dir / "flow_audit_init.json", flow_audit(model, cfg.seed))
    log = JsonlLogger(run_dir / "train_log.jsonl")
    log.log(event="start", run_dir=str(run_dir), arm=cfg.arm, seed=cfg.seed,
            **git)

    t0 = time.time()
    peak_rss = 0.0
    global_step = 0

    def train_step(batch: tuple, phase: int, local_step: int) -> None:
        nonlocal global_step, peak_rss
        xb, yb, tb, nb = batch
        lr = _lr(cfg, global_step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        result = model(xb, tb, nb)
        loss = F.nll_loss(result["logits"].reshape(-1, R.VOCAB),
                          yb.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_pre = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  cfg.grad_clip)
        optimizer.step()
        global_step += 1
        if global_step == 1 or global_step % cfg.log_every == 0:
            exact = float((result["probs"].argmax(-1) == yb)
                          .all(-1).float().mean())
            log.log(event="step", step=global_step, phase=phase,
                    phase_step=local_step + 1, lr=lr, loss=float(loss.detach()),
                    batch_acc=exact, grad_norm_preclip=float(grad_pre.detach()),
                    grad_norm_postclip=float(min(float(grad_pre.detach()),
                                                 cfg.grad_clip)))
        if (local_step + 1) % cfg.eval_every == 0:
            singleton_tasks = [td for td in bundle.seen_heldout
                               if td.task.n_tokens == 1]
            single_eval = _evaluate_tasks(model, singleton_tasks)
            payload = {"singleton": single_eval, "step": global_step,
                       "phase": phase, "phase_step": local_step + 1}
            if phase == 2:
                payload["standard"] = evaluate_standard(model, bundle)
            write_json(run_dir / "evals" / f"step{global_step:06d}.json",
                       payload)
            log.log(event="eval", step=global_step, phase=phase,
                    singleton_mean=single_eval["mean"],
                    singleton_min=single_eval["min"])
            peak_rss = max(peak_rss, check_rss())

    for local_step in range(cfg.phase1_steps):
        idx = curriculum["phase1_indices"][local_step]
        train_step(_batch(curriculum["singleton"], idx), 1, local_step)

    singleton_tasks = [td for td in bundle.seen_heldout
                       if td.task.n_tokens == 1]
    phase1_eval = _evaluate_tasks(model, singleton_tasks)
    phase1_pass = phase1_eval["min"] >= cfg.phase1_singleton_min
    write_json(run_dir / "phase1_gate.json", {
        "threshold": cfg.phase1_singleton_min, "passes": phase1_pass,
        "smoke_override": bool(cfg.smoke and not phase1_pass),
        "evaluation": phase1_eval})
    log.log(event="phase1_gate", step=global_step, passes=phase1_pass,
            singleton_min=phase1_eval["min"],
            smoke_override=bool(cfg.smoke and not phase1_pass))

    phase2_ran = phase1_pass or cfg.smoke
    if phase2_ran:
        for local_step in range(cfg.phase2_steps):
            train_step(_mixed_batch(curriculum, local_step), 2, local_step)

    standard = evaluate_standard(model, bundle)
    extrap = evaluate_extrapolation(model, panels)
    interface = evaluate_interface(model, bundle)
    audit = flow_audit(model, cfg.seed)
    robustness = evaluate_robustness(model, bundle, cfg, extrap)
    metrics = {
        "arm": cfg.arm, "seed": cfg.seed, "smoke": cfg.smoke,
        "experiment": cfg.experiment,
        "protocol_revision": cfg.protocol_revision,
        "final_step": global_step,
        "phase1_pass": phase1_pass,
        "phase2_ran": phase2_ran,
        "phase1_singleton_min": phase1_eval["min"],
        "acc_singleton_mean_hard": standard["singleton"]["mean"],
        "acc_singleton_min_hard": standard["singleton"]["min"],
        "acc_seen_pair_hard": standard["seen_pair"]["mean"],
        "acc_unseen_L1_hard": standard["L1"]["mean"],
        "acc_unseen_L2_hard": standard["L2"]["mean"],
        "acc_unseen_L3_hard": standard["L3"]["mean"],
        **{f"acc_extrap_len{length}_hard": extrap[str(length)]["mean"]
           for length in R.E11_EXTRAP_LENGTHS},
        "interface_prediction_agreement": interface["prediction_agreement"],
        "interface_probability_error_max": interface["probability_error_max"],
        "mass_conservation_relative_max":
            audit["mass_conservation_relative_max"],
        "permutation_argmax_match": audit["permutation_argmax_match"],
        "ideal_permutation_probability_mass":
            audit["ideal_permutation_probability_mass"],
        "acc_pair_noisy_hard": robustness["means"]["gaussian"]["exact"],
        "acc_pair_int8_hard": robustness["means"]["int8"]["exact"],
        "acc_pair_int4_hard": robustness["means"]["int4"]["exact"],
        "acc_pair_dropped_hard": robustness["means"]["dropped"]["exact"],
        "acc_pair_corrupted_hard": robustness["means"]["corrupted"]["exact"],
        "robustness_composite": robustness["composite"],
        "param_counts": _parameter_counts(model),
        "standard": standard, "extrapolation": extrap,
        "interface": interface, "flow_audit": audit,
        "robustness": robustness,
    }
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "step": global_step}, run_dir / "checkpoints" / "final.pt")
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "env.json", {
        **env_info(), **git, "train_seconds": time.time() - t0,
        "peak_rss_gb": peak_rss})
    log.log(event="done", step=global_step, phase1_pass=phase1_pass,
            phase2_ran=phase2_ran, train_seconds=time.time() - t0,
            peak_rss_gb=peak_rss)
    log.close()
    write_sha256sums(run_dir)
    return run_dir
