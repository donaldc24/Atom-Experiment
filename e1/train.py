"""Training driver for E1. Produces artifacts; computes NO headline metrics.

Usage:
    python -m e1.train --arm A0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import GENERATIONS, Config, config_for_arm, config_for_generation
from .data import (
    Bundle,
    build_bundle,
    load_split,
    split_hash,
    split_path_for,
    verify_split,
)
from .evaluate import emit_artifacts
from .model import AtomNet
from .primitives import K
from .utils import (
    RSSMonitor,
    env_info,
    git_info,
    require_clean_tree,
    param_counts,
    seed_everything,
    set_threads,
    write_json,
    write_sha256sums,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

class TrainArrays:
    def __init__(self, task_data_list, cfg):
        from .primitives import DEFAULT_SET, apply_composition
        pset = getattr(cfg, "primitive_set", DEFAULT_SET)

        self.x = np.concatenate([td.inputs for td in task_data_list])
        self.y = np.concatenate([td.targets for td in task_data_list])
        self.instr = np.concatenate([
            np.tile(np.asarray(td.task.instruction), (td.inputs.shape[0], 1))
            for td in task_data_list
        ])
        # Per-step targets: y_step[:, t] is the partial composition after step t.
        # Because a length-1 task is (p, identity), step 2 of a singleton repeats
        # step 1's target, so this is uniform across task kinds.
        self.y_step = np.stack([
            np.concatenate([
                apply_composition(td.task.instruction[:t + 1], td.inputs, pset)
                for td in task_data_list
            ])
            for t in range(cfg.depth)
        ], axis=1)
        self.n = self.x.shape[0]

    def torch_view(self):
        return (
            torch.from_numpy(np.ascontiguousarray(self.x)).long(),
            torch.from_numpy(np.ascontiguousarray(self.y)).long(),
            torch.from_numpy(np.ascontiguousarray(self.instr)).long(),
            torch.from_numpy(np.ascontiguousarray(self.y_step)).long(),
        )


def shuffle_targets(arrays: TrainArrays, seed: int) -> None:
    """A4 leakage detector: training targets are permuted across the whole set."""
    rng = np.random.default_rng(seed)
    arrays.y = arrays.y[rng.permutation(arrays.n)]


# --------------------------------------------------------------------------
# Optimiser construction
# --------------------------------------------------------------------------

def build_optimizer(model, cfg, include_atoms: bool):
    """Atom parameters always carry weight_decay=0.

    Decoupled weight decay would shrink a frozen atom even when its gradient is
    exactly zero, which would silently break A3's freezing guarantee. Applying
    wd=0 to atoms in *every* arm keeps the arms comparable. See DECISIONS.md D5.
    """
    scale = getattr(cfg, "codec_lr_scale", 1.0)
    if scale == 1.0:
        groups = [{"params": model.non_atom_parameters(),
                   "weight_decay": cfg.weight_decay}]
    else:
        # R3-fixed: encoder/decoder/state_norm become a slow-moving substrate once
        # phase 0 has trained them. Down-weighted rather than frozen -- a fully
        # frozen codec may be unable to accommodate the atoms at all. See D35.
        codec = list(model.encoder.parameters()) + list(model.decoder.parameters())
        if model.state_norm is not None:
            codec += list(model.state_norm.parameters())
        codec_ids = {id(p) for p in codec}
        rest = [p for p in model.non_atom_parameters() if id(p) not in codec_ids]
        groups = [
            {"params": rest, "weight_decay": cfg.weight_decay},
            {"params": codec, "weight_decay": cfg.weight_decay, "lr": cfg.lr * scale},
        ]
    if include_atoms:
        groups.append({"params": model.atom_parameters(), "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=cfg.lr)


def set_atom_grads_trainable(model, trainable_idx: set[int] | None) -> None:
    """Zero the gradient rows of atoms that are frozen at this stage (A3 phase 1)."""
    if trainable_idx is None:
        return
    n = model.atoms.n_atoms
    frozen = [i for i in range(n) if i not in trainable_idx]
    if not frozen:
        return
    for p in (model.atoms.w1, model.atoms.b1, model.atoms.w2,
              model.atoms.b2, model.atoms.keys):
        if p.grad is not None:
            p.grad[frozen] = 0.0


# --------------------------------------------------------------------------
# Core loop
# --------------------------------------------------------------------------

def run_phase(
    model, cfg, arrays: TrainArrays, *, epochs: int, log, state,
    atom_mask=None, trainable_atoms=None, include_atoms=True,
    mode="gumbel", anneal_tau=True, phase_name="main", rss=None,
):
    x, y, instr, y_step = arrays.torch_view()
    n = arrays.n
    optim = build_optimizer(model, cfg, include_atoms=include_atoms)
    steps_per_epoch = max(1, n // cfg.batch_size)
    total_steps = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, total_steps))

    gen = torch.Generator()
    gen.manual_seed(cfg.seed * 100003 + state["phase_counter"] * 7919 + 11)
    state["phase_counter"] += 1

    # A1 keeps one fixed co-occurrence order for every epoch; A2/A4 reshuffle.
    fixed_order = np.random.default_rng(cfg.seed * 31 + 5).permutation(n)

    plateau = 0
    model.train()
    for epoch in range(epochs):
        if cfg.reshuffle_each_epoch:
            order = np.random.default_rng(
                cfg.seed * 31 + 5 + (epoch + 1) * 104729
            ).permutation(n)
        else:
            order = fixed_order

        epoch_correct = 0
        epoch_seen = 0
        for b in range(steps_per_epoch):
            idx = torch.from_numpy(order[b * cfg.batch_size:(b + 1) * cfg.batch_size])
            xb, yb, ib, ysb = x[idx], y[idx], instr[idx], y_step[idx]

            tau = cfg.tau_start
            if anneal_tau and total_steps > 1:
                frac = state["global_step_in_phase"] / max(1, total_steps - 1)
                tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * min(1.0, frac)

            out = model(
                xb, ib,
                mode=mode,
                tau=tau,
                forced=ib if mode == "forced" else None,
                atom_mask=atom_mask,
                atom_dropout=cfg.atom_dropout,
                generator=gen,
            )
            # Each term is captured separately for the log. The accumulation order
            # into `loss` is unchanged, so the optimised quantity is bit-identical to
            # runs made before this logging existed (D27).
            terms = {}
            loss = F.cross_entropy(
                out["logits"].reshape(-1, cfg.vocab), yb.reshape(-1)
            )
            terms["loss_task"] = float(loss.detach())
            if mode == "forced" and cfg.routing_ce_weight > 0:
                route_ce = F.cross_entropy(
                    out["route_logits"].reshape(-1, cfg.n_atoms), ib.reshape(-1)
                )
                loss = loss + cfg.routing_ce_weight * route_ce
                terms["loss_route"] = float(route_ce.detach())
            if cfg.code_consistency_weight > 0:
                # E1b R2+: h_t must be a valid code -- it must survive a round trip
                # through decode and re-encode. Self-supervised: it says nothing about
                # WHICH point on the manifold h_t should be, only that it must be on
                # it. That separation is the whole point of E1b (see D24).
                code_raw = []
                for t in range(cfg.depth - 1):
                    h_t = out["states"][:, t]
                    # Straight-through HARD re-encode: the forward pass is the
                    # argmax round trip that code_residual measures, so the
                    # constraint cannot be satisfied by an uninformative decode (D27).
                    recoded = model.recode(h_t, hard=True)
                    rel_sq = (
                        (h_t - recoded).pow(2).sum(dim=1)
                        / h_t.pow(2).sum(dim=1).clamp_min(1e-6)
                    ).mean()
                    loss = loss + cfg.code_consistency_weight * rel_sq
                    code_raw.append(float(rel_sq.detach()))
                raw = sum(code_raw) / len(code_raw)
                # `loss_code_rel` is the UNWEIGHTED relative error, directly comparable
                # to the eval-time `code_residual` and to the 0.15 binding threshold at
                # every weight. The weighted contribution shrinks 10x from w=10 to w=1;
                # this does not, which is the point.
                terms["loss_code_rel"] = raw ** 0.5
                terms["loss_code_raw"] = raw
                terms["loss_code_weighted"] = cfg.code_consistency_weight * raw
            if cfg.state_consistency:
                # h_t must land on the encoder manifold at the partial composition.
                # The target is detached: this shapes the atoms, not the encoder.
                state_raw = []
                arb = getattr(cfg, "arbitrary_targets", False)
                # S-arb pulls only h_1 (the intermediate), toward the frozen code of
                # the step's FIRST primitive -- consistent per primitive, but
                # semantically arbitrary. The final state keeps the task loss only.
                steps = range(cfg.depth - 1) if arb else range(cfg.depth)
                for t in steps:
                    with torch.no_grad():
                        if arb:
                            target_state = model.arbitrary_targets[ib[:, t]]
                        else:
                            target_state = model.code(ysb[:, t])
                    # Normalised so the term is *relative* squared error and does not
                    # depend on the scale the encoder happens to settle on.
                    denom = target_state.pow(2).mean().clamp_min(1e-6)
                    rel_sq = F.mse_loss(out["states"][:, t], target_state) / denom
                    loss = loss + cfg.state_consistency_weight * rel_sq
                    state_raw.append(float(rel_sq.detach()))
                terms["loss_state_rel"] = (sum(state_raw) / len(state_raw)) ** 0.5
            if cfg.intermediate_supervision:
                # Force each intermediate state to decode to the partial composition.
                # This is what makes atom_i a closed map on the latent code rather
                # than a transform that only works after its training partners.
                for t in range(cfg.depth - 1):
                    loss = loss + F.cross_entropy(
                        model.decoder(out["states"][:, t]).reshape(-1, cfg.vocab),
                        ysb[:, t].reshape(-1),
                    )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            set_atom_grads_trainable(model, trainable_atoms)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            ).item()
            optim.step()
            sched.step()

            correct = (out["logits"].argmax(-1) == yb).all(dim=-1)
            epoch_correct += int(correct.sum())
            epoch_seen += correct.numel()

            state["step"] += 1
            state["global_step_in_phase"] += 1
            if state["step"] % cfg.log_every == 0:
                log({
                    "step": state["step"], "phase": phase_name, "epoch": epoch,
                    "loss": float(loss.item()), "lr": float(sched.get_last_lr()[0]),
                    "temp": float(tau), "train_acc": float(correct.float().mean()),
                    "grad_norm": float(grad_norm),
                    **terms,
                })

        epoch_acc = epoch_correct / max(1, epoch_seen)
        log({"step": state["step"], "phase": phase_name, "epoch": epoch,
             "epoch_train_acc": epoch_acc, "event": "epoch_end"})
        if rss is not None:
            rss.sample()

        if not cfg.early_stop:
            continue
        plateau = plateau + 1 if epoch_acc >= cfg.early_stop_acc else 0
        if plateau >= cfg.early_stop_patience:
            log({"step": state["step"], "phase": phase_name, "epoch": epoch,
                 "event": "early_stop", "epoch_train_acc": epoch_acc})
            break

    state["global_step_in_phase"] = 0


def run_codec_pretrain(model, cfg, arrays: TrainArrays, *, log, rss=None) -> None:
    """R3-fixed phase 0: train encoder+decoder (+state_norm) on reconstruction only.

    The R3 bottleneck routes every intermediate state through the decoder. With an
    untrained codec the atoms learn to write through a transcriber that
    mis-transcribes, and the two corrupt each other's signal -- the observed
    pre-fix signature was a flat ~0.001 accuracy curve for 58 epochs. Phase 0 gives
    the projection a working codec before any atom is asked to use it.

    Atoms and composer are excluded from the optimizer here: this is tokens -> enc ->
    [state_norm] -> dec -> tokens cross-entropy and nothing else. See D35.
    """
    x, _, _, _ = arrays.torch_view()
    codec = list(model.encoder.parameters()) + list(model.decoder.parameters())
    if model.state_norm is not None:
        codec += list(model.state_norm.parameters())
    optim = torch.optim.AdamW(codec, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, arrays.n // cfg.batch_size)
    order_rng = np.random.default_rng(cfg.seed * 977 + 3)

    model.train()
    for epoch in range(cfg.codec_pretrain_epochs):
        order = order_rng.permutation(arrays.n)
        correct = total = 0
        for b in range(steps_per_epoch):
            idx = torch.from_numpy(order[b * cfg.batch_size:(b + 1) * cfg.batch_size])
            xb = x[idx]
            logits = model.decoder(model.code(xb))
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), xb.reshape(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec, cfg.grad_clip)
            optim.step()
            correct += int((logits.argmax(-1) == xb).all(dim=-1).sum())
            total += xb.shape[0]
        acc = correct / max(1, total)
        log({"event": "codec_pretrain_epoch", "phase": "codec_pretrain",
             "epoch": epoch, "loss": float(loss.item()), "recon_acc": acc})
        if rss is not None:
            rss.sample()
    log({"event": "codec_pretrain_done", "recon_acc": acc})


def train(cfg: Config, run_dir: Path, allow_dirty: bool = False) -> AtomNet:
    dirty_diff = require_clean_tree(allow_dirty)
    set_threads(cfg.num_threads, cfg.num_interop_threads)
    seed_everything(cfg.seed)

    split_path = split_path_for(cfg.generation, cfg.split_seed)
    split = load_split(split_path)
    problems = verify_split(split, cfg.primitive_set)
    if problems:
        raise RuntimeError(f"split verification failed: {problems}")

    bundle = build_bundle(cfg, split)
    model = AtomNet(cfg)
    pc = param_counts(model)
    rss = RSSMonitor(cfg.rss_fail_gb)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    log_file = open(log_path, "w", encoding="utf-8", newline="\n")

    def log(rec):
        log_file.write(json.dumps(rec, sort_keys=True) + "\n")
        log_file.flush()   # live progress during long unattended batches

    state = {"step": 0, "phase_counter": 0, "global_step_in_phase": 0}
    t0 = time.time()

    if cfg.sequential:
        # --- A3 phase 1: grow the library one atom at a time on length-1 tasks.
        singletons = [td for td in bundle.train if td.task.kind == "singleton"]
        singletons.sort(key=lambda td: td.task.primitives[0])
        for i in range(cfg.n_atoms):
            stage_tasks = singletons[: i + 1]
            arrays = TrainArrays(stage_tasks, cfg)
            mask = torch.zeros(cfg.n_atoms, dtype=torch.bool)
            mask[: i + 1] = True
            # Equal step budget per stage regardless of how much data the stage has.
            # Early stopping normally ends a stage well before this cap.
            steps_per_epoch = max(1, arrays.n // cfg.batch_size)
            stage_epochs = max(1, -(-cfg.seq_steps_per_atom // steps_per_epoch))
            # A3b: pin primitive p -> atom p, with atom 0 (trained first, on the
            # identity task) as the trailing no-op. A3 leaves this to free routing,
            # which does not in fact discover it (D21/D23).
            run_phase(
                model, cfg, arrays,
                epochs=stage_epochs, log=log, state=state,
                atom_mask=mask, trainable_atoms={i}, include_atoms=True,
                mode="forced" if cfg.sequential_forced_assignment else "gumbel",
                phase_name=f"seq_stage_{i}", rss=rss,
            )
        # Snapshot the library at the end of phase 1, BEFORE phase 2 moves the code
        # underneath it. Without this, "phase 2 invalidated the library" is an
        # inference; with it, it is a measurement. See D20.
        from .evaluate import compute_alignment
        model.eval()
        A_p1, A1_p1, id_atom_p1 = compute_alignment(model, bundle.probe_inputs, cfg)
        model.train()
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "artifacts" / "alignment_matrix_phase1.npy", A_p1)
        np.save(run_dir / "artifacts" / "alignment_matrix_1step_phase1.npy", A1_p1)
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
                   run_dir / "checkpoints" / "phase1.pt")
        log({"event": "phase1_snapshot",
             "align_phase1": float(A_p1.max(axis=1).mean()),
             "primitives_covered_phase1": int(len(set(A_p1.argmax(axis=1).tolist()))),
             "identity_atom_phase1": id_atom_p1})

        # --- A3 phase 2: length-2 pairs with the whole library frozen.
        for p in model.atom_parameters():
            p.requires_grad_(False)
        arrays = TrainArrays(bundle.train, cfg)
        run_phase(
            model, cfg, arrays, epochs=cfg.epochs, log=log, state=state,
            include_atoms=False, phase_name="seq_frozen", rss=rss,
        )
        for p in model.atom_parameters():
            p.requires_grad_(True)
    else:
        arrays = TrainArrays(bundle.train, cfg)
        if cfg.shuffle_labels:
            shuffle_targets(arrays, seed=cfg.seed * 7 + 13)
        if getattr(cfg, "arbitrary_targets", False):
            # Must be built from the INITIAL encoder and frozen thereafter (D36).
            model.init_arbitrary_targets(cfg.split_seed)
        if getattr(cfg, "codec_pretrain_epochs", 0) > 0:
            run_codec_pretrain(model, cfg, arrays, log=log, rss=rss)
        run_phase(
            model, cfg, arrays, epochs=cfg.epochs, log=log, state=state,
            mode="forced" if cfg.forced_routing else "gumbel",
            phase_name="main", rss=rss,
        )

    train_seconds = time.time() - t0
    log({"event": "done", "train_seconds": train_seconds, "peak_rss_gb": rss.peak_gb})
    log_file.close()

    torch.save(
        {"model": model.state_dict(), "config": cfg.to_dict()},
        run_dir / "checkpoints" / "final.pt",
    )

    write_json(run_dir / "config.json", cfg.to_dict())
    env = env_info(cfg)
    env["dirty_diff_sha256"] = dirty_diff      # None when the tree was clean
    env["train_seconds"] = train_seconds
    env["peak_rss_gb"] = rss.peak_gb
    write_json(run_dir / "env.json", env)
    write_json(run_dir / "split_ref.json", {
        "path": str(split_path.relative_to(REPO_ROOT).as_posix()),
        "sha256": split_hash(split_path),
        "generation": cfg.generation,
        "primitive_set": cfg.primitive_set,
        "split_seed": cfg.split_seed,
        "n_train_pairs": split["n_train_pairs"],
        "n_heldout_pairs": split["n_heldout_pairs"],
    })

    emit_artifacts(model, bundle, cfg, run_dir, pc)
    rss.sample()
    write_sha256sums(run_dir)
    return model


def run_id_for(arm: str, seed: int, split_seed: int | None = None) -> str:
    """v1 keeps its historical single-split id so existing run names still resolve.

    Generations with more than one split carry the split seed in the id -- otherwise
    the same arm on two different splits collides on one directory and the second run
    silently overwrites the first.
    """
    base = f"{arm}_{seed}" if split_seed is None else f"{arm}_{seed}_s{split_seed}"
    return f"{base}_{git_info()['git_sha_short']}"


def run_dir_for(cfg: Config) -> Path:
    """runs/<generation>/<run_id>. See D40.

    Runs made before D40 sit flat at runs/<run_id> and are left there: they carry the
    Perro provenance (D39) and moving them would detach them from the report that
    cites them. Discovery is recursive, so both layouts aggregate together.
    """
    multi = len(GENERATIONS[cfg.generation]["split_seeds"]) > 1
    rid = run_id_for(cfg.arm, cfg.seed, cfg.split_seed if multi else None)
    return RUNS_DIR / cfg.generation / rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--generation", default="v1",
                    help="which experiment generation to run (D40)")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="which of the generation's frozen splits to use")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run from a dirty tree, recording the diff hash in env.json")
    ap.add_argument("--threads", type=int, default=None,
                    help="override num_threads. This is a DETERMINISM parameter, not "
                         "a performance knob -- thread count changes reduction order. "
                         "Sweep it once, freeze one value in Config, and do not vary "
                         "it within a batch. See DECISIONS.md D39.")
    args = ap.parse_args()

    cfg = config_for_generation(args.generation, args.arm, args.seed, args.split_seed)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.threads is not None:
        cfg.num_threads = args.threads
    run_dir = Path(args.out) if args.out else run_dir_for(cfg)
    print(f"[{args.generation}/{args.arm} seed={args.seed} "
          f"split={cfg.split_seed}] -> {run_dir}")
    train(cfg, run_dir, allow_dirty=args.allow_dirty)
    print(f"done in {json.load(open(run_dir / 'env.json'))['train_seconds']:.1f}s")


if __name__ == "__main__":
    main()
