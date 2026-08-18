"""E2 batch driver (H1-Experiment2.md): interface-noise battery.

Registered order:
  1. A6 no-noise equivalence gate - the completed E1b A6 runs may serve as
     the control ONLY if the new zero-noise code path is bit-identical to the
     E1b implementation. Verified mechanically: replay the first 50 training
     steps of A6 seed 0 under the current harness and compare the logged
     step-1 and step-50 loss records byte-for-byte against the completed
     run's train_log. Failure => rerun A6 under this harness (hard stop).
  2. A8/A9/A10 smoke runs + implementation gates (liveness telemetry present,
     p_max invariant, observed cosine within 0.005 of target, transmitted
     scale ~N(0,1), clean-eval determinism). Structural gates (noise
     isolation, no clean side channel, timing, masking, pass channel noise)
     are enforced by tests/test_e2.py.
  3. A8/A9/A10 x seeds 0/1/2, 20k steps, full certified panel + registered
     robustness sweep + analyze.
  4. Evaluation-only robustness sweep on the E1b A6 reference checkpoints.

An implementation-gate failure invalidates the run. Optimization failure
under correctly implemented noise is a result, not an invalid run.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from . import data as data_mod
from . import registered as R
from .aggregate import aggregate
from .analyze import analyze
from .config import E2_ARMS, config_for_arm, run_dir_for
from .model import AtomModel
from .noise import robustness_sweep
from .panel import run_all_panels
from .train import train_run
from .utils import (RESULTS_DIR, RUNS_DIR, read_json, seed_everything,
                    set_threads, stream_rng, stream_seed, write_json,
                    write_sha256sums)


def _find_run_dir(experiment: str, arm: str, seed: int):
    hits = sorted((RUNS_DIR / experiment).glob(f"{arm}_s{seed}_*"))
    hits = [h for h in hits if (h / "train_log.jsonl").exists()]
    if not hits:
        raise SystemExit(f"no completed {experiment} run found for "
                         f"{arm} seed {seed}")
    return hits[-1]


def _replay_first_steps(cfg, n_steps: int) -> dict:
    """Reproduce train_run's exact first n_steps (including the
    init-calibration forward that consumes the Gumbel stream) and return the
    {step: record} losses the training loop would have logged."""
    set_threads()
    seed_everything(cfg.seed)
    bundle = data_mod.build_bundle(cfg)
    arrays = data_mod.build_epoch_arrays(bundle, cfg, include_partials=False)
    n_examples = len(arrays["x"])
    model = AtomModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  betas=cfg.betas,
                                  weight_decay=cfg.weight_decay)
    gumbel_gen = torch.Generator().manual_seed(
        int(stream_seed(cfg.seed, "gumbel").generate_state(1)[0]))

    model.eval()  # init calibration consumes the same draws train_run does
    with torch.no_grad():
        xb = torch.from_numpy(arrays["x"][: cfg.batch_size])
        tb = torch.from_numpy(arrays["tokens"][: cfg.batch_size])
        nb = torch.from_numpy(arrays["n_tokens"][: cfg.batch_size])
        model(xb, tb, nb, mode="gumbel", tau=cfg.tau_start,
              generator=gumbel_gen)

    records = {}
    order = stream_rng(cfg.seed, "shuffle", 0).permutation(n_examples)
    step = 0
    for b in range(n_steps):
        idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
        xb = torch.from_numpy(arrays["x"][idx])
        yb = torch.from_numpy(arrays["y"][idx])
        tb = torch.from_numpy(arrays["tokens"][idx])
        nb = torch.from_numpy(arrays["n_tokens"][idx])
        lr = cfg.lr * (step + 1) / cfg.warmup_steps \
            if step < cfg.warmup_steps else cfg.lr
        for g in optimizer.param_groups:
            g["lr"] = lr
        model.train()
        out = model(xb, tb, nb, mode="gumbel", tau=cfg.tau_end,
                    generator=gumbel_gen, noise_sigma=cfg.state_noise_sigma,
                    noise_generator=None)
        loss_task = F.cross_entropy(out["logits"].reshape(-1, cfg.vocab),
                                    yb.reshape(-1))
        rent_raw = out["soft_atom_mass"].sum(dim=1).mean()
        loss = loss_task + cfg.lambda_use * rent_raw
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.grad_clip)
        optimizer.step()
        step += 1
        if step == 1 or step % cfg.log_every == 0:
            with torch.no_grad():
                batch_acc = (out["logits"].argmax(-1) == yb).all(-1
                             ).float().mean().item()
            records[step] = {"loss": loss.item(),
                             "loss_task": loss_task.item(),
                             "grad_norm": float(grad_norm),
                             "batch_acc": batch_acc}
    return records


def a6_equivalence_gate(out_dir) -> None:
    """Registered reuse condition: zero-noise path bit-identical to E1b A6."""
    cfg = config_for_arm(R.E2_BASE_ARM, 0)
    assert cfg.state_noise_sigma == 0.0
    a6_dir = _find_run_dir("e1b", R.E2_BASE_ARM, 0)
    logged = {}
    import json
    with open(a6_dir / "train_log.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "step" and rec["step"] in (1, 50):
                logged[rec["step"]] = rec
    if set(logged) != {1, 50}:
        raise SystemExit(f"A6 equivalence: steps 1/50 not found in "
                         f"{a6_dir}/train_log.jsonl")
    replayed = _replay_first_steps(cfg, n_steps=50)
    mismatches = []
    for step in (1, 50):
        for key in ("loss", "loss_task", "grad_norm", "batch_acc"):
            a, b = logged[step][key], replayed[step][key]
            if a != b:
                mismatches.append({"step": step, "key": key,
                                   "e1b_logged": a, "replayed": b})
    record = {"reference_run": str(a6_dir), "steps_replayed": 50,
              "compared_steps": [1, 50], "bit_identical": not mismatches,
              "mismatches": mismatches}
    write_json(out_dir / "e2_a6_equivalence.json", record)
    if mismatches:
        raise SystemExit(
            f"A6 EQUIVALENCE GATE FAILED: zero-noise replay diverges from the "
            f"E1b A6 run ({mismatches[:2]}...). The completed A6 runs may NOT "
            "serve as the E2 control; rerun A6 seeds 0/1/2 under this harness "
            "before interpreting treatment comparisons.")
    print("A6 equivalence gate: bit-identical (steps 1 and 50)")


def _smoke_gates(arm: str, allow_dirty: bool) -> None:
    cfg = config_for_arm(arm, 0, smoke=True)
    run_dir = run_dir_for(cfg)
    if not (run_dir / "checkpoints" / "final.pt").exists():
        print(f"=== smoke {arm} -> {run_dir}")
        train_run(cfg, allow_dirty=allow_dirty)
    lv = sorted((run_dir / "liveness").glob("step*.json"))
    if len(lv) < 2:
        raise SystemExit(f"smoke {arm}: <2 liveness evals")
    for p in lv:
        if not read_json(p)["base_geometry"]["pmax_invariant_ok"]:
            raise SystemExit(f"smoke {arm}: p_max invariant violated ({p})")
    nts = sorted((run_dir / "noise_telemetry").glob("step*.json"))
    if not nts:
        raise SystemExit(f"smoke {arm}: no noise telemetry written")
    for p in nts:
        nt = read_json(p)
        if not nt["handoff"]["cosine_within_tol"]:
            raise SystemExit(
                f"smoke {arm}: observed cosine "
                f"{nt['handoff']['cosine']['p50']:.6f} not within "
                f"{R.E2_COSINE_TOL} of target {nt['target_cosine']:.6f} ({p})")
        if nt["handoff"]["transmitted_pos_mean_abs"] > 0.1 \
                or abs(nt["handoff"]["transmitted_pos_var"] - 1.0) > 0.1:
            raise SystemExit(f"smoke {arm}: transmitted-state scale off "
                             f"(mean_abs={nt['handoff']['transmitted_pos_mean_abs']}, "
                             f"var={nt['handoff']['transmitted_pos_var']}) ({p})")
    # Clean-eval determinism: repeated clean hard forward is bit-identical
    # and contains no interface-noise draw.
    from .panel import load_checkpoint
    model, ck_cfg, _ = load_checkpoint(run_dir, "final.pt")
    diag_rng = stream_rng(ck_cfg.seed, "e1b_diag")
    x = torch.from_numpy(diag_rng.integers(0, 10, size=(64, 6)).astype("int64"))
    toks = torch.from_numpy(diag_rng.integers(0, 8, size=(64, 2)).astype("int64"))
    ntok = torch.full((64,), 2, dtype=torch.int64)
    with torch.no_grad():
        o1 = model(x, toks, ntok, mode="hard")
        o2 = model(x, toks, ntok, mode="hard")
    if not (torch.equal(o1["logits"], o2["logits"])
            and torch.equal(o1["choices"], o2["choices"])):
        raise SystemExit(f"smoke {arm}: clean hard eval is not deterministic")
    if "states_transmitted" in o1:
        raise SystemExit(f"smoke {arm}: clean eval carries a noise payload")
    print(f"    smoke {arm}: {len(lv)} liveness evals, {len(nts)} noise "
          "records, cosine + scale + determinism gates ok")


def _run_one(arm: str, seed: int, smoke: bool, allow_dirty: bool) -> dict:
    cfg = config_for_arm(arm, seed, smoke=smoke)
    run_dir = run_dir_for(cfg)
    if (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists():
        print(f"skip (complete): {run_dir}")
        return read_json(run_dir / "metrics.json")
    print(f"=== {arm} (state_noise_sigma={cfg.state_noise_sigma:.9f}, "
          f"target_cos={R.E2_TARGET_COSINE[arm]}) seed={seed} -> {run_dir}")
    if (run_dir / "checkpoints" / "final.pt").exists():
        print("    training already complete; resuming at panel")
    else:
        train_run(cfg, allow_dirty=allow_dirty)
    run_all_panels(run_dir)
    if not (run_dir / "noise_robustness.json").exists():
        write_json(run_dir / "noise_robustness.json", robustness_sweep(run_dir))
    metrics = analyze(run_dir)
    write_sha256sums(run_dir)
    print(f"    seen={metrics['acc_seen_hard']:.4f} "
          f"L1={metrics['acc_unseen_L1_hard']:.4f} "
          f"L2={metrics['acc_unseen_L2_hard']:.4f} "
          f"L3={metrics['acc_unseen_L3_hard']:.4f} "
          f"repairL3={metrics.get('canon_repair_acc_L3')} "
          f"cos_gate={metrics.get('e2_cosine_gate_ok')} "
          f"valid={metrics.get('e1b_run_valid')}")
    return metrics


def _a6_reference_sweeps(out_dir) -> None:
    sweeps = {}
    for seed in R.SEEDS:
        run_dir = _find_run_dir("e1b", R.E2_BASE_ARM, seed)
        path = run_dir / "noise_robustness.json"
        if not path.exists():
            print(f"=== robustness sweep on reference {run_dir.name}")
            write_json(path, robustness_sweep(run_dir))
            write_sha256sums(run_dir)
        sweeps[f"seed{seed}"] = read_json(path)
    write_json(out_dir / "a6_reference_robustness.json", sweeps)


def _validity_verdict(rows: list[dict]) -> dict:
    per_run, per_arm = {}, {}
    for m in rows:
        per_run[f"{m['arm']}_seed{m['seed']}"] = {
            "liveness_valid": bool(m.get("e1b_run_valid")),
            "deafness_violated": bool(m.get("e1b_deafness_violated")),
            "cosine_gate_ok": bool(m.get("e2_cosine_gate_ok")),
            "valid": bool(m.get("e1b_run_valid"))
            and bool(m.get("e2_cosine_gate_ok")),
        }
    for arm in E2_ARMS:
        seeds = [v for k, v in per_run.items() if k.startswith(f"{arm}_")]
        per_arm[arm] = {
            "n_runs": len(seeds),
            "n_valid": sum(v["valid"] for v in seeds),
            "liveness_premise_holds": bool(seeds) and not any(
                v["deafness_violated"] for v in seeds),
            "implementation_gates_ok": bool(seeds) and all(
                v["cosine_gate_ok"] for v in seeds),
        }
    return {"per_run": per_run, "per_arm": per_arm}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E2 battery "
                                             "(interface noise)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(R.SEEDS))
    ap.add_argument("--arms", nargs="*", default=list(E2_ARMS))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-equivalence", action="store_true")
    ap.add_argument("--skip-free-smoke", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    bad = [a for a in args.arms if a not in E2_ARMS]
    if bad:
        raise SystemExit(f"unknown E2 arms {bad}; battery arms are {E2_ARMS}. "
                         "A6 is the reused E1b control, never re-run here "
                         "unless the equivalence gate fails.")

    batch = [(arm, seed) for arm in args.arms for seed in args.seeds]
    if args.plan:
        print("A6 equivalence gate (replay 50 steps vs e1b A6 s0 train_log)")
        for arm in args.arms:
            print(f"smoke {arm} seed=0  (implementation gates)")
        for arm, seed in batch:
            print(f"{arm} (sigma={R.E2_STATE_NOISE_SIGMA[arm]:.9f}, "
                  f"cos={R.E2_TARGET_COSINE[arm]}) seed={seed}")
        print("robustness sweep on e1b A6 references")
        return

    out = RESULTS_DIR / ("smoke_e2" if args.smoke else "e2")
    out.mkdir(parents=True, exist_ok=True)

    if not args.smoke and not args.skip_equivalence:
        a6_equivalence_gate(out)
    if not args.smoke and not args.skip_free_smoke:
        for arm in args.arms:
            _smoke_gates(arm, args.allow_dirty)

    rows = [
        _run_one(arm, seed, args.smoke, args.allow_dirty)
        for arm, seed in batch
    ]

    verdict = _validity_verdict(rows)
    write_json(out / "e2_validity.json", verdict)
    for arm, v in verdict["per_arm"].items():
        print(f"{arm}: {v['n_valid']}/{v['n_runs']} valid, liveness "
              f"{'HOLDS' if v['liveness_premise_holds'] else 'FAILED'}, "
              f"gates {'ok' if v['implementation_gates_ok'] else 'FAILED'}")

    if not args.smoke:
        _a6_reference_sweeps(out)
    print(aggregate("e2", smoke=args.smoke))


if __name__ == "__main__":
    main()
