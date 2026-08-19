"""E4 batch driver (H1-Experiment4.md): catching + throwing stacked.

Both prior treatments coexist unchanged on the A6 base: E2 interface noise
(A9/A8 doses) + E3 atom sandbox (A12/A11 doses). Registered order:

  1. Three replay gates, all hard stops:
     a. A6 zero-path gate - state_noise_sigma = 0 AND lambda_sandbox_* = 0
        replays E1b A6 seed 0 bit-identical (steps 1 and 50). Reuses run_e2's
        mechanical gate.
     b. Noise-only gate - the A9 config replayed under THIS harness
        reproduces the completed E2 A9 seed-0 step-1/step-50 records.
     c. Sandbox-only gate - the A12 config replayed under THIS harness
        reproduces the completed E3 A12 seed-0 step-1/step-50 records.
  2. A14/A15 smoke runs + implementation gates (all E2 gates: cosine target,
     scale, clean-eval determinism; all E3 gates: telemetry finite, weights
     in range, init calibration carries sandbox scale; liveness present).
     Structural gates stay unit-enforced (tests/test_e2.py, test_e3.py,
     test_e4.py).
  3. A14/A15 x seeds 0/1/2, 30k steps, panels at 20k + final, registered
     robustness sweep, analyze (which emits the 20k like-for-like block and
     the per-P3-cell dax-crack check).

A6/A9/A12 attach as labelled reference rows at aggregation, never re-run.
Implementation failure invalidates a run; optimization failure under
correctly implemented pressures is a result, not an invalid run.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from . import data as data_mod
from . import registered as R
from .aggregate import aggregate
from .analyze import analyze
from .config import E4_ARMS, config_for_arm, run_dir_for
from .model import AtomModel
from .noise import robustness_sweep
from .panel import run_all_panels
from .run_e2 import _find_run_dir, a6_equivalence_gate
from .train import train_run
from .utils import (RESULTS_DIR, read_json, seed_everything, set_threads,
                    stream_rng, stream_seed, write_json, write_sha256sums)


def _replay_first_steps(cfg, n_steps: int) -> dict:
    """Reproduce train_run's exact first n_steps for ANY free-arm config -
    zero-path, noise-bearing, or sandbox-bearing - including the
    init-calibration forward (which consumes the Gumbel stream but no noise
    or sandbox training draws, exactly as train_run orders it)."""
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
    noise_gen = None
    if cfg.state_noise_sigma > 0:
        noise_gen = torch.Generator().manual_seed(
            int(stream_seed(cfg.seed, "state_noise").generate_state(1)[0]))
    sandbox_state = None
    if cfg.lambda_sandbox_valid > 0 or cfg.lambda_sandbox_unique > 0:
        from . import sandbox as sandbox_mod
        sandbox_state = sandbox_mod.SandboxState(cfg)

    model.eval()  # init calibration: same Gumbel draws train_run consumes
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
                    noise_generator=noise_gen)
        loss_task = F.cross_entropy(out["logits"].reshape(-1, cfg.vocab),
                                    yb.reshape(-1))
        rent_raw = out["soft_atom_mass"].sum(dim=1).mean()
        loss = loss_task + cfg.lambda_use * rent_raw
        record = {"loss_task": loss_task.item()}
        if sandbox_state is not None:
            sandbox_state.update_usage(out["choices"].detach())
            sb_terms = sandbox_mod.sandbox_losses(
                model, out["states"][0].detach(), sandbox_state)
            loss = (loss
                    + cfg.lambda_sandbox_valid
                    * sb_terms["loss_sandbox_valid"]
                    + cfg.lambda_sandbox_unique
                    * sb_terms["loss_sandbox_unique"])
            record["loss_sandbox_valid"] = float(
                sb_terms["loss_sandbox_valid"])
            record["loss_sandbox_unique"] = float(
                sb_terms["loss_sandbox_unique"])
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
            record.update({"loss": loss.item(), "grad_norm": float(grad_norm),
                           "batch_acc": batch_acc})
            records[step] = dict(record)
    return records


def _single_pressure_gate(exp: str, arm: str, out_dir, filename: str) -> None:
    """Gates 1b/1c: a single-treatment config replayed under THIS harness
    must reproduce the completed battery run's step-1/step-50 records."""
    cfg = config_for_arm(arm, 0)
    ref_dir = _find_run_dir(exp, arm, 0)
    logged = {}
    with open(ref_dir / "train_log.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "step" and rec["step"] in (1, 50):
                logged[rec["step"]] = rec
    if set(logged) != {1, 50}:
        raise SystemExit(f"{arm} gate: steps 1/50 not found in "
                         f"{ref_dir}/train_log.jsonl")
    replayed = _replay_first_steps(cfg, n_steps=50)
    keys = ["loss", "loss_task", "grad_norm", "batch_acc"]
    if cfg.lambda_sandbox_valid > 0:
        keys += ["loss_sandbox_valid", "loss_sandbox_unique"]
    mismatches = []
    for step in (1, 50):
        for key in keys:
            a, b = logged[step][key], replayed[step][key]
            if a != b:
                mismatches.append({"step": step, "key": key,
                                   "reference": a, "replayed": b})
    record = {"reference_run": str(ref_dir), "arm": arm,
              "compared_steps": [1, 50], "compared_keys": keys,
              "bit_identical": not mismatches, "mismatches": mismatches}
    write_json(out_dir / filename, record)
    if mismatches:
        raise SystemExit(
            f"{arm.upper()} SINGLE-PRESSURE GATE FAILED: the {arm} config "
            f"replayed under this harness diverges from the completed {exp} "
            f"run ({mismatches[:2]}...). The merged harness does not "
            f"reproduce the {exp} treatment; nothing proceeds until it does.")
    print(f"{arm} single-pressure gate: bit-identical (steps 1 and 50)")


def _smoke_gates(arm: str, allow_dirty: bool) -> None:
    """All E2 + E3 artifact-level implementation gates on one smoke run."""
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
            raise SystemExit(f"smoke {arm}: transmitted-state scale off ({p})")
    sts = sorted((run_dir / "sandbox_telemetry").glob("step*.json"))
    if not sts:
        raise SystemExit(f"smoke {arm}: no sandbox telemetry written")
    for p in sts:
        st = read_json(p)
        for key, val in (("standalone read", st["standalone"]["read_mean"]),
                         ("closure read", st["closure"]["read_mean"])):
            if not math.isfinite(val):
                raise SystemExit(f"smoke {arm}: non-finite {key} ({p})")
        if not all(0.0 <= w <= 1.0 for w in st["usage"]["weights"]):
            raise SystemExit(f"smoke {arm}: usage weight outside [0,1] ({p})")
    cal = read_json(run_dir / "init_calibration.json")
    if "loss_sandbox_valid_init" not in cal:
        raise SystemExit(f"smoke {arm}: init calibration lacks sandbox scale")
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
    print(f"    smoke {arm}: {len(lv)} liveness, {len(nts)} noise, "
          f"{len(sts)} sandbox records; all implementation gates ok")


def _run_one(arm: str, seed: int, smoke: bool, allow_dirty: bool) -> dict:
    cfg = config_for_arm(arm, seed, smoke=smoke)
    run_dir = run_dir_for(cfg)
    if (run_dir / "metrics.json").exists() and (run_dir / "SHA256SUMS").exists():
        print(f"skip (complete): {run_dir}")
        return read_json(run_dir / "metrics.json")
    print(f"=== {arm} (noise_sigma={cfg.state_noise_sigma:.9f}, "
          f"lambda_sandbox={cfg.lambda_sandbox_valid}) seed={seed} "
          f"steps={cfg.total_steps} -> {run_dir}")
    if (run_dir / "checkpoints" / "final.pt").exists():
        print("    training already complete; resuming at panel")
    else:
        train_run(cfg, allow_dirty=allow_dirty)
    run_all_panels(run_dir)
    if not (run_dir / "noise_robustness.json").exists():
        write_json(run_dir / "noise_robustness.json", robustness_sweep(run_dir))
    metrics = analyze(run_dir)
    write_sha256sums(run_dir)
    def _f(key):
        v = metrics.get(key)
        return "-" if v is None else f"{v:.4f}"
    print(f"    final: seen={metrics['acc_seen_hard']:.4f} "
          f"L1={metrics['acc_unseen_L1_hard']:.4f} "
          f"L3={metrics['acc_unseen_L3_hard']:.4f} | "
          f"20k: seen={_f('e4_acc_seen_hard_20k')} "
          f"L1={_f('e4_acc_unseen_L1_hard_20k')} | "
          f"dax_max={_f('e4_dax_max_cell_acc')} "
          f"crack={metrics.get('e4_dax_crack')} "
          f"valid={metrics.get('e1b_run_valid')}")
    return metrics


def _validity_verdict(rows: list[dict]) -> dict:
    per_run, per_arm = {}, {}
    for m in rows:
        per_run[f"{m['arm']}_seed{m['seed']}"] = {
            "liveness_valid": bool(m.get("e1b_run_valid")),
            "deafness_violated": bool(m.get("e1b_deafness_violated")),
            "cosine_gate_ok": bool(m.get("e2_cosine_gate_ok")),
            "dax_crack": bool(m.get("e4_dax_crack")),
            "valid": bool(m.get("e1b_run_valid"))
            and bool(m.get("e2_cosine_gate_ok")),
        }
    for arm in E4_ARMS:
        seeds = [v for k, v in per_run.items() if k.startswith(f"{arm}_")]
        per_arm[arm] = {
            "n_runs": len(seeds),
            "n_valid": sum(v["valid"] for v in seeds),
            "liveness_premise_holds": bool(seeds) and not any(
                v["deafness_violated"] for v in seeds),
            "dax_crack_any_seed": any(v["dax_crack"] for v in seeds),
        }
    return {"per_run": per_run, "per_arm": per_arm,
            "note": "outcome bins (STACK/REDUNDANT/INTERFERE/JOINT OVERDOSE/"
                    "DAX CRACK) are called against the registered taxonomy "
                    "from the 20k like-for-like columns; a dax crack in any "
                    "healthy arm is the headline regardless of other bins."}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the E4 battery "
                                             "(stacked pressures)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(R.SEEDS))
    ap.add_argument("--arms", nargs="*", default=list(E4_ARMS))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-equivalence", action="store_true")
    ap.add_argument("--skip-free-smoke", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    bad = [a for a in args.arms if a not in E4_ARMS]
    if bad:
        raise SystemExit(f"unknown E4 arms {bad}; battery arms are {E4_ARMS}. "
                         "A6/A9/A12 are completed references, never re-run.")

    batch = [(arm, seed) for arm in args.arms for seed in args.seeds]
    if args.plan:
        print("gate 1a: A6 zero-path replay vs e1b A6 s0 (bit-identical)")
        print("gate 1b: A9 noise-only replay vs e2 A9 s0 (bit-identical)")
        print("gate 1c: A12 sandbox-only replay vs e3 A12 s0 (bit-identical)")
        for arm in args.arms:
            print(f"smoke {arm} seed=0  (E2+E3 implementation gates)")
        for arm, seed in batch:
            print(f"{arm} (noise={R.E4_STATE_NOISE_SIGMA[arm]:.9f}, "
                  f"sandbox={R.E4_LAMBDA_VALID[arm]}) seed={seed} "
                  f"steps={R.E4_TOTAL_STEPS}")
        return

    out = RESULTS_DIR / ("smoke_e4" if args.smoke else "e4")
    out.mkdir(parents=True, exist_ok=True)

    if not args.smoke and not args.skip_equivalence:
        a6_equivalence_gate(out, filename="e4_a6_equivalence.json")
        _single_pressure_gate("e2", "A9", out, "e4_a9_equivalence.json")
        _single_pressure_gate("e3", "A12", out, "e4_a12_equivalence.json")
    if not args.smoke and not args.skip_free_smoke:
        for arm in args.arms:
            _smoke_gates(arm, args.allow_dirty)

    rows = [
        _run_one(arm, seed, args.smoke, args.allow_dirty)
        for arm, seed in batch
    ]

    verdict = _validity_verdict(rows)
    write_json(out / "e4_validity.json", verdict)
    for arm, v in verdict["per_arm"].items():
        print(f"{arm}: {v['n_valid']}/{v['n_runs']} valid, liveness "
              f"{'HOLDS' if v['liveness_premise_holds'] else 'FAILED'}, "
              f"dax crack {'YES' if v['dax_crack_any_seed'] else 'no'}")

    print(aggregate("e4", smoke=args.smoke))


if __name__ == "__main__":
    main()
