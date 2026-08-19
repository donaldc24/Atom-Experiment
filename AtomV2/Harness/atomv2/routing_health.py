"""Routing-health diagnostic: does the router still receive gradient?

Standalone on purpose. It is NOT wired into run_panel, because adding outputs
to the panel mid-battery would make early and late runs incomparable. Run it
over completed runs at any time; it reads checkpoints and train logs only.

What it exists to catch (found 2026-08-15 while A2 was mid-flight):

The rent term is an L1 penalty on the post-Gumbel softmax mass. As the router
converges and tau anneals to 0.5, that softmax saturates to exact one-hot in
float32. Two things then die at once:

  1. The rent's gradient becomes EXACTLY zero - the penalty degenerates into
     (live atom steps)/batch_size, a function of batch composition alone. No
     value of lambda_use can revive it, because lambda multiplies zero.
  2. The straight-through gradient to the router from the TASK loss dies too,
     since d(weights)/d(logits) = d(soft)/d(logits) ~ 0 when soft is one-hot.
     The composer freezes and the atoms go on training against fixed routes.

The registered calibration checks rent magnitude at INITIALISATION, where the
softmax is still soft, so it cannot see either failure. This module measures
them where they actually happen: at checkpoints, late in training.

On invariants: this is measurement, not training. It loads a checkpoint,
enables grad to read gradient MAGNITUDES, and discards everything. No optimizer
step is ever taken and nothing is written back to any model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as data_mod
from .panel import load_checkpoint
from .utils import RUNS_DIR, read_json, write_json

BATCH = 128


def tau_at(cfg, step: int) -> float:
    frac = min(step / max(cfg.tau_anneal_steps, 1), 1.0)
    return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac


def measure_checkpoint(run_dir: Path, ckpt_name: str, n_draws: int = 8) -> dict:
    """Measure over SEVERAL Gumbel draws and several batches.

    A single draw is not trustworthy here: the straight-through gradient
    depends on which routes the sample happened to take, and single-draw
    router/atom ratios were observed to swing by four orders of magnitude at
    neighbouring checkpoints. Report the spread, not one number.
    """
    model, cfg, step = load_checkpoint(run_dir, ckpt_name)
    for p in model.parameters():
        p.requires_grad_(True)
    bundle = data_mod.build_bundle(cfg)
    arrays = data_mod.build_epoch_arrays(bundle, cfg)
    tau = tau_at(cfg, step)
    model.train()   # reproduce the TRAINING forward, which is what we diagnose

    router = list(model.composer.parameters()) + [model.atoms.keys,
                                                  model.atoms.pass_key]
    atoms = [model.atoms.w1, model.atoms.b1, model.atoms.w2, model.atoms.b2]

    def grad_norm(loss, params):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        return float(torch.sqrt(sum((p.grad ** 2).sum()
                                    for p in params if p.grad is not None)))

    sat, passm, ratios, rentg, taskg = [], [], [], [], []
    rng = np.random.default_rng(0)
    for d in range(n_draws):
        idx = rng.choice(len(arrays["x"]), size=min(BATCH, len(arrays["x"])),
                         replace=False)
        xb = torch.from_numpy(arrays["x"][idx])
        yb = torch.from_numpy(arrays["y"][idx])
        tb = torch.from_numpy(arrays["tokens"][idx])
        nb = torch.from_numpy(arrays["n_tokens"][idx])
        out = model(xb, tb, nb, mode="gumbel", tau=tau,
                    generator=torch.Generator().manual_seed(d))
        with torch.no_grad():
            lm = out["soft_atom_mass"][out["live"]]
            sat.append(float((lm == 1.0).float().mean()))
            passm.append(float((1.0 - lm).mean()))
        task = F.cross_entropy(out["logits"].reshape(-1, cfg.vocab),
                               yb.reshape(-1))
        rent = out["soft_atom_mass"].sum(dim=1).mean()
        gr = grad_norm(task, router)
        ga = grad_norm(task, atoms)
        rentg.append(grad_norm(rent, router))  # UNWEIGHTED: lambda scales this
        taskg.append(gr)
        ratios.append(gr / ga if ga else float("nan"))
    model.zero_grad(set_to_none=True)

    def stat(v):
        return {"mean": float(np.mean(v)), "min": float(np.min(v)),
                "max": float(np.max(v))}

    return {
        "checkpoint": ckpt_name, "step": step, "tau": tau,
        "lambda_use": cfg.lambda_use, "n_draws": n_draws,
        "frac_soft_mass_exactly_one": stat(sat),
        # THE headline number: the share of routing probability left on the
        # pass key. The rent can only push mass toward pass, so its gradient
        # scales with this. Starved pass => inert rent, whatever lambda is.
        "mass_on_pass_key": stat(passm),
        "grad_task_router": stat(taskg),
        "router_over_atoms": stat(ratios),
        "grad_rent_router_unweighted": stat(rentg),
        "rent_contribution_at_lambda": stat(
            [r * cfg.lambda_use for r in rentg]),
    }


def saturation_step(run_dir: Path, batch_size: int = BATCH) -> dict:
    """Last step whose rent_raw was NOT an exact multiple of 1/batch_size.

    Once the softmax is one-hot, rent_raw * batch_size counts live atom steps
    and is exactly an integer; before that it carries fractional soft mass. The
    last fractional step is where the rent's gradient died.
    """
    last, total = None, 0
    for line in open(Path(run_dir) / "train_log.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("event") != "step":
            continue
        total += 1
        v = rec["loss_rent_raw"] * batch_size
        if abs(v - round(v)) > 1e-6:
            last = rec["step"]
    return {"last_unsaturated_logged_step": last, "logged_steps": total}


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    ckpts = sorted(p.name for p in (run_dir / "checkpoints").glob("step*.pt"))
    picks = [ckpts[0], ckpts[len(ckpts) // 2], ckpts[-1]] if ckpts else []
    if (run_dir / "checkpoints" / "final.pt").exists():
        picks.append("final.pt")
    report = {
        "run": run_dir.name,
        "saturation": saturation_step(run_dir),
        "checkpoints": [measure_checkpoint(run_dir, c) for c in dict.fromkeys(picks)],
    }
    report["router_frozen_at_end"] = report["checkpoints"][-1]["router_frozen"]
    report["rent_inert_at_end"] = report["checkpoints"][-1]["rent_inert"]
    write_json(run_dir / "routing_health.json", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Router gradient-health diagnostic")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--experiment", default=None, choices=["e0", "e1"])
    args = ap.parse_args()
    targets = ([Path(args.run_dir)] if args.run_dir
               else sorted(p.parent for p in
                           (RUNS_DIR / args.experiment).rglob("config.json")))
    for t in targets:
        if not (t / "train_log.jsonl").exists():
            continue
        r = run(t)
        last = r["checkpoints"][-1]
        print(f"{t.name}")
        print(f"   lambda={last['lambda_use']} "
              f"saturation completed by step "
              f"{r['saturation']['last_unsaturated_logged_step']}")
        print(f"   final: soft mass one-hot {last['frac_soft_mass_exactly_one']:.4f} "
              f"| pass mass {last['mean_mass_on_pass_key']:.3e}")
        print(f"   final: router/atoms grad {last['router_over_atoms']:.3e} "
              f"| rent grad {last['grad_rent_router_unweighted']:.3e}")
        print(f"   router_frozen={r['router_frozen_at_end']} "
              f"rent_inert={r['rent_inert_at_end']}")


if __name__ == "__main__":
    main()
