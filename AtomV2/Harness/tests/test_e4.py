"""E4 registrations (H1-Experiment4.md): stacked-pressure config wiring,
dose transcription-by-reference, the clean-sandbox interaction rule, and the
registered budget change."""
import numpy as np
import torch

from atomv2 import registered as R
from atomv2.config import config_for_arm
from atomv2.model import AtomModel
from atomv2.sandbox import SandboxState, sandbox_losses


def test_doses_are_references_not_transcriptions():
    assert R.E4_STATE_NOISE_SIGMA["A14"] is R.E2_STATE_NOISE_SIGMA["A9"]
    assert R.E4_STATE_NOISE_SIGMA["A15"] is R.E2_STATE_NOISE_SIGMA["A8"]
    assert R.E4_LAMBDA_VALID["A14"] == R.E3_LAMBDA_VALID["A12"] == 0.3
    assert R.E4_LAMBDA_VALID["A15"] == R.E3_LAMBDA_VALID["A11"] == 0.1
    assert R.EXPERIMENT_REVISIONS["e4"] == R.E4_PROTOCOL_REVISION


def test_e4_config_is_a6_plus_both_pressures_and_budget():
    for arm in R.E4_ARMS:
        e4 = config_for_arm(arm, 1).to_dict()
        a6 = config_for_arm("A6", 1).to_dict()
        diff = {k for k in e4 if e4[k] != a6[k]}
        assert diff == {"arm", "experiment", "protocol_revision",
                        "state_noise_sigma", "lambda_sandbox_valid",
                        "lambda_sandbox_unique", "total_steps",
                        "panel_steps"}, arm
        assert e4["total_steps"] == 30_000
        assert tuple(e4["panel_steps"]) == (20_000,)
        assert e4["lambda_use"] == 0.0
    # The smoke path keeps the smoke budget (pipeline checks stay cheap).
    sm = config_for_arm("A14", 0, smoke=True)
    assert sm.total_steps == 300
    assert sm.state_noise_sigma == R.E4_STATE_NOISE_SIGMA["A14"]
    assert sm.lambda_sandbox_valid == R.E4_LAMBDA_VALID["A14"]


def test_sandbox_sees_pre_noise_state():
    # Registered interaction rule: z0 is the task forward's PRE-noise encoder
    # state. A noisy forward's states[0] must equal code(x) bit-exactly, and
    # the sandbox chains themselves carry no channel noise (step_once has no
    # noise path at all).
    cfg = config_for_arm("A14", 0, smoke=True)
    torch.manual_seed(0)
    model = AtomModel(cfg)
    model.eval()
    g = np.random.default_rng(0)
    x = torch.from_numpy(g.integers(0, 10, size=(16, 6)).astype("int64"))
    toks = torch.from_numpy(g.integers(0, 8, size=(16, 2)).astype("int64"))
    ntok = torch.full((16,), 2, dtype=torch.int64)
    with torch.no_grad():
        out = model(x, toks, ntok, mode="hard",
                    noise_sigma=cfg.state_noise_sigma,
                    noise_generator=torch.Generator().manual_seed(1))
        assert torch.equal(out["states"][0], model.code(x))
    assert "states_transmitted" in out   # the treatment was actually on


def test_combined_training_step_runs_both_pressures():
    # One optimizer step with noise AND sandbox: both losses finite, atom
    # MLPs receive gradient, and the sandbox stream consumed is numpy-only
    # (the torch noise generator advances only via the forward's handoffs).
    cfg = config_for_arm("A15", 0, smoke=True)
    torch.manual_seed(0)
    model = AtomModel(cfg)
    sb = SandboxState(cfg)
    g = np.random.default_rng(1)
    x = torch.from_numpy(g.integers(0, 10, size=(32, 6)).astype("int64"))
    y = torch.from_numpy(g.integers(0, 10, size=(32, 6)).astype("int64"))
    toks = torch.from_numpy(g.integers(0, 8, size=(32, 2)).astype("int64"))
    ntok = torch.full((32,), 2, dtype=torch.int64)
    out = model(x, toks, ntok, mode="gumbel", tau=1.0,
                generator=torch.Generator().manual_seed(2),
                noise_sigma=cfg.state_noise_sigma,
                noise_generator=torch.Generator().manual_seed(3))
    loss_task = torch.nn.functional.cross_entropy(
        out["logits"].reshape(-1, cfg.vocab), y.reshape(-1))
    sb.update_usage(out["choices"].detach())
    terms = sandbox_losses(model, out["states"][0].detach(), sb)
    loss = (loss_task
            + cfg.lambda_sandbox_valid * terms["loss_sandbox_valid"]
            + cfg.lambda_sandbox_unique * terms["loss_sandbox_unique"])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.atoms.w1.grad is not None
    assert float(model.atoms.w1.grad.abs().sum()) > 0
    assert model.encoder.digit_emb.weight.grad is not None  # task path alive


def test_dax_threshold_registered():
    assert R.E4_DAX_CRACK_THRESHOLD == 0.01
    assert R.E4_TOTAL_STEPS == 30_000 and R.E4_PANEL_STEPS == (20_000,)
