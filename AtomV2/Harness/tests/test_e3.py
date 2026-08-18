"""E3 registrations: sandbox mechanics, gradient boundaries, streams
(H1-Experiment3.md structural checklist in unit form)."""
import numpy as np
import torch

from atomv2 import registered as R
from atomv2 import sandbox as sb_mod
from atomv2.config import config_for_arm
from atomv2.model import AtomModel
from atomv2.sandbox import (SandboxState, apply_chain, fingerprint,
                            sandbox_grad_boundary, sandbox_losses,
                            tv_distance, validity_terms)


def _model(arm="A12"):
    cfg = config_for_arm(arm, 0, smoke=True)
    torch.manual_seed(0)
    m = AtomModel(cfg)
    m.eval()
    return m, cfg


def _z0(model, n=16, seed=0):
    g = np.random.default_rng(seed)
    x = torch.from_numpy(g.integers(0, 10, size=(n, 6)).astype("int64"))
    with torch.no_grad():
        return model.code(x)


def _draws(**over):
    base = {
        "standalone_atom": 0,
        "chain_len": 3,
        "chain_atoms": [1, 2, 1, 4, 5, 6],
        "closure_atom": 7,
        "unique_atoms": [0, 3, 7, 11],
    }
    base.update(over)
    return base


def test_registered_values():
    assert R.E3_ARMS == ("A11", "A12", "A13")
    assert R.E3_BASE_ARM == "A6"
    for arm in R.E3_ARMS:
        assert R.E3_LAMBDA_VALID[arm] > 0
        assert R.E3_LAMBDA_UNIQUE[arm] > 0
    assert 0.0 < R.E3_UNIQUE_MARGIN <= 1.0      # TV lives in [0,1]
    assert 0.0 < R.E3_USAGE_EMA_DECAY < 1.0
    assert R.E3_USAGE_EMA_INIT > R.E3_USAGE_WEIGHT_EPS  # warm start: w_i = 1
    assert R.E3_CHAIN_MAX == 6                  # max atom apps on a pair task
    assert 2 <= R.E3_UNIQUE_ATOMS_PER_STEP <= R.N_ATOMS


def test_e3_config_is_a6_plus_sandbox_only():
    for arm in R.E3_ARMS:
        e3 = config_for_arm(arm, 1).to_dict()
        a6 = config_for_arm("A6", 1).to_dict()
        diff = {k for k in e3 if e3[k] != a6[k]}
        assert diff == {"arm", "experiment", "protocol_revision",
                        "lambda_sandbox_valid", "lambda_sandbox_unique"}, arm
        assert e3["lambda_sandbox_valid"] == R.E3_LAMBDA_VALID[arm]
        assert e3["lambda_sandbox_unique"] == R.E3_LAMBDA_UNIQUE[arm]
        assert e3["lambda_use"] == 0.0
        assert e3["state_noise_sigma"] == 0.0


def test_streams_registered():
    from atomv2.utils import _STREAMS
    assert _STREAMS["e3_sandbox"] == 11
    assert _STREAMS["e3_sandbox_eval"] == 12


def test_gradient_boundary():
    # Sandbox gradients reach the atom MLPs and NOTHING else - not the
    # encoder, decoder, composer, routing keys, or pass key - and the
    # boundary restores requires_grad afterwards.
    model, cfg = _model()
    sb = SandboxState(cfg)
    z0 = _z0(model)
    terms = sandbox_losses(model, z0, sb, draws=_draws())
    total = terms["loss_sandbox_valid"] + terms["loss_sandbox_unique"]
    total.backward()
    for p in (model.atoms.w1, model.atoms.b1, model.atoms.w2, model.atoms.b2):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0
    frozen = (list(model.encoder.parameters())
              + list(model.decoder.parameters())
              + list(model.composer.parameters())
              + [model.atoms.keys, model.atoms.pass_key])
    for p in frozen:
        assert p.grad is None
        assert p.requires_grad          # boundary restored


def test_boundary_is_exception_safe():
    model, _cfg = _model()
    try:
        with sandbox_grad_boundary(model):
            assert not model.atoms.pass_key.requires_grad
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert model.atoms.pass_key.requires_grad


def test_same_atom_update_as_execution():
    # apply_chain is sequential model.step_once - bit-identical to the
    # normal hard application Apply_i(z) = LN(z + A_i(z)); no separate
    # sandbox atom implementation.
    model, _cfg = _model()
    z0 = _z0(model)
    ids = [3, 3, 9, 0]                  # self-predecessors allowed
    z = z0
    for a in ids:
        z = model.step_once(z, a)
    assert torch.equal(apply_chain(model, z0, ids), z)


def test_closure_predecessors_carry_no_grad():
    # Only the closure TARGET atom learns; chain atoms cannot adapt
    # themselves to help it.
    model, cfg = _model()
    sb = SandboxState(cfg)
    z0 = _z0(model)
    draws = _draws(chain_len=6, chain_atoms=[3, 3, 3, 3, 3, 3],
                   closure_atom=5)
    terms = sandbox_losses(model, z0, sb, draws=draws)
    terms["loss_sandbox_closure"].backward()
    assert float(model.atoms.w1.grad[5].abs().sum()) > 0
    assert float(model.atoms.w1.grad[3].abs().sum()) == 0
    assert float(model.atoms.w2.grad[3].abs().sum()) == 0


def test_standalone_needs_no_predecessor():
    # The standalone branch trains its atom directly on the clean encoder
    # state: gradient lands on exactly that atom.
    model, cfg = _model()
    sb = SandboxState(cfg)
    z0 = _z0(model)
    terms = sandbox_losses(model, z0, sb,
                           draws=_draws(standalone_atom=13))
    terms["loss_sandbox_standalone"].backward()
    assert float(model.atoms.w1.grad[13].abs().sum()) > 0
    grads = model.atoms.w1.grad.abs().sum(dim=(1, 2))
    assert float(grads.sum()) == float(grads[13])


def test_nonidentity_active_for_identity_atom():
    # An atom behaving as identity has fingerprint == pass fingerprint, so
    # the nonidentity hinge is active: self-sufficiency cannot be satisfied
    # by learning identity.
    model, cfg = _model()
    with torch.no_grad():               # make atom 2 an exact no-op
        model.atoms.w1[2].zero_()
        model.atoms.b1[2].zero_()
        model.atoms.w2[2].zero_()
        model.atoms.b2[2].zero_()
    sb = SandboxState(cfg)
    sb.usage_ema.fill_(1.0)             # full uniqueness weight everywhere
    z0 = _z0(model)
    with torch.no_grad():
        z2 = model.step_once(z0, 2)
        d = tv_distance(fingerprint(model, z2), fingerprint(model, z0))
    assert float(d) < 1e-6
    terms = sandbox_losses(model, z0, sb,
                           draws=_draws(unique_atoms=[2, 5, 9, 14]))
    assert float(terms["loss_sandbox_nonidentity"]) > 0


def test_unused_atoms_feel_no_uniqueness_pressure():
    # w_i = 0 for every sampled atom -> the uniqueness term is exactly zero
    # and contributes no gradient: unused slots may remain unused.
    model, cfg = _model()
    sb = SandboxState(cfg)
    sb.usage_ema.zero_()
    z0 = _z0(model)
    terms = sandbox_losses(model, z0, sb, draws=_draws())
    assert float(terms["loss_sandbox_unique"]) == 0.0
    terms["loss_sandbox_unique"].backward()
    assert float(model.atoms.w1.grad.abs().sum()) == 0


def test_usage_ema_and_weights():
    cfg = config_for_arm("A12", 0, smoke=True)
    sb = SandboxState(cfg)
    w0 = sb.usage_weights()
    assert torch.all(w0 == 1.0)         # warm start: 1/16 > CENSUS_EPS
    # Only atom 2 is ever picked (plus pass and dead steps, both excluded
    # from the denominator): its weight stays 1, the rest decay to 0.
    choices = torch.full((8, 6), 2, dtype=torch.int64)
    choices[:, 3] = R.PASS_INDEX
    choices[:, 5] = -1
    for _ in range(1500):
        sb.update_usage(choices)
    w = sb.usage_weights()
    assert float(w[2]) == 1.0
    others = w[torch.arange(R.N_ATOMS) != 2]
    assert float(others.max()) < 1e-4   # decayed out (never exactly zero)
    # An all-pass batch must not touch the EMA.
    before = sb.usage_ema.clone()
    sb.update_usage(torch.full((8, 6), R.PASS_INDEX, dtype=torch.int64))
    assert torch.equal(sb.usage_ema, before)


def test_no_torch_rng_consumed_and_draws_deterministic():
    # Sandbox sampling is numpy-only on the dedicated stream: the torch RNG
    # state is untouched (routing/noise/init draws stay bit-identical to the
    # A6 pair) and two states with the same seed draw identically, with
    # FIXED per-step consumption.
    model, cfg = _model()
    z0 = _z0(model)
    before = torch.random.get_rng_state().clone()
    terms = sandbox_losses(model, z0, SandboxState(cfg))
    assert torch.equal(torch.random.get_rng_state(), before)
    a, b = SandboxState(cfg), SandboxState(cfg)
    for _ in range(5):
        da, db = a.draw(), b.draw()
        assert da == db
        assert 1 <= da["chain_len"] <= R.E3_CHAIN_MAX
        assert len(da["chain_atoms"]) == R.E3_CHAIN_MAX
        assert all(0 <= i < R.N_ATOMS for i in da["chain_atoms"])
        assert len(set(da["unique_atoms"])) == R.E3_UNIQUE_ATOMS_PER_STEP
    assert all(float(v) >= 0 for k, v in terms.items()
               if k.startswith("loss_"))


def test_validity_terms_shapes_and_finiteness():
    model, _cfg = _model()
    z0 = _z0(model)
    with torch.no_grad():
        read, cycle = validity_terms(model, model.step_once(z0, 4))
    assert read.ndim == 0 and cycle.ndim == 0
    assert torch.isfinite(read) and torch.isfinite(cycle)
    # TV distance is bounded by 1 and zero against itself.
    with torch.no_grad():
        f = fingerprint(model, z0)
        assert float(tv_distance(f, f)) == 0.0
        g = fingerprint(model, model.step_once(z0, 1))
        assert 0.0 <= float(tv_distance(f, g)) <= 1.0


def test_measure_is_read_only():
    # Telemetry never mutates parameters or the model's training flag and
    # returns the full distance matrix.
    model, cfg = _model()
    model.train()
    sb = SandboxState(cfg)
    diag = {"x": np.random.default_rng(0).integers(
        0, 10, size=(8, 6)).astype("int64")}
    params_before = [p.detach().clone() for p in model.parameters()]
    st = sb_mod.measure(model, cfg, diag, sb, step=100)
    assert model.training
    for p, b in zip(model.parameters(), params_before):
        assert torch.equal(p.detach(), b)
    m = np.asarray(st["uniqueness"]["pair_dist_matrix"])
    assert m.shape == (R.N_ATOMS, R.N_ATOMS)
    assert np.allclose(m, m.T)
    assert st["usage"]["n_weighted"] == R.N_ATOMS  # warm-start EMA
