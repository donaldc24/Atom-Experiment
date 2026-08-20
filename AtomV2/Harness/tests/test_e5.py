"""E5 registrations (H1-Experiment5.md): producer-branch mechanics, the
non-negotiable gradient path (through frozen chains into the producer atom
ONLY), config wiring, dose transcription, streams, and telemetry."""
import numpy as np
import torch

from atomv2 import producer as pr_mod
from atomv2 import registered as R
from atomv2.config import config_for_arm
from atomv2.model import AtomModel
from atomv2.producer import (ProducerState, frozen_continuation_boundary,
                             producer_losses)
from atomv2.sandbox import apply_chain


def _model(arm="A16"):
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


def _draws(producer_atom=5, branches=None):
    if branches is None:
        branches = [
            {"chain_len": 3, "chain_atoms": [1, 2, 3, 4, 6, 7]},
            {"chain_len": 1, "chain_atoms": [8, 9, 10, 11, 12, 13]},
            {"chain_len": 6, "chain_atoms": [14, 15, 0, 1, 2, 3]},
            {"chain_len": 2, "chain_atoms": [4, 6, 7, 8, 9, 10]},
        ]
    return {"producer_atom": producer_atom, "branches": branches}


def test_registered_values():
    assert R.E5_ARMS == ("A16", "A17")
    assert R.E5_BASE_ARM == "A14"
    assert R.E5_LAMBDA_PRODUCER == {"A16": 0.1, "A17": 0.3}
    assert R.E5_PRODUCER_BRANCHES == 4
    assert R.E5_CHAIN_MAX is R.E3_CHAIN_MAX     # reference, not transcription
    assert R.EXPERIMENT_REVISIONS["e5"] == R.E5_PROTOCOL_REVISION


def test_streams_registered():
    from atomv2.utils import _STREAMS
    assert _STREAMS["e5_producer"] == 13
    assert _STREAMS["e5_producer_eval"] == 14


def test_e5_config_is_a14_plus_producer_only():
    # Base inherited byte-for-byte: the resolved configs differ from A14 in
    # nothing but identity strings and the one new lambda.
    for arm in R.E5_ARMS:
        e5 = config_for_arm(arm, 1).to_dict()
        a14 = config_for_arm("A14", 1).to_dict()
        diff = {k for k in e5 if e5[k] != a14[k]}
        assert diff == {"arm", "experiment", "protocol_revision",
                        "lambda_producer"}, arm
        assert e5["lambda_producer"] == R.E5_LAMBDA_PRODUCER[arm]
        assert e5["total_steps"] == 30_000
        assert tuple(e5["panel_steps"]) == (20_000,)
        assert e5["lambda_use"] == 0.0
        assert e5["state_noise_sigma"] == R.E4_STATE_NOISE_SIGMA["A14"]
        assert e5["lambda_sandbox_valid"] == R.E4_LAMBDA_VALID["A14"]
    # Zero-path condition: the base arm itself carries no producer.
    assert config_for_arm("A14", 0).lambda_producer == 0.0
    # The smoke path keeps the smoke budget (pipeline checks stay cheap).
    sm = config_for_arm("A16", 0, smoke=True)
    assert sm.total_steps == 300
    assert sm.lambda_producer == R.E5_LAMBDA_PRODUCER["A16"]


def test_gradient_path_producer_only():
    # THE non-negotiable: gradient flows THROUGH the frozen chains into the
    # producer atom ONLY. The producer atom gets gradient; chain atoms,
    # decoder, encoder, composer, and keys get exactly none; every
    # requires_grad flag is restored afterwards.
    model, cfg = _model()
    ps = ProducerState(cfg)
    z0 = _z0(model)
    draws = _draws(producer_atom=5)     # no branch chain contains atom 5
    terms = producer_losses(model, z0, ps, draws=draws)
    terms["loss_producer"].backward()
    grads = model.atoms.w1.grad.abs().sum(dim=(1, 2))
    assert float(grads[5]) > 0
    assert float(grads.sum()) == float(grads[5])
    for tensor in (model.atoms.b1.grad, model.atoms.w2.grad,
                   model.atoms.b2.grad):
        per_atom = tensor.abs().reshape(R.N_ATOMS, -1).sum(dim=1)
        assert float(per_atom[5]) > 0
        assert float(per_atom.sum()) == float(per_atom[5])
    frozen = (list(model.encoder.parameters())
              + list(model.decoder.parameters())
              + list(model.composer.parameters())
              + [model.atoms.keys, model.atoms.pass_key])
    for p in frozen:
        assert p.grad is None
        assert p.requires_grad          # boundary restored
    for p in (model.atoms.w1, model.atoms.b1, model.atoms.w2, model.atoms.b2):
        assert p.requires_grad          # boundary restored


def test_gradient_flows_through_the_chain():
    # Every branch has chain_len >= 1, so the ONLY path from L_producer back
    # to the producer atom runs THROUGH frozen downstream applications
    # (frozen parameters, live activations). Gradient must arrive - unlike
    # the closure branch, which stop-grads before the target.
    model, cfg = _model()
    ps = ProducerState(cfg)
    z0 = _z0(model)
    draws = _draws(producer_atom=0, branches=[
        {"chain_len": 6, "chain_atoms": [7, 8, 9, 10, 11, 12]}
        for _ in range(R.E5_PRODUCER_BRANCHES)])
    terms = producer_losses(model, z0, ps, draws=draws)
    assert terms["loss_producer"].requires_grad
    terms["loss_producer"].backward()
    assert float(model.atoms.w1.grad[0].abs().sum()) > 0
    # ... and the chain atoms, though live in the graph, received none.
    for a in (7, 8, 9, 10, 11, 12):
        assert float(model.atoms.w1.grad[a].abs().sum()) == 0


def test_producer_atom_in_chain_gets_no_chain_gradient():
    # A producer atom appearing ONLY inside chains (a different atom emits)
    # receives exactly zero gradient: chain applications are frozen even for
    # slices that are trainable elsewhere in the step's graphs.
    model, cfg = _model()
    ps = ProducerState(cfg)
    z0 = _z0(model)
    draws = _draws(producer_atom=2, branches=[
        {"chain_len": 4, "chain_atoms": [5, 5, 5, 5, 5, 5]}
        for _ in range(R.E5_PRODUCER_BRANCHES)])
    terms = producer_losses(model, z0, ps, draws=draws)
    terms["loss_producer"].backward()
    assert float(model.atoms.w1.grad[2].abs().sum()) > 0
    assert float(model.atoms.w1.grad[5].abs().sum()) == 0


def test_same_atom_update_as_execution():
    # The emission and every chain application go through model.step_once -
    # no separate producer atom implementation. Reconstructing one branch by
    # hand from the draws must give bit-identical READ.
    model, cfg = _model()
    z0 = _z0(model)
    draws = _draws()
    terms = producer_losses(model, z0, ProducerState(cfg), draws=draws)
    from atomv2.sandbox import validity_terms
    with torch.no_grad():
        reads = []
        for br in draws["branches"]:
            z = model.step_once(z0, draws["producer_atom"])
            for a in br["chain_atoms"][:br["chain_len"]]:
                z = model.step_once(z, a)
            r, _c = validity_terms(model, z)
            reads.append(r)
        expected = torch.stack(reads).mean()
    assert torch.equal(terms["loss_producer"].detach(), expected)


def test_boundary_is_exception_safe():
    model, _cfg = _model()
    try:
        with frozen_continuation_boundary(model):
            assert not model.atoms.w1.requires_grad
            assert not model.decoder.head.weight.requires_grad
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert model.atoms.w1.requires_grad
    assert model.decoder.head.weight.requires_grad


def test_no_torch_rng_consumed_and_draws_deterministic():
    # Producer sampling is numpy-only on the dedicated stream: the torch RNG
    # state is untouched, two states with the same seed draw identically,
    # and consumption is FIXED per step (K branches, full-length chains).
    model, cfg = _model()
    z0 = _z0(model)
    before = torch.random.get_rng_state().clone()
    terms = producer_losses(model, z0, ProducerState(cfg))
    assert torch.equal(torch.random.get_rng_state(), before)
    a, b = ProducerState(cfg), ProducerState(cfg)
    for _ in range(5):
        da, db = a.draw(), b.draw()
        assert da == db
        assert 0 <= da["producer_atom"] < R.N_ATOMS
        assert len(da["branches"]) == R.E5_PRODUCER_BRANCHES
        for br in da["branches"]:
            assert 1 <= br["chain_len"] <= R.E5_CHAIN_MAX
            assert len(br["chain_atoms"]) == R.E5_CHAIN_MAX
            assert all(0 <= i < R.N_ATOMS for i in br["chain_atoms"])
    assert float(terms["loss_producer"].detach()) >= 0
    assert float(terms["producer_read_spread"]) >= 0


def test_producer_requires_detached_z0():
    model, cfg = _model()
    g = np.random.default_rng(0)
    x = torch.from_numpy(g.integers(0, 10, size=(8, 6)).astype("int64"))
    z0_live = model.code(x)             # carries the encoder graph
    try:
        producer_losses(model, z0_live, ProducerState(cfg))
        raise SystemExit("producer accepted a non-detached z0")
    except AssertionError:
        pass


def test_measure_is_read_only():
    # Telemetry never mutates parameters or the model's training flag, and
    # carries both registered rows (per-atom output variance with the z0
    # floor; branch-READ spread over K branches per emitted state).
    model, cfg = _model()
    model.train()
    diag = {"x": np.random.default_rng(0).integers(
        0, 10, size=(8, 6)).astype("int64")}
    params_before = [p.detach().clone() for p in model.parameters()]
    pt = pr_mod.measure(model, cfg, diag, step=100)
    assert model.training
    for p, b in zip(model.parameters(), params_before):
        assert torch.equal(p.detach(), b)
    assert len(pt["output_variance"]["per_atom"]) == R.N_ATOMS
    assert pt["output_variance"]["min"] >= 0
    assert pt["output_variance"]["z0_reference"] >= 0
    br = pt["branch_read"]
    assert len(br["spread_per_producer"]) == R.E5_TELEMETRY_PRODUCERS
    assert all(len(r) == R.E5_PRODUCER_BRANCHES
               for r in br["reads_per_producer"])
    assert np.isfinite(br["read_mean"]) and np.isfinite(br["spread_mean"])
    # Indexed eval stream: same step -> identical record, different step ->
    # different draws (the training stream is never touched).
    pt2 = pr_mod.measure(model, cfg, diag, step=100)
    assert pt == pt2


def test_chain_uses_step_once():
    model, _cfg = _model()
    z0 = _z0(model)
    ids = [3, 3, 9, 0]
    z = z0
    for a in ids:
        z = model.step_once(z, a)
    assert torch.equal(apply_chain(model, z0, ids), z)
