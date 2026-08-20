"""E7 registrations (H1-Experiment7.md): the implementation gates in unit
form - one route decision per token, no hidden atom calls, replacement mode
free of residual addition, pass unchanged, consistent narrow-state
dimensions, refactor equivalence at micro_steps = 3, config wiring."""
import numpy as np
import torch

from atomv2 import registered as R
from atomv2.config import config_for_arm
from atomv2.model import MAX_TOKENS, AtomModel


def _model(arm, seed=0):
    cfg = config_for_arm(arm, seed, smoke=True)
    torch.manual_seed(0)
    m = AtomModel(cfg)
    m.eval()
    return m, cfg


def _batch(n=16, seed=0):
    g = np.random.default_rng(seed)
    x = torch.from_numpy(g.integers(0, 10, size=(n, 6)).astype("int64"))
    toks = torch.from_numpy(g.integers(0, 8, size=(n, 2)).astype("int64"))
    return x, toks


def test_registered_values():
    assert R.E7_ARMS == ("A18", "A19", "A20", "A21")
    assert R.E7_BASE_ARM == "A6"
    assert all(R.E7_MICRO_STEPS[a] == 1 for a in R.E7_ARMS)
    assert R.E7_STATE_DIM == {"A18": 384, "A19": 64, "A20": 384, "A21": 64}
    assert R.E7_ATOM_UPDATE == {"A18": "residual", "A19": "residual",
                                "A20": "replacement", "A21": "replacement"}
    # narrow arms keep the registered 2:1 state:hidden ratio
    assert R.E7_ATOM_HIDDEN["A19"] * 2 == R.E7_STATE_DIM["A19"]
    assert R.E7_HEALTH_SEEN_MIN == 0.80
    assert R.E7_ICR_STRONG == 0.80 and R.E7_ICR_SEVERE == 0.30
    assert R.EXPERIMENT_REVISIONS["e7"] == R.E7_PROTOCOL_REVISION


def test_e7_configs_are_a6_plus_registered_deltas_only():
    a6 = config_for_arm("A6", 1).to_dict()
    expected_delta = {
        "A18": {"micro_steps"},
        "A19": {"micro_steps", "state_dim", "atom_hidden"},
        "A20": {"micro_steps", "atom_update"},
        "A21": {"micro_steps", "state_dim", "atom_hidden", "atom_update"},
    }
    for arm in R.E7_ARMS:
        e7 = config_for_arm(arm, 1).to_dict()
        diff = {k for k in e7 if e7[k] != a6[k]}
        assert diff == ({"arm", "experiment", "protocol_revision"}
                        | expected_delta[arm]), arm
        assert e7["lambda_use"] == 0.0
        assert e7["state_noise_sigma"] == 0.0
        assert e7["lambda_sandbox_valid"] == 0.0
        assert e7["lambda_producer"] == 0.0
        assert e7["total_steps"] == 20_000
        assert e7["router"] == "cosine"


def test_one_route_decision_per_token():
    # Gates 2-4: A18 makes exactly one route decision per token; a
    # two-token program executes exactly two, a singleton one (second
    # step dead, choice -1); no hidden atom call exists anywhere.
    model, cfg = _model("A18")
    x, toks = _batch()
    with torch.no_grad():
        o2 = model(x, toks, torch.full((16,), 2, dtype=torch.int64),
                   mode="hard")
        o1 = model(x, toks, torch.full((16,), 1, dtype=torch.int64),
                   mode="hard")
    assert o2["choices"].shape == (16, 2)
    assert len(o2["states"]) == 3                 # z0 + one per token
    assert (o2["choices"] >= 0).all()             # both decisions live
    assert (o1["choices"][:, 0] >= 0).all()
    assert (o1["choices"][:, 1] == -1).all()      # dead step, no call
    # dead step leaves the state bit-identical
    assert torch.equal(o1["states"][1], o1["states"][2])


def test_micro_steps_3_shapes_unchanged():
    # The refactored implementation at micro_steps = 3 keeps the certified
    # A6 execution shapes (the bit-level gate is run_e7's A6 replay).
    model, cfg = _model("A6")
    assert model.n_steps == 6
    assert model.composer.micro_emb.num_embeddings == 3
    x, toks = _batch()
    with torch.no_grad():
        out = model(x, toks, torch.full((16,), 2, dtype=torch.int64),
                    mode="hard")
    assert out["choices"].shape == (16, 6)
    assert len(out["states"]) == 7


def test_replacement_has_no_residual_addition():
    # Gate 7: in replacement mode the selected atom generates the COMPLETE
    # outgoing state: s_new = LN(F_i(s)), constructed independently here.
    model, cfg = _model("A20")
    assert cfg.atom_update == "replacement"
    x, _ = _batch()
    with torch.no_grad():
        z0 = model.code(x)
        out = model.atoms.outputs(z0)[:, 5]
        expected = model._norm(out)               # NO + state term
        assert torch.equal(model.step_once(z0, 5), expected)
        residual = model._norm(z0 + out)
        assert not torch.equal(expected, residual)


def test_pass_is_exact_identity_in_both_modes():
    # Gate 8: pass remains the explicit no-op route, bitwise, residual and
    # replacement alike; ablated picks behave as pass.
    for arm in ("A18", "A20", "A21"):
        model, cfg = _model(arm)
        x, toks = _batch()
        n = len(x)
        ntok = torch.full((n,), 2, dtype=torch.int64)
        sched = torch.full((n, model.n_steps), R.PASS_INDEX,
                           dtype=torch.int64)
        with torch.no_grad():
            out = model(x, toks, ntok, mode="forced", forced=sched)
            z0 = model.code(x)
        assert torch.equal(out["states"][-1], z0), arm
        # ablated atom == pass: force atom 3 everywhere, ablate it
        sched3 = torch.full((n, model.n_steps), 3, dtype=torch.int64)
        abl = torch.zeros(cfg.n_atoms, dtype=torch.bool)
        abl[3] = True
        with torch.no_grad():
            out_a = model(x, toks, ntok, mode="forced", forced=sched3,
                          ablate=abl)
        assert torch.equal(out_a["states"][-1], z0), arm


def test_narrow_state_dimensions_consistent():
    # Gate 6: A19/A21 change state width everywhere consistently; the
    # compress/expand heads exist ONLY on narrow arms (positional arms
    # keep the certified parameter set byte-for-byte).
    for arm, narrow in (("A19", True), ("A21", True),
                        ("A18", False), ("A20", False), ("A6", False)):
        model, cfg = _model(arm)
        assert hasattr(model.encoder, "compress") == narrow, arm
        assert hasattr(model.decoder, "expand") == narrow, arm
        x, toks = _batch()
        with torch.no_grad():
            z0 = model.code(x)
            out = model(x, toks, torch.full((16,), 2, dtype=torch.int64),
                        mode="hard")
        assert z0.shape == (16, cfg.state_dim), arm
        assert out["logits"].shape == (16, 6, 10), arm
        assert model.atoms.w1.shape == (cfg.n_atoms, cfg.state_dim,
                                        cfg.atom_hidden), arm
        # non-affine norm is idempotent up to the LayerNorm eps (~1e-5);
        # EXACT pass no-op is guaranteed by the noop bypass and verified
        # bitwise in test_pass_is_exact_identity_in_both_modes.
        assert torch.allclose(model._norm(z0), z0, atol=1e-4), arm


def test_gumbel_trains_in_all_arms():
    for arm in R.E7_ARMS:
        model, cfg = _model(arm)
        model.train()
        x, toks = _batch()
        out = model(x, toks, torch.full((16,), 2, dtype=torch.int64),
                    mode="gumbel", tau=1.0,
                    generator=torch.Generator().manual_seed(3))
        loss = torch.nn.functional.cross_entropy(
            out["logits"].reshape(-1, cfg.vocab),
            x.reshape(-1))
        assert torch.isfinite(loss), arm
        loss.backward()
        assert float(model.atoms.w1.grad.abs().sum()) > 0, arm
        assert float(model.composer.net[0].weight.grad.abs().sum()) > 0, arm


def test_execute_from_state_matches_forward_at_token_zero():
    # The panel probe path stays consistent under the refactor.
    for arm in ("A18", "A21"):
        model, cfg = _model(arm)
        x, toks = _batch()
        ntok = torch.full((16,), 2, dtype=torch.int64)
        with torch.no_grad():
            a = model(x, toks, ntok, mode="hard")
            b = model.execute_from_state(model.code(x), toks, ntok, 0,
                                         mode="hard")
        assert torch.equal(a["logits"], b["logits"]), arm
        assert torch.equal(a["choices"], b["choices"]), arm


def test_no_training_side_canonicalization():
    # Gate 9: nothing in the E7 config enables any auxiliary pressure; the
    # training loss surface is task CE only (lambdas all zero, oracle off).
    for arm in R.E7_ARMS:
        cfg = config_for_arm(arm, 0)
        assert cfg.lambda_use == 0.0
        assert cfg.lambda_sandbox_valid == 0.0
        assert cfg.lambda_sandbox_unique == 0.0
        assert cfg.lambda_producer == 0.0
        assert cfg.state_noise_sigma == 0.0
        assert not cfg.forced_routing
