"""Model invariants: param budget, memoryless composer, exact pass, routing."""
import dataclasses

import numpy as np
import pytest
import torch

from atomv2 import registered as R
from atomv2.config import config_for_arm
from atomv2.model import N_STEPS, AtomModel, param_counts
from atomv2.oracle import forced_schedule


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return AtomModel(config_for_arm("A1", seed=0))


def _batch(n=8, n_tokens=2):
    g = np.random.default_rng(7)
    digits = torch.from_numpy(g.integers(0, 10, size=(n, 6), dtype=np.int64))
    tokens = torch.from_numpy(g.integers(0, 8, size=(n, 2), dtype=np.int64))
    ntok = torch.full((n,), n_tokens, dtype=torch.int64)
    if n_tokens == 1:
        tokens[:, 1] = R.N_SURFACE  # PAD
    return digits, tokens, ntok


def test_param_budget(model):
    pc = param_counts(model)
    assert pc["atoms_each"] == 148_032          # 384->192->384 with biases
    assert pc["atoms_total"] == 16 * 148_032
    assert 25_000 < pc["composer"] < 32_000     # "~28k, logged separately"
    assert 40_000 < pc["encoder"] < 60_000
    assert 40_000 < pc["decoder"] < 60_000
    assert pc["state_norm"] == 0                # non-affine: idempotent norm
    assert 2_300_000 < pc["total"] < 2_700_000  # ~2.6M, CPU-sized


def test_composer_size_independent_of_atom_count():
    cfg16 = config_for_arm("A1", seed=0)
    cfg32 = dataclasses.replace(cfg16, n_atoms=32)
    c16 = param_counts(AtomModel(cfg16))["composer"]
    c32 = param_counts(AtomModel(cfg32))["composer"]
    assert c16 == c32


def test_pass_is_exact_noop_in_forced_mode(model):
    digits, tokens, ntok = _batch(n_tokens=2)
    all_pass = np.full((len(digits), N_STEPS), R.PASS_INDEX, dtype=np.int64)
    out = model(digits, tokens, ntok, mode="forced",
                forced=torch.from_numpy(all_pass))
    h0 = out["states"][0]
    for s in out["states"][1:]:
        assert torch.equal(s, h0)  # bitwise, not approx
    # and the decode of the final state equals the decode of h0
    assert torch.equal(out["logits"], model.decoder(h0))


def test_ablated_pick_is_exactly_pass(model):
    digits, tokens, ntok = _batch(n_tokens=2)
    sched = np.full((len(digits), N_STEPS), R.PASS_INDEX, dtype=np.int64)
    sched[:, 0] = 3  # force atom 3 at step 0
    ablate = torch.zeros(R.N_ATOMS, dtype=torch.bool)
    ablate[3] = True
    out_abl = model(digits, tokens, ntok, mode="forced",
                    forced=torch.from_numpy(sched), ablate=ablate)
    all_pass = np.full((len(digits), N_STEPS), R.PASS_INDEX, dtype=np.int64)
    out_pass = model(digits, tokens, ntok, mode="forced",
                     forced=torch.from_numpy(all_pass))
    assert torch.equal(out_abl["logits"], out_pass["logits"])


def test_forced_mode_routes_exactly_as_scheduled(model):
    digits, tokens, ntok = _batch(n_tokens=2)
    g = np.random.default_rng(3)
    sched = g.integers(0, R.N_ATOMS + 1, size=(len(digits), N_STEPS)).astype(np.int64)
    out = model(digits, tokens, ntok, mode="forced", forced=torch.from_numpy(sched))
    assert np.array_equal(out["choices"].numpy(), sched)


def test_singleton_dead_steps(model):
    digits, tokens, ntok = _batch(n_tokens=1)
    out = model(digits, tokens, ntok, mode="hard")
    choices = out["choices"].numpy()
    assert (choices[:, R.MICRO_STEPS:] == -1).all()
    for k in range(R.MICRO_STEPS, N_STEPS):
        assert torch.equal(out["states"][k + 1], out["states"][R.MICRO_STEPS])
    assert (out["soft_atom_mass"][:, R.MICRO_STEPS:] == 0).all()


def test_composer_is_memoryless(model):
    # The query is a pure function of (state, active token, micro-step): same
    # inputs, same logits, no matter what was routed before.
    digits, tokens, ntok = _batch(n_tokens=2)
    state = model.code(digits)
    q1 = model.composer.query(state, tokens[:, 0], 2)
    _ = model(digits, tokens, ntok, mode="hard")   # run history through model
    q2 = model.composer.query(state, tokens[:, 0], 2)
    assert torch.equal(q1, q2)


def test_content_code_is_task_independent(model):
    digits, tokens, ntok = _batch(n_tokens=2)
    other = tokens.roll(1, dims=0)
    all_pass = np.full((len(digits), N_STEPS), R.PASS_INDEX, dtype=np.int64)
    out_a = model(digits, tokens, ntok, mode="forced",
                  forced=torch.from_numpy(all_pass))
    out_b = model(digits, other, ntok, mode="forced",
                  forced=torch.from_numpy(all_pass))
    assert torch.equal(out_a["states"][0], out_b["states"][0])
    assert torch.equal(out_a["logits"], out_b["logits"])


def test_router_control_excludes_partner_and_absolute_position(model):
    digits, _, ntok = _batch(n=6, n_tokens=2)
    all_pass = torch.full((len(digits), N_STEPS), R.PASS_INDEX, dtype=torch.int64)

    def routed(a, b):
        tokens = torch.tensor([[a, b]] * len(digits), dtype=torch.int64)
        return model(digits, tokens, ntok, mode="forced",
                     forced=all_pass)["route_logits"]

    ab = routed(1, 2)
    ac = routed(1, 4)
    db = routed(3, 2)
    # First-token program cannot see its partner.
    assert torch.equal(ab[:, :R.MICRO_STEPS], ac[:, :R.MICRO_STEPS])
    # Second-token program cannot see its partner.
    assert torch.equal(ab[:, R.MICRO_STEPS:], db[:, R.MICRO_STEPS:])
    # The same active token and micro-steps are identical in either position.
    x1 = routed(1, 5)
    x2 = routed(5, 1)
    assert torch.equal(x1[:, :R.MICRO_STEPS], x2[:, R.MICRO_STEPS:])


def test_atom_mask_removes_atom_from_routing(model):
    digits, tokens, ntok = _batch(n=32, n_tokens=2)
    out = model(digits, tokens, ntok, mode="hard")
    picked = [int(c) for c in out["choices"].unique() if 0 <= c < R.N_ATOMS]
    if not picked:
        pytest.skip("untrained router picked only pass")
    mask = torch.zeros(R.N_ATOMS, dtype=torch.bool)
    mask[picked[0]] = True
    out_m = model(digits, tokens, ntok, mode="hard", atom_mask=mask)
    assert picked[0] not in out_m["choices"].unique().tolist()


def test_rent_mass_excludes_pass(model):
    digits, tokens, ntok = _batch(n_tokens=2)
    g = torch.Generator().manual_seed(0)
    out = model(digits, tokens, ntok, mode="gumbel", tau=1.0, generator=g)
    live_mass = out["soft_atom_mass"][out["live"]]
    assert (live_mass >= 0).all() and (live_mass <= 1).all()
    # forced all-pass: soft mass is the one-hot's atom part = 0
    all_pass = np.full((len(digits), N_STEPS), R.PASS_INDEX, dtype=np.int64)
    out_p = model(digits, tokens, ntok, mode="forced",
                  forced=torch.from_numpy(all_pass))
    assert (out_p["soft_atom_mass"] == 0).all()


def test_gumbel_determinism(model):
    digits, tokens, ntok = _batch(n_tokens=2)
    o1 = model(digits, tokens, ntok, mode="gumbel", tau=1.0,
               generator=torch.Generator().manual_seed(11))
    o2 = model(digits, tokens, ntok, mode="gumbel", tau=1.0,
               generator=torch.Generator().manual_seed(11))
    assert torch.equal(o1["choices"], o2["choices"])
    assert torch.equal(o1["logits"], o2["logits"])


def test_straight_through_gradient_flows_to_keys_and_composer(model):
    digits, tokens, ntok = _batch(n_tokens=2)
    out = model(digits, tokens, ntok, mode="gumbel", tau=1.0,
                generator=torch.Generator().manual_seed(5))
    loss = out["logits"].sum() + out["soft_atom_mass"].sum()
    model.zero_grad()
    loss.backward()
    assert model.atoms.keys.grad is not None
    assert model.atoms.keys.grad.abs().sum() > 0
    comp_grads = [p.grad.abs().sum() for p in model.composer.parameters()
                  if p.grad is not None]
    assert sum(comp_grads) > 0


def test_oracle_forced_schedule_shape():
    tokens = np.array([[2, 5], [7, R.N_SURFACE]], dtype=np.int64)
    ntok = np.array([2, 1], dtype=np.int64)
    sched = forced_schedule(tokens, ntok)
    assert sched[0].tolist() == [2, 16, 16, 5, 16, 16]
    assert sched[1].tolist() == [7, 16, 16, 16, 16, 16]
