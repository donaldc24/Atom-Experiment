"""E10 registration, Reynolds transport, and mechanical verdict tests."""
from __future__ import annotations

import torch

from atomv2 import registered as R
from atomv2.e10 import build_extrapolation, config_for_e10, flow_audit
from atomv2.flow_model import ReynoldsFlowModel
from atomv2.run_e10 import base_score, replication_verdict, screen_verdict


def test_e10_registration_and_smoke_do_not_mutate_full_constants():
    for arm in R.E10_ARMS:
        full = config_for_e10(arm, 1)
        smoke = config_for_e10(arm, 0, smoke=True)
        assert full.protocol_revision == R.E10_PROTOCOL_REVISION
        assert full.total_steps == 6000
        assert full.transport == R.E10_TRANSPORT[arm]
        assert full.viscosity == R.E10_VISCOSITY[arm]
        assert smoke.total_steps == 300
        assert smoke.smoke
    assert R.E10_BASE_GATES == {
        "seen": 0.90, "L1": 0.75, "L2": 0.60,
        "L3": 0.60, "triple": 0.60, "quad": 0.50}


def test_direct_and_reynolds_transport_parameter_counts_match():
    direct = ReynoldsFlowModel("A29")
    reynolds = ReynoldsFlowModel("A30")
    viscous = ReynoldsFlowModel("A31")
    assert direct.transport_parameter_count == 8 * 6 * 6
    assert reynolds.transport_parameter_count == 2 * 8 * 3 * 6
    assert viscous.transport_parameter_count == reynolds.transport_parameter_count
    assert sum(p.numel() for p in direct.parameters()) \
        == sum(p.numel() for p in reynolds.parameters())


def test_local_transport_is_exact_identity():
    model = ReynoldsFlowModel("A28")
    p = model.transport_matrix(torch.tensor([0, 7]))
    eye = torch.eye(6).expand(2, -1, -1)
    assert torch.equal(p, eye)


def test_reynolds_pulses_are_centered_and_transport_is_incompressible():
    model = ReynoldsFlowModel("A30")
    tokens = torch.arange(8)
    a, b = model.pulse_profiles(tokens)
    assert float(a.detach().mean(-1).abs().max()) < 1e-7
    assert float(b.detach().mean(-1).abs().max()) < 1e-7
    assert torch.equal(torch.stack([a, -a], 2).mean(2), torch.zeros_like(a))
    p = model.transport_matrix(tokens)
    assert float((p.detach().sum(-1) - 1).abs().max()) < 1e-6
    assert float((p.detach().sum(-2) - 1).abs().max()) < 1e-5
    z = torch.randn(8, 6, 11)
    moved = torch.einsum("bij,bjd->bid", p, z)
    assert torch.allclose(moved.sum(1), z.sum(1), atol=2e-5, rtol=2e-5)


def test_direct_control_only_guarantees_row_stochasticity():
    model = ReynoldsFlowModel("A29")
    with torch.no_grad():
        model.transport_logits.copy_(torch.randn_like(model.transport_logits))
    p = model.transport_matrix(torch.arange(8))
    assert float((p.detach().sum(-1) - 1).abs().max()) < 1e-6
    assert float((p.detach().sum(-2) - 1).abs().max()) > 1e-3


def test_viscosity_damps_registered_high_frequency_mode():
    inviscid = ReynoldsFlowModel("A30")
    viscous = ReynoldsFlowModel("A31")
    ai = flow_audit(inviscid, 3)
    av = flow_audit(viscous, 3)
    assert ai["viscous_high_frequency_ratio"] == 1.0
    assert av["viscous_high_frequency_ratio"] < 1.0


def test_model_accepts_arbitrary_token_horizon_and_pulse_gradients_live():
    model = ReynoldsFlowModel("A30")
    digits = torch.randint(0, 10, (5, 6))
    tokens = torch.randint(0, 8, (5, 4))
    out = model(digits, tokens)
    assert out["logits"].shape == (5, 6, 10)
    assert len(out["states"]) == 5
    assert len(out["transport_matrices"]) == 4
    out["logits"].square().mean().backward()
    assert model.pulse_a.grad is not None
    assert float(model.pulse_a.grad.abs().sum()) > 0
    assert float(model.pulse_b.grad.abs().sum()) > 0


def test_extrapolation_selection_and_data_are_arm_paired():
    a, ma = build_extrapolation(config_for_e10("A29", 1, smoke=True))
    b, mb = build_extrapolation(config_for_e10("A31", 1, smoke=True))
    assert ma == mb
    assert [x["task_id"] for x in a[3]] == [x["task_id"] for x in b[3]]
    assert [x["task_id"] for x in a[4]] == [x["task_id"] for x in b[4]]


def _metric(arm: str, good: bool = True, noisy: float = 0.70,
            mass: float = 1e-7) -> dict:
    value = 0.95 if good else 0.10
    return {
        "arm": arm, "seed": 1,
        "acc_seen_hard": value,
        "acc_unseen_L1_hard": value,
        "acc_unseen_L2_hard": value,
        "acc_unseen_L3_hard": value,
        "acc_extrap_triple_hard": value,
        "acc_extrap_quad_hard": value,
        "acc_pair_noisy_hard": noisy,
        "mass_conservation_relative_max": mass,
    }


def test_base_score_is_joint_and_outcome_order_is_mechanical():
    assert base_score(_metric("A30"))["passes"]
    bad = _metric("A30")
    bad["acc_extrap_quad_hard"] = 0.49
    assert not base_score(bad)["passes"]

    rows = {"A28": _metric("A28", False),
            "A29": _metric("A29"),
            "A30": _metric("A30", noisy=0.70),
            "A31": _metric("A31", noisy=0.76)}
    assert screen_verdict(rows)["outcome"] == "VISCOUS_REYNOLDS_FLOW"
    rows["A31"]["acc_pair_noisy_hard"] = 0.71
    assert screen_verdict(rows)["outcome"] == "INVISCID_REYNOLDS_FLOW"
    rows["A30"] = _metric("A30", False, noisy=0.70)
    assert screen_verdict(rows)["outcome"] \
        == "STRUCTURED_FLOW_ONLY_WITH_VISCOSITY"
    rows["A31"] = _metric("A31", False)
    assert screen_verdict(rows)["outcome"] == "DIRECT_TRANSPORT_ONLY"
    rows["A29"] = _metric("A29", False)
    assert screen_verdict(rows)["outcome"] == "NO_FLOW_SUCCESS"


def test_replication_requires_two_base_passes_and_all_conservation():
    rows = [_metric("A30") for _ in range(3)]
    for seed, row in enumerate(rows):
        row["seed"] = seed
    assert replication_verdict("A30", rows)["claim_holds"]
    rows[0]["mass_conservation_relative_max"] = 1e-3
    assert not replication_verdict("A30", rows)["claim_holds"]
