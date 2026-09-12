"""E11 exact exchange, canonical interface, and verdict tests."""
from __future__ import annotations

import torch

from atomv2 import registered as R
from atomv2.data import build_bundle
from atomv2.e11 import (build_curriculum, build_extrapolation, config_for_e11,
                        flow_audit, quantized_copy)
from atomv2.exchange_model import ConservativeExchangeModel, MATCHINGS
from atomv2.run_e11 import primary_score, replication_verdict, screen_verdict


def test_e11_registration_and_smoke_do_not_mutate_full_constants():
    for arm in R.E11_ARMS:
        full = config_for_e11(arm, 1)
        smoke = config_for_e11(arm, 0, smoke=True)
        assert full.phase1_steps == 4000
        assert full.phase2_steps == 4000
        assert full.transport == R.E11_TRANSPORT[arm]
        assert smoke.phase1_steps == 150
        assert smoke.phase2_steps == 150
        assert smoke.smoke
    assert R.E11_BASE_GATES["singleton_min"] == 0.99
    assert R.E11_EXTRAP_LENGTHS == (3, 4, 8, 16)


def test_registered_matching_schedule_reaches_every_permutation():
    states = {tuple(range(R.SEQ_LEN))}
    for matching in MATCHINGS:
        successors = set()
        for state in states:
            for mask in range(1 << len(matching)):
                nxt = list(state)
                for edge, (i, j) in enumerate(matching):
                    if (mask >> edge) & 1:
                        nxt[i], nxt[j] = nxt[j], nxt[i]
                successors.add(tuple(nxt))
        states = successors
    assert len(states) == 720


def test_every_exchange_stage_conserves_all_channels():
    for arm in ("A32", "A33", "A34"):
        model = ConservativeExchangeModel(arm)
        state = torch.rand(8, R.SEQ_LEN, R.VOCAB)
        tokens = torch.arange(R.N_SURFACE)
        _, _, stages = model.transport(state, tokens)
        before = state.sum(dim=1)
        for stage in stages:
            assert torch.allclose(stage.sum(dim=1), before,
                                  atol=2e-6, rtol=2e-6)
        audit = flow_audit(model, 9)
        assert audit["mass_conservation_relative_max"] \
            <= R.E11_CONSERVATION_MAX


def test_reynolds_profiles_are_centered_signed_and_gradients_live():
    model = ConservativeExchangeModel("A33")
    tokens = torch.arange(R.N_SURFACE)
    a, b = model.pulse_profiles(tokens)
    assert float(a.detach().mean(-1).abs().max()) < 1e-7
    assert float(b.detach().mean(-1).abs().max()) < 1e-7
    assert torch.equal(torch.stack([a, -a], 2).mean(2), torch.zeros_like(a))
    digits = torch.randint(0, R.VOCAB, (16, R.SEQ_LEN))
    sequence = torch.randint(0, R.N_SURFACE, (16, 2))
    model(digits, sequence)["probs"].square().mean().backward()
    assert model.pulse_a.grad is not None
    assert float(model.pulse_a.grad.abs().sum()) > 0
    assert float(model.pulse_b.grad.abs().sum()) > 0


def test_state_is_categorical_and_internal_execution_equals_restart():
    model = ConservativeExchangeModel("A33")
    model.eval()
    digits = torch.randint(0, R.VOCAB, (32, R.SEQ_LEN))
    tokens = torch.randint(0, R.N_SURFACE, (32, 2))
    ordinary = model(digits, tokens)["probs"]
    produced = model(digits, tokens[:, :1])["probs"].argmax(-1)
    restarted = model(produced, tokens[:, 1:])["probs"]
    assert torch.equal(ordinary, restarted)
    assert torch.allclose(ordinary.sum(-1), torch.ones_like(ordinary[..., 0]))
    boundary = model(digits, tokens)["boundaries"][1]
    assert torch.equal(boundary.sum(-1), torch.ones_like(boundary[..., 0]))
    assert torch.all((boundary == 0) | (boundary == 1))


def test_curriculum_and_extrapolation_are_arm_paired_and_exactly_balanced():
    ca = config_for_e11("A32", 1, smoke=True)
    cb = config_for_e11("A33", 1, smoke=True)
    ba, bb = build_bundle(ca), build_bundle(cb)
    aa, ma = build_curriculum(ba, ca)
    ab, mb = build_curriculum(bb, cb)
    assert ma == mb
    assert aa["phase2_single_indices"].shape[1] == ca.batch_size // 2
    assert aa["phase2_pair_indices"].shape[1] == ca.batch_size // 2
    pa, pma = build_extrapolation(ca)
    pb, pmb = build_extrapolation(cb)
    assert pma == pmb
    assert [x["task_id"] for x in pa[16]] == [x["task_id"] for x in pb[16]]


def test_weight_quantization_is_copy_only():
    model = ConservativeExchangeModel("A33")
    before = {name: value.detach().clone()
              for name, value in model.named_parameters()}
    qmodel = quantized_copy(model, 4)
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])
    assert any(not torch.equal(value, dict(qmodel.named_parameters())[name])
               for name, value in model.named_parameters())


def _metric(arm: str, good: bool = True, robust: float = 0.5) -> dict:
    value = 0.99 if good else 0.1
    return {
        "arm": arm, "seed": 1, "phase1_pass": good, "phase2_ran": good,
        "acc_singleton_min_hard": value,
        "acc_seen_pair_hard": value,
        "acc_unseen_L1_hard": value,
        "acc_unseen_L2_hard": value,
        "acc_unseen_L3_hard": value,
        **{f"acc_extrap_len{length}_hard": value
           for length in R.E11_EXTRAP_LENGTHS},
        "mass_conservation_relative_max": 1e-8,
        "interface_prediction_agreement": 1.0,
        "robustness_composite": robust,
    }


def test_screen_verdict_is_conditional_and_mechanical():
    rows = {"A32": _metric("A32"), "A33": _metric("A33")}
    pending = screen_verdict(rows)
    assert pending["outcome"] == "VISCOSITY_REQUIRED"
    assert pending["viscosity_eligible"]
    rows["A34"] = _metric("A34", robust=0.54)
    assert screen_verdict(rows)["outcome"] == "VISCOUS_CANONICAL_REYNOLDS"
    rows["A34"]["robustness_composite"] = 0.51
    assert screen_verdict(rows)["outcome"] == "INVISCID_CANONICAL_REYNOLDS"
    rows["A33"] = _metric("A33", False)
    assert screen_verdict(rows)["outcome"] == "DIRECT_CANONICAL_EXCHANGE_ONLY"
    rows["A32"] = _metric("A32", False)
    assert screen_verdict(rows)["outcome"] \
        == "SINGLETON_CRYSTALLIZATION_FAILED"


def test_primary_and_replication_require_all_joint_gates():
    assert primary_score(_metric("A33"))["passes"]
    rows = [_metric("A33") for _ in range(3)]
    for seed, row in enumerate(rows):
        row["seed"] = seed
    assert replication_verdict("A33", rows)["claim_holds"]
    rows[0]["interface_prediction_agreement"] = 0.99
    assert not replication_verdict("A33", rows)["claim_holds"]
