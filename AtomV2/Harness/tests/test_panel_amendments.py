"""Amendments C1-C4 plus the hard invariants they must not violate."""
import dataclasses

import numpy as np
import pytest
import torch

from atomv2 import data as data_mod
from atomv2 import ops
from atomv2 import panel as panel_mod
from atomv2 import registered as R
from atomv2.config import config_for_arm
from atomv2.model import N_STEPS, AtomModel


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = AtomModel(config_for_arm("A1", seed=0))
    m.eval()
    return m


def _batch(n=12, n_tokens=2):
    g = np.random.default_rng(7)
    digits = torch.from_numpy(g.integers(0, 10, size=(n, 6), dtype=np.int64))
    tokens = torch.from_numpy(g.integers(0, 8, size=(n, 2), dtype=np.int64))
    ntok = torch.full((n,), n_tokens, dtype=torch.int64)
    if n_tokens == 1:
        tokens[:, 1] = R.N_SURFACE
    return digits, tokens, ntok


# --- the refactor must be behaviour-preserving -------------------------------

@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_execute_from_state_start0_reproduces_forward_bitwise(model, mode):
    d, t, n = _batch()
    with torch.no_grad():
        f = model(d, t, n, mode=mode, tau=0.5)
        e = model.execute_from_state(model.code(d), t, n, 0, mode=mode, tau=0.5)
    assert torch.equal(f["logits"], e["logits"])
    assert torch.equal(f["choices"], e["choices"])
    assert torch.equal(f["route_logits"], e["route_logits"])
    assert torch.equal(f["soft_atom_mass"], e["soft_atom_mass"])
    assert all(torch.equal(a, b) for a, b in zip(f["states"], e["states"]))


def test_substituting_a_state_with_itself_is_an_exact_noop(model):
    """The anchor for C3: if h_canonical == h_pair, repair must be identical.

    This is what makes 'oracle repair_delta ~ 0' a claim about the MODEL
    rather than about the substitution machinery.
    """
    d, t, n = _batch()
    with torch.no_grad():
        base = model(d, t, n, mode="hard", tau=0.5)
        again = model.execute_from_state(base["states"][R.MICRO_STEPS], t, n, 1,
                                         mode="hard", tau=0.5)
    assert torch.equal(base["logits"], again["logits"])
    sl = slice(R.MICRO_STEPS, 2 * R.MICRO_STEPS)
    assert torch.equal(base["choices"][:, sl], again["choices"][:, sl])
    assert torch.equal(base["route_logits"][:, sl], again["route_logits"][:, sl])


def test_execute_from_state_refuses_training_mode(model):
    d, t, n = _batch()
    model.train()
    try:
        with pytest.raises(AssertionError, match="probes read, never write|eval mode"):
            model.execute_from_state(model.code(d), t, n, 1)
    finally:
        model.eval()


def test_execute_from_state_skips_steps_before_the_boundary(model):
    d, t, n = _batch()
    with torch.no_grad():
        e = model.execute_from_state(model.code(d), t, n, 1, mode="hard")
    # steps 0..2 are not executed: placeholder choices, state untouched
    assert (e["choices"][:, : R.MICRO_STEPS] == -1).all()
    assert torch.equal(e["states"][0], e["states"][R.MICRO_STEPS])


# --- C1 ----------------------------------------------------------------------

def test_c1_probe_seed_is_pinned_to_the_legacy_name():
    """The rename must not perturb a number: the seed derives from the OLD key."""
    import inspect
    src = inspect.getsource(panel_mod.decodability)
    assert '"subop_from_delta"' in src and '"subop_from_state"' in src
    assert '"subop_h0_floor"' in src
    assert "leakage_subop_identity_from_delta" in src
    assert "probe_seed" in src


def test_c1_emits_deprecated_aliases(model):
    cfg = config_for_arm("A1", seed=0, smoke=True)
    bundle = data_mod.build_bundle(cfg)
    dec = panel_mod.decodability(model, bundle.seen_heldout[:6], cfg.seed)
    assert "leakage_subop_identity_from_delta" in dec
    assert dec["deprecated_aliases"]["decodability_subop_from_delta"] == \
        "leakage_subop_identity_from_delta"
    assert "leakage_note" in dec


# --- C2 ----------------------------------------------------------------------

def test_c2_carrier_map_is_derived_not_transcribed():
    derived = panel_mod.carrier_map()
    expected = {s: sorted(p for p in ops.SURFACE_NAMES
                          if s in R.SURFACE_RECIPES[p]) for s in ops.SUBOP_NAMES}
    assert derived == expected
    # every sub-op has >= 2 carriers, or a transfer split is impossible
    assert all(len(v) >= 2 for v in derived.values())
    assert derived["A"] == ["P2", "P3", "P7"] and derived["T"] == ["P2", "P4", "P8"]


def test_c2_transfer_split_trains_and_tests_on_different_carriers(model):
    cfg = config_for_arm("A1", seed=0, smoke=True)
    bundle = data_mod.build_bundle(cfg)
    res = panel_mod.transfer_split_subop_probes(model, bundle.seen_heldout,
                                                cfg.seed)
    assert res["chance"] == 0.5
    for sub in ops.SUBOP_NAMES:
        entry = res["per_subop"][sub]
        assert entry["carriers"] == panel_mod.carrier_map()[sub]
        for holdout, v in entry["per_holdout"].items():
            if v is None:
                continue
            # balanced 50/50 on both sides, so 0.5 really is chance
            assert v["train_pos_frac"] == pytest.approx(0.5)
            assert v["test_pos_frac"] == pytest.approx(0.5)
    assert f"{ops.SUBOP_NAMES[0]}_taskscope" in res["per_subop"]


def test_c2_untrained_model_reads_near_chance(model):
    cfg = config_for_arm("A1", seed=0, smoke=True)
    bundle = data_mod.build_bundle(cfg)
    res = panel_mod.transfer_split_subop_probes(model, bundle.seen_heldout,
                                                cfg.seed)
    assert res["mean_across_subops"] is None or 0.3 < res["mean_across_subops"] < 0.7


# --- C3 ----------------------------------------------------------------------

def test_c3_canonical_substitution_shape_and_levels(model):
    cfg = config_for_arm("A1", seed=0, smoke=True)
    bundle = data_mod.build_bundle(cfg)
    res = panel_mod.canonical_substitution(model, bundle, cfg)
    assert set(res["per_level"]) <= {"train", "L1", "L2", "L3"}
    assert "L3" in res["per_level"]
    for tid, cell in res["per_cell"].items():
        assert len(ops.task_surface_ops(tid)) == 2      # depth-2 only
        for tag in ("", "_alt"):
            assert 0.0 <= cell[f"canon_route_agree_hard{tag}"] <= 1.0
            assert cell[f"canon_route_kl{tag}"] >= 0.0
            assert cell[f"canon_repair_delta{tag}"] == pytest.approx(
                cell[f"canon_repair_acc{tag}"] - cell["baseline_acc"])


def test_c3_primary_and_alt_agree_under_r8(model):
    """R8 made the encoder digit-only and the router position-blind, so
    continuing as token 2 and continuing as a singleton must coincide."""
    cfg = config_for_arm("A1", seed=0, smoke=True)
    bundle = data_mod.build_bundle(cfg)
    res = panel_mod.canonical_substitution(model, bundle, cfg)
    assert res["all_variants_agree"]


# --- C4 ----------------------------------------------------------------------

def test_c4_isolates_partner_variance():
    # outcome depends ONLY on the partner: partner variance high, input zero
    c = np.zeros((2, 4, 10), dtype=bool)
    c[:, :2, :] = True
    d = panel_mod.variance_decomposition(c)
    assert d["partner_variance_mean"] == pytest.approx(0.25)
    assert d["input_variance_mean"] == pytest.approx(0.0)


def test_c4_isolates_input_variance():
    # outcome depends ONLY on the input: input variance high, partner zero
    c = np.zeros((2, 4, 10), dtype=bool)
    c[:, :, :5] = True
    d = panel_mod.variance_decomposition(c)
    assert d["input_variance_mean"] == pytest.approx(0.25)
    assert d["partner_variance_mean"] == pytest.approx(0.0)


def test_c4_constant_outcome_has_zero_of_both():
    d = panel_mod.variance_decomposition(np.ones((3, 4, 8), dtype=bool))
    assert d["partner_variance_mean"] == 0.0
    assert d["input_variance_mean"] == 0.0


# --- hard invariants ---------------------------------------------------------

def test_no_gradient_reaches_the_model_from_any_new_metric():
    """Invariant 1: probes read, never write."""
    cfg = config_for_arm("A1", seed=0, smoke=True)
    torch.manual_seed(0)
    m = AtomModel(cfg)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
        p.grad = None
    bundle = data_mod.build_bundle(cfg)
    panel_mod.transfer_split_subop_probes(m, bundle.seen_heldout, cfg.seed)
    panel_mod.canonical_substitution(m, bundle, cfg)
    assert all(p.grad is None for p in m.parameters())
    assert not any(p.requires_grad for p in m.parameters())
    assert not m.training


def test_load_checkpoint_freezes_and_evals(tmp_path):
    """Invariant 4: panel work runs on saved checkpoints, frozen."""
    from atomv2.train import train_run
    from atomv2.panel import load_checkpoint
    cfg = dataclasses.replace(config_for_arm("A1", 0, smoke=True),
                              total_steps=4, eval_every=2, ckpt_every=2,
                              warmup_steps=1, tau_anneal_steps=2,
                              panel_steps=(2,), log_every=2,
                              examples_per_train_task=16,
                              examples_per_eval_task=8, n_probe_examples=8)
    rd = train_run(cfg, out=str(tmp_path), allow_dirty=True)
    m, _, _ = load_checkpoint(rd, "final.pt")
    assert not m.training
    assert not any(p.requires_grad for p in m.parameters())
