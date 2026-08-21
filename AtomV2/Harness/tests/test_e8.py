"""E8 registrations (H1-Experiment8.md): depth-parameterized atoms, zero-depth
byte-identity, config wiring."""
import numpy as np
import torch

from atomv2 import registered as R
from atomv2.config import config_for_arm
from atomv2.model import AtomModel


def test_e8_config_is_a18_plus_depth_or_budget():
    a18 = config_for_arm("A18", 1).to_dict()
    a22 = config_for_arm("A22", 1).to_dict()
    a23 = config_for_arm("A23", 1).to_dict()
    a24 = config_for_arm("A24", 1).to_dict()
    assert {k for k in a22 if a22[k] != a18[k]} == {
        "arm", "experiment", "protocol_revision", "total_steps"}
    assert a22["total_steps"] == 40_000 and a22["atom_layers"] == 1
    for d in (a23, a24):
        assert {k for k in d if d[k] != a18[k]} == {
            "arm", "experiment", "protocol_revision", "atom_layers"}
    assert a23["atom_layers"] == 2 and a24["atom_layers"] == 3
    assert a23["micro_steps"] == 1 and a23["atom_update"] == "residual"
    # Smoke keeps the smoke budget for the control too.
    assert config_for_arm("A22", 0, smoke=True).total_steps == 300


def test_zero_depth_is_byte_identical_to_a18():
    # Same torch seed -> identical parameters AND identical outputs: the
    # depth machinery must not exist at atom_layers = 1 (the E8 gate's
    # structural half).
    torch.manual_seed(7)
    m18 = AtomModel(config_for_arm("A18", 0, smoke=True))
    torch.manual_seed(7)
    m22 = AtomModel(config_for_arm("A22", 0, smoke=True))
    s18, s22 = m18.state_dict(), m22.state_dict()
    assert set(s18) == set(s22)                     # no extra parameters
    assert all(torch.equal(s18[k], s22[k]) for k in s18)
    assert not hasattr(m22.atoms, "wm")
    g = np.random.default_rng(0)
    x = torch.from_numpy(g.integers(0, 10, size=(8, 6)).astype("int64"))
    with torch.no_grad():
        assert torch.equal(m18.atoms.outputs(m18.code(x)),
                           m22.atoms.outputs(m22.code(x)))


def test_deep_atoms_forward_and_page_as_one_block():
    for arm, depth in (("A23", 2), ("A24", 3)):
        cfg = config_for_arm(arm, 0, smoke=True)
        torch.manual_seed(0)
        model = AtomModel(cfg)
        model.eval()
        assert model.atoms.n_layers == depth
        assert model.atoms.wm.shape == (depth - 1, cfg.n_atoms,
                                        cfg.atom_hidden, cfg.atom_hidden)
        g = np.random.default_rng(1)
        x = torch.from_numpy(g.integers(0, 10, size=(8, 6)).astype("int64"))
        toks = torch.from_numpy(g.integers(0, 8, size=(8, 2)).astype("int64"))
        ntok = torch.full((8,), 2, dtype=torch.int64)
        with torch.no_grad():
            out = model(x, toks, ntok, mode="hard")
            assert torch.isfinite(out["logits"]).all()
            # One route decision per token, unchanged from A18.
            assert out["choices"].shape[1] == 2
            # step_once uses the same deep path (panel probes inherit depth).
            z = model.step_once(model.code(x), 3)
            assert z.shape == (8, cfg.state_dim)
        # Depth changes the atom MLP only: composer/keys/encoder/decoder
        # parameter counts match A18's exactly.
        base = AtomModel(config_for_arm("A18", 0, smoke=True))
        for mod in ("encoder", "composer", "decoder"):
            assert (sum(p.numel() for p in getattr(model, mod).parameters())
                    == sum(p.numel()
                           for p in getattr(base, mod).parameters())), mod


def test_e8_registration_constants():
    assert R.E8_ARMS == ("A22", "A23", "A24")
    assert R.E8_ATOM_LAYERS == {"A22": 1, "A23": 2, "A24": 3}
    assert R.E8_TOTAL_STEPS["A22"] == 40_000
    assert R.E8_SCREEN_SEED == 1 and R.E8_HEALTH_SEEN_MIN == 0.80
    assert R.EXPERIMENT_REVISIONS["e8"] == R.E8_PROTOCOL_REVISION
