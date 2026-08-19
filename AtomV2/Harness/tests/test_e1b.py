"""E1b registrations: cosine router bounds, derived constants, config wiring.

H1-E1bExperiment.md: alpha is fixed by anchoring A5 at (p_max=0.99, sigma=1);
each arm's sigma is DERIVED from its full-precision p_max, never transcribed;
base logits are bounded in [-alpha, alpha]; the certified scaled_dot router is
untouched by default.
"""
import math

import numpy as np
import torch

from atomv2 import registered as R
from atomv2.config import Config, config_for_arm
from atomv2.liveness import registered_p_max
from atomv2.model import AtomModel


def test_alpha_anchored_at_a5():
    assert abs(R.E1B_ALPHA
               - 0.5 * math.log(16 * 0.99 / 0.01)) < 1e-12
    assert abs(R.E1B_SIGMA["A5"] - 1.0) < 1e-12


def test_sigma_derivation_roundtrips_p_max():
    # p_max = exp(2a/s) / (exp(2a/s) + 16) must recover the registered target.
    for arm, p in R.E1B_P_MAX.items():
        s = R.E1B_SIGMA[arm]
        e = math.exp(2.0 * R.E1B_ALPHA / s)
        assert abs(e / (e + 16) - p) < 1e-12, arm
    # A6's target is 0.90 ** (1/6): 10% chance of any exploratory decision in 6.
    assert abs((1 - (1 - R.E1B_P_MAX["A6"])) ** 6 - 0.90) < 1e-12
    # Ordering: more exploration = larger sigma.
    assert R.E1B_SIGMA["A5"] < R.E1B_SIGMA["A6"] < R.E1B_SIGMA["A7"]


def test_worst_case_base_prob_equals_p_max():
    # Maximum separation puts one route at +alpha, 16 at -alpha; the base
    # softmax(z/sigma) at that point IS p_max, so the invariant is attainable
    # but not exceedable.
    for arm in R.E1B_ARMS:
        z = np.full(17, -R.E1B_ALPHA)
        z[3] = R.E1B_ALPHA
        p = np.exp(z / R.E1B_SIGMA[arm])
        p /= p.sum()
        assert p.max() <= R.E1B_P_MAX[arm] + R.E1B_PMAX_TOL
        assert abs(p.max() - R.E1B_P_MAX[arm]) < 1e-9


def test_cosine_logits_bounded_by_alpha():
    cfg = config_for_arm("A5", 0, smoke=True)
    torch.manual_seed(0)
    model = AtomModel(cfg)
    model.eval()
    # Inflate a key norm: bounded logits must not care (the anti-saturation
    # property the router exists for).
    with torch.no_grad():
        model.atoms.keys[0] *= 1e3
    x = torch.randint(0, 10, (8, 6))
    toks = torch.randint(0, 8, (8, 2))
    ntok = torch.full((8,), 2, dtype=torch.int64)
    with torch.no_grad():
        out = model(x, toks, ntok, mode="hard")
    z = out["route_logits"][out["live"]]
    assert float(z.abs().max()) <= R.E1B_ALPHA + 1e-5
    # And therefore the marginal hard-choice probability respects p_max.
    p = torch.softmax(z / cfg.router_sigma, dim=-1)
    assert float(p.max()) <= registered_p_max(cfg) + R.E1B_PMAX_TOL


def test_certified_router_is_default_and_unchanged():
    for arm in ("A0-free", "A0-oracle", "A1", "A2", "A3", "A4"):
        cfg = config_for_arm(arm, 0)
        assert cfg.router == "scaled_dot", arm
        assert cfg.tau_start == R.TAU_START and cfg.tau_end == R.TAU_END, arm


def test_e1b_config_wiring():
    for arm in R.E1B_ARMS:
        cfg = config_for_arm(arm, 1)
        assert cfg.experiment == "e1b"
        assert cfg.protocol_revision == R.E1B_PROTOCOL_REVISION
        assert cfg.router == "cosine"
        assert cfg.lambda_use == 0.0            # no rent intervention in E1b
        assert cfg.router_alpha == R.E1B_ALPHA
        assert cfg.router_sigma == R.E1B_SIGMA[arm]
        assert cfg.router_tau_backward == 1.0
        assert cfg.tau_start == 1.0 and cfg.tau_end == 1.0  # no annealing
        assert not cfg.forced_routing
    oracle = config_for_arm(R.E1B_ORACLE_ARM, 0)
    assert oracle.forced_routing
    assert oracle.router == "cosine"
    assert oracle.router_sigma == R.E1B_SIGMA["A5"]
    # Round-trip through the saved-config path panels use.
    rt = Config.from_dict(config_for_arm("A6", 2).to_dict())
    assert rt == config_for_arm("A6", 2)
    # Old checkpoints (no router keys) resolve to the certified router.
    d = config_for_arm("A2", 0).to_dict()
    for k in list(d):
        if k.startswith("router"):
            del d[k]
    assert Config.from_dict(d).router == "scaled_dot"


def test_arm_difference_is_sigma_only():
    a5 = config_for_arm("A5", 0).to_dict()
    a7 = config_for_arm("A7", 0).to_dict()
    diff = {k for k in a5 if a5[k] != a7[k]}
    assert diff == {"arm", "router_sigma"}
