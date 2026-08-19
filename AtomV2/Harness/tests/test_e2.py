"""E2 registrations: interface-noise mechanics, streams, and gates
(H1-Experiment2.md smoke-test checklist items 1-6 in unit form)."""
import math

import numpy as np
import torch

from atomv2 import registered as R
from atomv2.config import config_for_arm
from atomv2.model import AtomModel
from atomv2.noise import sigma_for_cosine, target_cosine


def _batch(n=16, n_tokens=2, seed=0):
    g = np.random.default_rng(seed)
    x = torch.from_numpy(g.integers(0, 10, size=(n, 6)).astype("int64"))
    toks = torch.from_numpy(g.integers(0, 8, size=(n, 2)).astype("int64"))
    ntok = torch.full((n,), n_tokens, dtype=torch.int64)
    return x, toks, ntok


def _model(arm="A9"):
    cfg = config_for_arm(arm, 0, smoke=True)
    torch.manual_seed(0)
    m = AtomModel(cfg)
    m.eval()
    return m, cfg


def test_sigma_derivation_matches_registration():
    # sigma = sqrt(1/c^2 - 1), receipt values from the doc's table.
    for arm, expected in (("A8", 0.044754933), ("A9", 0.142492283),
                          ("A10", 0.328684105)):
        assert abs(R.E2_STATE_NOISE_SIGMA[arm] - expected) < 1e-9, arm
        c = R.E2_TARGET_COSINE[arm]
        assert abs(sigma_for_cosine(c) - R.E2_STATE_NOISE_SIGMA[arm]) < 1e-15
        assert abs(target_cosine(R.E2_STATE_NOISE_SIGMA[arm]) - c) < 1e-12


def test_e2_config_is_a6_plus_noise_only():
    for arm in R.E2_ARMS:
        e2 = config_for_arm(arm, 1).to_dict()
        a6 = config_for_arm("A6", 1).to_dict()
        diff = {k for k in e2 if e2[k] != a6[k]}
        assert diff == {"arm", "experiment", "protocol_revision",
                        "state_noise_sigma"}, arm
        assert e2["state_noise_sigma"] == R.E2_STATE_NOISE_SIGMA[arm]
        assert e2["lambda_use"] == 0.0


def test_zero_noise_bypasses_generation_completely():
    model, _cfg = _model("A6")
    x, toks, ntok = _batch()
    gen = torch.Generator().manual_seed(7)
    before = gen.get_state().clone()
    with torch.no_grad():
        o1 = model(x, toks, ntok, mode="hard",
                   noise_sigma=0.0, noise_generator=gen)
        o2 = model(x, toks, ntok, mode="hard")
    assert torch.equal(gen.get_state(), before)     # never touched
    assert torch.equal(o1["logits"], o2["logits"])
    assert "states_transmitted" not in o1           # no noise payload


def test_noise_isolation_from_routing_stream():
    # Changing the interface-noise seed changes transmitted states but not
    # the pre-noise route Gumbel draws (step-0 decisions and the routing
    # generator's own state).
    model, cfg = _model("A10")
    x, toks, ntok = _batch()
    outs, gum_states = [], []
    for noise_seed in (1, 2):
        gum = torch.Generator().manual_seed(123)
        ng = torch.Generator().manual_seed(noise_seed)
        with torch.no_grad():
            out = model(x, toks, ntok, mode="gumbel", tau=1.0, generator=gum,
                        noise_sigma=cfg.state_noise_sigma, noise_generator=ng)
        outs.append(out)
        gum_states.append(gum.get_state().clone())
    assert torch.equal(gum_states[0], gum_states[1])   # routing stream intact
    assert torch.equal(outs[0]["choices"][:, 0], outs[1]["choices"][:, 0])
    assert not torch.equal(outs[0]["states_transmitted"][0],
                           outs[1]["states_transmitted"][0])


def test_noise_timing_and_masking():
    model, _ = _model("A9")
    sigma = 5.0                                     # exaggerated: visible
    x, toks, _ = _batch()
    for n_tok, last_noisy in ((1, 1), (2, 4)):      # noisy handoffs 0..last
        ntok = torch.full((16,), n_tok, dtype=torch.int64)
        with torch.no_grad():
            out = model(x, toks, ntok, mode="hard", noise_sigma=sigma,
                        noise_generator=torch.Generator().manual_seed(0))
        for k, t in enumerate(out["states_transmitted"]):
            differs = not torch.equal(t, out["states"][k + 1])
            assert differs == (k <= last_noisy), (n_tok, k)
        # No noise is ever applied after the final live step: the decoder's
        # input is exactly the recorded final state.
        with torch.no_grad():
            assert torch.equal(out["logits"],
                               model.decoder(out["states"][-1]))
    # Fixed RNG consumption regardless of masks: singleton and pair batches
    # advance the noise stream identically.
    ends = []
    for n_tok in (1, 2):
        ng = torch.Generator().manual_seed(9)
        ntok = torch.full((16,), n_tok, dtype=torch.int64)
        with torch.no_grad():
            model(x, toks, ntok, mode="hard", noise_sigma=sigma,
                  noise_generator=ng)
        ends.append(ng.get_state().clone())
    assert torch.equal(ends[0], ends[1])


def test_pass_gets_channel_noise():
    # An all-pass forced program still receives interface noise: pass cannot
    # dodge corruption.
    model, _ = _model("A9")
    x, toks, ntok = _batch()
    sched = torch.full((16, 6), R.PASS_INDEX, dtype=torch.int64)
    with torch.no_grad():
        out = model(x, toks, ntok, mode="forced", forced=sched,
                    noise_sigma=0.5,
                    noise_generator=torch.Generator().manual_seed(0))
        clean = model(x, toks, ntok, mode="forced", forced=sched)
    assert not torch.equal(out["states_transmitted"][0], clean["states"][1])
    assert not torch.equal(out["logits"], clean["logits"])


def test_observed_cosine_matches_target():
    # Gate 8 in unit form: on LayerNormed states, the observed median
    # clean/transmitted cosine sits within 0.005 of the registered target.
    model, cfg = _model("A9")
    x, toks, ntok = _batch(n=64)
    cos = []
    for d in range(8):
        with torch.no_grad():
            out = model(x, toks, ntok, mode="hard",
                        noise_sigma=cfg.state_noise_sigma,
                        noise_generator=torch.Generator().manual_seed(d))
        for k, t in enumerate(out["states_transmitted"]):
            c = torch.nn.functional.cosine_similarity(
                out["states"][k + 1], t, dim=-1)
            cos.append(c)
    med = float(torch.cat(cos).median())
    assert abs(med - R.E2_TARGET_COSINE["A9"]) <= R.E2_COSINE_TOL
    # And the transmitted state stays properly normalized (no amplitude
    # side channel): per-position mean ~0, var ~1.
    h = out["states_transmitted"][0].view(-1, cfg.seq_len, cfg.d_model)
    assert float(h.mean(dim=-1).abs().max()) < 1e-4
    assert abs(float(h.var(dim=-1, unbiased=False).mean()) - 1.0) < 1e-3


def test_streams_registered():
    from atomv2.utils import _STREAMS
    assert _STREAMS["state_noise"] == 9
    assert _STREAMS["e2_noise_eval"] == 10
