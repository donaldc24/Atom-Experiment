"""Metric machinery: census arithmetic, closed-map, probes, E0 verdict logic."""
import numpy as np
import pytest
import torch

from atomv2 import aggregate as aggregate_mod
from atomv2 import data as data_mod
from atomv2 import registered as R
from atomv2.aggregate import e0_verdict
from atomv2.config import config_for_arm
from atomv2.evaluate import _closed_map_task, _forward_task, _routing_stats, \
    task_usage_matrix, run_eval
from atomv2.model import AtomModel
from atomv2.panel import _train_linear_probe
from atomv2.utils import write_json


def _fake_results(choices_by_task):
    return {tid: {"choices": np.array(ch), "n_tokens": nt}
            for tid, (ch, nt) in choices_by_task.items()}


def test_census_denominator_excludes_pass():
    # 16 live picks total: atom 0 picked 3x, atom 1 once, pass 12x, rest dead.
    results = _fake_results({
        "T1": ([[0, 16, -1, -1, -1, -1], [0, 1, -1, -1, -1, -1]], 1),
        "T2": ([[0, 16, 16, 16, 16, 16], [16, 16, 16, 16, 16, 16]], 2),
    })
    stats = _routing_stats(results)
    assert stats["n_atom_picks"] == 4                # pass NOT in denominator
    assert stats["atom_selection_share"][0] == pytest.approx(3 / 4)
    assert stats["atom_selection_share"][1] == pytest.approx(1 / 4)
    assert stats["atoms_in_use"] == 2                # both above eps=2%
    # pass rate over live picks: 12 pass of 16 live
    assert stats["pass_rate"] == pytest.approx(12 / 16)


def test_census_eps_excludes_rare_atoms():
    # atom 5 picked once in 100 atom-picks -> 1% < eps=2% -> not in use
    rows = [[0, 16, -1, -1, -1, -1]] * 99 + [[5, 16, -1, -1, -1, -1]]
    stats = _routing_stats(_fake_results({"T": (rows, 1)}))
    assert stats["atoms_in_use"] == 1
    assert not stats["in_use_mask"][5]


def test_steps_per_token():
    results = _fake_results({
        "T1": ([[0, 1, 16, 2, 16, 16]], 2),   # 3 atom steps / 2 tokens = 1.5
        "T2": ([[16, 16, 16, -1, -1, -1]], 1),  # 0 atom steps
    })
    stats = _routing_stats(results)
    assert stats["steps_per_token"] == pytest.approx((1.5 + 0.0) / 2)


def test_task_usage_matrix():
    results = _fake_results({
        "T1": ([[0, 16, -1, -1, -1, -1], [0, 16, -1, -1, -1, -1],
                [1, 16, -1, -1, -1, -1], [0, 16, -1, -1, -1, -1]], 1),
    })
    usage, tids = task_usage_matrix(results)
    assert tids == ["T1"]
    assert usage[0, 0] == pytest.approx(0.75)   # atom 0 in 3 of 4 examples
    assert usage[1, 0] == pytest.approx(0.25)


def test_closed_map_trajectory_properties():
    torch.manual_seed(0)
    cfg = config_for_arm("A1", seed=0, smoke=True)
    model = AtomModel(cfg)
    model.eval()
    bundle = data_mod.build_bundle(cfg)
    td = bundle.seen_heldout[10]  # some pair task
    res = _forward_task(model, td, "hard", 0.5)
    cm = _closed_map_task(model, td, res["states"])
    n_live = td.task.n_tokens * R.MICRO_STEPS
    assert cm["error"] >= 0
    assert len(cm["error_per_step"]) == n_live
    assert cm["n_prefixes"] == 2 * td.task.n_tokens + 1
    assert np.array(cm["prefix_visit_hist"]).shape == (n_live, cm["n_prefixes"])
    assert cm["final_dist_to_target"] >= 0
    assert cm["per_example_min"].shape == (len(td.x), n_live)


def test_closed_map_all_pass_is_caught_by_target_companion():
    """An all-pass dead trajectory: nearest-prefix error ~0 (state = enc(x) =
    prefix 0), but distance-to-target stays high. The companion is what makes
    the headline ungameable by doing nothing."""
    torch.manual_seed(0)
    cfg = config_for_arm("A1", seed=0, smoke=True)
    model = AtomModel(cfg)
    model.eval()
    bundle = data_mod.build_bundle(cfg)
    td = bundle.seen_heldout[10]
    n = len(td.x)
    # fabricate an all-pass trajectory: every state equals h0
    toks = torch.from_numpy(np.tile(td.task.tokens, (n, 1)))
    with torch.no_grad():
        h0 = model.code(torch.from_numpy(td.x)).numpy()
    states = np.repeat(h0[:, None, :], 7, axis=1)
    cm = _closed_map_task(model, td, states)
    assert cm["error"] == pytest.approx(0.0, abs=1e-5)
    assert cm["final_dist_to_target"] > 0.1


def test_linear_probe_learns_separable_labels():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 20)).astype(np.float32)
    w = rng.normal(size=(20, 3))
    y = ((x @ w) > 0).astype(np.float64)
    res = _train_linear_probe(x, y, "multilabel", seed=1)
    assert res["score"] > 0.9
    y_shuf = y[rng.permutation(len(y))]
    res_s = _train_linear_probe(x, y_shuf, "multilabel", seed=1)
    assert res_s["score"] < 0.65


def test_e0_verdict_logic():
    def summary(o_l1, f_l1, o_cm, f_cm):
        return {"A0-oracle": {"acc_unseen_L1_hard_mean": o_l1,
                              "closed_map_target_seen_mean": o_cm},
                "A0-free": {"acc_unseen_L1_hard_mean": f_l1,
                            "closed_map_target_seen_mean": f_cm}}
    good = e0_verdict([], summary(0.9, 0.01, 0.05, 0.8))
    assert good["passed"]
    # free arm scoring HIGH is a fail (the world leaks somewhere)
    leak = e0_verdict([], summary(0.9, 0.4, 0.05, 0.8))
    assert not leak["passed"]
    # oracle failing to hit the ceiling is a fail (rig broken)
    broken = e0_verdict([], summary(0.5, 0.01, 0.05, 0.8))
    assert not broken["passed"]
    # closed-map direction must match
    wrongdir = e0_verdict([], summary(0.9, 0.01, 0.9, 0.1))
    assert not wrongdir["passed"]


def test_collect_rejects_duplicate_arm_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_mod, "RUNS_DIR", tmp_path)
    for name in ("one", "two"):
        d = tmp_path / "e0" / name
        write_json(d / "metrics.json", {
            "arm": "A0-oracle", "seed": 0, "smoke": False,
            "protocol_revision": R.PROTOCOL_REVISION,
            "param_counts": {"composer": 1, "atoms_total": 2},
        })
        write_json(d / "env.json", {
            "hostname": "test-host", "git_sha": "same-source"})
    with pytest.raises(SystemExit, match="duplicate run key"):
        aggregate_mod.collect("e0")


def test_run_eval_summary_shape():
    torch.manual_seed(0)
    cfg = config_for_arm("A1", seed=0, smoke=True)
    model = AtomModel(cfg)
    bundle = data_mod.build_bundle(cfg)
    summary, traces = run_eval(model, bundle, cfg, step=0)
    assert set(summary["sets"]) == {"seen_heldout", "unseen_L1", "unseen_L2",
                                    "unseen_L3"}
    assert len(summary["sets"]["seen_heldout"]["tasks"]) == 42
    assert len(summary["sets"]["unseen_L3"]["tasks"]) == 15
    assert "dissociation_gap_hard" in summary
    assert summary["routing"]["census_eps"] == R.CENSUS_EPS
    # traces carry per-example arrays for every task
    assert any(k.endswith("/choices") for k in traces)
