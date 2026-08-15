"""Data generation: volumes, dedup, determinism, opaqueness, oversampling."""
import numpy as np
import pytest

from atomv2 import data as data_mod
from atomv2 import ops
from atomv2 import registered as R
from atomv2.config import config_for_arm


@pytest.fixture(scope="module")
def bundle():
    return data_mod.build_bundle(config_for_arm("A1", seed=0, smoke=True))


def test_volumes_and_membership(bundle):
    cfg = config_for_arm("A1", seed=0, smoke=True)
    assert len(bundle.train) == 42
    assert len(bundle.seen_heldout) == 42
    assert len(bundle.unseen["L1"]) == 8
    assert len(bundle.unseen["L2"]) == 6
    assert len(bundle.unseen["L3"]) == 15
    for td in bundle.train:
        assert len(td.x) == cfg.examples_per_train_task
    for td in bundle.seen_heldout + [t for l in bundle.unseen.values() for t in l]:
        assert len(td.x) == cfg.examples_per_eval_task


def test_train_and_seen_heldout_disjoint_per_task(bundle):
    for tr, ev in zip(bundle.train, bundle.seen_heldout):
        assert tr.task.task_id == ev.task.task_id
        tr_keys = {row.tobytes() for row in tr.x}
        ev_keys = {row.tobytes() for row in ev.x}
        assert not (tr_keys & ev_keys)


def test_targets_match_the_algebra(bundle):
    for td in bundle.train + bundle.seen_heldout:
        assert np.array_equal(td.y, ops.apply_task(td.task.task_id, td.x))


def test_task_tokens_are_opaque_ids(bundle):
    # Tokens are surface indices 0..7 + PAD; no sub-op structure anywhere.
    for td in bundle.train:
        assert td.task.tokens.max() <= data_mod.PAD_TOKEN
        if td.task.kind == "singleton":
            assert td.task.tokens[1] == data_mod.PAD_TOKEN


def test_determinism_and_seed_sensitivity():
    cfg = config_for_arm("A1", seed=0, smoke=True)
    m1 = data_mod.data_manifest(data_mod.build_bundle(cfg))
    m2 = data_mod.data_manifest(data_mod.build_bundle(cfg))
    assert m1 == m2
    cfg2 = config_for_arm("A1", seed=1, smoke=True)
    m3 = data_mod.data_manifest(data_mod.build_bundle(cfg2))
    assert m1 != m3


def test_data_shared_across_arms_at_same_seed():
    # E0 requirement: both arms run on the same data; lambda arms likewise.
    m_free = data_mod.data_manifest(
        data_mod.build_bundle(config_for_arm("A0-free", seed=0, smoke=True)))
    m_oracle = data_mod.data_manifest(
        data_mod.build_bundle(config_for_arm("A0-oracle", seed=0, smoke=True)))
    assert m_free == m_oracle


def test_p3_presentation_oversampling(bundle):
    cfg = config_for_arm("A1", seed=0, smoke=True)
    arrays = data_mod.build_epoch_arrays(bundle, cfg)
    n = cfg.examples_per_train_task
    expected = (41 + cfg.p3_oversample_factor) * n
    assert len(arrays["x"]) == expected
    # count P3 presentations via its token pattern
    p3_tok = data_mod.make_task("P3", "train").tokens
    is_p3 = (arrays["tokens"] == p3_tok).all(axis=1) & (arrays["n_tokens"] == 1)
    assert is_p3.sum() == cfg.p3_oversample_factor * n


def test_oracle_partials(bundle):
    cfg = config_for_arm("A0-oracle", seed=0, smoke=True)
    arrays = data_mod.build_epoch_arrays(bundle, cfg, include_partials=True)
    # after token 1 of a pair, y_partial[:,0] must equal the first surface op
    pair_rows = arrays["n_tokens"] == 2
    x = arrays["x"][pair_rows][:50]
    yp = arrays["y_partial"][pair_rows][:50]
    toks = arrays["tokens"][pair_rows][:50]
    for b in range(len(x)):
        p_first = ops.SURFACE_NAMES[toks[b, 0]]
        assert np.array_equal(yp[b, 0], ops.SURFACE_FNS[p_first](x[b][None, :])[0])
    # y_partial[:,1] is the final target for pairs
    assert np.array_equal(arrays["y_partial"][pair_rows][:, 1],
                          arrays["y"][pair_rows])
