"""Negative controls for the leak audit: every check must FAIL when sabotaged.

An audit that cannot fail proves nothing. Each test here injects exactly the
contamination the corresponding check exists to catch and asserts it fires.
"""
import numpy as np
import pytest

from atomv2 import data as data_mod
from atomv2 import leak_audit as la
from atomv2 import ops
from atomv2 import registered as R
from atomv2 import split as split_mod
from atomv2.config import config_for_arm


@pytest.fixture(scope="module")
def smoke_bundle():
    cfg = config_for_arm("A0-free", seed=0, smoke=True)
    return data_mod.build_bundle(cfg), cfg


def _manifest(bundle):
    return data_mod.data_manifest(bundle)


# --- A ----------------------------------------------------------------------

def test_a_passes_clean_and_catches_injected_dax_pair(smoke_bundle):
    bundle, _ = smoke_bundle
    m = _manifest(bundle)
    assert la.check_a_no_dax_pairs({"run": m})["ok"]

    bad = {k: dict(v) for k, v in m.items() if isinstance(v, dict)}
    bad["train"] = dict(m["train"])
    bad["train"]["P3_P1"] = {"n": 1000, "x": "deadbeef", "y": "deadbeef"}
    res = la.check_a_no_dax_pairs({"run": bad})
    assert not res["ok"]
    assert res["dax_pairs_in_train"]["run"] == ["P3_P1"]


def test_a_catches_dax_pair_hiding_in_seen_heldout(smoke_bundle):
    bundle, _ = smoke_bundle
    m = _manifest(bundle)
    bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in m.items()}
    bad["seen_heldout"] = dict(m["seen_heldout"])
    bad["seen_heldout"]["P8_P3"] = {"n": 400, "x": "x", "y": "y"}
    assert not la.check_a_no_dax_pairs({"run": bad})["ok"]


def test_a_catches_missing_dax_singleton(smoke_bundle):
    """P3 IS trained as a singleton; its absence is a different bug."""
    bundle, _ = smoke_bundle
    m = _manifest(bundle)
    bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in m.items()}
    bad["train"] = {k: v for k, v in m["train"].items() if k != R.DAX}
    assert not la.check_a_no_dax_pairs({"run": bad})["ok"]


# --- B ----------------------------------------------------------------------

def test_b_catches_tampered_world_block(monkeypatch):
    s = split_mod.load()
    tampered = {**s, "world": {**s["world"],
                               "subops": {**s["world"]["subops"],
                                          "R": {"pi": [0, 1, 2, 3, 4, 5],
                                                "a": 1, "b": [0] * 6}}}}
    monkeypatch.setattr(split_mod, "load", lambda: tampered)
    res = la.check_b_split_and_world([])
    assert not res["ok"]
    assert any(m.get("subop") == "R" for m in res["world_block_mismatches"])


def test_b_catches_tampered_surface_recipe(monkeypatch):
    s = split_mod.load()
    tampered = {**s, "world": {**s["world"],
                               "surface_recipes": {**s["world"]["surface_recipes"],
                                                   "P1": ["I", "R"]}}}
    monkeypatch.setattr(split_mod, "load", lambda: tampered)
    res = la.check_b_split_and_world([])
    assert not res["ok"]
    assert any(m.get("recipe") == "P1" for m in res["world_block_mismatches"])


# --- C ----------------------------------------------------------------------

def test_c_passes_clean(smoke_bundle):
    bundle, cfg = smoke_bundle
    assert la.check_c_dax_oversampling({0: (bundle, cfg)},
                                       {"run": _manifest(bundle)})["ok"]


def test_c_catches_missing_oversampling(smoke_bundle, monkeypatch):
    bundle, cfg = smoke_bundle
    real = data_mod.build_epoch_arrays

    def no_oversample(b, c, include_partials=False):
        import dataclasses
        return real(b, dataclasses.replace(c, p3_oversample_factor=1),
                    include_partials=include_partials)

    monkeypatch.setattr(data_mod, "build_epoch_arrays", no_oversample)
    res = la.check_c_dax_oversampling({0: (bundle, cfg)},
                                      {"run": _manifest(bundle)})
    assert not res["ok"]
    assert res["per_seed"]["0"]["presentations"] != \
        res["per_seed"]["0"]["expected_presentations"]


def test_c_catches_inflated_unique_examples(smoke_bundle, monkeypatch):
    """The R4 failure mode: 7,000 DISTINCT examples instead of 7 copies."""
    bundle, cfg = smoke_bundle
    real = data_mod.build_epoch_arrays

    def distinct_copies(b, c, include_partials=False):
        arrays = real(b, c, include_partials=include_partials)
        dax = data_mod.make_task(R.DAX, "train")
        is_dax = ((arrays["tokens"] == dax.tokens).all(axis=1)
                  & (arrays["n_tokens"] == 1))
        rng = np.random.default_rng(0)
        arrays["x"][is_dax] = rng.integers(
            0, ops.MOD, size=(int(is_dax.sum()), ops.L), dtype=np.int64)
        return arrays

    monkeypatch.setattr(data_mod, "build_epoch_arrays", distinct_copies)
    res = la.check_c_dax_oversampling({0: (bundle, cfg)},
                                      {"run": _manifest(bundle)})
    assert not res["ok"]
    assert not res["per_seed"]["0"]["blocks_are_exact_copies"]


def test_c_catches_manifest_showing_inflated_unique_count(smoke_bundle):
    bundle, cfg = smoke_bundle
    m = _manifest(bundle)
    bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in m.items()}
    bad["train"] = dict(m["train"])
    bad["train"][R.DAX] = {**m["train"][R.DAX], "n": 7000}
    assert not la.check_c_dax_oversampling({0: (bundle, cfg)}, {"run": bad})["ok"]


# --- D ----------------------------------------------------------------------

def test_d_passes_clean(smoke_bundle):
    bundle, _ = smoke_bundle
    assert la.check_d_seen_heldout_fresh({0: (bundle, None)},
                                         {"run": _manifest(bundle)})["ok"]


def test_d_catches_train_eval_input_overlap(smoke_bundle):
    import copy
    bundle, _ = smoke_bundle
    dirty = copy.deepcopy(bundle)
    # leak one training input into the same task's seen_heldout set
    dirty.seen_heldout[3].x[0] = dirty.train[3].x[0]
    res = la.check_d_seen_heldout_fresh({0: (dirty, None)},
                                        {"run": _manifest(dirty)})
    assert not res["ok"]
    assert res["per_seed"]["0"]["tasks_with_input_overlap"]


def test_d_catches_colliding_manifest_hashes(smoke_bundle):
    bundle, _ = smoke_bundle
    m = _manifest(bundle)
    bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in m.items()}
    bad["seen_heldout"] = dict(m["seen_heldout"])
    tid = next(iter(m["train"]))
    bad["seen_heldout"][tid] = {**m["seen_heldout"][tid], "x": m["train"][tid]["x"]}
    assert not la.check_d_seen_heldout_fresh({0: (bundle, None)},
                                             {"run": bad})["ok"]


# --- F ----------------------------------------------------------------------

def test_f_passes_clean():
    assert la.check_f_no_function_level_leak()["ok"]


def test_f_catches_function_level_leak(monkeypatch):
    s = split_mod.load()
    # P4_P8 and P8_P4 are one function; hold one out while the other trains
    tampered = {**s,
                "train_pairs": [t for t in s["train_pairs"] if t != "P8_P4"],
                "heldout": {**s["heldout"],
                            "L1": list(s["heldout"]["L1"]) + ["P8_P4"]}}
    monkeypatch.setattr(split_mod, "load", lambda: tampered)
    res = la.check_f_no_function_level_leak()
    assert not res["ok"]
    assert res["leaked_cells"]["L1"] == ["P8_P4"]
