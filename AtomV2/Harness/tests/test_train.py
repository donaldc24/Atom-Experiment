"""Micro end-to-end training tests (tiny steps, tmp output dir)."""
import dataclasses
import json

import numpy as np
import pytest
import torch

from atomv2.analyze import analyze
from atomv2.config import config_for_arm
from atomv2.panel import run_all_panels
from atomv2.train import train_run
from atomv2.utils import read_json, write_json


def _micro_cfg(arm, seed=0):
    cfg = config_for_arm(arm, seed, smoke=True)
    return dataclasses.replace(cfg, total_steps=6, eval_every=3, ckpt_every=3,
                               warmup_steps=2, tau_anneal_steps=4,
                               panel_steps=(3,), log_every=2,
                               examples_per_train_task=16,
                               examples_per_eval_task=8, n_probe_examples=8)


@pytest.mark.parametrize("arm", ["A1", "A0-oracle"])
def test_micro_run_emits_all_artifacts(tmp_path, arm):
    cfg = _micro_cfg(arm)
    run_dir = train_run(cfg, out=str(tmp_path), allow_dirty=True)
    for name in ("config.json", "env.json", "split_ref.json",
                 "data_manifest.json", "init_calibration.json",
                 "param_counts.json", "train_log.jsonl"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "checkpoints" / "final.pt").exists()
    assert (run_dir / "evals" / "step000003.json").exists()
    assert (run_dir / "evals" / "step000006.json").exists()
    assert (run_dir / "traces" / "step000006.npz").exists()
    m = analyze(run_dir)
    assert 0.0 <= m["acc_seen_hard"] <= 1.0
    assert "per_cell" in m and len(m["per_cell"]) == 42 + 29
    assert len(m["curve"]) == 2


def test_micro_run_is_deterministic(tmp_path):
    cfg = _micro_cfg("A1")
    d1 = train_run(cfg, out=str(tmp_path / "a"), allow_dirty=True)
    d2 = train_run(cfg, out=str(tmp_path / "b"), allow_dirty=True)
    l1 = [json.loads(x) for x in open(d1 / "train_log.jsonl", encoding="utf-8")]
    l2 = [json.loads(x) for x in open(d2 / "train_log.jsonl", encoding="utf-8")]
    s1 = [r for r in l1 if r.get("event") == "step"]
    s2 = [r for r in l2 if r.get("event") == "step"]
    assert len(s1) == len(s2) > 0
    for a, b in zip(s1, s2):
        assert a["loss"] == b["loss"], (a["step"], a["loss"], b["loss"])
    e1 = read_json(d1 / "evals" / "step000006.json")
    e2 = read_json(d2 / "evals" / "step000006.json")
    assert e1 == e2


def test_free_arm_never_touches_oracle_terms(tmp_path):
    cfg = _micro_cfg("A1")
    run_dir = train_run(cfg, out=str(tmp_path), allow_dirty=True)
    logs = [json.loads(x) for x in open(run_dir / "train_log.jsonl",
                                        encoding="utf-8")]
    for r in logs:
        assert "loss_state_rel" not in r
        assert "loss_route_ce" not in r
        assert "loss_intermediate_ce" not in r


def test_oracle_arm_logs_oracle_terms(tmp_path):
    cfg = _micro_cfg("A0-oracle")
    run_dir = train_run(cfg, out=str(tmp_path), allow_dirty=True)
    steps = [json.loads(x) for x in open(run_dir / "train_log.jsonl",
                                         encoding="utf-8")
             if json.loads(x).get("event") == "step"]
    assert all("loss_state_rel" in r and "loss_route_ce" in r for r in steps)


def test_panel_checkpoint_saved_off_regular_cadence(tmp_path):
    cfg = dataclasses.replace(_micro_cfg("A1"), total_steps=6,
                              eval_every=3, ckpt_every=4, panel_steps=(3,))
    run_dir = train_run(cfg, out=str(tmp_path), allow_dirty=True)
    assert (run_dir / "checkpoints" / "step000003.pt").exists()
    assert (run_dir / "traces" / "step000003.npz").exists()
    assert (run_dir / "checkpoints" / "step000004.pt").exists()


def test_panel_refuses_missing_registered_checkpoint(tmp_path):
    write_json(tmp_path / "config.json", {"panel_steps": [5]})
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(FileNotFoundError, match="registered panel checkpoint missing"):
        run_all_panels(tmp_path)
