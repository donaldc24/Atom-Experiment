"""Amendments R9 (A1 not in the E1 battery) and R10 (pooling identity)."""
import subprocess
import sys

import pytest

from atomv2 import aggregate as aggregate_mod
from atomv2 import registered as R
from atomv2 import utils
from atomv2.backfill import backfill_run
from atomv2.config import E1_ARMS, E1_BATTERY_ARMS, config_for_arm
from atomv2.utils import read_json, write_json


# --- R9 ---------------------------------------------------------------------

def test_a0_free_is_configurationally_identical_to_a1():
    """The premise of R9. If this ever fails, A1 must go back in the battery."""
    a = config_for_arm("A0-free", 0).to_dict()
    b = config_for_arm("A1", 0).to_dict()
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"arm", "experiment"}, differing


def test_a1_excluded_from_battery_but_still_constructible():
    assert "A1" not in E1_BATTERY_ARMS
    assert E1_BATTERY_ARMS == ("A2", "A3", "A4")
    assert "A1" in E1_ARMS                       # still a registered arm
    assert R.LAMBDA_GRID["A1"] == 0.0            # grid itself unchanged
    assert config_for_arm("A1", 0).lambda_use == 0.0


def test_run_e1_refuses_a1():
    proc = subprocess.run(
        [sys.executable, "-m", "atomv2.run_e1", "--arms", "A1", "--plan"],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "A1 is not part of the E1 battery" in (proc.stdout + proc.stderr)


def test_run_e1_default_plan_omits_a1():
    proc = subprocess.run(
        [sys.executable, "-m", "atomv2.run_e1", "--plan"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "A1 " not in proc.stdout
    for arm in ("A2", "A3", "A4"):
        assert arm in proc.stdout


# --- R10 --------------------------------------------------------------------

def test_status_path_keeps_the_leading_space_column():
    """Regression: `_git(...).strip()` ate the first porcelain line's leading
    space, so line[3:] lost a character. An output path could then be misread
    as source and spuriously refuse the run."""
    # X is a SPACE for worktree-only modifications - the load-bearing case
    assert utils._status_path(" M AtomV2/runs/e0/x/metrics.json") == \
        "AtomV2/runs/e0/x/metrics.json"
    assert utils._is_output_path(
        utils._status_path(" M AtomV2/runs/e0/x/metrics.json"))
    # staged, untracked, and renamed forms
    assert utils._status_path("M  AtomV2/Harness/atomv2/model.py") == \
        "AtomV2/Harness/atomv2/model.py"
    assert utils._status_path("?? AtomV2/results/e1/") == "AtomV2/results/e1/"
    assert utils._status_path("R  old/path.py -> AtomV2/Harness/new.py") == \
        "AtomV2/Harness/new.py"
    # a genuinely non-output source file is still detected as source
    assert not utils._is_output_path(
        utils._status_path(" M AtomV2/H1Experiments.md"))


def test_git_status_is_read_without_stripping():
    raw = utils._git("status", "--porcelain", "--untracked-files=all",
                     strip=False)
    stripped = utils._git("status", "--porcelain", "--untracked-files=all")
    if raw.splitlines() and raw.splitlines()[0].startswith(" "):
        # the unstripped read must preserve the column that strip() removed
        assert raw.splitlines()[0] != stripped.splitlines()[0]
    for line in raw.splitlines():
        assert len(line) > 3 and line[2] == " "


def test_source_fingerprint_is_content_based(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "HARNESS_ROOT", tmp_path)
    (tmp_path / "atomv2").mkdir()
    (tmp_path / "splits").mkdir()
    (tmp_path / "atomv2" / "model.py").write_text("x = 1\n")
    (tmp_path / "splits" / "split_v2.json").write_text("{}")
    first = utils.harness_source_sha256()
    assert first == utils.harness_source_sha256()          # stable

    # A file that cannot change a number does not change the identity.
    (tmp_path / "README.md").write_text("docs")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("assert True")
    assert utils.harness_source_sha256() == first

    # A harness module's CONTENT does.
    (tmp_path / "atomv2" / "model.py").write_text("x = 2\n")
    assert utils.harness_source_sha256() != first


def _run(tmp_path, name, env_extra, seed=0):
    d = tmp_path / "e0" / name
    write_json(d / "metrics.json", {
        "arm": "A0-oracle", "seed": seed, "smoke": False,
        "protocol_revision": R.PROTOCOL_REVISION,
        "param_counts": {"composer": 1, "atoms_total": 2},
    })
    write_json(d / "env.json", {"hostname": "h", **env_extra})
    return d


def test_collect_refuses_runs_missing_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_mod, "RUNS_DIR", tmp_path)
    _run(tmp_path, "old", {"git_sha": "abc", "dirty_source_sha256": "46c8"})
    with pytest.raises(SystemExit, match="harness_source_sha256"):
        aggregate_mod.collect("e0")


def test_collect_refuses_differing_source_content(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_mod, "RUNS_DIR", tmp_path)
    _run(tmp_path, "a", {"harness_source_sha256": "content-1"}, seed=0)
    _run(tmp_path, "b", {"harness_source_sha256": "content-2"}, seed=1)
    with pytest.raises(SystemExit, match="multiple source snapshots"):
        aggregate_mod.collect("e0")


def test_collect_pools_same_content_across_dirty_and_committed(tmp_path,
                                                               monkeypatch):
    """The R10 case: identical code, one run dirty, one committed."""
    monkeypatch.setattr(aggregate_mod, "RUNS_DIR", tmp_path)
    _run(tmp_path, "dirty", {"harness_source_sha256": "same",
                             "git_sha": "d354ffd",
                             "dirty_source_sha256": "46c8"}, seed=0)
    _run(tmp_path, "clean", {"harness_source_sha256": "same",
                             "git_sha": "ff6a1b3"}, seed=1)
    rows = aggregate_mod.collect("e0")
    assert len(rows) == 2


def test_backfill_marks_provenance_and_refuses_conflict(tmp_path):
    d = _run(tmp_path, "old", {"git_sha": "abc"})
    assert backfill_run(d, "content-1", "ff6a1b3", "replay evidence") == "backfilled"
    env = read_json(d / "env.json")
    assert env["harness_source_sha256"] == "content-1"
    # A backfilled value is never indistinguishable from a self-recorded one.
    assert env["harness_source_provenance"]["recorded_by"] == "backfill"
    assert env["harness_source_provenance"]["revision"] == "ff6a1b3"
    assert env["git_sha"] == "abc"           # original provenance preserved
    assert (d / "SHA256SUMS").exists()

    assert backfill_run(d, "content-1", "ff6a1b3", "r") == "already recorded"
    assert "CONFLICT" in backfill_run(d, "content-2", "ff6a1b3", "r")
    assert read_json(d / "env.json")["harness_source_sha256"] == "content-1"
