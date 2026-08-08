"""Fast acceptance tests: T4, T5, T7, T8 plus the static half of T3.

These run in CI on every commit. T1-T3 require training runs and gate the
experiment rather than each commit -- see tests/test_gates.py.

Runnable with pytest, or standalone:  python tests/test_fast.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e1.config import config_for_arm                                  # noqa: E402
from e1.data import build_bundle, load_split, verify_split            # noqa: E402
from e1.evaluate import compute_ablation                              # noqa: E402
from e1.model import AtomNet                                          # noqa: E402
from e1.primitives import (                                           # noqa: E402
    K, apply_composition, check_primitive_independence, random_inputs,
)
from e1.utils import RSSMonitor, param_counts, seed_everything, set_threads  # noqa: E402


# --------------------------------------------------------------------------
# T4 -- primitive independence
# --------------------------------------------------------------------------

def test_t4_primitive_independence():
    violations, notes = check_primitive_independence(n_samples=10_000, seed=0)
    assert violations == [], f"primitive set is degenerate: {violations}"
    assert notes["n_samples"] == 10_000


def test_t4_identity_exclusion_is_the_only_exclusion():
    """identity is excluded as a T4 target because it IS reverse@reverse, by design."""
    rng = np.random.default_rng(0)
    x = random_inputs(rng, 2_000)
    involutions = [p for p in range(1, K)
                   if np.array_equal(apply_composition((p, p), x), x)]
    assert involutions, "expected at least one involution justifying the exclusion"


# --------------------------------------------------------------------------
# T5 -- split constraints (and the static half of T3)
# --------------------------------------------------------------------------

def test_t5_split_constraints():
    split = load_split()
    problems = verify_split(split)
    assert problems == [], f"split violates constraints: {problems}"


def test_t5_split_sizes():
    split = load_split()
    assert split["n_train_pairs"] == 40
    assert split["n_heldout_pairs"] == 24
    train = {tuple(p) for p in split["train_pairs"]}
    heldout = {tuple(p) for p in split["heldout_pairs"]}
    assert not (train & heldout)
    assert len(train | heldout) == K * K


def test_t3_static_no_eval_input_equals_a_training_input_for_the_same_task():
    cfg = config_for_arm("A1", 0)
    cfg.examples_per_train_task = 200
    cfg.examples_per_eval_task = 100
    bundle = build_bundle(cfg)
    train_by_task = {td.task.task_id: {r.tobytes() for r in td.inputs}
                     for td in bundle.train}
    for td in bundle.seen_heldout + bundle.singleton:
        overlap = train_by_task[td.task.task_id] & {r.tobytes() for r in td.inputs}
        assert not overlap, f"{td.task.task_id}: {len(overlap)} eval inputs seen in training"
    for td in bundle.unseen:
        assert td.task.task_id not in train_by_task


def test_targets_match_ground_truth():
    cfg = config_for_arm("A1", 0)
    cfg.examples_per_train_task = 50
    cfg.examples_per_eval_task = 50
    bundle = build_bundle(cfg)
    for td in bundle.train + bundle.unseen:
        assert np.array_equal(apply_composition(td.task.primitives, td.inputs), td.targets)


# --------------------------------------------------------------------------
# T7 -- ablation sanity
# --------------------------------------------------------------------------

def _tiny_bundle_and_model():
    set_threads(4, 1)
    seed_everything(0)
    cfg = config_for_arm("A1", 0)
    cfg.examples_per_train_task = 20
    cfg.examples_per_eval_task = 20
    cfg.n_probe_examples = 20
    bundle = build_bundle(cfg)
    return cfg, bundle, AtomNet(cfg)


def test_t7_ablating_all_atoms_drops_to_chance():
    cfg, bundle, model = _tiny_bundle_and_model()
    model.eval()
    res = compute_ablation(model, bundle.unseen, cfg)
    # Exact match on 8 tokens over a 10-symbol vocabulary: chance is 1e-8.
    assert res["ablate_all_acc"].mean() <= 0.02, res["ablate_all_acc"].mean()


def test_t7_ablating_none_reproduces_logged_accuracy():
    """Ablating no atom must be bit-identical to the plain forward pass."""
    cfg, bundle, model = _tiny_bundle_and_model()
    model.eval()
    from e1.evaluate import _exact_match, _run_task
    for td in bundle.unseen[:4]:
        preds_plain, _, _ = _run_task(model, td, "hard", ablate=None)
        preds_noabl, _, _ = _run_task(
            model, td, "hard", ablate=torch.zeros(cfg.n_atoms, dtype=torch.bool))
        assert torch.equal(preds_plain, preds_noabl)
        assert _exact_match(preds_plain, td.targets).mean() == \
            _exact_match(preds_noabl, td.targets).mean()


# --------------------------------------------------------------------------
# T8 -- memory ceiling
# --------------------------------------------------------------------------

def test_t8_memory_ceiling_on_a_short_run():
    from e1.train import train
    tmp = Path(tempfile.mkdtemp(prefix="e1_t8_"))
    try:
        cfg = config_for_arm("A1", 0)
        cfg.examples_per_train_task = 200
        cfg.examples_per_eval_task = 60
        cfg.n_probe_examples = 60
        cfg.epochs = 1
        train(cfg, tmp, allow_dirty=True)  # fixture, not a research run
        import json
        env = json.loads((tmp / "env.json").read_text())
        assert env["peak_rss_gb"] < cfg.rss_fail_gb, env["peak_rss_gb"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rss_monitor_fails_loudly():
    mon = RSSMonitor(fail_gb=0.0001)
    try:
        mon.sample()
    except MemoryError:
        return
    raise AssertionError("RSSMonitor did not raise above its ceiling")


# --------------------------------------------------------------------------
# Architecture invariants
# --------------------------------------------------------------------------

def test_composer_size_is_independent_of_atom_count():
    """H5: content-based routing puts the per-atom cost in the library, not the composer."""
    sizes = {}
    for n in (4, 8, 16):
        cfg = config_for_arm("A1", 0)
        cfg.n_atoms = n
        sizes[n] = param_counts(AtomNet(cfg))
    composer = {s["composer"] for s in sizes.values()}
    assert len(composer) == 1, f"composer grew with N: {composer}"
    assert sizes[16]["atoms_total"] == 2 * sizes[8]["atoms_total"]


def test_atom_size_matches_spec():
    cfg = config_for_arm("A1", 0)
    pc = param_counts(AtomNet(cfg))
    assert pc["atoms_each"] == 262_912, pc["atoms_each"]
    assert pc["keys_each"] == 32
    assert pc["n_atoms"] == 8


def test_a3_phase2_leaves_atoms_bit_identical():
    """A3's structural guarantee: a frozen library cannot reshape to meet new atoms."""
    from e1.train import TrainArrays, run_phase
    set_threads(4, 1)
    seed_everything(0)
    cfg = config_for_arm("A3", 0)
    cfg.examples_per_train_task = 40
    cfg.examples_per_eval_task = 20
    bundle = build_bundle(cfg)
    model = AtomNet(cfg)

    before = [p.detach().clone() for p in model.atom_parameters()]
    for p in model.atom_parameters():
        p.requires_grad_(False)
    state = {"step": 0, "phase_counter": 0, "global_step_in_phase": 0}
    run_phase(model, cfg, TrainArrays(bundle.train, cfg), epochs=1,
              log=lambda r: None, state=state, include_atoms=False,
              phase_name="frozen_check")
    for b, p in zip(before, model.atom_parameters()):
        assert torch.equal(b, p.detach()), "frozen atom changed during phase 2"


def test_a3_phase1_only_the_new_atom_moves():
    from e1.train import TrainArrays, run_phase
    set_threads(4, 1)
    seed_everything(0)
    cfg = config_for_arm("A3", 0)
    cfg.examples_per_train_task = 40
    bundle = build_bundle(cfg)
    model = AtomNet(cfg)
    singles = sorted([td for td in bundle.train if td.task.kind == "singleton"],
                     key=lambda td: td.task.primitives[0])

    before = model.atoms.w1.detach().clone()
    mask = torch.zeros(cfg.n_atoms, dtype=torch.bool)
    mask[:2] = True
    state = {"step": 0, "phase_counter": 0, "global_step_in_phase": 0}
    run_phase(model, cfg, TrainArrays(singles[:2], cfg), epochs=2,
              log=lambda r: None, state=state, atom_mask=mask,
              trainable_atoms={1}, include_atoms=True, phase_name="stage1")
    after = model.atoms.w1.detach()
    assert not torch.equal(before[1], after[1]), "the trainable atom did not move"
    for i in [0] + list(range(2, cfg.n_atoms)):
        assert torch.equal(before[i], after[i]), f"frozen atom {i} moved"


def test_routing_modes_are_consistent():
    cfg, bundle, model = _tiny_bundle_and_model()
    model.eval()
    td = bundle.unseen[0]
    x = torch.from_numpy(td.inputs).long()
    instr = torch.tensor(td.task.instruction).unsqueeze(0).expand(x.shape[0], -1)
    with torch.no_grad():
        hard = model(x, instr, mode="hard")
        forced = model(x, instr, mode="forced",
                       forced=hard["routing_hard"])
    assert torch.equal(hard["routing_hard"], forced["routing_hard"])
    assert torch.allclose(hard["logits"], forced["logits"], atol=1e-5)


# Post-review regression tests (DECISIONS.md D32). Each pins one of the three
# defects found in external review so it cannot silently return.
# --------------------------------------------------------------------------

def test_b1_every_trainable_param_is_in_the_optimizer():
    """B1: state_norm was omitted from the optimizer but included in grad clipping.

    Its LayerNorm gains stayed frozen at init while its gradients inflated the
    global norm, shrinking the clip coefficient and suppressing every other
    parameter's update.
    """
    from e1.config import config_for_rung
    from e1.train import build_optimizer

    for rung, weight in [("R1", 0.0), ("R2", 10.0), ("R3", 0.0)]:
        cfg = config_for_rung(rung, 0, weight)
        model = AtomNet(cfg)
        assert model.state_norm is not None, f"{rung} should have state_norm"
        opt = build_optimizer(model, cfg, include_atoms=True)
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        missing = trainable - in_opt
        assert not missing, (
            f"{rung}: {len(missing)} trainable params absent from the optimizer "
            "while still entering clip_grad_norm_")


def test_b1_e1_arms_have_no_state_norm():
    """The B1 bug path cannot have executed for any E1 arm."""
    for arm in ("A0", "A1", "A2", "A3", "A3b", "A4"):
        cfg = config_for_arm(arm, 0)
        assert AtomNet(cfg).state_norm is None, f"{arm} unexpectedly has state_norm"


def test_b2_e1b_runs_are_excluded_from_the_e1_table():
    """B2: E1b cells keep arm="A1", so collect() must skip anything with a rung."""
    import json
    from e1.aggregate import collect

    tmp = Path(tempfile.mkdtemp(prefix="e1_b2_"))
    try:
        for name, extra in [("A1_0_test", {}), ("e1b_R2_w10_0_test", {"rung": "R2"})]:
            d = tmp / name / "artifacts"
            d.mkdir(parents=True)
            m = {"arm": "A1", "seed": 0,
                 "params": {"composer": 1, "atoms_total": 1, "encoder": 1,
                            "decoder": 1, "composer_over_atoms": 1.0}}
            m.update(extra)
            (tmp / name / "metrics.json").write_text(json.dumps(m))
        df = collect(tmp)
        ids = list(df["run_id"])
        assert ids == ["A1_0_test"], f"E1b run leaked into the E1 table: {ids}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_b3_r3_evaluation_is_deterministic():
    """B3: the R3 projection sampled Gumbel at eval, so every pass measured a
    different model (82% token disagreement). Eval must be deterministic; training
    must remain stochastic."""
    from e1.config import config_for_rung

    seed_everything(0)
    cfg = config_for_rung("R3", 0, 0.0)
    model = AtomNet(cfg)
    x = torch.randint(0, cfg.vocab, (32, cfg.seq_len))
    instr = torch.zeros(32, cfg.depth, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        a = model(x, instr, mode="hard")
        b = model(x, instr, mode="hard")
    assert torch.equal(a["states"], b["states"]), "R3 eval is not deterministic"
    assert torch.equal(a["logits"], b["logits"])

    model.train()
    with torch.no_grad():
        c = model(x, instr, mode="gumbel", tau=1.5)
        d = model(x, instr, mode="gumbel", tau=1.5)
    assert not torch.equal(c["states"], d["states"]), \
        "training projection should still sample"


def test_b3_non_bottleneck_rungs_were_never_affected():
    from e1.config import config_for_rung
    seed_everything(0)
    cfg = config_for_rung("R2", 0, 10.0)
    model = AtomNet(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab, (32, cfg.seq_len))
    instr = torch.zeros(32, cfg.depth, dtype=torch.long)
    with torch.no_grad():
        a = model(x, instr, mode="hard")
        b = model(x, instr, mode="hard")
    assert torch.equal(a["logits"], b["logits"])


def test_dirty_tree_guard_blocks_by_default():
    """Provenance: a dirty tree must refuse to train unless explicitly allowed."""
    from e1.utils import require_clean_tree
    import e1.utils as U

    real = U.git_info
    try:
        U.git_info = lambda: {"git_dirty": True, "git_sha": "x", "git_sha_short": "x"}
        try:
            require_clean_tree(False)
        except SystemExit:
            pass
        else:
            raise AssertionError("guard did not block on a dirty tree")
        assert require_clean_tree(True), "--allow-dirty should return a diff hash"
        U.git_info = lambda: {"git_dirty": False, "git_sha": "x", "git_sha_short": "x"}
        assert require_clean_tree(False) is None
    finally:
        U.git_info = real


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)


# --------------------------------------------------------------------------
