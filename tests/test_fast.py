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


def test_d43_batch_output_does_not_dirty_the_tree():
    """A batch's own artifacts must not stop the next run in that batch.

    `git status --porcelain` lists untracked run output, so the original guard made
    run #2 of ANY batch refuse to start. It was never caught because D33's guard
    postdates the only multi-run battery ever executed.
    """
    from e1.utils import source_dirt
    batch_output = "?? runs/v2/\n?? results/v2/summary.csv\n"
    assert source_dirt(batch_output) == [], source_dirt(batch_output)
    # ...but real source changes still count, including inside output dirs when the
    # file is TRACKED (a rewritten committed artifact must stop a run).
    for line in (" M e1/train.py", "?? e1/new_module.py", " M runs/v1/A0_0/metrics.json",
                 " M splits/pairs_split.json", "?? scratch.py"):
        assert source_dirt(line + "\n") == [line], line
    # Renames are parsed on the destination path.
    assert source_dirt("R  runs/old -> runs/new\n") == ["R  runs/old -> runs/new"]


def test_dirty_tree_guard_blocks_by_default():
    """Provenance: a dirty tree must refuse to train unless explicitly allowed."""
    from e1.utils import require_clean_tree
    import e1.utils as U

    def info(source_dirty, dirty=True):
        # The guard keys on git_source_dirty, not the raw git_dirty (D43): mid-batch
        # the raw flag is True purely because earlier runs wrote their artifacts.
        return lambda: {"git_dirty": dirty, "git_source_dirty": source_dirty,
                        "git_sha": "x", "git_sha_short": "x"}

    real = U.git_info
    try:
        U.git_info = info(source_dirty=True)
        try:
            require_clean_tree(False)
        except SystemExit:
            pass
        else:
            raise AssertionError("guard did not block on uncommitted source")
        assert require_clean_tree(True), "--allow-dirty should return a diff hash"
        U.git_info = info(source_dirty=False, dirty=False)
        assert require_clean_tree(False) is None
        # Mid-batch: raw tree dirty from output, source clean -> must still run.
        U.git_info = info(source_dirty=False, dirty=True)
        assert require_clean_tree(False) is None, \
            "a batch's own output must not block the next run"
    finally:
        U.git_info = real


# --------------------------------------------------------------------------
# D35/D36 regression -- the newest code carried no coverage at all
# --------------------------------------------------------------------------

def test_d35_r3_carries_the_fix_knobs():
    """R3 must run the FIXED configuration, not the pre-fix one.

    The three knobs default to off in Config so no other rung is disturbed. That
    left `config_for_rung("R3")` silently producing the pre-fix setup -- the one
    whose flat 0.7%-at-epoch-58 curve D35 exists to remove -- so an R3 run would
    have hit the D35 kill rule without ever testing the fix.
    """
    from e1.config import config_for_rung
    cfg = config_for_rung("R3", 0, 0.0)
    assert cfg.code_bottleneck is True
    assert cfg.codec_pretrain_epochs == 10, cfg.codec_pretrain_epochs
    assert cfg.codec_lr_scale == 0.1, cfg.codec_lr_scale
    assert cfg.project_tau_floor == 1.0, cfg.project_tau_floor
    # And the knobs must stay off everywhere else.
    for rung, w in (("R0", 0.0), ("R1", 0.0), ("R2", 10.0)):
        other = config_for_rung(rung, 0, w)
        assert other.codec_pretrain_epochs == 0, rung
        assert other.codec_lr_scale == 1.0, rung
        assert other.project_tau_floor == 0.0, rung


def test_d37_sarb_is_blocked():
    """S-arb must not run until its target design is replaced. See D37."""
    from e1.config import config_for_rung
    try:
        config_for_rung("Sarb", 0, 0.0)
    except NotImplementedError as exc:
        assert "D37" in str(exc)
    else:
        raise AssertionError("Sarb built a config; the D36 target is ill-posed")


def test_d37_arbitrary_targets_are_input_independent():
    """Pin the defect itself, so a future redesign cannot reintroduce it quietly.

    The D36 target is one constant vector per primitive. With depth=2, h_1 is the
    only path from input to output, so pulling h_1 onto a constant demands it carry
    zero information about x -- guaranteeing acc ~ 0 for a reason that has nothing
    to do with semantic correctness. Any replacement MUST make this test fail.
    """
    from e1.config import Config
    from e1.model import AtomNet
    seed_everything(0)
    cfg = Config(arm="A1", seed=0, atom_layernorm=True, arbitrary_targets=True)
    model = AtomNet(cfg)
    model.init_arbitrary_targets(cfg.split_seed)
    targets = model.arbitrary_targets
    assert targets.shape == (cfg.n_primitives, cfg.state_dim)
    same_primitive = targets[torch.full((6,), 3)]
    spread = float((same_primitive - same_primitive[0]).abs().max())
    assert spread == 0.0, (
        "targets now vary with the input -- if this is the intended redesign, "
        "supersede D37 and re-register a prediction before running the rung"
    )


def test_d37_unschedulable_rung_is_not_a_silent_no_op():
    """`--rungs X` filters LADDER, so an off-ladder name planned zero runs and

    exited 0 -- success and "did nothing" were indistinguishable.
    """
    import subprocess
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "e1.run_e1b", "--rungs", "Sarb", "--plan"],
        cwd=str(root), capture_output=True, text=True,
    )
    assert proc.returncode != 0, "off-ladder rung exited 0 instead of failing"
    assert "no ladder cell" in (proc.stderr + proc.stdout)


# --------------------------------------------------------------------------
# D40 -- generations: v1 immutable, v2 sound, the two never mixed
# --------------------------------------------------------------------------

V1_SPLIT_SHA = "a7b1ca7cca42e242a604c0c8541c3b248119665d4afd662948675a4afae0eabb"


def test_d40_v1_primitive_set_is_frozen():
    """v1's primitive table is load-bearing for 30 committed runs.

    Every v1 artifact, and the sha256 in every v1 split_ref.json, is a fact about
    exactly this table in exactly this order. Ids are positional, so reordering it
    would silently relabel every task in the E1 report.
    """
    from e1.primitives import PRIMITIVE_SETS, primitive_names
    assert primitive_names("v1") == (
        "identity", "reverse", "increment", "sort_asc",
        "rotate_left", "swap_halves", "double", "reflect",
    )
    # The default must stay v1: every un-parameterised call site resolves through it.
    from e1.primitives import DEFAULT_SET, PRIMITIVES
    assert DEFAULT_SET == "v1"
    assert PRIMITIVES is PRIMITIVE_SETS["v1"]


def test_d40_v1_split_hash_is_unchanged():
    """The frozen split must survive every refactor byte-for-byte."""
    from e1.data import split_hash, split_path_for
    assert split_hash(split_path_for("v1", 1234)) == V1_SPLIT_SHA


def test_d40_v2_differs_from_v1_in_exactly_one_slot():
    from e1.primitives import primitive_names
    v1, v2 = primitive_names("v1"), primitive_names("v2")
    differing = [i for i, (a, b) in enumerate(zip(v1, v2)) if a != b]
    assert differing == [3], differing
    assert v2[3] == "index_shift"
    assert v1[0] == v2[0] == "identity", "the identity slot must be shared"


def test_d40_v2_primitives_pass_t4_and_raise_resolution():
    """The reason for the swap: sort_asc is idempotent and order-destroying (D10)."""
    from e1.primitives import (
        apply_primitive, check_primitive_independence, distinct_pair_functions,
        random_inputs,
    )
    violations, _ = check_primitive_independence(2000, seed=0, pset="v2")
    assert violations == [], violations
    n_v1 = len(distinct_pair_functions(2000, pset="v1"))
    n_v2 = len(distinct_pair_functions(2000, pset="v2"))
    assert n_v1 == 39 and n_v2 == 42, (n_v1, n_v2)
    x = random_inputs(np.random.default_rng(0), 500)
    once = apply_primitive(3, x, "v2")
    assert not np.array_equal(apply_primitive(3, once, "v2"), once), \
        "index_shift must be non-idempotent"


def test_d40_v2_splits_exist_and_are_distinct():
    from e1.config import GENERATIONS
    from e1.data import load_split, split_path_for, verify_split
    heldouts = []
    for ss in GENERATIONS["v2"]["split_seeds"]:
        path = split_path_for("v2", ss)
        assert path.exists(), f"{path} missing -- run `make_split --generation v2`"
        split = load_split(path)
        assert verify_split(split, "v2") == []
        assert split["primitive_set"] == "v2"
        heldouts.append({tuple(p) for p in split["heldout_pairs"]})
    # Three splits that mostly agree would not be three independent samples of the
    # task space, which is the whole reason for having more than one (E1_REPORT 6b).
    for i in range(len(heldouts)):
        for j in range(i + 1, len(heldouts)):
            shared = len(heldouts[i] & heldouts[j])
            assert shared < 20, f"splits {i},{j} share {shared}/24 held-out pairs"


def test_d40_split_and_primitive_set_cannot_be_mismatched():
    """A v1 split used with v2 functions would generate targets from the wrong maps."""
    from e1.config import Config
    from e1.data import build_bundle, load_split, split_path_for, verify_split
    v1_split = load_split(split_path_for("v1", 1234))
    assert verify_split(v1_split, "v2"), "mismatch was not reported"
    assert verify_split(v1_split, "v1") == []
    cfg = Config(arm="A1", seed=0, generation="v2", primitive_set="v2",
                 examples_per_train_task=2, examples_per_eval_task=2,
                 n_probe_examples=2)
    try:
        build_bundle(cfg, v1_split)
    except ValueError as exc:
        assert "primitive set" in str(exc)
    else:
        raise AssertionError("build_bundle accepted a mismatched split")


def test_d40_generation_config_rejects_unfrozen_split_seed():
    from e1.config import config_for_generation
    cfg = config_for_generation("v2", "A1", 0, 5678)
    assert cfg.primitive_set == "v2" and cfg.split_seed == 5678
    for bad in (("v3", "A1", 0, None), ("v2", "A1", 0, 4321)):
        try:
            config_for_generation(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")


def test_d40_run_dirs_separate_generations_and_split_seeds():
    from e1.config import config_for_generation
    from e1.train import run_dir_for
    v1 = run_dir_for(config_for_generation("v1", "A1", 0))
    a = run_dir_for(config_for_generation("v2", "A1", 0, 1234))
    b = run_dir_for(config_for_generation("v2", "A1", 0, 5678))
    assert v1.parent.name == "v1" and a.parent.name == "v2"
    # Same arm and seed on two splits must not collide -- otherwise the second run
    # silently overwrites the first.
    assert a != b, (a, b)
    assert "s1234" in a.name and "s5678" in b.name
    # v1 has one split, so its ids keep their historical shape.
    assert "_s" not in v1.name


def test_d40_aggregate_never_mixes_generations():
    """v1 and v2 measure different task families; a pooled mean describes neither."""
    import json
    from e1.aggregate import collect
    tmp = Path(tempfile.mkdtemp())
    try:
        for gen, acc in (("v1", 0.10), ("v2", 0.90)):
            d = tmp / gen / f"A1_0_{gen}"
            d.mkdir(parents=True)
            (d / "config.json").write_text(json.dumps({"generation": gen}))
            (d / "metrics.json").write_text(json.dumps({
                "arm": "A1", "seed": 0, "M1_acc_unseen": acc,
                "params": {"composer": 1, "atoms_total": 1, "encoder": 1,
                           "decoder": 1, "composer_over_atoms": 1.0},
            }))
        for gen, acc in (("v1", 0.10), ("v2", 0.90)):
            df = collect(tmp, generation=gen)
            assert len(df) == 1, df
            assert float(df["M1_acc_unseen"].iloc[0]) == acc
            assert set(df["generation"]) == {gen}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_d40_make_split_refuses_to_regenerate_v1():
    """v1's split hash is pinned by 30 committed runs; --force must not touch it."""
    from e1.data import split_hash, split_path_for
    from e1.make_split import main as make_split_main
    before = split_hash(split_path_for("v1", 1234))
    assert make_split_main(["--generation", "v1", "--force"]) == 0
    assert split_hash(split_path_for("v1", 1234)) == before == V1_SPLIT_SHA


# --------------------------------------------------------------------------
# D41 -- archived batteries are frozen and never pooled
# --------------------------------------------------------------------------

def _fake_run(d: Path, *, generation="v1", host="Perro", acc=0.5):
    import json
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"generation": generation}))
    (d / "env.json").write_text(json.dumps({"hostname": host}))
    (d / "metrics.json").write_text(json.dumps({
        "arm": "A1", "seed": 0, "M1_acc_unseen": acc,
        "params": {"composer": 1, "atoms_total": 1, "encoder": 1,
                   "decoder": 1, "composer_over_atoms": 1.0},
    }))


def test_d41_archived_runs_are_skipped_by_default():
    from e1.aggregate import collect
    tmp = Path(tempfile.mkdtemp())
    try:
        _fake_run(tmp / "archive_perro_v1" / "A1_0_old", host="Perro", acc=0.11)
        _fake_run(tmp / "v1" / "A1_0_new", host="Perrito", acc=0.99)
        df = collect(tmp, generation="v1")
        assert len(df) == 1 and float(df["M1_acc_unseen"].iloc[0]) == 0.99, df
        # ...but pointing --runs AT the archive reads it: opting in is explicit.
        arch = collect(tmp / "archive_perro_v1", generation="v1")
        assert len(arch) == 1 and float(arch["M1_acc_unseen"].iloc[0]) == 0.11
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_d39_aggregation_refuses_to_span_two_hosts():
    """Determinism is a within-platform guarantee; mixing hosts is not a comparison."""
    from e1.aggregate import collect
    tmp = Path(tempfile.mkdtemp())
    try:
        _fake_run(tmp / "v1" / "A1_0_a", host="Perro")
        _fake_run(tmp / "v1" / "A1_1_b", host="Perrito")
        try:
            collect(tmp, generation="v1")
        except SystemExit as exc:
            assert "D39" in str(exc) and "Perro" in str(exc)
        else:
            raise AssertionError("aggregated across two hosts")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_d41_archive_layout_is_intact():
    """The committed v1 record must stay where the report says it is."""
    root = Path(__file__).resolve().parents[1]
    runs = root / "runs" / "archive_perro_v1"
    res = root / "results" / "archive_perro_v1"
    assert runs.is_dir() and res.is_dir()
    assert len([d for d in runs.iterdir() if (d / "metrics.json").exists()]) == 36
    assert (res / "E1_REPORT.md").exists()
    assert (res / "summary.csv").exists()
    # Checkpoints must stay ignored at the deeper path, or a batch commits 90 of them.
    ignore = (root / ".gitignore").read_text()
    assert "runs/**/checkpoints/" in ignore, "checkpoint ignore is not recursive"


def test_d42_seed_counts_are_per_generation():
    """v1's five seeds stay pinned; v2 is 3 seeds x 3 splits = 9 runs per arm."""
    from e1.config import GENERATIONS, seeds_for
    assert seeds_for("v1") == (0, 1, 2, 3, 4)
    assert seeds_for("v2") == (0, 1, 2)
    per_arm = {g: len(s["seeds"]) * len(s["split_seeds"])
               for g, s in GENERATIONS.items()}
    assert per_arm["v1"] == 5
    assert per_arm["v2"] == 9, per_arm
    # The whole justification for the trade is that v2 has MORE runs per arm.
    assert per_arm["v2"] > per_arm["v1"]


def test_d42_run_all_plans_the_registered_batch():
    """A stale global seed list would silently plan the wrong batch size.

    Uses --plan, which must print the batch and exit WITHOUT training -- a test that
    shells out to run_all without it would launch the real 54-run battery.
    """
    import subprocess
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "e1.run_all", "--generation", "v2", "--plan"],
        cwd=str(root), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[:400]
    assert "generation v2: 54 runs" in proc.stdout, proc.stdout[:400]
    assert "9 per arm" in proc.stdout, proc.stdout[:400]
    # One indented line per planned run (the header also says "splits", so match on
    # the indent rather than the word), and no run may be started by --plan.
    planned = [l for l in proc.stdout.splitlines() if l.startswith("  ") and "seed" in l]
    assert len(planned) == 54, len(planned)
    v1 = subprocess.run(
        [sys.executable, "-m", "e1.run_all", "--generation", "v1", "--plan"],
        cwd=str(root), capture_output=True, text=True, timeout=120,
    )
    assert "generation v1: 30 runs" in v1.stdout, v1.stdout[:400]


# --------------------------------------------------------------------------
# D44 -- every ground-truth computation must honour cfg.primitive_set
# --------------------------------------------------------------------------

def test_d44_no_bare_primitive_calls_in_the_package():
    """Static guard: `apply_primitive`/`apply_composition` must be passed a pset.

    evaluate.py called them with two arguments, so every diagnostic silently scored
    v2 runs against v1 functions. Only slot 3 differs, so the corruption was confined
    to one primitive -- which is exactly why it produced clean-looking k/8 fractions
    rather than obvious garbage. A behavioural test catches today's call sites; this
    catches the ones added tomorrow.
    """
    import ast
    pkg = Path(__file__).resolve().parents[1] / "e1"
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        if path.name == "primitives.py":       # where they are defined
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name in ("apply_primitive", "apply_composition"):
                if len(node.args) < 3:
                    offenders.append(f"{path.name}:{node.lineno} {name} "
                                     f"({len(node.args)} args, needs pset)")
    assert not offenders, "ground truth computed from the default primitive set:\n" \
        + "\n".join(offenders)


def test_d44_diagnostics_actually_depend_on_the_primitive_set():
    """Behavioural: identical weights must score differently under v1 vs v2.

    The two sets differ only in slot 3, so a diagnostic that ignores `primitive_set`
    returns byte-identical output for both -- which is the signature this asserts
    against. Complements the static check: this one fails if a call site threads the
    argument but the value never reaches the ground truth.
    """
    import dataclasses
    from e1.config import Config
    from e1.evaluate import compute_state_alignment
    from e1.model import AtomNet
    from e1.primitives import apply_primitive, random_inputs

    seed_everything(0)
    cfg1 = Config(arm="A1", seed=0, generation="v1", primitive_set="v1",
                  n_probe_examples=64)
    cfg2 = dataclasses.replace(cfg1, generation="v2", primitive_set="v2")
    model = AtomNet(cfg1)          # ONE model; only the scoring set changes
    model.eval()
    probe = random_inputs(np.random.default_rng(0), 64)

    # The two sets must actually differ on slot 3, or the test proves nothing.
    assert not np.array_equal(apply_primitive(3, probe, "v1"),
                              apply_primitive(3, probe, "v2"))

    # Closed-map error is a CONTINUOUS distance to enc(p(x)), so it responds to the
    # target changing even on an untrained model. Exact-match alignment does not:
    # atoms initialise near zero, so h0+atom(h0) ~ h0 and every atom reads as
    # identity whatever the targets are (D24) -- a degenerate probe, not a signal.
    err1, _ = compute_state_alignment(model, probe, cfg1)
    err2, _ = compute_state_alignment(model, probe, cfg2)
    assert not np.array_equal(err1, err2), \
        "compute_state_alignment ignores primitive_set"
    # Only slot 3's column may move; if others do, the generations differ by more
    # than one primitive and are not comparable at all.
    shared = [p for p in range(cfg1.n_primitives) if p != 3]
    assert np.array_equal(err1[:, shared], err2[:, shared]), \
        "v1 and v2 disagree outside slot 3"
    assert not np.array_equal(err1[:, 3], err2[:, 3]), "slot 3 column did not move"


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
