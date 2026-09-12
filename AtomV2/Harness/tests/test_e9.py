"""E9 vortex-crystallization registration, copy, loss, and audit gates."""
import torch
import torch.nn.functional as F

from atomv2 import registered as R
from atomv2.config import config_for_arm
from atomv2.crystallization import (
    copy_teacher_to_student,
    distillation_losses,
    gradient_concentration,
    measure,
    prepare_student,
    tensor_concentration,
)
from atomv2.model import AtomModel
from atomv2.run_e9 import screen_verdict


def test_e9_configs_are_one_step_and_separate_the_cause():
    a25 = config_for_arm("A25", 1)
    a26 = config_for_arm("A26", 1)
    a27 = config_for_arm("A27", 1)
    for cfg in (a25, a26, a27):
        assert cfg.experiment == "e9"
        assert cfg.protocol_revision == R.E9_PROTOCOL_REVISION
        assert cfg.micro_steps == 1
        assert cfg.crystal_teacher_experiment == "e0"
        assert cfg.crystal_teacher_arm == "A0-free"
    assert not a25.crystal_warm_start
    assert a26.crystal_warm_start and a27.crystal_warm_start
    assert a25.lambda_crystal_state == a26.lambda_crystal_state == 0.0
    assert a25.lambda_crystal_logits == a26.lambda_crystal_logits == 0.0
    assert a27.lambda_crystal_state == 1.0
    assert a27.lambda_crystal_logits == 0.25
    assert config_for_arm("A27", 0, smoke=True).crystal_force_ramp_steps == 50


def test_teacher_copy_has_one_explicit_microstep_slice():
    torch.manual_seed(4)
    teacher = AtomModel(config_for_arm("A0-free", 0, smoke=True))
    torch.manual_seed(8)
    student = AtomModel(config_for_arm("A26", 0, smoke=True))
    receipt = copy_teacher_to_student(student, teacher.state_dict())
    assert receipt["n_copied_elements"] == receipt["n_student_elements"]
    assert [x["key"] for x in receipt["sliced_tensors"]] == [
        "composer.micro_emb.weight"]
    assert torch.equal(student.composer.micro_emb.weight,
                       teacher.composer.micro_emb.weight[:1])
    for key, value in student.state_dict().items():
        if key != "composer.micro_emb.weight":
            assert torch.equal(value, teacher.state_dict()[key]), key


def test_prepare_pins_teacher_and_freezes_it(tmp_path):
    tcfg = config_for_arm("A0-free", 0, smoke=True)
    teacher = AtomModel(tcfg)
    ckpt = tmp_path / "run" / "checkpoints" / "final.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"model": teacher.state_dict(), "config": tcfg.to_dict(),
                "step": 7}, ckpt)
    scfg = config_for_arm("A27", 0, smoke=True)
    student = AtomModel(scfg)
    frozen, receipt = prepare_student(
        student, scfg, teacher_checkpoint=ckpt,
        require_teacher_metrics=False)
    assert frozen is not None and not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    assert receipt["teacher_checkpoint_sha256"]
    assert receipt["mode"] == "teacher_forced"
    assert receipt["copy"]["n_copied_elements"] == receipt["copy"][
        "n_student_elements"]


def test_distillation_loss_connects_only_the_student():
    tcfg = config_for_arm("A0-free", 0, smoke=True)
    scfg = config_for_arm("A27", 0, smoke=True)
    teacher = AtomModel(tcfg)
    student = AtomModel(scfg)
    x = torch.randint(0, 10, (5, 6))
    tokens = torch.randint(0, 8, (5, 2))
    n_tokens = torch.tensor([1, 2, 1, 2, 2])
    with torch.no_grad():
        tout = teacher(x, tokens, n_tokens, mode="hard")
    sout = student(x, tokens, n_tokens, mode="soft")
    losses = distillation_losses(sout, tout, n_tokens, 1, 3)
    total = losses["loss_crystal_state"] + losses["loss_crystal_logits"]
    total.backward()
    assert torch.isfinite(total)
    assert any(p.grad is not None for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_concentration_metrics_have_known_endpoints():
    uniform = tensor_concentration(torch.ones(3, 4))
    core = tensor_concentration(torch.tensor([[2.0, 0.0, 0.0, 0.0]]))
    assert uniform["energy_l2_sq_mean"] == 4.0
    assert uniform["peak_energy_fraction_mean"] == 0.25
    assert uniform["effective_coordinates_mean"] == 4.0
    assert core["peak_energy_fraction_mean"] == 1.0
    assert core["effective_coordinates_mean"] == 1.0


def test_one_step_measurement_and_gradient_partition():
    cfg = config_for_arm("A25", 0, smoke=True)
    model = AtomModel(cfg)
    diag = {
        "x": torch.randint(0, 10, (6, 6)).numpy(),
        "y": torch.randint(0, 10, (6, 6)).numpy(),
        "tokens": torch.randint(0, 8, (6, 2)).numpy(),
        "n_tokens": torch.tensor([1, 1, 2, 2, 2, 1]).numpy(),
    }
    rec = measure(model, diag, step=0)
    assert rec["routing"]["one_route_per_token"]
    assert rec["routing"]["n_live_decisions"] == sum(diag["n_tokens"])
    x = torch.from_numpy(diag["x"])
    tokens = torch.from_numpy(diag["tokens"])
    n_tokens = torch.from_numpy(diag["n_tokens"])
    out = model(x, tokens, n_tokens, mode="soft")
    F.cross_entropy(out["logits"].reshape(-1, 10),
                    torch.from_numpy(diag["y"]).reshape(-1)).backward()
    grad = gradient_concentration(model, cfg.grad_clip)
    assert grad["grad_global_norm_postclip"] <= cfg.grad_clip
    assert 0.0 <= grad["grad_atom_peak_fraction_within_atoms"] <= 1.0
    assert 0.0 <= grad["grad_atom_effective_support"] <= cfg.n_atoms


def _rows(a25, a26, a27):
    return [
        {"arm": "A25", "acc_seen_hard": a25[0],
         "acc_unseen_L1_hard": a25[1], "e9_one_route_per_token": True},
        {"arm": "A26", "acc_seen_hard": a26[0],
         "acc_unseen_L1_hard": a26[1], "e9_one_route_per_token": True},
        {"arm": "A27", "acc_seen_hard": a27[0],
         "acc_unseen_L1_hard": a27[1], "e9_one_route_per_token": True},
    ]


def test_screen_verdict_attributes_the_first_sufficient_mechanism():
    teacher = {"arm": "A0-free", "seed": 1, "acc_seen_hard": 1.0,
               "acc_unseen_L1_hard": 0.8}
    v = screen_verdict(
        _rows((0.2, 0.0), (0.5, 0.2), (0.95, 0.70)), teacher, "teacher")
    assert v["outcome"] == "CONTROLLED_CRYSTALLIZATION"
    assert v["registered_winner"] == "A27"
    v = screen_verdict(
        _rows((0.95, 0.70), (0.95, 0.70), (0.95, 0.70)), teacher, "teacher")
    assert v["outcome"] == "ONE_STEP_BASE_LEARNS"
