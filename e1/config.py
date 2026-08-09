"""Resolved configuration for E1. Every default is explicit and gets written to config.json."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

ARMS = ("A0", "A1", "A2", "A3", "A3b", "A4")
SEEDS = (0, 1, 2, 3, 4)


@dataclass
class Config:
    # --- identity -------------------------------------------------------
    arm: str = "A1"
    seed: int = 0
    # Generation: which *design* of the experiment this run belongs to. Both
    # generations live in one working tree and run from one checkout -- switching
    # generations must never require switching commits, or old results stop being
    # reproducible the moment the code moves on. See D40.
    #   v1  the committed E1 battery: sort_asc in slot 3, one split (seed 1234)
    #   v2  index_shift in slot 3 (39 -> 42 distinct pair functions), 3 split seeds
    generation: str = "v1"
    primitive_set: str = "v1"

    # --- data -----------------------------------------------------------
    seq_len: int = 8
    vocab: int = 10
    n_primitives: int = 8
    n_train_pairs: int = 40
    n_heldout_pairs: int = 24
    examples_per_train_task: int = 1000
    examples_per_eval_task: int = 400
    n_probe_examples: int = 400
    split_seed: int = 1234
    data_seed: int = 20250101

    # --- architecture ---------------------------------------------------
    d_model: int = 64
    n_heads: int = 4
    ffn_dim: int = 256
    n_atoms: int = 8
    atom_hidden: int = 256
    d_key: int = 32
    composer_hidden: int = 64
    composer_instr_dim: int = 32
    composer_state_dim: int = 32
    depth: int = 2  # T, max composition depth

    # --- training -------------------------------------------------------
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 80          # raised from the spec's 30 per §12 item 3; see D17
    grad_clip: float = 1.0
    tau_start: float = 2.0
    tau_end: float = 0.5
    early_stop: bool = True
    early_stop_acc: float = 0.99
    early_stop_patience: int = 3
    log_every: int = 50

    # --- arm-specific ---------------------------------------------------
    atom_dropout: float = 0.15          # A2, A4
    reshuffle_each_epoch: bool = False  # A2, A4 -> True; A1 keeps a fixed order
    forced_routing: bool = True         # A0 only
    routing_ce_weight: float = 1.0      # A0 only
    intermediate_supervision: bool = False  # A0 only
    state_consistency: bool = False         # A0 only
    state_consistency_weight: float = 40.0   # D18: final authorised step
    sequential: bool = False            # A3, A3b
    sequential_forced_assignment: bool = False   # A3b only; see D23
    # A3 phase 1 is budgeted in optimizer STEPS, not epochs: stage i trains on
    # (i+1)*1000 examples, so a fixed epoch count would have given atom 0 only 42
    # steps and the whole library 1,662 against A0's 30,000. See D19.
    seq_steps_per_atom: int = 3000      # A3 phase 1, per stage (early stopping applies)
    shuffle_labels: bool = False        # A4

    # --- E1b manifold ladder (all default off, so E1 behaviour is unchanged) ---
    rung: str = ""                          # "R0".."R3", "Sarb"; empty = an E1 arm
    atom_layernorm: bool = False            # R1+
    code_consistency_weight: float = 0.0    # R2+
    code_bottleneck: bool = False           # R3

    # --- R3-fixed: make the structural bottleneck trainable (D35) -------------
    # The projection routes through the decoder, so an untrained codec means atoms
    # learn to write through a transcriber that mis-transcribes and the two corrupt
    # each other's signal. Phase 0 trains the codec alone first.
    codec_pretrain_epochs: int = 0          # phase 0: enc+dec reconstruction only
    codec_lr_scale: float = 1.0             # enc/dec/state_norm lr multiplier after phase 0
    project_tau_floor: float = 0.0          # projection uses max(tau, floor)

    # --- S-arb: consistency without correctness (D36) -------------------------
    arbitrary_targets: bool = False         # pull h_1 to a frozen per-primitive code

    # --- execution ------------------------------------------------------
    num_threads: int = 4
    num_interop_threads: int = 1
    rss_fail_gb: float = 4.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Rebuild from a saved config.json, tolerating schema drift.

        Fields that no longer exist are dropped and fields added since the run was
        written take their defaults, so an old run stays loadable for re-analysis.
        Dropped keys are surfaced rather than silently ignored.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            print(f"[config] ignoring fields absent from the current schema: {unknown}")
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def state_dim(self) -> int:
        return self.d_model * self.seq_len


def config_for_arm(arm: str, seed: int) -> Config:
    """Arms differ ONLY in training procedure -- architecture, data and splits are shared."""
    cfg = Config(arm=arm, seed=seed)
    cfg.forced_routing = False
    cfg.atom_dropout = 0.0
    cfg.reshuffle_each_epoch = False
    cfg.intermediate_supervision = False
    cfg.state_consistency = False
    cfg.early_stop = True
    cfg.sequential_forced_assignment = False
    cfg.arbitrary_targets = False

    if arm == "A0":
        # Oracle: routing forced to the ground-truth primitive at every step, AND
        # every intermediate state supervised to decode to the partial composition.
        # That second signal is what makes each atom a *closed* map on the latent
        # code, which is the property a hand-built library would have had.
        cfg.forced_routing = True
        cfg.intermediate_supervision = True
        # Every intermediate state must EQUAL the encoding of its partial
        # composition, not merely decode to it. This is what makes atom_i the
        # closed map enc(x) -> enc(p_i(x)), i.e. the hand-built library of A0's
        # original description, and it is what makes composition generalise.
        cfg.state_consistency = True
        # A0's objective carries auxiliary terms that task accuracy does not track:
        # it can reach 0.99 train accuracy while state consistency is still
        # converging, and stopping there leaves the drift that breaks composition.
        cfg.early_stop = False
    elif arm == "A1":
        # Naive joint. Fixed co-occurrence order, no countermeasures.
        pass
    elif arm == "A2":
        cfg.atom_dropout = 0.15
        cfg.reshuffle_each_epoch = True
    elif arm == "A3":
        cfg.sequential = True
    elif arm == "A3b":
        # A3 as spec 4 intended it. A3's free routing does not in fact discover one
        # atom per primitive -- it routes (p, identity) to (atom_p, atom_p) and builds
        # half-primitives with no identity slot (D21). A3b forces the assignment so
        # phase 2 tests the question A3 was built for: does a CLEAN frozen library
        # compose? See D23.
        cfg.sequential = True
        cfg.sequential_forced_assignment = True
    elif arm == "A4":
        cfg.atom_dropout = 0.15
        cfg.reshuffle_each_epoch = True
        cfg.shuffle_labels = True
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return cfg


# --- generations ----------------------------------------------------------
# A generation fixes the task family and the split policy. Everything else -- arms,
# architecture, optimiser, thresholds -- is shared, which is what keeps v1 and v2
# comparable and keeps a single `analyze.py` scoring both.
GENERATIONS = {
    "v1": {
        "primitive_set": "v1",
        "split_seeds": (1234,),
        "note": "committed E1 battery; sort_asc in slot 3; ONE split, inspected "
                "during development (E1_REPORT 6b) -- development evidence",
    },
    "v2": {
        "primitive_set": "v2",
        "split_seeds": (1234, 5678, 9012),
        "note": "index_shift replaces sort_asc (39 -> 42 distinct pair functions, "
                "0 T4 violations); three independent splits so the reported spread "
                "covers split variance, not just optimisation variance",
    },
}


def config_for_generation(generation: str, arm: str, seed: int,
                          split_seed: int | None = None) -> Config:
    """An arm config bound to a generation's task family and one of its splits.

    `split_seed` must be one the generation actually froze: a run pointing at an
    ungenerated split would otherwise fail deep inside data loading, or worse, load a
    neighbouring generation's file.
    """
    try:
        spec = GENERATIONS[generation]
    except KeyError:
        raise ValueError(
            f"unknown generation {generation!r}; known: {sorted(GENERATIONS)}"
        ) from None
    seeds = spec["split_seeds"]
    if split_seed is None:
        split_seed = seeds[0]
    if split_seed not in seeds:
        raise ValueError(
            f"generation {generation!r} froze split seeds {list(seeds)}, "
            f"not {split_seed}"
        )
    cfg = config_for_arm(arm, seed)
    cfg.generation = generation
    cfg.primitive_set = spec["primitive_set"]
    cfg.split_seed = split_seed
    return cfg


RUNGS = ("R0", "R1", "R2", "R3", "Sarb")
R2_WEIGHTS = (1.0, 10.0, 40.0)


def config_for_rung(rung: str, seed: int, weight: float = 0.0) -> Config:
    """E1b: A1 (naive joint) plus one cumulative on-manifold constraint.

    A1 is the base because it has no dropout, no curriculum and no countermeasures,
    so any recovery is attributable to the rung's constraint alone. Everything else --
    architecture, optimizer, schedule, split, data, epoch budget, thread pinning --
    is identical to the E1 battery.
    """
    cfg = config_for_arm("A1", seed)
    cfg.rung = rung
    if rung == "R0":
        pass                                    # unchanged A1; the harness check
    elif rung == "R1":
        cfg.atom_layernorm = True
    elif rung == "R2":
        cfg.atom_layernorm = True
        cfg.code_consistency_weight = weight
    elif rung == "R3":
        # Bottleneck ONLY -- deliberately NOT cumulative with R2's penalty (D30).
        # The projection enforces code validity structurally, so the gradient penalty
        # is redundant here; worse, it is the term that collapsed the code at w=10, and
        # inheriting it would produce a failure misattributed to the bottleneck.
        # This is also the more faithful reading of the spec, which defines R3 as
        # "R2's *architecture*" -- a loss term is not architecture.
        cfg.atom_layernorm = True
        cfg.code_consistency_weight = weight   # ladder passes 0.0; see run_e1b.LADDER
        cfg.code_bottleneck = True
        # D35's three knobs. They default to off in Config so no other rung is
        # touched, which meant R3 was still running the PRE-FIX configuration --
        # the one that produced the flat 0.7%-at-epoch-58 curve the fix exists to
        # remove. Wiring them here is what makes "R3-fixed" actually fixed.
        cfg.codec_pretrain_epochs = 10   # phase 0: encoder+decoder reconstruction only
        cfg.codec_lr_scale = 0.1         # codec becomes a slow-moving substrate after
        cfg.project_tau_floor = 1.0      # routing anneal must not close the proj channel
    elif rung == "Sarb":
        # BLOCKED -- the D36 design is ill-posed and would return a confident wrong
        # answer. See D37. `arbitrary_targets` is one CONSTANT vector per primitive
        # (verified: shape [8, 512], byte-identical across inputs), so the constraint
        # "h_1 == T[p_1] for all x" demands h_1 carry zero information about x. With
        # depth=2, h_1 is the only path from input to output, so at weight 40 the run
        # is guaranteed to land at acc_unseen ~ 0 with loss_state_rel low -- which is
        # exactly D36's registered signature for "semantic correctness is
        # load-bearing", and would have redirected the program to prospectus 10 on
        # the strength of an information-destroying target. Same family as D29.
        #
        # The replacement is not a patch: D37 shows any admissible target must be
        # input-dependent AND determine p_1(x), which forces correctness up to
        # isomorphism. Do not re-enable without a superseding decision entry and a
        # freshly registered prediction.
        raise NotImplementedError(
            "rung 'Sarb' is blocked: the D36 target is input-independent and "
            "guarantees an uninterpretable failure. See DECISIONS.md D37."
        )
    else:
        raise ValueError(f"unknown rung {rung!r}")
    return cfg


# --- E1b pre-registered interpretation (frozen before the first E1b run) ---
# Judged on the mean over 5 seeds at the BEST rung. Not to be moved after seeing
# results; ambiguity is resolved by iterating on the training procedure.
#
# The closed-map gate keys on the MATCHED variant plus a coverage floor, not on the
# bare error. D24 showed the bare error is near-zero for a dead library -- every atom
# trivially "implements identity" -- so a bare-error gate could be cleared by a
# library that does nothing. The `acc_unseen` conjunction would have caught that, but
# by luck rather than design; this makes it explicit. See D24, D26.
E1B_THRESHOLDS = {
    "recovers": {
        "M3_closed_map_error_matched": ("le", 0.15),
        "M3_closed_map_coverage": ("ge", 6),
        "M1_acc_unseen": ("ge", 0.50),
    },
    "partial": {
        "M3_closed_map_error_matched": ("le", 0.30),
        "M3_closed_map_coverage": ("ge", 4),
        "M1_acc_unseen": ("ge", 0.15),
    },
    # DOES NOT RECOVER requires code_residual to be LOW -- the constraint bound and
    # still did not help. If it stays high the result is INCONCLUSIVE: a
    # training-procedure problem, not a finding. Judged on R2 ONLY: R3 projects
    # through the code, so its residual is near-zero by construction and cannot
    # distinguish "bound" from "learned nothing" (D25).
    "code_bound_max": 0.15,
    "code_bound_judged_on": "R2",
}


# --- Pre-registered thresholds (frozen before the first training run) -----
# Judged on the mean across 5 seeds. PASS requires ALL six.
THRESHOLDS = {
    "M1_acc_unseen": {"pass": ("ge", 0.85), "fail": ("le", 0.50)},
    "M1_gap":        {"pass": ("le", 0.05), "fail": ("ge", 0.20)},
    "M2_cv":         {"pass": ("le", 0.35), "fail": ("ge", 0.75)},
    "M3_align":      {"pass": ("ge", 0.85), "fail": ("le", 0.50)},
    "M3_purity":     {"pass": ("ge", 0.50), "fail": ("le", 0.20)},
    "M5_dead":       {"pass": ("le", 1),    "fail": ("ge", 3)},
}
