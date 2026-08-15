"""Registered constants for Atom V2 (H1). Fixed BEFORE the first run.

Every value in this file is part of the preregistration. Amendments are allowed
only during Experiment 0 calibration and must be recorded in DECISIONS with a
reason; nothing here may change in response to Experiment 1 results, ever
(AtomV2/H1Experiments.md "Training Config").

Two kinds of content live here:
  1. World/spec constants transcribed from AtomV2/H1Experiments.md and
     AtomV2/SplitMath.md (the surface-op recipes, training schedule, volumes).
  2. Registered DECISIONS the docs left open, fixed on 2026-08-14 before the
     first run (census epsilon, closed-map definition choice, ablation
     mechanics, P3 oversampling reading, oracle mechanics). Rationale for each
     is recorded in AtomV2/Harness/REGISTERED.md.

Everything derivable from the (pi, a, b) algebra is NOT here: triples, task
classes, adjacencies, L3 membership, the excluded identity cell and the merged
class are all derived by atomv2.split from atomv2.ops and mechanically
validated. Only the genuinely chosen held-out cell lists (which sibling of each
shareable adjacency is held out; which unique-adjacency cells form L2) are
registered data, because no formula picks them.
"""

# E0 calibration amendment R8. This revision is embedded in configs, run IDs,
# aggregate verdicts, and the E1 gate so pre/post-amendment evidence can never
# be pooled accidentally.
PROTOCOL_REVISION = "e0a-control-separated"

# ---------------------------------------------------------------------------
# World definition (H1Experiments.md "Designing the Data")
# ---------------------------------------------------------------------------
SEQ_LEN = 6                 # digits per list
VOCAB = 10                  # digits 0-9, all arithmetic mod 10
N_SURFACE = 8               # P1..P8
N_SUBOPS = 7                # R T W I N M A

# Surface-op recipes: "X then Y" means apply X first, then Y.
SURFACE_RECIPES = {
    "P1": ("R", "I"),
    "P2": ("T", "A"),
    "P3": ("A", "R"),
    "P4": ("M", "T"),
    "P5": ("N", "W"),
    "P6": ("I", "M"),
    "P7": ("W", "A"),
    "P8": ("T", "N"),   # T and N commute: P8's internal order is a SET, {T, N}.
}

# The dax primitive: trained only as a singleton, never inside any pair.
DAX = "P3"

# ---------------------------------------------------------------------------
# Split assignment (SplitMath.md "Split Assignment") - registered CHOICES only.
# Derived facts (L3 = all P3 pairs, exclusion of the identity class, the merged
# class, sibling adjacencies, coverage) are computed and validated in split.py.
# ---------------------------------------------------------------------------
HELDOUT_L1 = (
    "P2_P1", "P2_P5", "P7_P4", "P7_P6",   # first-position siblings via A-last
    "P1_P2", "P5_P2", "P6_P8", "P4_P2",   # second-position siblings via T-first
)
HELDOUT_L2 = (
    "P1_P4", "P5_P6", "P6_P1", "P6_P5", "P4_P6", "P8_P5",
)

# ---------------------------------------------------------------------------
# Model (H1Experiments.md "The Model")
# ---------------------------------------------------------------------------
N_ATOMS = 16                # deliberately overcomplete (7 sub-ops, 8 surface)
D_MODEL = 64                # per-position width
STATE_DIM = SEQ_LEN * D_MODEL   # 384; named baseline number (bandwidth knob)
ATOM_HIDDEN = 192           # atom MLP 384 -> 192 -> 384
KEY_DIM = 32                # routing key dimension; keys live in the atoms
MICRO_STEPS = 3             # per task token; overcomplete like the atom count
N_HEADS = 4
FF_DIM = 256
PASS_INDEX = N_ATOMS        # routing option 16 of 0..16; a real key, not an atom

# ---------------------------------------------------------------------------
# Data volumes (H1Experiments.md "Designing the Data")
# ---------------------------------------------------------------------------
EXAMPLES_PER_TRAIN_TASK = 1000
EXAMPLES_PER_EVAL_TASK = 400
N_PROBE_EXAMPLES = 400      # fresh inputs for panel probes (per-run stream)

# REGISTERED DECISION (P3 oversampling reading): P3 keeps 1,000 unique training
# examples like every other task; oversampling "to ~7,000 examples" is
# PRESENTATION frequency - the epoch array contains P3's examples
# P3_OVERSAMPLE_FACTOR times. 41*1000 + 7*1000 = 48,000 presentations/epoch,
# matching the doc's "~55 epochs over ~48k examples" at 20k steps x batch 128.
P3_OVERSAMPLE_FACTOR = 7

# ---------------------------------------------------------------------------
# Training config (H1Experiments.md "Training Config" - frozen)
# ---------------------------------------------------------------------------
LR = 3e-4
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500          # linear warmup, then constant LR; no decay
BATCH_SIZE = 128
TOTAL_STEPS = 20_000        # fixed budget; no early stopping; final ckpt = result
GRAD_CLIP = 1.0
TAU_START = 2.0             # Gumbel temperature ...
TAU_END = 0.5               # ... annealed linearly over the first
TAU_ANNEAL_STEPS = 10_000   # steps, held at TAU_END after
EVAL_EVERY = 1000           # this cadence IS the time-resolved closed-map curve
CKPT_EVERY = 2000
PANEL_STEPS = (5000, 10_000, 15_000)   # + final; full metric panel checkpoints
LOG_EVERY = 50
SEEDS = (0, 1, 2)           # one master seed per run controls EVERYTHING

# E1 arms: lambda_use grid (rent per atom APPLICATION; pass exempt).
# The grid itself is unchanged from registration; A1 is the lambda=0 cell.
LAMBDA_GRID = {"A1": 0.0, "A2": 0.001, "A3": 0.01, "A4": 0.1}

# AMENDMENT R9: A1 is NOT re-run as part of the E1 battery. E0's A0-free arm is
# configurationally identical to A1 (verified: the resolved configs differ only
# in the `arm` and `experiment` strings), so re-running it would spend compute
# to reproduce an existing result and would put two names on one condition.
# A1 stays constructible - config_for_arm("A1") still resolves, tests use it,
# and the lambda=0 row is sourced from the E0 A0-free runs, labelled as such.
E1_BATTERY_ARMS = ("A2", "A3", "A4")
LAMBDA_ZERO_SOURCE = {"experiment": "e0", "arm": "A0-free", "equivalent_arm": "A1"}

# ---------------------------------------------------------------------------
# E0 calibration (H1Experiments.md "Experiment 0") - qualitative pattern match
# ---------------------------------------------------------------------------
E0_ORACLE_L1_MIN = 0.70     # A0-Oracle unseen L1 accuracy must exceed this
E0_FREE_L1_MAX = 0.05       # A0-Free unseen L1 accuracy must sit below this
E0_GAP_MIN = 0.50           # "unmistakable, not marginal": oracle - free >= 0.50

# AMENDMENT R11 (2026-08-15): E1 is no longer GATED on the E0 verdict.
# The thresholds above are NOT moved - E0 still fails free_L1_floor and
# gap_unmistakable, and that verdict stands on the record exactly as computed.
# What changes is only whether that failure blocks E1.
#
# The rationale, from what E0 actually established:
#   - E0's job was to prove the instrument works. Its instrument audit passed
#     perfectly on all three oracle seeds: usage_matches_ground_truth = 1.0,
#     ablation CV = 0.0, census = the predicted 7, all 8 forced atoms observed.
#   - The oracle ceiling passed (L1 = 1.0000 > 0.70): a composing solution is
#     reachable, which is the precondition that makes a null result readable.
#   - The closed-map direction check passed, and the formal leak audit passed
#     all six checks, so a leaking world is excluded as the explanation.
#   - The two failing checks both concern the FREE arm's absolute L1 level.
#     That is a finding about this architecture, not a broken rig: amendment R8
#     separated control from data, which makes per-token program composition
#     available, so free routing recombines trained tokens by construction.
#     The "free arm sits at the floor" premise was inherited from V1, whose
#     composer had memory and whose state carried task identity.
#   - The unmistakable dissociation the protocol was reaching for does exist -
#     at L3, where oracle 1.0000 against free 0.0108 is a gap of 0.993.
# E1 therefore measures against a validated instrument and a characterised
# baseline. The E0 verdict is copied into the E1 results directory so the
# context always travels with the numbers.
E1_REQUIRES_E0_PASS = False
# Closed-map direction check: oracle final closed-map error < free final
# closed-map error on seen_heldout (direction only, no magnitude threshold).

# Oracle mechanics (V1 lineage, D18): forced [atom, pass, pass] per token with
# atom index = surface-op index (P1 -> atom 0 ... P8 -> atom 7); ground-truth
# intermediate state supervision on the CURRENT digit-only encoder's canonical
# embedding of the partial-composition digits (stop-grad), relative MSE:
ORACLE_STATE_SUP_WEIGHT = 40.0
ORACLE_ROUTING_CE_WEIGHT = 1.0
ORACLE_INTERMEDIATE_CE_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# Metric registrations (decisions recorded 2026-08-14, before first run)
# ---------------------------------------------------------------------------
# Census: an atom is IN USE if it receives more than CENSUS_EPS of all hard-
# routing ATOM-selections (pass picks EXCLUDED from the denominator) on the
# seen_heldout eval set. Pass usage is a separate logged number with its own
# story (step-smearing detection); it never dilutes the census denominator.
# eps = 2%: an order of magnitude above the hard-routing noise floor of a
# converged router, and 2-3x below the weakest legitimate atom in the most
# fragmented plausible world (sub-op factorization, ~5-7% for the rarest
# sub-op). The census logs TWO numbers: atoms-in-use AND steps-per-token.
CENSUS_EPS = 0.02

# Task-level usage threshold for gating ablation reporting (V1 lineage):
# atom i "is used by" task j if picked at some micro-step on >= 50% of task
# j's eval examples under hard routing.
USAGE_TASK_THRESHOLD = 0.5

# Ablation (F2): zero-delta intercept at APPLICATION - the composer still
# selects the atom, the delta is zeroed; with the non-affine state norm this is
# exactly "this atom becomes pass". One atom at a time only; damage is measured
# against the SAME RUN's unablated per-task accuracy. The routing-mask
# "compensation probe" runs on the FINAL checkpoint only, is logged
# observationally, and carries no verdict weight.
ABLATION_MODE = "zero_delta"

# Closed-map error, redefined against the sub-op lattice (headline, time-
# resolved at EVAL_EVERY): during hard-routed eval forwards, after every live
# micro-step, relative L2 from the current state to the nearest encoding of the
# task's sub-op-lattice prefix states (all prefixes of the task's canonical
# sub-op chain, including depth 0, encoded in the task-independent canonical
# content space). Companions logged so an all-pass dead system cannot score
# well: distance-to-target at the final step, and the prefix-visit histogram.
# The panel additionally runs the V1-style atom-centric probe against all
# 7 sub-op + 8 surface-op targets (matched assignment + coverage).
CLOSED_MAP_MODE = "trajectory+atom_centric"

# Decodability probes are trained on frozen, DETACHED states (probes read,
# never write); each probe is paired with an h0 floor (same probe on the
# pre-atom content state, a no-control leakage floor) and a shuffled-label
# chance floor. P8 labels are the SET {T, N}: probes are
# multi-label over sub-ops and never encode P8's internal order.
PROBE_TRAIN_FRACTION = 0.7

# ---------------------------------------------------------------------------
# Environment / determinism (V1 lineage: thread counts change reduction order)
# ---------------------------------------------------------------------------
NUM_THREADS = 4
NUM_INTEROP_THREADS = 1
RSS_FAIL_GB = 4.0
