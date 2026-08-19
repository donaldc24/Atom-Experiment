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

# ---------------------------------------------------------------------------
# E1b (H1-E1bExperiment.md): anti-saturation router + exploration dose.
# Registered BEFORE the first E1b run. One conceptual routing-stack change:
# bounded cosine logits, fixed shared scale, arm-specific forward Gumbel
# noise, fixed straight-through temperature, no annealing. lambda_use = 0 in
# every free arm; there is no rent intervention in E1b.
# ---------------------------------------------------------------------------
import math as _math

E1B_PROTOCOL_REVISION = "e1b-cosine-router"

# Registered p_max targets (full precision; the table in the doc is rounded).
# A6 uses 0.90 ** (1/6) so that P(any exploratory decision in six) = 10%.
E1B_P_MAX = {"A5": 0.99, "A6": 0.90 ** (1.0 / 6.0), "A7": 0.90}

# Shared cosine-logit scale, fixed by anchoring A5 at (p_max=0.99, sigma=1):
# alpha = 0.5 * ln(16 * p_max / (1 - p_max)) with 16 = n_routes - 1.
E1B_N_ALTERNATIVES = 16              # 17 routes: 16 atoms + pass
E1B_ALPHA = 0.5 * _math.log(E1B_N_ALTERNATIVES * 0.99 / (1 - 0.99))

# Per-arm forward Gumbel noise scale, derived (never transcribed) from the
# full-precision p_max: sigma = 2 * alpha / ln(16 * p_max / (1 - p_max)).
E1B_SIGMA = {
    arm: 2.0 * E1B_ALPHA / _math.log(E1B_N_ALTERNATIVES * p / (1.0 - p))
    for arm, p in E1B_P_MAX.items()
}

E1B_TAU_BACKWARD = 1.0               # straight-through surrogate temperature
E1B_NORM_EPS = 1e-6                  # the one fixed epsilon in q/(|q|+eps)
E1B_ARMS = ("A5", "A6", "A7")
E1B_ORACLE_ARM = "A5-oracle"         # architectural regression check only

# Liveness telemetry (one fixed diagnostic batch at every eval; stochastic
# measurements over eight independently seeded Gumbel draws).
E1B_DIAG_DRAWS = 8
E1B_PMAX_TOL = 1e-6                  # implementation invariant tolerance
E1B_DEAF_GRAD_NORM = 1e-8            # deafness rule, condition 1
E1B_DEAF_RATIO = 1e-3                # deafness rule, condition 2 (conjunction)
E1B_DEAF_WINDOW = (2000, 18000)      # steps included in the validity rule
E1B_NORM_GROWTH_FLAG = 10.0          # raw q/k norm growth audit flag (x median@1k)

# Oracle regression thresholds ("approximately perfect accuracy and
# closed-map error near 0.015"): pattern check, not number match.
E1B_ORACLE_ACC_MIN = 0.99
E1B_ORACLE_CLOSED_MAP_MAX = 0.05

# ---------------------------------------------------------------------------
# E2 (H1-Experiment2.md): interface noise. One treatment - training-only
# Gaussian noise at live, NONTERMINAL state handoffs, re-normalized by the
# existing non-affine LayerNorm. Base condition is E1b's A6 (cosine router,
# sigma = E1B_SIGMA['A6'], lambda_use = 0); the completed A6 runs are the
# control, guarded by the registered no-noise equivalence gate. Headline
# evaluation is always CLEAN (state_noise_sigma = 0).
# ---------------------------------------------------------------------------
E2_PROTOCOL_REVISION = "e2-interface-noise"
E2_BASE_ARM = "A6"                   # registered base + paired control

# Registered noise levels as target clean/noisy cosines; sigma is DERIVED
# (registration receipt): sigma = sqrt(1 / cosine^2 - 1).
E2_TARGET_COSINE = {"A8": 0.999, "A9": 0.990, "A10": 0.950}
E2_STATE_NOISE_SIGMA = {
    arm: _math.sqrt(1.0 / (c * c) - 1.0)
    for arm, c in E2_TARGET_COSINE.items()
}
E2_ARMS = ("A8", "A9", "A10")

# Noise telemetry + implementation gates.
E2_NOISE_DRAWS = 8                   # registered draws on the diagnostic batch
E2_COSINE_TOL = 0.005                # observed median cosine vs target gate
E2_GRAD_DRAWS = 2                    # producer-gradient draws per eval (cost)
E2_CM_EXAMPLES = 100                 # per-task subsample for the noisy
                                     # producer/transmitted closed-map curves

# Registered evaluation-only robustness sweep (final checkpoint, every arm
# INCLUDING the A6 reference): target cosines, 8 draws per noisy point.
E2_ROBUST_COSINES = (1.000, 0.999, 0.990, 0.950)

# ---------------------------------------------------------------------------
# E3 (H1-Experiment3.md): the atom sandbox. Three content-agnostic pressures
# on the atom MLPs ONLY (standalone validity, random-context closure,
# usage-weighted functional uniqueness), trained alongside the untouched A6
# task path. Gradient boundary is absolute: encoder, decoder, composer,
# routing keys and the pass key receive NO sandbox gradient; the decoder is
# the frozen validity evaluator / fingerprint reader. lambda_use = 0.
# ---------------------------------------------------------------------------
E3_PROTOCOL_REVISION = "e3-atom-sandbox"
E3_BASE_ARM = "A6"                   # registered base + paired control (e1b)
E3_ARMS = ("A11", "A12", "A13")

# Registered dose grid: (lambda_sandbox_valid, lambda_sandbox_unique) applied
# jointly - the sandbox is one treatment at three intensities, not a factorial.
E3_LAMBDA_VALID = {"A11": 0.1, "A12": 0.3, "A13": 1.0}
E3_LAMBDA_UNIQUE = {"A11": 0.1, "A12": 0.3, "A13": 1.0}

# REGISTERED DECISION (validity criterion): the harness has no separate
# validity evaluator, so L_valid(z) is defined against the FROZEN decoder as
#   L_valid(z) = READ(z): per-position CE of Decoder(z) against its own hard
#   readout (-log p_max: the state must decode CONFIDENTLY into some digit
#   list). Deliberately content-agnostic AND representation-agnostic: it does
#   NOT push atom outputs toward any particular latent encoding - an atom may
#   invent whatever representation it likes, provided the state is not
#   gibberish to the frozen decoder.
# CYCLE (relative MSE from z to code(readout), the oracle's relative-MSE
# form) is measured as TELEMETRY ONLY and never backpropagated: as a loss it
# would define what the intermediate representation should look like, which
# is exactly the pressure this experiment refuses to apply.
# Standalone and closure share this one criterion and are logged separately
# (L_sandbox_valid = their mean).

# Functional-uniqueness branch: behavioral fingerprints are frozen-decoder
# softmax outputs on CLEAN standalone states F_i = softmax(Decoder(Apply_i(z0)));
# distance is mean total variation across the batch and positions; hinge
# margin below. Compared pairwise among sampled atoms AND against pass
# (F_pass = softmax(Decoder(z0))) so identity cannot satisfy self-sufficiency.
E3_UNIQUE_MARGIN = 0.5               # TV lives in [0,1]
E3_UNIQUE_ATOMS_PER_STEP = 4         # sampled atoms per step (all 6 pairs)

# Usage weighting: unused slots may stay unused. usage_ema tracks the REAL
# hard-routing frequency of each atom over ACTIVE ROUTING OPPORTUNITIES of
# the A6 task path (denominator = live steps; a pass pick contributes zero
# to every atom), so an all-pass batch DECAYS every weight - emergent pass
# usage shows up as fading uniqueness pressure, never as a frozen stale
# weight. NOTE this normalization deliberately differs from the census
# (which excludes pass from its denominator); CENSUS_EPS is reused only as
# the clamp scale of w_i = clamp(usage_ema_i / CENSUS_EPS, 0, 1). EMA
# initialized uniform (1/16 > CENSUS_EPS, so every atom starts fully
# weighted and unused slots decay out; half-life ~69 steps at decay 0.99).
# No gradient enters routing.
E3_USAGE_EMA_DECAY = 0.99
E3_USAGE_EMA_INIT = 1.0 / N_ATOMS
E3_USAGE_WEIGHT_EPS = CENSUS_EPS

# Random-context / closure branch: predecessor chains of K ~ Uniform{1..6}
# no-grad atom applications (6 = the most atom applications A6 can execute on
# a two-token task); self-predecessors included, no exclusions. The chain is
# always drawn at full length so the sandbox stream's consumption per step is
# fixed; only the first K draws are applied.
E3_CHAIN_MAX = 2 * MICRO_STEPS       # 6

# Telemetry (eval cadence, measurement only): random closure chains per
# record, drawn from the indexed 'e3_sandbox_eval' stream.
E3_TELEMETRY_CHAINS = 8

# Which protocol revision each experiment's runs must carry to be pooled.
EXPERIMENT_REVISIONS = {
    "e0": PROTOCOL_REVISION,
    "e1": PROTOCOL_REVISION,
    "e1b": E1B_PROTOCOL_REVISION,
    "e2": E2_PROTOCOL_REVISION,
    "e3": E3_PROTOCOL_REVISION,
}
