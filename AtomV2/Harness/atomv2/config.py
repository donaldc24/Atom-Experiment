"""Resolved run configuration. Every default explicit; full dict -> config.json.

Arms differ ONLY in training procedure - architecture, data and split are
shared (V1 contract, config.py:126 lineage):

  E0 (calibration):  A0-oracle  forced routing + intermediate state supervision
                                (the ONLY place oracle machinery exists)
                     A0-free    identical config, free routing, task loss only
                     lambda_use = 0 for both arms so cost is not a variable.
  E1 (the sweep):    A1..A4     free routing, lambda_use from the registered
                                grid {0, 0.001, 0.01, 0.1}.

The `smoke` flag shrinks volumes/steps for end-to-end pipeline verification;
smoke runs are marked in config.json and aggregate.py refuses to mix them with
real runs.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from . import registered as R
from .utils import RUNS_DIR, git_info

E0_ARMS = ("A0-oracle", "A0-free")
E1_ARMS = tuple(R.LAMBDA_GRID)  # ('A1','A2','A3','A4')


@dataclass
class Config:
    # identity
    arm: str = "A1"
    seed: int = 0                       # THE master seed; everything derives from it
    experiment: str = "e1"              # 'e0' | 'e1'
    smoke: bool = False
    protocol_revision: str = R.PROTOCOL_REVISION

    # world / model (registered; do not tune)
    seq_len: int = R.SEQ_LEN
    vocab: int = R.VOCAB
    n_atoms: int = R.N_ATOMS
    d_model: int = R.D_MODEL
    state_dim: int = R.STATE_DIM
    atom_hidden: int = R.ATOM_HIDDEN
    key_dim: int = R.KEY_DIM
    micro_steps: int = R.MICRO_STEPS
    n_heads: int = R.N_HEADS
    ff_dim: int = R.FF_DIM

    # data volumes (registered)
    examples_per_train_task: int = R.EXAMPLES_PER_TRAIN_TASK
    examples_per_eval_task: int = R.EXAMPLES_PER_EVAL_TASK
    n_probe_examples: int = R.N_PROBE_EXAMPLES
    p3_oversample_factor: int = R.P3_OVERSAMPLE_FACTOR

    # training schedule (registered)
    lr: float = R.LR
    betas: tuple = R.BETAS
    weight_decay: float = R.WEIGHT_DECAY
    warmup_steps: int = R.WARMUP_STEPS
    batch_size: int = R.BATCH_SIZE
    total_steps: int = R.TOTAL_STEPS
    grad_clip: float = R.GRAD_CLIP
    tau_start: float = R.TAU_START
    tau_end: float = R.TAU_END
    tau_anneal_steps: int = R.TAU_ANNEAL_STEPS
    eval_every: int = R.EVAL_EVERY
    ckpt_every: int = R.CKPT_EVERY
    panel_steps: tuple = R.PANEL_STEPS
    log_every: int = R.LOG_EVERY

    # loss
    lambda_use: float = 0.0

    # oracle machinery - ONLY config_for_arm('A0-oracle') may enable these.
    forced_routing: bool = False
    oracle_state_sup_weight: float = 0.0
    oracle_routing_ce_weight: float = 0.0
    oracle_intermediate_ce_weight: float = 0.0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["betas"] = list(d["betas"])
        d["panel_steps"] = list(d["panel_steps"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        dropped = sorted(set(d) - known)
        if dropped:
            print(f"config: dropping unknown saved keys {dropped}")
        kw = {k: v for k, v in d.items() if k in known}
        if "betas" in kw:
            kw["betas"] = tuple(kw["betas"])
        if "panel_steps" in kw:
            kw["panel_steps"] = tuple(kw["panel_steps"])
        return cls(**kw)


def config_for_arm(arm: str, seed: int, smoke: bool = False) -> Config:
    """Build the resolved config for an arm. Resets every flag to baseline
    first: no arm inherits another arm's machinery by accident."""
    cfg = Config(arm=arm, seed=seed, smoke=smoke)
    if arm in E0_ARMS:
        cfg.experiment = "e0"
        cfg.lambda_use = 0.0            # cost is not a variable in E0
        if arm == "A0-oracle":
            cfg.forced_routing = True
            cfg.oracle_state_sup_weight = R.ORACLE_STATE_SUP_WEIGHT
            cfg.oracle_routing_ce_weight = R.ORACLE_ROUTING_CE_WEIGHT
            cfg.oracle_intermediate_ce_weight = R.ORACLE_INTERMEDIATE_CE_WEIGHT
    elif arm in E1_ARMS:
        cfg.experiment = "e1"
        cfg.lambda_use = R.LAMBDA_GRID[arm]
    else:
        raise ValueError(f"unknown arm {arm!r}; E0 arms {E0_ARMS}, E1 arms {E1_ARMS}")

    if smoke:
        cfg.examples_per_train_task = 48
        cfg.examples_per_eval_task = 24
        cfg.n_probe_examples = 24
        cfg.total_steps = 300
        cfg.warmup_steps = 20
        cfg.tau_anneal_steps = 150
        cfg.eval_every = 100
        cfg.ckpt_every = 100
        cfg.panel_steps = (100, 200)
        cfg.log_every = 20
    return cfg


def run_id_for(cfg: Config) -> str:
    git = git_info()
    source = git["git_sha_short"]
    if git.get("dirty_source_sha256"):
        source += f"-dirty{git['dirty_source_sha256'][:8]}"
    return f"{cfg.arm}_s{cfg.seed}_{cfg.protocol_revision}_{source}"


def run_dir_for(cfg: Config, out: str | None = None) -> Path:
    base = Path(out) if out else RUNS_DIR
    sub = f"smoke_{cfg.experiment}" if cfg.smoke else cfg.experiment
    return base / sub / run_id_for(cfg)
