"""Reproducibility helpers shared by data loading and training."""

from __future__ import annotations

import os
import random
from collections.abc import Callable

import numpy as np
import torch


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic algorithms.

    PyTorch's ``warn_only`` mode keeps the CPU-first experiment usable if a
    backend lacks a deterministic implementation, while still surfacing that
    operation for the run record.
    """
    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Required for deterministic CUDA matrix multiplication on relevant CUDA
    # versions.  ``setdefault`` preserves an explicit environment choice.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a separately seeded generator for deterministic loader order."""
    _validate_seed(seed)
    return torch.Generator().manual_seed(seed)


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Return a deterministic ``DataLoader`` worker initializer.

    The worker seed is intentionally derived from the experiment seed rather
    than mutable global RNG state, making construction order irrelevant.
    """
    _validate_seed(seed)

    def seed_worker(worker_id: int) -> None:
        if not isinstance(worker_id, int) or isinstance(worker_id, bool) or worker_id < 0:
            raise ValueError("worker_id must be a non-negative integer")
        worker_seed = (seed + worker_id) % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker
