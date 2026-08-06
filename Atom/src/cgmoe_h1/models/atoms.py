"""Shared rank-one atom dictionaries for frozen linear layers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validated_task_ids(task_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(task_ids)
    if not normalized:
        raise ValueError("task_ids must contain at least one task")
    if any(not isinstance(task_id, str) or not task_id for task_id in normalized):
        raise ValueError("every task ID must be a non-empty string")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"task_ids must be unique, got {normalized!r}")
    return normalized


@dataclass
class TaskContext:
    """Model-owned task selection shared by all injected atom layers.

    This is intentionally a plain object rather than a global or an
    :class:`~torch.nn.Module`: it owns no learned state and must not inflate
    checkpoint parameter counts.  ``top_k`` is an inference-time capacity
    constraint; it never mutates or removes stored coefficients.
    """

    task_ids: tuple[str, ...]
    current_task_id: str | None = None
    top_k: int | None = None

    def __init__(
        self,
        task_ids: Sequence[str],
        current_task_id: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self.task_ids = _validated_task_ids(task_ids)
        self.current_task_id = None
        self.top_k = None
        if current_task_id is not None:
            self.set_active_task(current_task_id, top_k=top_k)
        elif top_k is not None:
            self.set_top_k(top_k)

    def validate_task(self, task_id: str) -> str:
        if task_id not in self.task_ids:
            expected = ", ".join(self.task_ids)
            raise KeyError(f"unsupported task ID {task_id!r}; expected one of: {expected}")
        return task_id

    @staticmethod
    def _validate_top_k(top_k: int | None) -> int | None:
        if top_k is None:
            return None
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError(f"top_k must be a non-negative integer or None, got {top_k!r}")
        return top_k

    def set_active_task(self, task_id: str, top_k: int | None = None) -> None:
        """Select a task and optionally a deterministic top-k atom budget."""

        self.current_task_id = self.validate_task(task_id)
        self.top_k = self._validate_top_k(top_k)

    set_task = set_active_task

    def clear(self) -> None:
        """Clear both active task and top-k selection."""

        self.current_task_id = None
        self.top_k = None

    def set_top_k(self, top_k: int | None) -> None:
        """Set the atom budget without changing the active task."""

        self.top_k = self._validate_top_k(top_k)

    def clear_top_k(self) -> None:
        """Restore unpruned atom evaluation."""

        self.top_k = None

    @contextmanager
    def use(self, task_id: str, top_k: int | None = None) -> Iterator["TaskContext"]:
        """Temporarily select a task, restoring the previous state afterward."""

        previous = (self.current_task_id, self.top_k)
        self.set_active_task(task_id, top_k=top_k)
        try:
            yield self
        finally:
            self.current_task_id, self.top_k = previous


class AtomLinear(nn.Module):
    """Add a task-weighted dictionary of shared rank-one operators.

    ``atom_v`` stores the input vectors, ``atom_u`` the output vectors, and one
    row of ``coefficients`` is selected per task.  The base linear layer is
    frozen.  A direct ``task_id`` is useful in standalone tests, while injected
    transformer layers read from their shared :class:`TaskContext`.
    """

    def __init__(
        self,
        base: nn.Linear,
        task_ids: Sequence[str],
        atom_count: int,
        scaling: float = 1.0,
        task_context: TaskContext | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be nn.Linear, got {type(base).__name__}")
        if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
            raise ValueError(f"atom_count must be a positive integer, got {atom_count!r}")
        if scaling <= 0:
            raise ValueError(f"scaling must be positive, got {scaling!r}")

        self.base = base
        self.task_ids = _validated_task_ids(task_ids)
        self.task_to_index = {task_id: index for index, task_id in enumerate(self.task_ids)}
        self.atom_count = atom_count
        self.scaling = float(scaling)
        self.task_context = task_context
        if task_context is not None and task_context.task_ids != self.task_ids:
            raise ValueError(
                "task_context task order must match AtomLinear task_ids exactly: "
                f"{task_context.task_ids!r} != {self.task_ids!r}"
            )

        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.atom_v = nn.Parameter(torch.empty(atom_count, base.in_features, **factory_kwargs))
        self.atom_u = nn.Parameter(torch.empty(atom_count, base.out_features, **factory_kwargs))
        self.coefficients = nn.Parameter(torch.empty(len(self.task_ids), atom_count, **factory_kwargs))

        nn.init.normal_(self.atom_v, mean=0.0, std=0.02)
        nn.init.normal_(self.atom_u, mean=0.0, std=0.02)
        nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self._top_k: int | None = None
        self.freeze_base()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def freeze_base(self) -> None:
        """Freeze every parameter in the wrapped base linear layer."""

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def _resolve_task_id(self, task_id: str | None) -> str:
        if task_id is None and self.task_context is not None:
            task_id = self.task_context.current_task_id
        if task_id is None:
            raise RuntimeError(
                "no active task; pass task_id to AtomLinear.forward or call "
                "model.set_active_task(task_id) before the forward pass"
            )
        if task_id not in self.task_to_index:
            expected = ", ".join(self.task_ids)
            raise KeyError(f"unsupported task ID {task_id!r}; expected one of: {expected}")
        return task_id

    def coefficient_row(self, task_id: str | None = None) -> Tensor:
        """Return the live (not copied) coefficient row for a task."""

        resolved = self._resolve_task_id(task_id)
        return self.coefficients[self.task_to_index[resolved]]

    def coefficient_l1(self, task_id: str | None = None) -> Tensor:
        """Mean absolute coefficient for one task, before any top-k mask."""

        return self.coefficient_row(task_id).abs().mean()

    def topk_mask(self, task_id: str | None, top_k: int) -> Tensor:
        """Return a deterministic boolean largest-magnitude coefficient mask.

        Exact ties are broken by lower atom index, as required by the locked H1
        contract.  The returned mask is fixed and carries no gradient.
        """

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError(f"top_k must be a non-negative integer, got {top_k!r}")
        row = self.coefficient_row(task_id)
        selected_count = min(top_k, self.atom_count)
        mask = torch.zeros(self.atom_count, dtype=torch.bool, device=row.device)
        if selected_count == 0:
            return mask
        if selected_count == self.atom_count:
            return torch.ones_like(mask)

        # Sorting a Python list makes the secondary (atom-index) tie break
        # explicit and independent of backend-specific torch.topk tie behavior.
        magnitudes = row.detach().abs().cpu().tolist()
        selected = sorted(range(self.atom_count), key=lambda index: (-magnitudes[index], index))[
            :selected_count
        ]
        mask[selected] = True
        return mask

    get_top_k_mask = topk_mask

    def set_top_k(self, top_k: int | None) -> None:
        """Set a layer-local top-k override used by subsequent forwards."""

        self._top_k = TaskContext._validate_top_k(top_k)

    def clear_top_k(self) -> None:
        """Clear the layer-local top-k override."""

        self._top_k = None

    def _effective_top_k(self) -> int | None:
        if self._top_k is not None:
            return self._top_k
        if self.task_context is not None:
            return self.task_context.top_k
        return None

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        """Yield shared atom vectors and all task coefficient rows."""

        yield self.atom_v
        yield self.atom_u
        yield self.coefficients

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def get_extra_state(self) -> dict[str, Any]:
        """Persist task-row semantics alongside coefficient tensors."""

        return {"task_ids": self.task_ids}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        checkpoint_task_ids = tuple(state.get("task_ids", ()))
        if checkpoint_task_ids != self.task_ids:
            raise RuntimeError(
                "atom checkpoint task order does not match the constructed layer: "
                f"{checkpoint_task_ids!r} != {self.task_ids!r}"
            )

    def forward(self, x: Tensor, task_id: str | None = None) -> Tensor:
        resolved_task = self._resolve_task_id(task_id)
        coefficient_row = self.coefficient_row(resolved_task)
        top_k = self._effective_top_k()
        if top_k is not None and top_k < self.atom_count:
            coefficient_row = coefficient_row * self.topk_mask(resolved_task, top_k)

        # [..., d_in] -> [..., N] -> [..., d_out].  This is algebraically the
        # weighted sum c[k] * u[k] * dot(v[k], x), vectorized over all leading
        # dimensions (including BERT batch and sequence axes).
        atom_activations = F.linear(x, self.atom_v)
        correction = F.linear(atom_activations * coefficient_row, self.atom_u.transpose(0, 1))
        return self.base(x) + correction * self.scaling


def iter_atom_layers(module: nn.Module) -> Iterator[AtomLinear]:
    """Yield each atom wrapper below ``module`` exactly once."""

    for child in module.modules():
        if isinstance(child, AtomLinear):
            yield child


def atom_parameter_count(module: nn.Module) -> int:
    """Count dictionary vectors and all coefficient rows, excluding the base."""

    return sum(layer.adapter_parameter_count() for layer in iter_atom_layers(module))


def coefficient_l1_penalty(module: nn.Module, task_id: str | None = None) -> Tensor:
    """Contract-defined mean ``abs(coefficient[m, task, k])`` over layers/atoms."""

    rows = [layer.coefficient_row(task_id).reshape(-1) for layer in iter_atom_layers(module)]
    if not rows:
        raise ValueError("coefficient L1 penalty requires at least one AtomLinear layer")
    return torch.cat(rows).abs().mean()


def coefficient_l1_regularization(
    module: nn.Module,
    task_id: str | None = None,
    weight: float = 1.0,
) -> Tensor:
    """Return ``weight * coefficient_l1_penalty`` with coefficient gradients."""

    if weight < 0:
        raise ValueError(f"regularization weight must be non-negative, got {weight!r}")
    return coefficient_l1_penalty(module, task_id) * float(weight)


def set_atom_top_k(module: nn.Module, top_k: int | None) -> None:
    """Set or clear the same layer-local top-k budget on every atom layer."""

    found = False
    for layer in iter_atom_layers(module):
        layer.set_top_k(top_k)
        found = True
    if not found:
        raise ValueError("cannot set atom top-k: model contains no AtomLinear layers")


def clear_atom_top_k(module: nn.Module) -> None:
    """Restore unpruned behavior on every atom layer."""

    set_atom_top_k(module, None)


# Readable aliases used by training and evaluation call sites.
atom_coefficient_l1 = coefficient_l1_penalty
atom_regularization_loss = coefficient_l1_regularization


__all__ = [
    "AtomLinear",
    "TaskContext",
    "atom_coefficient_l1",
    "atom_parameter_count",
    "atom_regularization_loss",
    "clear_atom_top_k",
    "coefficient_l1_penalty",
    "coefficient_l1_regularization",
    "iter_atom_layers",
    "set_atom_top_k",
]
