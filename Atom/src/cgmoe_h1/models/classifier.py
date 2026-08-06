"""Frozen BERT encoder with initialized task-specific classification heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from torch import Tensor, nn

from .atoms import AtomLinear, TaskContext
from .lora import LoRALinear


class BertTaskClassifier(nn.Module):
    """Classify the first-token representation of a frozen BERT-like encoder.

    Parameters
    ----------
    encoder:
        A Hugging Face BERT model or a compatible synthetic module.  Its output
        may expose ``last_hidden_state`` or return that tensor as item zero.
    num_labels:
        Either a single label count or, for convenience, a mapping from task ID
        to label count.  When it is an integer and ``task_ids`` is omitted, the
        sole task ID is ``"default"``.
    task_num_labels:
        Explicit per-task label counts.  This is mutually exclusive with a
        mapping passed as ``num_labels``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int | Mapping[str, int] | None = None,
        *,
        task_num_labels: Mapping[str, int] | None = None,
        task_ids: Sequence[str] | None = None,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError(f"encoder must be nn.Module, got {type(encoder).__name__}")

        label_counts = self._resolve_label_counts(num_labels, task_num_labels, task_ids)
        self.encoder = encoder
        self.task_ids = tuple(label_counts)
        self.task_context = TaskContext(self.task_ids)
        # Injection is sometimes called on classifier.encoder rather than on
        # the wrapper.  Publishing the same plain context there keeps both
        # workflows correct without registering duplicate modules.
        setattr(self.encoder, "task_context", self.task_context)
        for module in self.encoder.modules():
            if isinstance(module, AtomLinear):
                if module.task_ids != self.task_ids:
                    raise ValueError(
                        "pre-injected atom layer tasks do not match classifier heads: "
                        f"{module.task_ids!r} != {self.task_ids!r}"
                    )
                module.task_context = self.task_context

        resolved_hidden_size = hidden_size or self._encoder_hidden_size(encoder)
        if isinstance(resolved_hidden_size, bool) or not isinstance(resolved_hidden_size, int):
            raise TypeError("hidden_size must be an integer")
        if resolved_hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = resolved_hidden_size

        self.heads = nn.ModuleDict(
            {
                task_id: nn.Linear(resolved_hidden_size, label_count)
                for task_id, label_count in label_counts.items()
            }
        )
        self.reset_head_parameters()
        self.freeze_base_encoder()

    @staticmethod
    def _resolve_label_counts(
        num_labels: int | Mapping[str, int] | None,
        task_num_labels: Mapping[str, int] | None,
        task_ids: Sequence[str] | None,
    ) -> dict[str, int]:
        if isinstance(num_labels, Mapping):
            if task_num_labels is not None:
                raise ValueError("pass task label mapping only once")
            task_num_labels = num_labels
            num_labels = None

        if task_num_labels is not None:
            if num_labels is not None or task_ids is not None:
                raise ValueError(
                    "task_num_labels cannot be combined with num_labels or task_ids"
                )
            label_counts = dict(task_num_labels)
        else:
            if num_labels is None:
                raise ValueError("num_labels or task_num_labels is required")
            if isinstance(num_labels, bool) or not isinstance(num_labels, int):
                raise TypeError("num_labels must be an integer or a task mapping")
            ids = tuple(task_ids) if task_ids is not None else ("default",)
            label_counts = {task_id: num_labels for task_id in ids}

        if not label_counts:
            raise ValueError("at least one task head is required")
        if any(not isinstance(task_id, str) or not task_id for task_id in label_counts):
            raise ValueError("task IDs must be non-empty strings")
        for task_id, count in label_counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 1:
                raise ValueError(
                    f"label count for {task_id!r} must be an integer greater than one"
                )
        return label_counts

    @staticmethod
    def _encoder_hidden_size(encoder: nn.Module) -> int:
        config = getattr(encoder, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(encoder, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "cannot infer encoder hidden size; pass hidden_size explicitly or provide "
                "encoder.config.hidden_size"
            )
        return hidden_size

    @property
    def bert(self) -> nn.Module:
        """Compatibility alias for the held encoder without duplicate registration."""

        return self.encoder

    @property
    def head(self) -> nn.Linear:
        """Return the sole classification head for single-task models."""

        if len(self.heads) != 1:
            raise AttributeError("multitask classifier has no singular head; use heads[task_id]")
        return next(iter(self.heads.values()))

    def reset_head_parameters(self) -> None:
        """Apply the contract initialization to every task head."""

        for head in self.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(head.bias)

    def freeze_base_encoder(self) -> None:
        """Freeze base weights while preserving any already-injected adapters."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for module in self.encoder.modules():
            if isinstance(module, LoRALinear):
                for parameter in module.adapter_parameters():
                    parameter.requires_grad_(True)
            elif isinstance(module, AtomLinear):
                for parameter in module.adapter_parameters():
                    parameter.requires_grad_(True)

    freeze_encoder = freeze_base_encoder

    def set_active_task(self, task_id: str, top_k: int | None = None) -> None:
        """Select both the classification head and atom coefficient row."""

        self.task_context.set_active_task(task_id, top_k=top_k)

    set_task = set_active_task

    def set_atom_top_k(self, top_k: int | None) -> None:
        """Apply an inference-time top-k budget through the shared context."""

        self.task_context.set_top_k(top_k)
        for module in self.encoder.modules():
            if isinstance(module, AtomLinear):
                module.set_top_k(top_k)

    def clear_atom_top_k(self) -> None:
        self.task_context.clear_top_k()
        for module in self.encoder.modules():
            if isinstance(module, AtomLinear):
                module.clear_top_k()

    @contextmanager
    def use_task(self, task_id: str, top_k: int | None = None):
        """Temporarily select a task and optional top-k evaluation budget."""

        with self.task_context.use(task_id, top_k=top_k):
            yield self

    def _resolve_task_id(self, task_id: str | None) -> str:
        if task_id is not None:
            self.set_active_task(task_id)
            return task_id
        if self.task_context.current_task_id is not None:
            return self.task_context.current_task_id
        if len(self.task_ids) == 1:
            sole_task = self.task_ids[0]
            self.set_active_task(sole_task)
            return sole_task
        expected = ", ".join(self.task_ids)
        raise RuntimeError(f"task_id is required for multitask classifier; expected one of: {expected}")

    @staticmethod
    def _last_hidden_state(outputs: Any) -> Tensor:
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, Mapping):
            hidden = outputs["last_hidden_state"]
        elif isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
        else:
            raise TypeError(
                "encoder output must expose last_hidden_state or contain it at index zero"
            )
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise ValueError("encoder last_hidden_state must have shape [batch, sequence, hidden]")
        return hidden

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        task_id: str | None = None,
        **encoder_kwargs: Any,
    ) -> Tensor:
        resolved_task = self._resolve_task_id(task_id)
        if attention_mask is not None:
            encoder_kwargs["attention_mask"] = attention_mask
        outputs = self.encoder(input_ids=input_ids, **encoder_kwargs)
        first_token = self._last_hidden_state(outputs)[:, 0, :]
        if first_token.shape[-1] != self.hidden_size:
            raise ValueError(
                f"encoder returned hidden width {first_token.shape[-1]}, expected {self.hidden_size}"
            )
        return self.heads[resolved_task](first_token)


__all__ = ["BertTaskClassifier"]
