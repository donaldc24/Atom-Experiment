"""Independent low-rank adapters for frozen linear layers.

The implementation deliberately keeps the frozen base layer inside the wrapper.
That makes injection reversible in principle and, more importantly, ensures a
checkpoint contains only one copy of the base model rather than one copy per
task adapter.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Add a trainable LoRA branch to a frozen :class:`~torch.nn.Linear`.

    The correction is ``(alpha / rank) * B(A(dropout(x)))``.  ``A`` is
    initialized from ``Normal(0, 0.02)`` and ``B`` is initialized to zero, as
    fixed by the H1 experiment contract.  Consequently, wrapping a layer does
    not change its output before training.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be nn.Linear, got {type(base).__name__}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank!r}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / rank
        self.dropout_probability = float(dropout)

        # Construct on the same device and with the same floating dtype as the
        # wrapped layer.  BERT-tiny uses float32, while this also keeps unit
        # tests and mixed-precision inference unsurprising.
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            **factory_kwargs,
        )
        self.lora_b = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            **factory_kwargs,
        )
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_b.weight)
        self.freeze_base()

    @property
    def in_features(self) -> int:
        """Input width, mirroring :class:`~torch.nn.Linear`."""

        return self.base.in_features

    @property
    def out_features(self) -> int:
        """Output width, mirroring :class:`~torch.nn.Linear`."""

        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        """The frozen base weight (for compatibility with linear consumers)."""

        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        """The frozen base bias, if present."""

        return self.base.bias

    @property
    def A(self) -> nn.Linear:
        """Alias for the input-side LoRA projection."""

        return self.lora_a

    @property
    def B(self) -> nn.Linear:
        """Alias for the output-side LoRA projection."""

        return self.lora_b

    def freeze_base(self) -> None:
        """Freeze every parameter belonging to the wrapped base layer."""

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only persistent learned LoRA parameters."""

        yield from self.lora_a.parameters()
        yield from self.lora_b.parameters()

    def adapter_parameter_count(self) -> int:
        """Return the exact number of learned LoRA scalar values."""

        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base(x)
        correction = self.lora_b(self.lora_a(self.dropout(x)))
        return base_output + correction * self.scaling


def iter_lora_layers(module: nn.Module) -> Iterator[LoRALinear]:
    """Yield each LoRA wrapper below ``module`` exactly once."""

    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def lora_parameter_count(module: nn.Module) -> int:
    """Count LoRA parameters without counting any frozen base tensor."""

    return sum(layer.adapter_parameter_count() for layer in iter_lora_layers(module))


__all__ = ["LoRALinear", "iter_lora_layers", "lora_parameter_count"]
