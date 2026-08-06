#!/usr/bin/env python3
"""Inspect BERT linear modules and the exact H1 adapter targets."""

from __future__ import annotations

import argparse

from torch import nn
from transformers import AutoModel


DEFAULT_MODEL = "prajjwal1/bert-tiny"
TARGET_SUFFIXES = ("attention.self.query", "attention.self.value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = AutoModel.from_pretrained(args.model)

    linears: list[tuple[str, nn.Linear]] = []
    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        linears.append((name, module))
        trainable = any(parameter.requires_grad for parameter in module.parameters())
        print(
            f"{name}: Linear({module.in_features}, {module.out_features}), "
            f"requires_grad={trainable}"
        )
        if name.endswith(TARGET_SUFFIXES):
            targets.append((name, module))

    config = model.config
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print("\nSummary")
    print(f"Model: {args.model}")
    print(f"Transformer layers: {config.num_hidden_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Linear modules: {len(linears)}")
    print(f"Query targets: {sum(name.endswith('attention.self.query') for name, _ in targets)}")
    print(f"Value targets: {sum(name.endswith('attention.self.value') for name, _ in targets)}")
    print(f"Adapter targets: {len(targets)}")
    for name, module in targets:
        print(f"  {name}: ({module.out_features}, {module.in_features})")
    print(f"Total base-model parameters: {total_parameters:,}")


if __name__ == "__main__":
    main()
