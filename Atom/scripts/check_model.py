#!/usr/bin/env python3
"""Download BERT Tiny and run a single inference smoke test."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import Any


MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_SENTENCE = "This tiny model is ready for an experiment."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download prajjwal1/bert-tiny, tokenize one sentence, and run one "
            "forward pass."
        )
    )
    parser.add_argument(
        "--sentence",
        default=DEFAULT_SENTENCE,
        help="sentence to tokenize (default: %(default)s)",
    )
    return parser


def _tensor_shapes(values: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for name, value in values.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes[name] = tuple(shape)
    return shapes


def _fail(stage: str, exc: Exception) -> int:
    print(
        f"Model smoke test FAILED while {stage}: "
        f"{type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    print(
        "Install the project dependencies and check network access to the "
        "Hugging Face Hub, then rerun this script.",
        file=sys.stderr,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.sentence.strip():
        print("Model smoke test FAILED: --sentence must not be empty.", file=sys.stderr)
        return 2

    try:
        import torch
    except Exception as exc:  # Import-time binary/linker errors are not ImportError.
        return _fail("importing PyTorch", exc)

    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        return _fail("importing Transformers", exc)

    print(f"Model: {MODEL_NAME}")
    print("Downloading/loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:
        return _fail(f"loading the {MODEL_NAME} tokenizer", exc)

    print("Downloading/loading model...")
    try:
        model = AutoModel.from_pretrained(MODEL_NAME)
    except Exception as exc:
        return _fail(f"loading the {MODEL_NAME} model", exc)

    print(f"Sentence: {args.sentence}")
    try:
        encoded = tokenizer(args.sentence, return_tensors="pt")
    except Exception as exc:
        return _fail("tokenizing the sentence", exc)

    input_shapes = _tensor_shapes(encoded)
    if not input_shapes:
        return _fail("inspecting tokenizer output", RuntimeError("no tensors returned"))
    print(f"Input tensor shapes: {input_shapes}")

    try:
        model.eval()
        with torch.inference_mode():
            outputs = model(**encoded)
    except Exception as exc:
        return _fail("running the forward pass", exc)

    output_values = outputs if hasattr(outputs, "items") else {}
    output_shapes = _tensor_shapes(output_values)
    if not output_shapes:
        return _fail(
            "inspecting model output", RuntimeError("no output tensors returned")
        )
    print(f"Output tensor shapes: {output_shapes}")

    try:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    except Exception as exc:
        return _fail("counting model parameters", exc)

    print(f"Model parameter count: {parameter_count:,}")
    print("Model smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
