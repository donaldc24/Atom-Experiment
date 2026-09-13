"""Generic behavioral compiler: independent source-position and value tables.

This module imports no experiment model, operation generator, or task recipes.
It sees only calibration and validation input/output examples. Its deliberately
restricted hypothesis class has six output coordinates, each depending on one
input coordinate through an arbitrary ten-entry lookup table.
"""
from __future__ import annotations

import numpy as np

SEQ_LEN = 6
VOCAB = 10
VALIDATION_MIN = 0.98
SOURCE_MARGIN_MIN = 0.10
PROGRAM_BYTES = SEQ_LEN + SEQ_LEN * VOCAB


def _digits(values, name: str = "digits") -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != SEQ_LEN:
        raise ValueError(f"{name} must have shape (N, {SEQ_LEN})")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    if array.size and (array.min() < 0 or array.max() >= VOCAB):
        raise ValueError(f"{name} entries must be in [0, {VOCAB - 1}]")
    return array.astype(np.int64, copy=False)


def _parts(program: dict) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(program["source"])
    table = np.asarray(program["table"])
    if source.shape != (SEQ_LEN,) or table.shape != (SEQ_LEN, VOCAB):
        raise ValueError("program must contain source[6] and table[6][10]")
    if (not np.issubdtype(source.dtype, np.integer)
            or not np.issubdtype(table.dtype, np.integer)):
        raise ValueError("program source and table must contain integers")
    if source.min() < 0 or source.max() >= SEQ_LEN:
        raise ValueError("program source indices must be in [0, 5]")
    if table.min() < 0 or table.max() >= VOCAB:
        raise ValueError("program table values must be in [0, 9]")
    return source.astype(np.int64), table.astype(np.int64)


def apply_program(program: dict, x: np.ndarray) -> np.ndarray:
    """Run a candidate or accepted program, with no neural dependencies."""
    inputs = _digits(x)
    source, table = _parts(program)
    return table[np.arange(SEQ_LEN)[None, :], inputs[:, source]]


def fit_program(x_cal, y_cal, x_val, y_val) -> dict:
    """Fit from calibration only; validation measures agreement with the teacher.

    Every source's LUT uses conditional majority labels. np.argmax chooses the
    lowest digit on label ties and the lowest source index on source-score ties.
    Neither validation labels nor any ground-truth task semantics affect fitting.
    """
    x_cal, y_cal = _digits(x_cal, "x_cal"), _digits(y_cal, "y_cal")
    x_val, y_val = _digits(x_val, "x_val"), _digits(y_val, "y_val")
    if len(x_cal) == 0 or len(x_val) == 0:
        raise ValueError("calibration and validation must both be nonempty")
    if x_cal.shape != y_cal.shape or x_val.shape != y_val.shape:
        raise ValueError("inputs and outputs must have matching shapes")

    source, table, per_position = [], [], []
    for output_pos in range(SEQ_LEN):
        tables, scores = [], []
        for input_pos in range(SEQ_LEN):
            counts = np.bincount(
                x_cal[:, input_pos] * VOCAB + y_cal[:, output_pos],
                minlength=VOCAB * VOCAB,
            ).reshape(VOCAB, VOCAB)
            lut = counts.argmax(axis=1)
            tables.append(lut)
            scores.append(float((lut[x_cal[:, input_pos]] == y_cal[:, output_pos]).mean()))
        winner = int(np.argmax(scores))
        runner = max(score for idx, score in enumerate(scores) if idx != winner)
        source.append(winner)
        table.append(tables[winner].tolist())
        per_position.append({
            "output_position": output_pos,
            "source": winner,
            "source_fit_scores": scores,
            "winner_fit_score": scores[winner],
            "runnerup_fit_score": runner,
            "source_margin": scores[winner] - runner,
        })

    program = {"source": source, "table": table}
    fit_agreement = float((apply_program(program, x_cal) == y_cal).all(axis=1).mean())
    validation_agreement = float((apply_program(program, x_val) == y_val).all(axis=1).mean())
    min_margin = min(p["source_margin"] for p in per_position)
    gates = {
        "unique_sources": len(set(source)) == SEQ_LEN,
        "bijective_tables": all(len(set(row)) == VOCAB for row in table),
        "validation_agreement": validation_agreement >= VALIDATION_MIN,
        "source_predictability_margin": min_margin >= SOURCE_MARGIN_MIN,
    }
    return {
        **program,
        "fit_agreement": fit_agreement,
        "validation_agreement": validation_agreement,
        "accepted_candidate": bool(all(gates.values())),
        "gates": gates,
        "thresholds": {
            "validation_agreement_min": VALIDATION_MIN,
            "source_predictability_margin_min": SOURCE_MARGIN_MIN,
        },
        "min_source_margin": min_margin,
        "per_position": per_position,
        "n_calibration": len(x_cal),
        "n_validation": len(x_val),
    }


def program_to_bytes(program: dict) -> bytes:
    """Encode exactly 66 uint8 values: six sources then sixty LUT entries."""
    source, table = _parts(program)
    return np.concatenate((source, table.reshape(-1))).astype(np.uint8).tobytes()


def program_from_bytes(blob: bytes) -> dict:
    """Load an untrusted candidate encoding; this does not certify semantics."""
    if len(blob) != PROGRAM_BYTES:
        raise ValueError(f"expected exactly {PROGRAM_BYTES} bytes")
    values = np.frombuffer(blob, dtype=np.uint8)
    program = {"source": values[:SEQ_LEN].tolist(),
               "table": values[SEQ_LEN:].reshape(SEQ_LEN, VOCAB).tolist()}
    _parts(program)
    return program
