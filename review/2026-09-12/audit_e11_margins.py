"""Post-hoc fresh-review audit; not a registered experiment or training run.

Run from any directory with the repository's Python environment:
    .venv/Scripts/python.exe review/2026-09-12/audit_e11_margins.py

The default invocation only reads original checkpoints and prints JSON. Pass
--output PATH to save this review's result. No original run artifact is edited.

For a token, E11 has input-independent nonnegative transport T and a reaction
kernel K at each output position j. Before positive row normalization,
    q[j,y] = sum_i T[j,i] K[j,x[i],y].
The desired output is y = (a*d+b[j]) mod 10, d=x[pi[j]]. For competitor c,
the EXACT minimum of q[j,y]-q[j,c] over the remaining input digits is
    T[j,pi[j]] * (K[j,d,y]-K[j,d,c])
    + sum_{i != pi[j]} T[j,i] * min_v (K[j,v,y]-K[j,v,c]).
Nonnegative transport makes independent minimization valid. Positive row
normalization preserves the argmax. Strict positivity for all j,d,c is a
sufficient certificate in real arithmetic for every one of the 10^6 inputs.

LIMITATIONS: this script evaluates that analytic expression in ordinary
float64 using saved float32 weights. It is NOT an interval-arithmetic proof or
a machine-checked certificate of all float32 execution paths. Original-dtype
witness execution is checked separately. Positive margins imply arbitrary
finite-length composition over certified tokens in exact arithmetic because
each nonterminal boundary is one-hot. That implication assumes unchanged
weights, clean interfaces, exact argmax canonicalization, and sequences using
only certified tokens. It says nothing about uncertified tokens, learned
sub-operation discovery, noisy interfaces, or other task families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "AtomV2" / "Harness"))
from atomv2 import ops  # noqa: E402
from atomv2.exchange_model import ConservativeExchangeModel  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def audit(path: Path) -> dict:
    saved = torch.load(path, weights_only=False, map_location="cpu")
    arm = saved["config"]["arm"]
    original = ConservativeExchangeModel(arm).eval()
    original.load_state_dict(saved["model"])
    model = ConservativeExchangeModel(arm).double().eval()
    model.load_state_dict(saved["model"])
    tokens = torch.arange(len(ops.SURFACE_NAMES))
    n = model.left_index.max().item() + 1
    identity = torch.eye(n, dtype=torch.float64).expand(len(tokens), -1, -1)
    transport = model.transport(identity, tokens)[0]
    if model.viscosity:
        transport = ((1 - 2 * model.viscosity) * transport
                     + model.viscosity * torch.roll(transport, 1, 1)
                     + model.viscosity * torch.roll(transport, -1, 1))
    assert float(transport.min()) >= 0
    kernels = model.reaction_kernel(tokens)
    report = {}
    for t, name in enumerate(ops.SURFACE_NAMES):
        pi, a, b = ops.SURFACE_TRIPLES[name]
        candidates = []
        position_minima = []
        for j in range(n):
            position_candidates = []
            for digit in range(10):
                target = (a * digit + b[j]) % 10
                for rival in range(10):
                    if rival == target:
                        continue
                    differences = kernels[t, j, :, target] - kernels[t, j, :, rival]
                    distractor_digit = int(differences.argmin())
                    source_mass = transport[t, j, pi[j]]
                    other_mass = transport[t, j].sum() - source_mass
                    margin = float(source_mass * differences[digit]
                                   + other_mass * differences[distractor_digit])
                    x = [distractor_digit] * n
                    x[pi[j]] = digit
                    entry = (margin, x, j, target, rival)
                    candidates.append(entry)
                    position_candidates.append(margin)
            position_minima.append(min(position_candidates))
        margin, x, j, target, rival = min(candidates, key=lambda entry: entry[0])
        inputs = torch.tensor([x])
        token = torch.tensor([[t]])
        probs64 = model(inputs, token)["probs"][0]
        probs32 = original(inputs, token)["probs"][0]
        truth = ops.apply_triple(ops.SURFACE_TRIPLES[name], np.asarray([x]))[0].tolist()
        pred64 = probs64.argmax(-1).tolist()
        pred32 = probs32.argmax(-1).tolist()
        actual_margin64 = float(probs64[j, target] - probs64[j, rival])
        # Row normalization can scale the analytic unnormalized margin; these
        # stochastic matrices have sums 1 to float64 precision.
        assert abs(actual_margin64 - margin) < 1e-12
        if margin < -1e-10:
            assert pred64 != truth
        report[name] = {
            "numeric_sufficient_bound_positive": margin > 0,
            "worst_case_margin_float64": margin,
            "position_minima_float64": position_minima,
            "transport_argmax": transport[t].argmax(-1).tolist(),
            "truth_permutation": list(pi),
            "worst_case_witness": {
                "input": x, "output_position_zero_based": j,
                "target_digit": target, "competitor_digit": rival,
                "truth": truth, "prediction_float64": pred64,
                "prediction_original_float32": pred32,
                "float64_exact_match": pred64 == truth,
                "original_float32_exact_match": pred32 == truth,
                "observed_margin_float64": actual_margin64,
                "observed_margin_original_float32": float(
                    probs32[j, target] - probs32[j, rival]),
            },
        }
    certified = [name for name, value in report.items()
                 if value["numeric_sufficient_bound_positive"]]
    return {
        "arm": arm, "seed": saved["config"]["seed"],
        "checkpoint_step": saved["step"],
        "checkpoint_path": path.relative_to(REPO).as_posix(),
        "checkpoint_sha256": sha256(path),
        "tokens_with_positive_numeric_bound": certified,
        "minimum_positive_token_margin": min(
            (report[name]["worst_case_margin_float64"] for name in certified),
            default=None),
        "tokens": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = sorted((REPO / "AtomV2" / "runs" / "e11").glob("*/checkpoints/final.pt"))
    if not paths:
        raise SystemExit("No real E11 final checkpoints found")
    result = {
        "review_date": "2026-09-12", "post_hoc": True,
        "original_run_artifacts_modified": False,
        "method": "Analytic independent-digit worst-case margin; float64 evaluation",
        "scope": "All six-digit inputs for each token independently",
        "limitations": (
            "Not interval-verified or a formal floating-point proof. Arbitrary-length "
            "composition follows in exact arithmetic only for positive-bound tokens, "
            "unchanged weights, clean interfaces, and one-hot boundary projection. "
            "This post-hoc audit does not change the registered E11 failure verdict."
        ),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__},
        "source_sha256": {name: sha256(REPO / name) for name in (
            "AtomV2/Harness/atomv2/exchange_model.py",
            "AtomV2/Harness/atomv2/ops.py",
            "AtomV2/Harness/atomv2/registered.py")},
        "runs": [audit(path) for path in paths],
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
