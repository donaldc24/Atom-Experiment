"""Unified command-line interface for the stabilized H1 workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from cgmoe_h1 import __version__


def _add_common_training_budget(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--validation-examples", type=int)
    parser.add_argument("--epochs", type=int)


def build_parser() -> argparse.ArgumentParser:
    """Build the package CLI without importing the ML stack."""
    parser = argparse.ArgumentParser(
        prog="cgmoe-h1",
        description="Train, evaluate, and summarize the CGMoE H1 experiment.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    inspect_parser = commands.add_parser(
        "inspect-model", help="inspect BERT linear modules and adapter targets"
    )
    inspect_parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    inspect_parser.add_argument("--model", help="override the model named by the config")
    inspect_parser.set_defaults(handler=_inspect_model)

    independent = commands.add_parser(
        "train-independent", help="train independent task-specific LoRA adapters"
    )
    independent.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    independent.add_argument("--task", action="append", dest="tasks")
    independent.add_argument("--seed", type=int, default=17)
    independent.add_argument("--rank", type=int)
    independent.add_argument(
        "--output-root", type=Path, default=Path("results/independent_lora")
    )
    independent.add_argument("--development", action="store_true")
    independent.add_argument("--force", action="store_true")
    _add_common_training_budget(independent)
    independent.set_defaults(handler=_train_independent)

    atoms = commands.add_parser(
        "train-atoms", help="train a shared atom dictionary across selected tasks"
    )
    atoms.add_argument("--config", type=Path, default=Path("configs/atoms.yaml"))
    atoms.add_argument("--task", action="append", dest="tasks")
    atoms.add_argument("--seed", type=int, default=17)
    atoms.add_argument("--atom-count", type=int)
    atoms.add_argument("--top-k", type=int)
    atoms.add_argument("--sparsity-lambda", type=float)
    atoms.add_argument("--freeze-atoms", action="store_true")
    atoms.add_argument("--shuffle-training-labels", action="store_true")
    atoms.add_argument("--output-root", type=Path, default=Path("results/shared_atoms"))
    atoms.add_argument("--development", action="store_true")
    atoms.add_argument("--force", action="store_true")
    _add_common_training_budget(atoms)
    atoms.set_defaults(handler=_train_atoms)

    evaluate = commands.add_parser(
        "evaluate-top-k", help="reload a shared checkpoint and apply a fixed top-k mask"
    )
    evaluate.add_argument("--config", type=Path, default=Path("configs/atoms.yaml"))
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=17)
    evaluate.add_argument("--k", type=int, default=4)
    evaluate.set_defaults(handler=_evaluate_top_k)

    summarize = commands.add_parser(
        "summarize", help="regenerate the paired-seed H1 decision report"
    )
    summarize.add_argument(
        "--independent-root", type=Path, default=Path("results/independent_lora")
    )
    summarize.add_argument("--shared-root", type=Path, default=Path("results/shared_atoms"))
    summarize.add_argument("--output-dir", type=Path, default=Path("results/h1_report"))
    summarize.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    summarize.set_defaults(handler=_summarize)

    return parser


def _with_budget_overrides(config: object, args: argparse.Namespace):
    changes = {
        field: value
        for field, value in (
            ("train_examples_per_task", args.train_examples),
            ("validation_examples_per_task", args.validation_examples),
            ("epochs", args.epochs),
        )
        if value is not None
    }
    return config.with_overrides(**changes) if changes else config


def _inspect_model(args: argparse.Namespace) -> int:
    from torch import nn
    from transformers import AutoModel

    from cgmoe_h1.config import load_config

    config = load_config(args.config)
    model_name = args.model or config.base_model
    model = AutoModel.from_pretrained(model_name)
    target_suffixes = tuple(f"attention.self.{name}" for name in config.target_modules)
    linears: list[tuple[str, nn.Linear]] = []
    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linears.append((name, module))
            if name.endswith(target_suffixes):
                targets.append((name, module))

    print(f"Model: {model_name}")
    print(f"Transformer layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Linear modules: {len(linears)}")
    print(f"Adapter targets: {len(targets)}")
    for name, module in targets:
        print(f"  {name}: ({module.out_features}, {module.in_features})")
    print(f"Total base-model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return 0


def _train_independent(args: argparse.Namespace) -> int:
    from cgmoe_h1.config import load_config
    from cgmoe_h1.experiments import run_independent_lora

    config = _with_budget_overrides(load_config(args.config).with_overrides(seed=args.seed), args)
    result = run_independent_lora(
        config,
        args.output_root,
        tasks=tuple(args.tasks or config.tasks),
        run_kind="development" if args.development else "confirmatory",
        rank=args.rank,
        force=args.force,
    )
    for task, record in result["tasks"].items():
        print(f"{task}: {record['best']['metrics']['primary_score']:.6f}")
    return 0


def _train_atoms(args: argparse.Namespace) -> int:
    from cgmoe_h1.config import load_config
    from cgmoe_h1.experiments import run_shared_atoms

    config = _with_budget_overrides(load_config(args.config).with_overrides(seed=args.seed), args)
    result = run_shared_atoms(
        config,
        args.output_root,
        tasks=tuple(args.tasks or config.tasks),
        run_kind="development" if args.development else "confirmatory",
        atom_count=args.atom_count,
        top_k=args.top_k,
        sparsity_lambda=args.sparsity_lambda,
        freeze_atoms=args.freeze_atoms,
        shuffle_training_labels=args.shuffle_training_labels,
        force=args.force,
    )
    for task, record in result["tasks"].items():
        print(
            f"{task}: all={record['all_atoms']['metrics']['primary_score']:.6f}, "
            f"top-k={record['top_k']['metrics']['primary_score']:.6f}"
        )
    return 0


def _evaluate_top_k(args: argparse.Namespace) -> int:
    from cgmoe_h1.config import load_config
    from cgmoe_h1.experiments import evaluate_atom_checkpoint

    config = load_config(args.config).with_overrides(seed=args.seed)
    record = evaluate_atom_checkpoint(config, args.run_dir, top_k=args.k)
    for task, result in record["tasks"].items():
        print(f"{task}: {result['metrics']['primary_score']:.6f}")
    return 0


def _summarize(args: argparse.Namespace) -> int:
    from cgmoe_h1.reporting import generate_h1_report

    summary, summary_path, report_path = generate_h1_report(
        args.independent_root,
        args.shared_root,
        args.output_dir,
        seeds=tuple(args.seeds),
    )
    verdict = "PASS" if summary["preregistered_pass"] else "FAIL"
    print(f"H1 decision: {verdict} - {summary['decision']}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command and return its process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``
    raise SystemExit(main())
