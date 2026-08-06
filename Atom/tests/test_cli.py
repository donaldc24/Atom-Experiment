"""Fast tests for the unified command-line parser."""

from __future__ import annotations

from cgmoe_h1.cli import build_parser, main


def test_cli_lists_stabilized_workflows(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    for command in (
        "inspect-model",
        "train-independent",
        "train-atoms",
        "evaluate-top-k",
        "summarize",
    ):
        assert command in output


def test_cli_parses_top_k_without_importing_model_stack() -> None:
    args = build_parser().parse_args(
        ["evaluate-top-k", "--run-dir", "results/shared_atoms/seed_17", "--k", "2"]
    )
    assert args.command == "evaluate-top-k"
    assert args.seed == 17
    assert args.k == 2


def test_cli_summary_uses_canonical_report_directory() -> None:
    args = build_parser().parse_args(["summarize"])
    assert args.output_dir.as_posix() == "results/h1_report"
