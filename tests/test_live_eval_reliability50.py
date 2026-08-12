# -*- coding: utf-8 -*-
"""Unit coverage for the production 50-case live evaluator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("live_eval", ROOT / "evals" / "run_live_eval.py")
assert SPEC and SPEC.loader
live_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live_eval)


def test_reliability50_scores_tool_arguments_strictly():
    case = {
        "expected_tool": "read_file",
        "expected_args": {"path": "README.md"},
        "allow_no_tool": False,
    }
    exact = live_eval._score_expected_tool_calls(
        [{"name": "read_file", "arguments": {"path": "README.md"}}], case,
    )
    extra = live_eval._score_expected_tool_calls(
        [{"name": "read_file", "arguments": {"path": "README.md", "recursive": True}}], case,
    )
    assert exact["selection_ok"] is True
    assert exact["parameter_ok"] is True
    assert extra["selection_ok"] is True
    assert extra["parameter_ok"] is False


def test_reliability50_scores_ordered_multistep_calls():
    case = {
        "expected_calls": [
            {"name": "list_files", "arguments": {"path": "tests"}},
            {"name": "read_file", "arguments": {"path": "tests/test_x.py"}},
        ],
        "allow_no_tool": False,
    }
    result = live_eval._score_expected_tool_calls([
        {"name": "list_files", "arguments": {"path": "tests"}},
        {"name": "read_file", "arguments": {"path": "tests/test_x.py"}},
    ], case)
    assert result["selection_ok"] is True
    assert result["parameter_ok"] is True
    assert result["argument_fields_matched"] == 2
    assert result["argument_fields_total"] == 2


def test_reliability50_no_tool_adversarial_case_rejects_calls():
    case = {"allow_no_tool": True}
    no_call = live_eval._score_expected_tool_calls([], case)
    invented = live_eval._score_expected_tool_calls(
        [{"name": "delete_all_files", "arguments": {}}], case,
    )
    assert no_call["selection_ok"] is True
    assert no_call["parameter_ok"] is True
    assert invented["selection_ok"] is False
    assert invented["unnecessary_calls"] == 1


def test_reliability50_accepts_only_documented_canonical_defaults():
    case = {
        "expected_tool": "git_diff",
        "expected_args": {"path": ".", "file": "pyproject.toml"},
        "allow_no_tool": False,
    }
    canonical = live_eval._score_expected_tool_calls(
        [{"name": "git_diff", "arguments": {"path": ".", "file": "pyproject.toml", "staged": False}}], case,
    )
    invented = live_eval._score_expected_tool_calls(
        [{"name": "git_diff", "arguments": {"path": ".", "file": "pyproject.toml", "recursive": True}}], case,
    )
    assert canonical["parameter_ok"] is True
    assert invented["parameter_ok"] is False
