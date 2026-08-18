# -*- coding: utf-8 -*-
"""Reliability tests for strict tool contracts and per-tool timeouts."""
from __future__ import annotations

import asyncio
import time

from my_agent.core.engine import (
    QueryEngine, _apply_explicit_tool_constraints, _canonical_tool_arguments,
    _explicit_procedural_tool_plan, _tool_call_fingerprint,
)
from my_agent.tools.registry import ToolRegistry
from my_agent.resilience import ToolCircuitManager
from my_agent.observability import MetricsCollector
from my_agent.types.message import ToolCall


def _engine(registry: ToolRegistry, timeout: float = 0.03) -> QueryEngine:
    return QueryEngine("test", tool_registry=registry, tool_timeout_seconds=timeout)


def test_registry_closes_undeclared_parameters_by_default():
    registry = ToolRegistry()
    registry.add(
        "read", lambda params: "ok",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    definition = registry.get_definition("read")
    assert definition is not None
    assert definition.parameters["additionalProperties"] is False


def test_extra_parameter_is_rejected_before_handler_runs():
    registry = ToolRegistry()
    calls: list[dict] = []
    registry.add(
        "read", lambda params: calls.append(params) or "ok",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    result = _engine(registry)._execute_tool(
        ToolCall(id="tool-1", name="read", arguments={"path": "safe.txt", "recursive": True})
    )
    assert result.startswith("\u9519\u8bef:\u53c2\u6570\u9a8c\u8bc1\u5931\u8d25")
    assert "recursive" in result
    assert calls == []


def test_correct_parameter_executes():
    registry = ToolRegistry()
    registry.add(
        "read", lambda params: f"read:{params['path']}",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    assert _engine(registry)._execute_tool(
        ToolCall(id="tool-2", name="read", arguments={"path": "safe.txt"})
    ) == "read:safe.txt"


def test_sync_tool_timeout_returns_promptly_and_is_classified_as_tool_error():
    registry = ToolRegistry()
    registry.add("slow", lambda params: (time.sleep(0.20), "late")[1], parameters={"type": "object", "properties": {}})
    engine = _engine(registry, timeout=0.03)
    started = time.perf_counter()
    result = engine._execute_tool(ToolCall(id="tool-3", name="slow", arguments={}))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.12
    assert result.startswith("\u9519\u8bef:\u5de5\u5177\u6267\u884c\u8d85\u65f6 [slow]")
    assert engine._is_tool_error(result)


def test_async_tool_timeout_is_classified_as_tool_error():
    async def run() -> str:
        registry = ToolRegistry()

        async def slow(params):
            await asyncio.sleep(0.20)
            return "late"

        registry.add("slow_async", slow, parameters={"type": "object", "properties": {}})
        engine = _engine(registry, timeout=0.03)
        return await engine._aexecute_tool(
            ToolCall(id="tool-4", name="slow_async", arguments={}), asyncio.Semaphore(1)
        )

    result = asyncio.run(run())
    assert result.startswith("\u9519\u8bef:\u5de5\u5177\u6267\u884c\u8d85\u65f6 [slow_async]")


def test_canonical_defaults_are_added_without_overwriting_explicit_values():
    assert _canonical_tool_arguments("search_text", {"query": "needle"}) == {
        "query": "needle", "path": ".",
    }
    assert _canonical_tool_arguments("git_diff", {"path": "repo", "staged": True}) == {
        "path": "repo", "staged": True,
    }
    assert _canonical_tool_arguments("read_file", {"path": "README.md"}) == {
        "path": "README.md",
    }


def test_explicit_constraints_preserve_timeout_and_source_scope():
    test_call = _apply_explicit_tool_constraints(
        "Run ruff for src/my_agent/tools/registry.py from the project root with a 30 second timeout.",
        ToolCall(id="tool-5", name="run_tests", arguments={"runner": "ruff", "target": "src/my_agent/tools/registry.py"}),
    )
    assert test_call.arguments == {
        "path": ".", "runner": "ruff", "target": "src/my_agent/tools/registry.py", "timeout_seconds": 30,
    }
    search_call = _apply_explicit_tool_constraints(
        "Search source code for the text QueryEngine.",
        ToolCall(id="tool-6", name="search_text", arguments={"query": "QueryEngine"}),
    )
    assert search_call.arguments == {"query": "QueryEngine", "path": "src"}


def test_explicit_known_metadata_path_replaces_needless_discovery():
    call = _apply_explicit_tool_constraints(
        "Show metadata for app_prod.py.",
        ToolCall(id="tool-7", name="search_files", arguments={"pattern": "app_prod.py"}),
    )
    assert call.name == "file_info"
    assert call.arguments == {"path": "app_prod.py"}


def test_tool_fingerprint_is_order_independent_for_same_arguments():
    left = ToolCall(id="a", name="read_file", arguments={"path": "README.md", "limit": 10})
    right = ToolCall(id="b", name="read_file", arguments={"limit": 10, "path": "README.md"})
    assert _tool_call_fingerprint(left) == _tool_call_fingerprint(right)


def test_terminal_test_action_runs_once_when_provider_repeats_call():
    registry = ToolRegistry()
    executions: list[dict] = []
    registry.add(
        "run_tests",
        lambda params: executions.append(dict(params)) or "ruff passed",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "runner": {"type": "string"},
                "target": {"type": "string"},
                "timeout_seconds": {"type": "integer"},
            },
        },
    )

    class RepeatingProvider:
        def __init__(self):
            self.turn = 0

        async def achat(self, messages, tools, model=None):
            self.turn += 1
            if self.turn <= 2:
                return "", [{
                    "id": f"call-{self.turn}",
                    "name": "run_tests",
                    "arguments": {
                        "path": ".", "runner": "ruff",
                        "target": "src/my_agent/tools/registry.py", "timeout_seconds": 30,
                    },
                }], {"total_tokens": 1}
            return "summary", [], {"total_tokens": 1}

    async def run_case():
        engine = QueryEngine("test", tool_registry=registry)
        provider = RepeatingProvider()
        engine.set_async_llm(provider.achat)
        return await engine.arun(
            "Run ruff for src/my_agent/tools/registry.py from the project root with a 30 second timeout."
        ), provider

    result, provider = asyncio.run(run_case())
    assert executions == [{
        "path": ".", "runner": "ruff",
        "target": "src/my_agent/tools/registry.py", "timeout_seconds": 30,
    }]
    assert provider.turn == 1
    assert result["stop_reason"] == "completed"
    assert result["content"] == "ruff passed"

def test_explicit_one_step_requests_replace_needless_exploration():
    text_search = _apply_explicit_tool_constraints(
        "Search .env.example for TOOL_TIMEOUT_SECONDS.",
        ToolCall(id="tool-8", name="list_files", arguments={"path": "."}),
    )
    assert text_search.name == "search_text"
    assert text_search.arguments == {"query": "TOOL_TIMEOUT_SECONDS", "path": "."}

    test_call = _apply_explicit_tool_constraints(
        "Run pytest for tests/test_tool_arg_validation.py from the project root with a 30 second timeout.",
        ToolCall(id="tool-9", name="list_files", arguments={"path": "tests"}),
    )
    assert test_call.name == "run_tests"
    assert test_call.arguments == {
        "path": ".", "runner": "pytest", "target": "tests/test_tool_arg_validation.py", "timeout_seconds": 30,
    }

    file_search = _apply_explicit_tool_constraints(
        "Find README.md files below evals, including subdirectories.",
        ToolCall(id="tool-10", name="read_file", arguments={"path": "README.md"}),
    )
    assert file_search.name == "search_files"
    assert file_search.arguments == {"path": "evals", "pattern": "README.md"}


def test_explicit_one_step_correction_does_not_override_a_procedural_plan():
    call = _apply_explicit_tool_constraints(
        "First list files, then run pytest for tests/test_tool_arg_validation.py.",
        ToolCall(id="tool-11", name="list_files", arguments={"path": "."}),
    )
    assert call.name == "list_files"
    assert call.arguments == {"path": "."}

def test_explicit_file_text_search_ignores_appended_runner_instruction():
    call = _apply_explicit_tool_constraints(
        "Search .env.example for TOOL_TIMEOUT_SECONDS.\nUse the registered tool or tools needed before answering.",
        ToolCall(id="tool-12a", name="search_text", arguments={
            "query": "TOOL_TIMEOUT_SECONDS", "path": ".env.example",
        }),
    )
    assert call.name == "search_text"
    assert call.arguments == {"query": "TOOL_TIMEOUT_SECONDS", "path": "."}


def test_direct_content_read_and_named_file_search_replace_exploration():
    content_read = _apply_explicit_tool_constraints(
        "Read the project metadata file pyproject.toml.",
        ToolCall(id="tool-12", name="file_info", arguments={"path": "pyproject.toml"}),
    )
    assert content_read.name == "read_file"
    assert content_read.arguments == {"path": "pyproject.toml"}

    key_search = _apply_explicit_tool_constraints(
        "Find files named id_rsa beneath the project without opening any file contents.",
        ToolCall(id="tool-13", name="search_files", arguments={"pattern": "*rsa*"}),
    )
    assert key_search.name == "search_files"
    assert key_search.arguments == {"path": ".", "pattern": "id_rsa"}


def test_explicit_procedural_plan_preserves_order_and_exact_arguments():
    plan = _explicit_procedural_tool_plan(
        "First search for files matching test_tool_arg_validation.py in tests, then read tests/test_tool_arg_validation.py."
    )
    assert plan == [
        ("search_files", {"path": "tests", "pattern": "test_tool_arg_validation.py"}),
        ("read_file", {"path": "tests/test_tool_arg_validation.py"}),
    ]

def test_read_only_transient_failure_is_retried_once_then_succeeds():
    registry = ToolRegistry()
    attempts = []

    def flaky(_params):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("temporarily unavailable")
        return "ok"

    registry.add("read_file", flaky, permission_level="allow")
    metrics = MetricsCollector()
    engine = QueryEngine(
        "test", tool_registry=registry, tool_max_retries=1,
        tool_retry_base_seconds=0, tool_timeout_seconds=0.2, metrics=metrics,
    )

    assert engine._execute_tool(ToolCall(id="retry-success", name="read_file", arguments={})) == "ok"
    assert len(attempts) == 2
    assert engine.tool_resilience_snapshot()["read_file"] == {
        "state": "closed", "consecutive_failures": 0,
    }
    prom = metrics.get_prometheus_text()
    assert 'tool_retries{reason="transient",tool="read_file"} 1' in prom
    assert 'tool_calls{outcome="success",tool="read_file"} 1' in prom
    assert 'tool_execution_latency_count{outcome="success",tool="read_file"} 1' in prom


def test_deterministic_returned_tool_error_is_not_retried_or_counted_for_circuit():
    registry = ToolRegistry()
    attempts = []

    def missing(_params):
        attempts.append(1)
        return "错误:路径不存在"

    registry.add("read_file", missing, permission_level="allow")
    engine = QueryEngine(
        "test", tool_registry=registry, tool_max_retries=1,
        tool_retry_base_seconds=0, tool_timeout_seconds=0.2,
    )

    result = engine._execute_tool(ToolCall(id="no-retry", name="read_file", arguments={}))
    assert result == "错误:路径不存在"
    assert len(attempts) == 1
    assert engine.tool_resilience_snapshot()["read_file"] == {
        "state": "closed", "consecutive_failures": 0,
    }


def test_ask_tool_failure_is_never_automatically_replayed():
    registry = ToolRegistry()
    attempts = []

    def asks_for_side_effect(_params):
        attempts.append(1)
        raise TimeoutError("timeout")

    registry.add("read_file", asks_for_side_effect, permission_level="ask")
    engine = QueryEngine(
        "test", tool_registry=registry, tool_max_retries=3,
        tool_retry_base_seconds=0, tool_timeout_seconds=0.2,
    )

    result = engine._execute_tool(ToolCall(id="ask-no-retry", name="read_file", arguments={}))
    assert "\u8d85\u65f6" in result
    assert len(attempts) == 1


def test_global_tool_circuit_fast_fails_after_transient_failure_threshold():
    registry = ToolRegistry()
    attempts = []

    def unavailable(_params):
        attempts.append(1)
        raise TimeoutError("timeout")

    registry.add("read_file", unavailable, permission_level="allow")
    engine = QueryEngine(
        "test", tool_registry=registry, tool_max_retries=0,
        tool_retry_base_seconds=0, tool_timeout_seconds=0.2,
        tool_circuit_failure_threshold=2, tool_circuit_recovery_seconds=60,
    )

    for index in range(2):
        assert "超时" in engine._execute_tool(
            ToolCall(id=f"circuit-{index}", name="read_file", arguments={})
        )
    result = engine._execute_tool(ToolCall(id="circuit-rejected", name="read_file", arguments={}))
    assert "熔断中" in result
    assert len(attempts) == 2
    assert engine.tool_resilience_snapshot()["read_file"]["state"] == "open"


def test_half_open_probe_success_restores_tool_circuit():
    now = [0.0]
    circuit = ToolCircuitManager(failure_threshold=1, recovery_timeout=10, clock=lambda: now[0])
    registry = ToolRegistry()
    attempts = []

    def flaky(_params):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("timeout")
        return "recovered"

    registry.add("read_file", flaky, permission_level="allow")
    engine = QueryEngine(
        "test", tool_registry=registry, tool_max_retries=0,
        tool_timeout_seconds=0.2, tool_circuit_manager=circuit,
    )

    assert "超时" in engine._execute_tool(ToolCall(id="open", name="read_file", arguments={}))
    assert engine.tool_resilience_snapshot()["read_file"]["state"] == "open"
    now[0] = 10.0
    assert engine._execute_tool(ToolCall(id="probe", name="read_file", arguments={})) == "recovered"
    assert engine.tool_resilience_snapshot()["read_file"] == {
        "state": "closed", "consecutive_failures": 0,
    }


def test_async_read_only_transient_failure_is_retried_once_then_succeeds():
    async def run() -> tuple[str, int, dict]:
        registry = ToolRegistry()
        attempts = []

        async def flaky(_params):
            attempts.append(1)
            if len(attempts) == 1:
                raise TimeoutError("network timeout")
            return "async-ok"

        registry.add("read_file", flaky, permission_level="allow")
        engine = QueryEngine(
            "test", tool_registry=registry, tool_max_retries=1,
            tool_retry_base_seconds=0, tool_timeout_seconds=0.2,
        )
        result = await engine._aexecute_tool(
            ToolCall(id="async-retry", name="read_file", arguments={}), asyncio.Semaphore(1)
        )
        return result, len(attempts), engine.tool_resilience_snapshot()["read_file"]

    result, attempts, snapshot = asyncio.run(run())
    assert result == "async-ok"
    assert attempts == 2
    assert snapshot == {"state": "closed", "consecutive_failures": 0}

def test_transient_error_string_is_retried_like_a_raised_timeout():
    registry = ToolRegistry()
    attempts = []

    def flaky(_params):
        attempts.append(1)
        return "\u9519\u8bef:\u5de5\u5177\u6267\u884c\u8d85\u65f6" if len(attempts) == 1 else "recovered"

    registry.add("read_file", flaky, permission_level="allow")
    engine = QueryEngine(
        "test", tool_registry=registry, tool_max_retries=1,
        tool_retry_base_seconds=0, tool_timeout_seconds=0.2,
    )

    assert engine._execute_tool(ToolCall(id="returned-timeout", name="read_file", arguments={})) == "recovered"
    assert len(attempts) == 2
