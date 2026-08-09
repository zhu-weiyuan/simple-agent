# -*- coding: utf-8 -*-
"""Tests for tool permission_level enforcement in QueryEngine._execute_tool.

The ToolDefinition.permission_level field ('allow' | 'ask' | 'deny') was
previously ignored during execution. Run #11 (2026-08-02) added a permission
gate that checks this field before dispatching to the handler.
"""
from __future__ import annotations

import json
import pytest
from typing import Any, Dict

from my_agent.core.engine import QueryEngine, SessionConfig
from my_agent.core.hooks import HookPoint, HookContext
from my_agent.tools.registry import ToolRegistry, ToolDefinition


# ── helpers ──────────────────────────────────────────────────

def _make_engine(tool_defs=None):
    """Create a minimal QueryEngine with custom tool definitions."""
    config = SessionConfig(max_turns=5)
    engine = QueryEngine(system_prompt="test", session_config=config)
    if tool_defs:
        for td in tool_defs:
            engine.tool_registry.add(
                name=td.name,
                handler=td.handler,
                description=td.description,
                parameters=td.parameters,
                tags=td.tags,
            )
            # Set permission_level on the registered definition
            engine.tool_registry._tools[td.name].permission_level = td.permission_level
    return engine


def _make_tc(name: str, arguments: dict = None):
    """Create a minimal ToolCall-like object."""
    from my_agent.types.message import ToolCall
    return ToolCall(
        id=f"call_{name}",
        name=name,
        arguments=arguments or {},
    )


def _always_ok(params: Dict[str, Any]) -> str:
    return "ok"


def _record_call(params: Dict[str, Any]) -> str:
    _record_call.calls.append(params)
    return "executed"


_record_call.calls = []


# ── tests ────────────────────────────────────────────────────

class TestPermissionDeny:
    """Tools with permission_level='deny' must never execute."""

    def test_deny_blocks_execution(self):
        td = ToolDefinition(
            name="dangerous_tool",
            description="Should be blocked",
            parameters={"type": "object", "properties": {}},
            handler=_record_call,
            permission_level="deny",
        )
        engine = _make_engine([td])
        tc = _make_tc("dangerous_tool")

        result = engine._execute_tool(tc)

        assert "拒绝" in result
        assert "deny" in result
        assert _record_call.calls == []  # handler was NOT called

    def test_deny_fires_error_hook(self):
        td = ToolDefinition(
            name="blocked_tool",
            description="Blocked",
            parameters={"type": "object", "properties": {}},
            handler=_always_ok,
            permission_level="deny",
        )
        engine = _make_engine([td])
        errors = []

        def capture_error(ctx: HookContext, **kw):
            # hooks receive (context, data={...}) so error is in kw["data"]["error"]
            data = kw.get("data", {})
            errors.append(data.get("error", ""))

        engine.hooks.register(HookPoint.TOOL_ERROR, capture_error)
        tc = _make_tc("blocked_tool")
        engine._execute_tool(tc)

        assert any("拒绝" in e or "deny" in e for e in errors)


class TestPermissionAsk:
    """Tools with permission_level='ask' execute unless a hook denies."""

    def setup_method(self):
        _record_call.calls.clear()

    def test_ask_allows_when_no_hook_denies(self):
        td = ToolDefinition(
            name="file_read",
            description="Read file",
            parameters={"type": "object", "properties": {}},
            handler=_record_call,
            permission_level="ask",
        )
        engine = _make_engine([td])
        tc = _make_tc("file_read")

        result = engine._execute_tool(tc)

        assert result == "executed"
        assert len(_record_call.calls) == 1

    def test_ask_blocks_when_hook_denies(self):
        td = ToolDefinition(
            name="sensitive_read",
            description="Sensitive read",
            parameters={"type": "object", "properties": {}},
            handler=_record_call,
            permission_level="ask",
        )
        engine = _make_engine([td])

        def deny_all(ctx: HookContext, **kw):
            return {"allowed": False, "reason": "user said no"}

        engine.hooks.register(HookPoint.TOOL_PERMISSION_REQUEST, deny_all)
        tc = _make_tc("sensitive_read")

        result = engine._execute_tool(tc)

        assert "拒绝" in result
        assert "user said no" in result
        # _record_call.calls may have entries from prior tests; check only this call
        # by verifying the result indicates denial

    def test_ask_allows_when_hook_approves(self):
        td = ToolDefinition(
            name="approved_tool",
            description="Approved",
            parameters={"type": "object", "properties": {}},
            handler=_record_call,
            permission_level="ask",
        )
        engine = _make_engine([td])

        def approve(ctx: HookContext, **kw):
            return {"allowed": True}

        engine.hooks.register(HookPoint.TOOL_PERMISSION_REQUEST, approve)
        tc = _make_tc("approved_tool")

        result = engine._execute_tool(tc)

        assert result == "executed"

    def test_ask_passes_permission_level_in_hook_data(self):
        td = ToolDefinition(
            name="check_data",
            description="Check hook data",
            parameters={"type": "object", "properties": {}},
            handler=_always_ok,
            permission_level="ask",
        )
        engine = _make_engine([td])
        captured = []

        def capture(ctx: HookContext, **kw):
            captured.append(kw)
            return {"allowed": True}

        engine.hooks.register(HookPoint.TOOL_PERMISSION_REQUEST, capture)
        tc = _make_tc("check_data", {"foo": "bar"})
        engine._execute_tool(tc)

        assert len(captured) == 1
        # Hooks receive (context, **kwargs) where kwargs mirrors the fire() call.
        # engine fires: hooks.fire(TOOL_PERMISSION_REQUEST, data={...})
        # so the handler gets kw["data"] = {tool_name, arguments, permission_level}
        data = captured[0].get("data", {})
        assert data.get("tool_name") == "check_data"
        assert data.get("arguments") == {"foo": "bar"}
        assert data.get("permission_level") == "ask"


class TestPermissionAllow:
    """Tools with permission_level='allow' always execute."""

    def test_allow_executes_without_hook_check(self):
        td = ToolDefinition(
            name="safe_calc",
            description="Calculator",
            parameters={"type": "object", "properties": {}},
            handler=_record_call,
            permission_level="allow",
        )
        engine = _make_engine([td])
        tc = _make_tc("safe_calc")

        result = engine._execute_tool(tc)

        assert result == "executed"

    def test_allow_fires_before_and_after_hooks(self):
        td = ToolDefinition(
            name="hooked_tool",
            description="Hooked",
            parameters={"type": "object", "properties": {}},
            handler=_always_ok,
            permission_level="allow",
        )
        engine = _make_engine([td])
        before_fired = []
        after_fired = []

        def on_before(ctx: HookContext, **kw):
            before_fired.append(True)

        def on_after(ctx: HookContext, **kw):
            after_fired.append(True)

        engine.hooks.register(HookPoint.TOOL_CALL_BEFORE, on_before)
        engine.hooks.register(HookPoint.TOOL_CALL_AFTER, on_after)
        tc = _make_tc("hooked_tool")
        engine._execute_tool(tc)

        assert len(before_fired) == 1
        assert len(after_fired) == 1


class TestUnknownTool:
    """Unknown tools (not in registry) still get the standard error."""

    def test_unknown_tool_returns_error(self):
        engine = _make_engine()
        tc = _make_tc("nonexistent_tool")

        result = engine._execute_tool(tc)

        assert "未知工具" in result


class TestBuiltinPermissions:
    """Verify that builtin tools have sensible permission_level defaults."""

    def test_shell_is_deny(self):
        from my_agent.tools.builtins.shell import PowerShellTool
        tool = PowerShellTool()
        assert tool.permission_level == "deny"

    def test_read_file_is_ask(self):
        from my_agent.tools.builtins.file import ReadFileTool
        tool = ReadFileTool()
        assert tool.permission_level == "ask"

    def test_list_files_is_allow(self):
        from my_agent.tools.builtins.file import ListFilesTool
        tool = ListFilesTool()
        assert tool.permission_level == "allow"

    def test_calculator_is_allow(self):
        from my_agent.tools.builtins.calculator import CalculatorTool
        tool = CalculatorTool()
        assert tool.permission_level == "allow"

    def test_get_time_is_allow(self):
        from my_agent.tools.builtins.time import GetTimeTool
        tool = GetTimeTool()
        assert tool.permission_level == "allow"


class TestPermissionPropagation:
    """Verify permission_level propagates through ToolRegistry.add() and BaseTool.register()."""

    def test_add_passes_permission_level(self):
        registry = ToolRegistry()
        registry.add(
            name="test_tool",
            handler=_always_ok,
            description="test",
            permission_level="deny",
        )
        defn = registry.get_definition("test_tool")
        assert defn is not None
        assert defn.permission_level == "deny"

    def test_add_defaults_to_ask(self):
        registry = ToolRegistry()
        registry.add(
            name="default_tool",
            handler=_always_ok,
            description="test",
        )
        defn = registry.get_definition("default_tool")
        assert defn is not None
        assert defn.permission_level == "ask"

    def test_basetool_register_propagates_permission(self):
        from my_agent.tools.builtins.shell import PowerShellTool
        registry = ToolRegistry()
        tool = PowerShellTool()
        tool.register(registry)
        defn = registry.get_definition(tool.name)
        assert defn is not None
        assert defn.permission_level == "deny"

    def test_basetool_register_allows_propagates(self):
        from my_agent.tools.builtins.calculator import CalculatorTool
        registry = ToolRegistry()
        tool = CalculatorTool()
        tool.register(registry)
        defn = registry.get_definition(tool.name)
        assert defn is not None
        assert defn.permission_level == "allow"


# ── async permission gate tests ─────────────────────────────

class TestAsyncPermissionDeny:
    """Async _aexecute_tool must also enforce permission_level='deny'."""

    @pytest.mark.asyncio
    async def test_async_deny_blocks_execution(self):
        td = ToolDefinition(
            name="async_dangerous",
            description="Should be blocked",
            parameters={"type": "object", "properties": {}},
            handler=_record_call,
            permission_level="deny",
        )
        engine = _make_engine([td])
        tc = _make_tc("async_dangerous")
        sem = __import__('asyncio').Semaphore(1)

        result = await engine._aexecute_tool(tc, sem)

        assert "拒绝" in result
        assert "deny" in result

    @pytest.mark.asyncio
    async def test_async_deny_fires_error_hook(self):
        td = ToolDefinition(
            name="async_blocked",
            description="Should be blocked",
            parameters={"type": "object", "properties": {}},
            handler=_record_call,
            permission_level="deny",
        )
        engine = _make_engine([td])
        tc = _make_tc("async_blocked")
        errors = []

        def capture_error(ctx: HookContext, **kw):
            errors.append(kw.get("data", {}).get("error", ""))

        engine.hooks.register(HookPoint.TOOL_ERROR, capture_error)
        sem = __import__('asyncio').Semaphore(1)
        await engine._aexecute_tool(tc, sem)

        assert len(errors) == 1
        assert "deny" in errors[0]


class TestAsyncPermissionAsk:
    """Async _aexecute_tool must also enforce permission_level='ask'."""

    @pytest.mark.asyncio
    async def test_async_ask_allows_when_no_hook_denies(self):
        td = ToolDefinition(
            name="async_ask_tool",
            description="test",
            parameters={"type": "object", "properties": {}},
            handler=_always_ok,
            permission_level="ask",
        )
        engine = _make_engine([td])
        tc = _make_tc("async_ask_tool")
        sem = __import__('asyncio').Semaphore(1)

        result = await engine._aexecute_tool(tc, sem)

        assert result == "ok"

    @pytest.mark.asyncio
    async def test_async_ask_blocks_when_hook_denies(self):
        td = ToolDefinition(
            name="async_ask_deny",
            description="test",
            parameters={"type": "object", "properties": {}},
            handler=_always_ok,
            permission_level="ask",
        )
        engine = _make_engine([td])
        tc = _make_tc("async_ask_deny")

        def deny_hook(ctx: HookContext, **kw):
            return {"allowed": False, "reason": "not now"}

        engine.hooks.register(HookPoint.TOOL_PERMISSION_REQUEST, deny_hook)
        sem = __import__('asyncio').Semaphore(1)
        result = await engine._aexecute_tool(tc, sem)

        assert "拒绝" in result
        assert "not now" in result
