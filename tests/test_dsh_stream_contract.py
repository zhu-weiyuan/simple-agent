from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from my_agent.compaction import CompactionConfig, CompactionEngine
from my_agent.core.engine import QueryEngine
from my_agent.dsh_state_machine import AgentLoop as DSHAgentLoop
from my_agent.types.message import Message
from my_agent.types.session import SessionState
from my_agent.tools.registry import ToolRegistry


def _response(content: str, tool_calls=None):
    calls = []
    for tc in tool_calls or []:
        calls.append(SimpleNamespace(
            id=tc["id"],
            function=SimpleNamespace(name=tc["name"], arguments=tc.get("arguments", "{}")),
        ))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_dsh_always_sends_system_first_and_keeps_user_query():
    seen = []
    calls = 0

    def llm(messages, tools):
        nonlocal calls
        calls += 1
        seen.append(messages)
        if calls == 1:
            return _response("", [{"id": "c1", "name": "list_files", "arguments": '{"path":"."}'}])
        if calls == 2:
            return _response("Let me continue exploring.")
        return _response("任务完成：已检查项目。")

    registry = ToolRegistry()
    registry.add("list_files", lambda args: "[DIR] src\n[FILE] README.md")
    engine = QueryEngine("BASE SYSTEM", tool_registry=registry)
    engine.set_llm(llm)

    result = asyncio.run(engine.arun_with_dsh(
        "完整全面的看一遍你工作区项目",
        session=SessionState.create("BASE SYSTEM"),
        max_tool_calls=10,
        session_id="system-contract",
    ))

    assert result["content"].startswith("任务完成")
    assert len(seen) == 3
    for messages in seen:
        assert messages[0]["role"] == "system"
        assert "BASE SYSTEM" in messages[0]["content"]
        assert any(m["role"] == "user" and "完整全面" in (m.get("content") or "") for m in messages)
        assert sum(1 for m in messages if m["role"] == "system") == 1
    # A continuation must be a synthetic user turn.  This preserves the
    # assistant -> user -> assistant alternation required by strict local
    # chat templates, while the real system prompt remains at index zero.
    continuation = [m for m in seen[2] if m["role"] == "user" and "探索未完成" in (m.get("content") or "")]
    assert continuation
    assert seen[2][-1]["role"] == "user"


def test_exploration_mode_requires_broad_scope_not_a_single_directory_listing():
    # A concise file listing should be allowed to finish immediately after one
    # tool call; only explicitly broad project/workspace inspection uses the
    # long-exploration safety floor.
    assert not DSHAgentLoop._is_broad_exploration_request(
        "列出当前目录中前 5 个文件或文件夹，只返回名称。"
    )
    assert not DSHAgentLoop._is_broad_exploration_request(
        "List the files in the current directory."
    )
    assert DSHAgentLoop._is_broad_exploration_request(
        "完整全面的看一遍你工作区项目"
    )
    assert DSHAgentLoop._is_broad_exploration_request(
        "Explore the workspace directory and read every file."
    )


def test_parallel_dsh_requests_keep_their_loops_and_results_isolated():
    seen = []
    lock = threading.Lock()

    def llm(messages, tools):
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        time.sleep(0.05)
        with lock:
            seen.append(user)
        return _response(f"完成：{user}")

    engine = QueryEngine("parallel system")
    engine.set_llm(llm)
    first = SessionState.create("parallel system")
    second = SessionState.create("parallel system")

    async def run_both():
        return await asyncio.gather(
            engine.arun_with_dsh("会话 A", session=first, session_id="A"),
            engine.arun_with_dsh("会话 B", session=second, session_id="B"),
        )

    a, b = asyncio.run(run_both())
    assert a["content"] == "完成：会话 A"
    assert b["content"] == "完成：会话 B"
    assert set(seen) == {"会话 A", "会话 B"}
    assert first.messages[-1].content == "完成：会话 A"
    assert second.messages[-1].content == "完成：会话 B"


def test_stream_frames_separate_thinking_tool_and_final_message():
    async def stream(messages, tools):
        assert messages[0]["role"] == "system"
        yield {"type": "final", "content": "正式答案", "tool_calls": [], "usage": {"total_tokens": 3}}

    engine = QueryEngine("stream system")
    engine.set_async_llm(lambda messages, tools, model=None: ("", [], {}), stream)
    session = SessionState.create("stream system")
    chunks = asyncio.run(_collect(engine.arun_stream_with_dsh(
        "你好", session=session, session_id="stream-contract", max_tool_calls=5
    )))

    assert chunks[0]["progress"] == "thinking"
    assert chunks[1]["final"] is True
    assert chunks[1]["content"] == "正式答案"
    assert chunks[-1]["done"] is True
    assert all("token" not in chunk for chunk in chunks)


async def _collect(iterator):
    return [chunk async for chunk in iterator]


def test_dsh_preflight_compaction_is_reported_and_retains_useful_fallback_context():
    captured = []

    async def stream(messages, tools):
        captured.extend(messages)
        yield {"type": "final", "content": "已继续完成", "tool_calls": [], "usage": {}}

    engine = QueryEngine("stream system", context_window=4096)
    engine.set_async_llm(lambda messages, tools, model=None: ("", [], {}), stream)
    engine._compaction_engine = CompactionEngine(
        engine,
        CompactionConfig(threshold_ratio=0.8, retain_ratio=0.16, auto=True),
    )
    session = SessionState.create("stream system")
    for index in range(12):
        session.append(Message.user(f"第 {index} 个历史任务：检查模块 {index}"))
        session.append(Message.tool_result(f"call-{index}", "重要工具结果 " * 700))

    chunks = asyncio.run(_collect(engine.arun_stream_with_dsh(
        "继续处理当前任务", session=session, session_id="compact-contract", max_tool_calls=5
    )))

    compact_events = [c for c in chunks if c.get("progress") == "context:compacted"]
    assert compact_events
    assert compact_events[0]["context"]["dropped_messages"] > 0
    assert any(m["role"] == "user" and "compacted-summary" in (m.get("content") or "") for m in captured)
    assert chunks[-1]["done"] is True


def test_compaction_skips_orphan_tool_results_instead_of_looping():
    engine = QueryEngine("stream system", context_window=1024)
    engine._compaction_engine = CompactionEngine(
        engine,
        CompactionConfig(threshold_ratio=0.1, retain_ratio=0.2, auto=True),
    )
    session = SessionState.create("stream system")
    for index in range(8):
        session.append(Message.user(f"历史任务 {index}"))
        # Simulate legacy/interrupted history: a tool result without the
        # assistant tool-call that should have preceded it.
        session.append(Message.tool_result(f"orphan-{index}", "旧结果 " * 200))

    result = engine._compaction_engine.maybe_compact(session, threshold_ratio=0.1)

    assert result is not None
    assert any("compacted-summary" in (message.content or "") for message in session.messages)
    assert session.messages[0].role.value == "system"