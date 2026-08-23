import asyncio
import json
from types import SimpleNamespace

from my_agent.dsh_state_machine import AgentLoop


def _tool_call(call_id, name="read_file"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"path": "src/file.py"})},
    }


def _large_history():
    messages = [{"role": "system", "content": "You are a careful workspace agent."}]
    for i in range(80):
        call_id = f"call-{i}"
        messages.append({
            "role": "user",
            "content": f"探索第 {i} 轮\n" + ("workspace context " * 1200),
        })
        messages.append({
            "role": "assistant",
            "content": "继续检查项目结构。" + (" planning " * 900),
            "tool_calls": [_tool_call(call_id)],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": "[FILE] src/my_agent/core/engine.py\n" + ("result " * 1200),
        })
    messages.append({"role": "user", "content": "请给出最终完整总结"})
    return messages


def test_dsh_fits_large_history_before_provider_call():
    loop = AgentLoop(
        system_prompt="You are a careful workspace agent.",
        context_window=131072,
        context_target_ratio=0.60,
        reserved_output_tokens=8192,
        context_safety_margin=1024,
    )
    request = {"system": "You are a careful workspace agent.", "messages": _large_history(), "tools": []}
    fitted = loop._prepare_request_messages(request)

    assert fitted[0]["role"] == "system"
    assert sum(1 for m in fitted if m["role"] == "system") == 1
    assert fitted[-1]["role"] == "user"
    assert fitted[-1]["content"] == "请给出最终完整总结"
    assert loop._last_context_stats["estimated_tokens"] <= loop._last_context_stats["budget"]
    assert loop._last_context_stats["dropped_messages"] > 0


def test_dsh_keeps_tool_call_result_pairs_and_drops_orphans():
    loop = AgentLoop(context_window=4096, reserved_output_tokens=512, context_safety_margin=128)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "call", "tool_calls": [_tool_call("ok")]},
        {"role": "tool", "tool_call_id": "orphan", "content": "bad"},
        {"role": "tool", "tool_call_id": "ok", "content": "good"},
        {"role": "user", "content": "next"},
    ]
    fitted = loop._prepare_request_messages({"system": "system", "messages": messages, "tools": []})
    ids = {m.get("tool_call_id") for m in fitted if m["role"] == "tool"}
    assert "orphan" not in ids
    assert "ok" in ids
    assistant = next(m for m in fitted if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "ok"


def test_streaming_path_uses_same_fitted_payload():
    captured = {}

    async def stream_fn(messages, tools):
        captured["messages"] = messages
        yield {"type": "final", "content": "ok"}

    loop = AgentLoop(
        llm_stream_fn=stream_fn,
        system_prompt="system",
        context_window=2048,
        reserved_output_tokens=256,
        context_safety_margin=128,
    )
    request = {
        "system": "system",
        "messages": [{"role": "user", "content": "hello " * 5000}],
        "tools": [],
    }

    async def run():
        return await loop._call_llm_stream(request, None)

    asyncio.run(run())
    assert captured["messages"][0]["role"] == "system"
    assert loop._last_context_stats["estimated_tokens"] <= loop._last_context_stats["budget"]


def test_dsh_tiny_budget_never_exceeds_hard_cap():
    for window in (1, 8, 16, 32, 64, 128):
        loop = AgentLoop(
            system_prompt="系统提示 " * 500,
            context_window=window,
            context_target_ratio=1.0,
            reserved_output_tokens=0,
            context_safety_margin=0,
        )
        fitted = loop._prepare_request_messages({
            "system": "系统提示 " * 500,
            "messages": [{"role": "user", "content": "用户请求 " * 5000}],
            "tools": [],
        })
        assert loop._last_context_stats["estimated_tokens"] <= loop._last_context_stats["budget"]
        assert loop._estimate_payload_tokens(fitted) + sum(
            4 for _ in fitted
        ) <= loop._last_context_stats["budget"]


def test_dsh_large_tool_envelope_still_fits_messages():
    loop = AgentLoop(
        context_window=256,
        context_target_ratio=1.0,
        reserved_output_tokens=0,
        context_safety_margin=0,
    )
    fitted = loop._prepare_request_messages({
        "system": "system",
        "messages": [{"role": "user", "content": "latest " * 2000}],
        "tools": [],
    })
    assert fitted and fitted[-1]["role"] == "user"
    assert loop._last_context_stats["estimated_tokens"] <= loop._last_context_stats["budget"]


def test_dsh_cjk_payload_estimate_is_conservative_and_is_fitted():
    """Regression: escaped Unicode bounds once treated Chinese as ASCII."""
    chinese = "\u4e2d\u6587\u5de5\u4f5c\u533a\u5185\u5bb9"
    loop = AgentLoop(
        system_prompt="\u4f60\u662f\u8c28\u614e\u7684\u4ee3\u7801\u52a9\u624b\u3002",
        context_window=131072,
        context_target_ratio=0.60,
        reserved_output_tokens=8192,
        context_safety_margin=1024,
    )
    request = {
        "system": "\u4f60\u662f\u8c28\u614e\u7684\u4ee3\u7801\u52a9\u624b\u3002",
        "messages": [{"role": "user", "content": chinese * 40000}],
        "tools": [],
    }
    fitted = loop._prepare_request_messages(request)

    assert loop._estimate_payload_tokens("\u4e2d\u6587") >= 2
    assert loop._last_context_stats["estimated_tokens"] <= loop._last_context_stats["budget"]
    assert loop._last_context_stats["fitted"] is True
    assert fitted[-1]["role"] == "user"
    assert len(fitted[-1]["content"]) < len(request["messages"][0]["content"])



def _success_response(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


def test_sync_provider_context_overflow_retries_once_with_smaller_payload():
    calls = []

    def llm(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError(
                "request (147856 tokens) exceeds the available context size (131072 tokens)"
            )
        return _success_response("重试成功")

    loop = AgentLoop(
        llm_call_fn=llm,
        system_prompt="system",
        context_window=131072,
        context_target_ratio=0.90,
        reserved_output_tokens=1024,
        context_safety_margin=256,
    )
    request = {
        "system": "system",
        "messages": [{"role": "user", "content": "workspace context " * 30000}],
        "tools": [],
    }

    content, tool_calls, usage = asyncio.run(loop._call_llm(request, None))

    assert content == "重试成功"
    assert tool_calls == []
    assert usage["total_tokens"] == 12
    assert len(calls) == 2
    assert loop._messages_payload_tokens(calls[1]) < loop._messages_payload_tokens(calls[0])
    assert loop._last_context_stats["reason"] == "provider_context_limit_retry"


def test_sync_provider_context_overflow_does_not_retry_forever():
    calls = []

    def llm(messages, tools):
        calls.append(messages)
        raise RuntimeError(
            "request (147856 tokens) exceeds the available context size (131072 tokens)"
        )

    loop = AgentLoop(llm_call_fn=llm, system_prompt="system")

    try:
        asyncio.run(loop._call_llm({
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
        }, None))
    except RuntimeError as exc:
        assert "exceeds the available context size" in str(exc)
    else:
        raise AssertionError("the second provider context rejection must be surfaced")

    assert len(calls) == 2


def test_non_context_provider_error_is_not_retried():
    calls = []

    def llm(messages, tools):
        calls.append(messages)
        raise RuntimeError("401 unauthorized")

    loop = AgentLoop(llm_call_fn=llm, system_prompt="system")

    try:
        asyncio.run(loop._call_llm({
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
        }, None))
    except RuntimeError as exc:
        assert str(exc) == "401 unauthorized"
    else:
        raise AssertionError("the provider error must be surfaced")

    assert len(calls) == 1


def test_stream_provider_context_overflow_retries_once_with_smaller_payload():
    calls = []

    async def stream(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError(
                "request (147856 tokens) exceeds the available context size (131072 tokens)"
            )
        yield {"type": "final", "content": "流式重试成功", "usage": {"total_tokens": 9}}

    loop = AgentLoop(
        llm_stream_fn=stream,
        system_prompt="system",
        context_window=131072,
        context_target_ratio=0.90,
        reserved_output_tokens=1024,
        context_safety_margin=256,
    )
    request = {
        "system": "system",
        "messages": [{"role": "user", "content": "workspace context " * 30000}],
        "tools": [],
    }

    content, tool_calls, usage = asyncio.run(loop._call_llm_stream(request, None))

    assert content == "流式重试成功"
    assert tool_calls == []
    assert usage["total_tokens"] == 9
    assert len(calls) == 2
    assert loop._messages_payload_tokens(calls[1]) < loop._messages_payload_tokens(calls[0])


def test_stream_non_context_provider_error_is_not_retried():
    calls = []

    async def stream(messages, tools):
        calls.append(messages)
        raise RuntimeError("500 internal server error")
        yield  # pragma: no cover

    loop = AgentLoop(llm_stream_fn=stream, system_prompt="system")

    try:
        asyncio.run(loop._call_llm_stream({
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
        }, None))
    except RuntimeError as exc:
        assert str(exc) == "500 internal server error"
    else:
        raise AssertionError("the provider error must be surfaced")

    assert len(calls) == 1



def test_dsh_passes_dynamic_max_tokens_to_sync_callback():
    captured = {}

    def llm_fn(messages, tools, max_tokens=None):
        captured["messages"] = messages
        captured["tools"] = tools
        captured["max_tokens"] = max_tokens
        message = SimpleNamespace(content="ok", tool_calls=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    loop = AgentLoop(
        llm_call_fn=llm_fn,
        system_prompt="system",
        context_window=2048,
        reserved_output_tokens=512,
        context_safety_margin=128,
    )

    async def run():
        return await loop._call_llm({
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }, None)

    asyncio.run(run())
    assert captured["max_tokens"] is not None
    assert 1 <= captured["max_tokens"] <= 512
    assert captured["max_tokens"] + loop._messages_payload_tokens(captured["messages"]) \
        + loop._estimate_payload_tokens(captured["tools"]) \
        + loop._context_safety_margin <= loop._context_window


def test_dsh_stream_callback_receives_dynamic_max_tokens():
    captured = {}

    async def stream_fn(messages, tools, max_tokens=None):
        captured["max_tokens"] = max_tokens
        captured["messages"] = messages
        yield {"type": "final", "content": "ok"}

    loop = AgentLoop(
        llm_stream_fn=stream_fn,
        system_prompt="system",
        context_window=1024,
        reserved_output_tokens=256,
        context_safety_margin=64,
    )

    async def run():
        return await loop._call_llm_stream({
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
        }, None)

    content, _, _ = asyncio.run(run())
    assert content == "ok"
    assert 1 <= captured["max_tokens"] <= 256


def test_dsh_dynamic_budget_keeps_legacy_two_argument_callbacks():
    captured = {}

    def llm_fn(messages, tools):
        captured["called"] = True
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    loop = AgentLoop(llm_call_fn=llm_fn, system_prompt="system")

    async def run():
        return await loop._call_llm({
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
        }, None)

    content, _, _ = asyncio.run(run())
    assert content == "ok"
    assert captured["called"] is True


def test_dsh_never_drops_all_user_turns_when_latest_group_is_tool_result():
    """Long tool runs must remain valid for strict Jinja chat templates."""
    loop = AgentLoop(
        system_prompt="system",
        context_window=4096,
        context_target_ratio=1.0,
        reserved_output_tokens=128,
        context_safety_margin=64,
    )
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "请完整探索当前项目"}]
    for i in range(12):
        call_id = f"call-{i}"
        messages.extend([
            {
                "role": "assistant",
                "content": "继续检查",
                "tool_calls": [_tool_call(call_id)],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "result " * 120,
            },
        ])

    fitted = loop._prepare_request_messages({
        "system": "system", "messages": messages, "tools": []
    })

    assert any(m["role"] == "user" for m in fitted)
    assert fitted[0]["role"] == "system"
    assert loop._messages_payload_tokens(fitted) <= loop._last_context_stats["budget"]


def test_dsh_inserts_recovery_user_for_malformed_checkpoint():
    loop = AgentLoop(system_prompt="system", context_window=4096, reserved_output_tokens=128)
    fitted = loop._prepare_request_messages({
        "system": "system",
        "messages": [{"role": "assistant", "content": "orphan"}],
        "tools": [],
    })
    assert any(m["role"] == "user" for m in fitted)
