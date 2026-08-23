# -*- coding: utf-8 -*-
"""Regression tests for local LLM chat-template failures."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from my_agent.core.context_assembler import fit_messages_to_budget
from my_agent.core.engine import QueryEngine
from my_agent.llm import AsyncLLMClient
from my_agent.types.message import Message
from my_agent.types.session import SessionState


def test_message_fitting_moves_every_system_message_to_the_front():
    messages = [
        Message.user("历史问题"),
        Message.system("系统约束"),
        Message.assistant("历史回答"),
        Message.user("当前问题"),
    ]

    fitted = fit_messages_to_budget(messages, context_window=4096)

    assert fitted[0].role.value == "system"
    assert [m.role.value for m in fitted] == [
        "system", "user", "assistant", "user"
    ]
    # The normalizer must not mutate a live session.
    assert messages[0].role.value == "user"


def test_engine_payload_never_puts_system_after_user():
    engine = QueryEngine(system_prompt="基础系统提示", context_window=4096)
    session = SessionState(messages=[
        Message.user("旧问题"),
        Message.system("恢复的系统提示"),
        Message.assistant("旧回答"),
        Message.user("新问题"),
    ])

    payload = engine.build_openai_messages(session)

    assert payload[0]["role"] == "system"
    assert payload[-1]["role"] == "user"
    assert any(item["content"] == "新问题" for item in payload)


@pytest.mark.asyncio
async def test_jinja_template_500_is_not_retried_and_keeps_upstream_detail():
    client = AsyncLLMClient(
        api_key="test", base_url="http://fake/v1", model="test-model"
    )
    request = httpx.Request("POST", "http://fake/v1/chat/completions")
    response = httpx.Response(
        500,
        json={"error": {"code": 500, "message": "Jinja Exception: System message must be at the beginning."}},
        request=request,
    )
    fake = AsyncMock()
    fake.post = AsyncMock(return_value=response)
    fake.is_closed = False

    with patch.object(client, "_get_client", return_value=fake):
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.achat([{"role": "user", "content": "hello"}])

    assert fake.post.await_count == 1
    assert "System message must be at the beginning" in str(exc_info.value)


def test_llm_template_error_is_actionable_and_not_a_network_hint():
    message = QueryEngine._stop_message(
        "llm_error",
        "HTTPStatusError: 500 | upstream: Jinja Exception: System message must be at the beginning.",
    )

    assert "system 消息必须位于第一条" in message
    assert "检查网络和配置" not in message
    assert "HTTPStatusError" not in message


def test_missing_user_template_error_is_actionable():
    message = QueryEngine._stop_message(
        "llm_error",
        "HTTPStatusError: 500 | upstream: Jinja Exception: No user query found in messages.",
    )

    assert "没有 user 消息" in message
    assert "新建会话" in message

@pytest.mark.asyncio
async def test_streaming_http_error_reads_body_before_formatting_detail():
    """Streaming 4xx bodies must not cause httpx.ResponseNotRead."""
    client = AsyncLLMClient(
        api_key="test", base_url="http://fake/v1", model="test-model"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "No user query found in messages."}},
            request=request,
        )

    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with patch.object(client, "_get_client", return_value=transport_client):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                _ = [chunk async for chunk in client.astream([
                    {"role": "user", "content": "hello"}
                ])]
    finally:
        await transport_client.aclose()

    assert "No user query found in messages" in str(exc_info.value)
    assert "Attempted to access streaming response content" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_streaming_tool_call_is_aggregated_without_prereading_sse_body():
    """The successful SSE body remains available to the normal stream parser."""
    client = AsyncLLMClient(
        api_key="test", base_url="http://fake/v1", model="test-model"
    )
    body = b"\n".join([
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"list_files","arguments":"{\\\"path\\\":\\\".\\\"}"}}]}}]}',
        b"",
        b'data: {"choices":[{"delta":{}}]}',
        b"",
        b"data: [DONE]",
        b"",
    ])

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, request=request)

    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with patch.object(client, "_get_client", return_value=transport_client):
            chunks = [chunk async for chunk in client.astream([
                {"role": "user", "content": "列出文件"}
            ])]
    finally:
        await transport_client.aclose()

    assert chunks[-1]["type"] == "final"
    assert chunks[-1]["tool_calls"] == [{
        "id": "call_1", "name": "list_files", "arguments": {"path": "."},
    }]

