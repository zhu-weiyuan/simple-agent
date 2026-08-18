# -*- coding: utf-8 -*-
"""
tests.test_llm_idle_timeout — regression tests for LLM idle timeout wiring.

The user-facing complaint is that a chat request can hang for minutes when the
upstream LLM gateway stalls.  These tests verify that the engine can pass an
idle timeout through to both the non-streaming and streaming async LLM call
paths, and that the real AsyncLLMClient honors it for both achat and astream.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_agent.core.engine import QueryEngine  # noqa: E402

try:
    import httpx
    from my_agent.llm import AsyncLLMClient
    _has_async_client = True
except ImportError:
    _has_async_client = False

SYS = "You are a test agent."


def _make_engine_with_llm(achat=None, astream=None):
    eng = QueryEngine(system_prompt=SYS, context_window=32768)
    eng.set_async_llm(achat, astream)
    return eng


class _FakeAsyncLLM:
    """Scripted mock that records the idle_timeout it receives."""

    def __init__(self, scripts, usage_tokens=10):
        self.scripts = list(scripts)
        self.calls = 0
        self.usage_tokens = usage_tokens
        self.achat_idle_timeout = None
        self.astream_idle_timeout = None

    async def achat(self, messages, tools, model=None, idle_timeout=None):
        self.calls += 1
        self.achat_idle_timeout = idle_timeout
        content, tool_calls = self.scripts[min(self.calls - 1, len(self.scripts) - 1)]
        usage = {"prompt_tokens": self.usage_tokens,
                 "completion_tokens": self.usage_tokens,
                 "total_tokens": self.usage_tokens * 2}
        return content, tool_calls, usage

    async def astream(self, messages, tools, model=None, idle_timeout=None):
        self.calls += 1
        self.astream_idle_timeout = idle_timeout
        content, tool_calls = self.scripts[min(self.calls - 1, len(self.scripts) - 1)]
        for i in range(0, len(content), 3):
            yield {"type": "delta", "content": content[i:i + 3]}
        yield {"type": "final", "content": content, "tool_calls": tool_calls,
               "usage": {"prompt_tokens": self.usage_tokens,
                         "completion_tokens": self.usage_tokens,
                         "total_tokens": self.usage_tokens * 2}}


class TestEngineIdleTimeoutWiring:
    """Verify the engine forwards llm_idle_timeout to the LLM call functions."""

    @pytest.mark.asyncio
    async def test_arun_passes_idle_timeout_to_achat(self):
        llm = _FakeAsyncLLM([("ok", [])])
        eng = _make_engine_with_llm(achat=llm.achat)
        result = await eng.arun("hi", llm_idle_timeout=7.5)
        assert result["content"] == "ok"
        assert llm.achat_idle_timeout == 7.5

    @pytest.mark.asyncio
    async def test_arun_passes_idle_timeout_to_astream(self):
        llm = _FakeAsyncLLM([("ok", [])])
        eng = _make_engine_with_llm(astream=llm.astream)
        result = await eng.arun("hi", llm_idle_timeout=12.0)
        assert result["content"] == "ok"
        assert llm.astream_idle_timeout == 12.0

    @pytest.mark.asyncio
    async def test_arun_omits_idle_timeout_when_none(self):
        llm = _FakeAsyncLLM([("ok", [])])
        eng = _make_engine_with_llm(achat=llm.achat)
        result = await eng.arun("hi")
        assert result["content"] == "ok"
        assert llm.achat_idle_timeout is None


@pytest.mark.skipif(not _has_async_client, reason="httpx not installed")
class TestAsyncLLMClientIdleTimeout:
    """Verify AsyncLLMClient honors the idle_timeout parameter."""

    def _make_client(self, **kwargs):
        return AsyncLLMClient(
            api_key="test-key",
            base_url="http://fake-llm/v1",
            model="test-model",
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_achat_uses_idle_timeout_as_http_timeout(self):
        client = self._make_client(timeout=30.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello"}}]
        }

        captured_kwargs = {}

        async def mock_post(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_client.is_closed = False
        client._client = mock_client

        content, tool_calls, usage = await client.achat(
            [{"role": "user", "content": "hi"}], idle_timeout=15.0)
        assert content == "hello"
        # The httpx timeout passed to post should be the idle timeout (15.0),
        # not the client's default timeout (30.0).
        assert captured_kwargs["timeout"] == 15.0

    @pytest.mark.asyncio
    async def test_astream_uses_idle_timeout_as_http_timeout(self):
        client = self._make_client(timeout=30.0)

        # Build a fake streaming response that yields one line then finishes.
        async def fake_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
            yield 'data: [DONE]'

        fake_stream = MagicMock()
        fake_stream.raise_for_status = MagicMock()
        fake_stream.aiter_lines = fake_aiter_lines

        async def fake_aenter(self):
            return fake_stream

        async def fake_aexit(self, *args):
            return False

        fake_response = MagicMock()
        fake_response.__aenter__ = fake_aenter
        fake_response.__aexit__ = fake_aexit

        captured_kwargs = {}

        def mock_stream(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_response

        mock_client = MagicMock()
        mock_client.stream = mock_stream
        mock_client.is_closed = False
        client._client = mock_client

        events = []
        async for ev in client.astream(
                [{"role": "user", "content": "hi"}], idle_timeout=20.0):
            events.append(ev)

        assert events[0] == {"type": "delta", "content": "hi"}
        assert events[-1]["type"] == "final"
        # The httpx stream timeout should be the idle timeout (20.0).
        assert captured_kwargs["timeout"] == 20.0

    @pytest.mark.asyncio
    async def test_astream_idle_timeout_triggers_on_silent_stream(self):
        """Astream must raise TimeoutError if the stream goes silent."""
        client = self._make_client(timeout=30.0)

        async def fake_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"start"}}]}'
            # Simulate a silent upstream by waiting past the idle timeout.
            # The idle check happens on each iteration, so sleeping here
            # before yielding the next line should trigger the timeout.
            await asyncio.sleep(0.3)
            yield 'data: [DONE]'

        fake_stream = MagicMock()
        fake_stream.raise_for_status = MagicMock()
        fake_stream.aiter_lines = fake_aiter_lines

        async def fake_aenter(self):
            return fake_stream

        async def fake_aexit(self, *args):
            return False

        fake_response = MagicMock()
        fake_response.__aenter__ = fake_aenter
        fake_response.__aexit__ = fake_aexit

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=fake_response)
        mock_client.is_closed = False
        client._client = mock_client

        with pytest.raises(TimeoutError):
            async for _ in client.astream(
                    [{"role": "user", "content": "hi"}], idle_timeout=0.1):
                pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
