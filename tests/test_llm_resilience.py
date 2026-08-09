# -*- coding: utf-8 -*-
"""
tests.test_llm_resilience — Tests for LLM client circuit breaker integration

Verifies that LLMClient correctly integrates CircuitBreaker from resilience.py.
The existing retry/backoff logic is already in the package; this tests the
new circuit breaker gate.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from my_agent.llm import LLMClient
from my_agent.resilience import CircuitBreaker, CircuitOpenError, CircuitState


# ── Helpers ──────────────────────────────────────────────────

def _make_response(status_code=200, json_data=None):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.reason = "OK" if status_code == 200 else "Internal Server Error"
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        resp.raise_for_status.side_effect = HTTPError(
            f"{status_code}", response=resp
        )
    if json_data is None:
        json_data = {"choices": [{"message": {"content": "hello"}}]}
    resp.json.return_value = json_data
    return resp


def _make_timeout():
    from requests.exceptions import Timeout
    return Timeout("timed out")


def _make_connection_error():
    from requests.exceptions import ConnectionError
    return ConnectionError("connection refused")


# ── Tests ────────────────────────────────────────────────────


class TestLLMCircuitBreaker:
    """LLMClient circuit breaker integration."""

    def _make_client(self, **kwargs):
        return LLMClient(
            api_key="test-key",
            base_url="http://fake-llm/v1",
            model="test-model",
            **kwargs,
        )

    @patch("my_agent.llm.requests.post")
    def test_success_records_circuit_success(self, mock_post):
        """Successful call records success on circuit breaker."""
        mock_post.return_value = _make_response(200)
        client = self._make_client()

        result = client.chat([{"role": "user", "content": "hi"}])

        assert result["choices"][0]["message"]["content"] == "hello"
        assert mock_post.call_count == 1
        assert client._circuit_breaker.state == CircuitState.CLOSED

    @patch("my_agent.llm.requests.post")
    def test_circuit_open_blocks_request(self, mock_post):
        """Circuit already open — should raise CircuitOpenError immediately."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100)
        cb.record_failure()  # open the circuit

        client = self._make_client(circuit_breaker=cb)

        with pytest.raises(CircuitOpenError):
            client.chat([{"role": "user", "content": "hi"}])

        # Should not even attempt the HTTP call
        assert mock_post.call_count == 0

    @patch("my_agent.llm.requests.post")
    def test_all_retries_exhausted_records_circuit_failure(self, mock_post):
        """All retries fail — circuit records failure."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        client = self._make_client(circuit_breaker=cb)
        mock_post.side_effect = _make_timeout()

        with pytest.raises(Exception):
            client.chat([{"role": "user", "content": "hi"}])

        # At least one failure recorded
        assert cb.consecutive_failures >= 1

    @patch("my_agent.llm.requests.post")
    def test_shared_circuit_breaker_across_clients(self, mock_post):
        """Shared circuit breaker tracks failures across LLMClient instances."""
        # failure_threshold=1 so a single exhausted-retry session opens the circuit
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100)
        client1 = self._make_client(circuit_breaker=cb)
        client2 = self._make_client(circuit_breaker=cb)

        mock_post.side_effect = _make_timeout()

        # First client exhausts retries → circuit opens
        with pytest.raises(Exception):
            client1.chat([{"role": "user", "content": "a"}])

        assert cb.state == CircuitState.OPEN

        # Second client should be blocked by the open circuit
        with pytest.raises(CircuitOpenError):
            client2.chat([{"role": "user", "content": "b"}])

    @patch("my_agent.llm.requests.post")
    def test_default_circuit_breaker_created(self, mock_post):
        """LLMClient creates its own CircuitBreaker if none provided."""
        client = self._make_client()
        assert isinstance(client._circuit_breaker, CircuitBreaker)

    @patch("my_agent.llm.requests.post")
    def test_400_not_retried(self, mock_post):
        """400 client error — not retried (only 429/5xx are)."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        client = self._make_client(circuit_breaker=cb)
        mock_post.return_value = _make_response(400)

        with pytest.raises(Exception):
            client.chat([{"role": "user", "content": "hi"}])

        # 400 is not retried — only 1 call
        assert mock_post.call_count == 1

    @patch("my_agent.llm.requests.post")
    def test_stream_success(self, mock_post):
        """chat_stream works with circuit breaker."""
        resp = _make_response(200)
        resp.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"chunk1"}}]}',
            b'data: [DONE]',
        ]
        mock_post.return_value = resp
        client = self._make_client()

        chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))

        assert chunks == ["chunk1"]

    def test_no_circuit_breaker_graceful_fallback(self):
        """When resilience module is unavailable, circuit_breaker is None."""
        import my_agent.llm as llm_mod
        original = llm_mod.CircuitBreaker
        try:
            llm_mod.CircuitBreaker = None
            client = LLMClient(
                api_key="k",
                base_url="http://fake/v1",
                model="m",
            )
            assert client._circuit_breaker is None
        finally:
            llm_mod.CircuitBreaker = original

    @patch("my_agent.llm.requests.post")
    def test_circuit_half_open_allows_probe(self, mock_post):
        """Half-open state allows one probe request through."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
        cb.record_failure()  # open the circuit
        # After recovery_timeout (0.0), it transitions to half-open on next check
        mock_post.return_value = _make_response(200)

        client = self._make_client(circuit_breaker=cb)
        result = client.chat([{"role": "user", "content": "hi"}])

        assert result["choices"][0]["message"]["content"] == "hello"
        assert cb.state == CircuitState.CLOSED  # probe succeeded → closed

    @patch("my_agent.llm.requests.post")
    def test_circuit_open_error_not_retried_in_loop(self, mock_post):
        """CircuitOpenError raised inside the retry loop must not be retried.

        If the circuit opens mid-retry (e.g. from a concurrent thread calling
        record_failure), the next allow_request() call rejects with
        CircuitOpenError. Without the fix, the generic except Exception handler
        would catch it and retry — defeating the circuit breaker's fast-fail.
        """
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=100.0)
        client = self._make_client(circuit_breaker=cb)

        # First call: success (circuit stays closed)
        mock_post.return_value = _make_response(200)
        client.chat([{"role": "user", "content": "a"}])
        assert mock_post.call_count == 1

        # Simulate circuit opening mid-flight: make allow_request() return False
        # on the second call (which would be inside the retry loop if the first
        # attempt failed).
        cb.record_failure()  # failures=1, still closed (threshold=2)
        cb.record_failure()  # failures=2, now OPEN

        # Second call: circuit is open — should raise immediately, not retry
        with pytest.raises(CircuitOpenError):
            client.chat([{"role": "user", "content": "b"}])

        # No HTTP call should have been made (circuit blocked before POST)
        assert mock_post.call_count == 1  # only the first successful call


# ── Async tests ─────────────────────────────────────────────


try:
    import httpx
    from my_agent.llm import AsyncLLMClient
    _has_async_client = True
except ImportError:
    _has_async_client = False


@pytest.mark.skipif(not _has_async_client, reason="httpx not installed")
class TestAsyncLLMCircuitBreaker:
    """AsyncLLMClient circuit breaker integration."""

    def _make_client(self, **kwargs):
        return AsyncLLMClient(
            api_key="test-key",
            base_url="http://fake-llm/v1",
            model="test-model",
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_circuit_open_error_not_retried(self):
        """AsyncLLMClient: CircuitOpenError must not be retried in the loop."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=100.0)
        client = self._make_client(circuit_breaker=cb)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            await client.achat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_circuit_open_error_propagates_immediately(self):
        """CircuitOpenError raised before the loop propagates without retry."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100.0)
        cb.record_failure()  # circuit is now OPEN
        client = self._make_client(circuit_breaker=cb)

        with pytest.raises(CircuitOpenError, match="Circuit breaker open"):
            await client.achat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_success_records_circuit_success(self):
        """Successful async call records success on circuit breaker."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        client = self._make_client(circuit_breaker=cb)

        # Mock the HTTP client with async post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello"}}]
        }

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_client.is_closed = False
        client._client = mock_client

        content, tool_calls, usage = await client.achat(
            [{"role": "user", "content": "hi"}]
        )
        assert content == "hello"
        assert cb.state == CircuitState.CLOSED
