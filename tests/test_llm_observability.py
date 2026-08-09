# -*- coding: utf-8 -*-
"""
Tests for LLM client observability metrics integration.

Verifies that MetricsCollector is wired into LLMClient and AsyncLLMClient,
recording llm_call_latency, llm_calls, llm_retries, and llm_circuit_open.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from my_agent.llm import LLMClient, AsyncLLMClient
from my_agent.observability import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_response(status_code: int = 200, json_data: dict = None):
    """Create a mock requests.Response with a given status code."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {
        "choices": [{"message": {"content": "hello"}}]
    }
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        err = HTTPError(response=resp)
        resp.raise_for_status.side_effect = err
    return resp


def _make_fake_httpx_response(status_code: int = 200, json_data: dict = None):
    """Create a mock httpx.Response with a given status code."""
    import httpx
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {
        "choices": [{"message": {"content": "hello"}}]
    }
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        err = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp)
        resp.raise_for_status.side_effect = err
    return resp


# Use a fixed model name to avoid env-var leakage in metric keys
_MODEL = "test-model"


# ---------------------------------------------------------------------------
# Sync LLMClient metrics tests
# ---------------------------------------------------------------------------

class TestLLMClientMetrics:
    """Metrics instrumentation for synchronous LLMClient."""

    def test_success_records_latency_and_counter(self):
        """Successful LLM call records llm_call_latency and llm_calls counter."""
        metrics = MetricsCollector()
        client = LLMClient(api_key="test", base_url="http://fake/v1",
                           model=_MODEL, metrics=metrics)
        fake_resp = _make_fake_response(200)

        with patch("my_agent.llm.requests.post", return_value=fake_resp):
            client.chat([{"role": "user", "content": "hi"}])

        counters = metrics.get_metrics()["counters"]
        # MetricsCollector._build_key wraps values in quotes
        assert counters.get(f'llm_calls{{model="{_MODEL}",status="success"}}') == 1

        hists = metrics.get_metrics()["histograms"]
        key = f'llm_call_latency{{model="{_MODEL}",status="success"}}'
        assert key in hists
        assert hists[key]["count"] == 1
        assert hists[key]["sum_ms"] >= 0  # mock completes instantly, may round to 0

    def test_error_records_latency_and_counter(self):
        """Failed LLM call (non-retryable) records error metrics."""
        metrics = MetricsCollector()
        client = LLMClient(api_key="test", base_url="http://fake/v1",
                           model=_MODEL, metrics=metrics)
        fake_resp = _make_fake_response(400)

        with patch("my_agent.llm.requests.post", return_value=fake_resp):
            with pytest.raises(Exception):
                client.chat([{"role": "user", "content": "hi"}])

        counters = metrics.get_metrics()["counters"]
        assert counters.get(f'llm_calls{{model="{_MODEL}",status="error"}}') == 1

    def test_retry_increments_retry_counter(self):
        """Retryable error increments llm_retries counter."""
        metrics = MetricsCollector()
        client = LLMClient(api_key="test", base_url="http://fake/v1",
                           model=_MODEL, metrics=metrics)
        retry_resp = _make_fake_response(500)
        ok_resp = _make_fake_response(200)

        with patch("my_agent.llm.requests.post",
                   side_effect=[retry_resp, ok_resp]):
            with patch("my_agent.llm.time.sleep"):
                client.chat([{"role": "user", "content": "hi"}])

        counters = metrics.get_metrics()["counters"]
        assert counters.get(f'llm_retries{{model="{_MODEL}",status="500"}}') == 1

    def test_circuit_open_increments_counter(self):
        """Circuit breaker open increments llm_circuit_open counter."""
        from my_agent.resilience import CircuitBreaker, CircuitOpenError
        metrics = MetricsCollector()
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.record_failure()  # trip immediately
        client = LLMClient(api_key="test", base_url="http://fake/v1",
                           model=_MODEL, circuit_breaker=cb, metrics=metrics)

        with pytest.raises(CircuitOpenError):
            client.chat([{"role": "user", "content": "hi"}])

        counters = metrics.get_metrics()["counters"]
        assert counters.get(f'llm_circuit_open{{model="{_MODEL}"}}') == 1

    def test_no_metrics_when_none(self):
        """When metrics=None, no error is raised and call works."""
        client = LLMClient(api_key="test", base_url="http://fake/v1",
                           model=_MODEL, metrics=None)
        fake_resp = _make_fake_response(200)

        with patch("my_agent.llm.requests.post", return_value=fake_resp):
            client.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Async AsyncLLMClient metrics tests
# ---------------------------------------------------------------------------

class TestAsyncLLMClientMetrics:
    """Metrics instrumentation for async AsyncLLMClient."""

    @pytest.mark.asyncio
    async def test_success_records_latency_and_counter(self):
        """Successful async LLM call records llm_call_latency and llm_calls."""
        metrics = MetricsCollector()
        client = AsyncLLMClient(api_key="test", base_url="http://fake/v1",
                                model=_MODEL, metrics=metrics)
        fake_resp = _make_fake_httpx_response(200)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=fake_resp)
        mock_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_client):
            await client.achat([{"role": "user", "content": "hi"}])

        counters = metrics.get_metrics()["counters"]
        assert counters.get(f'llm_calls{{model="{_MODEL}",status="success"}}') == 1

        hists = metrics.get_metrics()["histograms"]
        key = f'llm_call_latency{{model="{_MODEL}",status="success"}}'
        assert key in hists

    @pytest.mark.asyncio
    async def test_circuit_open_increments_counter(self):
        """Circuit breaker open increments llm_circuit_open counter."""
        from my_agent.resilience import CircuitBreaker, CircuitOpenError
        metrics = MetricsCollector()
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.record_failure()
        client = AsyncLLMClient(api_key="test", base_url="http://fake/v1",
                                model=_MODEL, circuit_breaker=cb,
                                metrics=metrics)

        with pytest.raises(CircuitOpenError):
            await client.achat([{"role": "user", "content": "hi"}])

        counters = metrics.get_metrics()["counters"]
        assert counters.get(f'llm_circuit_open{{model="{_MODEL}"}}') == 1

    @pytest.mark.asyncio
    async def test_error_records_metrics(self):
        """Non-retryable HTTP error records error metrics."""
        import httpx
        metrics = MetricsCollector()
        client = AsyncLLMClient(api_key="test", base_url="http://fake/v1",
                                model=_MODEL, metrics=metrics)
        fake_resp = _make_fake_httpx_response(400)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=fake_resp)
        mock_client.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await client.achat([{"role": "user", "content": "hi"}])

        counters = metrics.get_metrics()["counters"]
        assert counters.get(f'llm_calls{{model="{_MODEL}",status="error"}}') == 1
