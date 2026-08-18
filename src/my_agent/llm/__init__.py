# -*- coding: utf-8 -*-
"""
my_agent.llm — LLM 客户端层 (P6)

约定: ``base_url`` **包含 /v1** (e.g. https://api.openai.com/v1)。
chat 与 list_models 的拼接已统一:
    chat        -> {base_url}/chat/completions
    list_models -> {base_url}/models
(旧版 bug: chat 用 {base}/chat/completions 而 list_models 用 {base}/v1/models)

新增 AsyncLLMClient (httpx.AsyncClient, 守卫导入):
    achat(messages, tools)  -> (content, tool_calls, usage)
    astream(messages, tools)-> AsyncIterator[dict]  (OpenAI 兼容 SSE 解析,
                               增量 delta 与工具调用聚合)
重试策略 (同步/异步一致): 最多 2 次重试, 指数退避 + jitter, 总 deadline 45s。
同步 LLMClient 保留兼容。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    httpx = None

logger = logging.getLogger(__name__)

try:
    from ..resilience import CircuitBreaker, CircuitOpenError
except ImportError:  # pragma: no cover
    CircuitBreaker = None  # type: ignore[assignment,misc]
    CircuitOpenError = RuntimeError  # type: ignore[assignment,misc]

try:
    from ..observability import get_metrics as _get_obs_metrics
except ImportError:  # pragma: no cover
    _get_obs_metrics = None  # type: ignore[assignment]

DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
TOTAL_DEADLINE = float(os.getenv("LLM_TOTAL_DEADLINE_SECONDS", "45"))
MAX_RETRIES = 2


def _normalize_base_url(base_url: str) -> str:
    """约定 base_url 含 /v1;若调用方漏掉则补上(localhost 裸端口除外由env自定)。"""
    url = base_url.rstrip("/")
    return url


def _backoff_delay(attempt: int) -> float:
    """指数退避 + jitter: 0.5*2^attempt + U(0, 0.25)"""
    return 0.5 * (2 ** attempt) + random.uniform(0, 0.25)


def _merge_reasoning(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize completion payload without exposing hidden reasoning.

    Some OpenAI-compatible providers return ``reasoning_content`` alongside
    the user-facing ``content``. It is intentionally discarded at the client
    boundary so internal chain-of-thought is never persisted or streamed.
    """
    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        message.pop("reasoning_content", None)
        delta = choice.get("delta") or {}
        delta.pop("reasoning_content", None)
    return data


class LLMClient:
    """OpenAI 兼容 API 客户端（同步, 基于 requests, 保留兼容）

    Integrates circuit breaker for failure isolation alongside the existing
    retry-with-backoff and deadline logic.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        circuit_breaker: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "xxx")
        self.base_url = _normalize_base_url(
            base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if circuit_breaker is not None:
            self._circuit_breaker = circuit_breaker
        elif CircuitBreaker is not None:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=5, recovery_timeout=30.0
            )
        else:
            self._circuit_breaker = None
        # Observability: optional MetricsCollector for LLM-specific metrics
        self._metrics = metrics or (_get_obs_metrics() if _get_obs_metrics else None)

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _post_with_retry(self, url: str, payload: Dict[str, Any],
                         stream: bool = False):
        if requests is None:
            raise RuntimeError("requests 未安装,无法使用同步 LLMClient")

        # Circuit breaker gate — fail fast when upstream is known-bad
        if self._circuit_breaker is not None:
            if not self._circuit_breaker.allow_request():
                if self._metrics is not None:
                    self._metrics.increment_counter(
                        "llm_circuit_open", {"model": self.model})
                raise CircuitOpenError("Circuit breaker open for LLM upstream")

        call_start = time.monotonic()
        deadline = call_start + TOTAL_DEADLINE
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                resp = requests.post(
                    url, json=payload, headers=self._headers(),
                    stream=stream, timeout=min(DEFAULT_TIMEOUT, remaining))
                if resp.status_code in (429, 500, 502, 503, 504) \
                        and attempt < MAX_RETRIES:
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                    if self._metrics is not None:
                        self._metrics.increment_counter(
                            "llm_retries", {"model": self.model,
                                            "status": str(resp.status_code)})
                    logger.warning(
                        "LLM upstream %d, retrying (attempt %d/%d)",
                        resp.status_code, attempt + 1, MAX_RETRIES,
                    )
                    time.sleep(min(_backoff_delay(attempt),
                                   max(0, deadline - time.monotonic())))
                    continue
                resp.raise_for_status()
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()
                if self._metrics is not None:
                    elapsed_ms = (time.monotonic() - call_start) * 1000
                    self._metrics.observe_histogram(
                        "llm_call_latency", elapsed_ms,
                        labels={"model": self.model, "status": "success"})
                    self._metrics.increment_counter(
                        "llm_calls", {"model": self.model, "status": "success"})
                return resp
            except CircuitOpenError:
                # Circuit breaker rejection must NOT be retried — fail immediately.
                raise
            except Exception as e:  # noqa: BLE001
                last_exc = e
                # Only retry on transient / network errors, not on HTTPError
                # from non-retryable status codes (400, 401, 403, etc.).
                _is_http_error = (
                    requests is not None
                    and isinstance(e, requests.exceptions.HTTPError)
                )
                _retryable_http = (
                    _is_http_error
                    and getattr(e, "response", None) is not None
                    and e.response.status_code in (429, 500, 502, 503, 504)
                )
                _is_transient = not _is_http_error or _retryable_http
                if not _is_transient:
                    break  # non-retryable — stop immediately
                if attempt < MAX_RETRIES and time.monotonic() < deadline:
                    logger.warning(
                        "LLM call failed (%s), retrying (attempt %d/%d)",
                        type(e).__name__, attempt + 1, MAX_RETRIES,
                    )
                    time.sleep(min(_backoff_delay(attempt),
                                   max(0, deadline - time.monotonic())))
                    continue

        # All retries exhausted — record failure for circuit breaker
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_failure()
        if self._metrics is not None:
            elapsed_ms = (time.monotonic() - call_start) * 1000
            self._metrics.observe_histogram(
                "llm_call_latency", elapsed_ms,
                labels={"model": self.model, "status": "error"})
            self._metrics.increment_counter(
                "llm_calls", {"model": self.model, "status": "error"})
        raise last_exc or RuntimeError("LLM 请求超出总 deadline")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """调用 chat completions API (base_url 含 /v1)"""
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        resp = self._post_with_retry(url, payload)
        return _merge_reasoning(resp.json())

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        model: Optional[str] = None,
    ):
        """流式调用 chat completions API,逐段 yield 文本增量"""
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        resp = self._post_with_retry(url, payload, stream=True)
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8", errors="replace")
                if not text.startswith("data: "):
                    continue
                data_str = text[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
        finally:
            # Generator cancellation/close must release the underlying socket.
            resp.close()

    def list_models(self) -> List[str]:
        """列出可用模型 (与 chat 统一: base_url 已含 /v1)"""
        if requests is None:
            return []
        url = f"{self.base_url}/models"
        try:
            resp = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"},
                                timeout=10)
        except Exception:
            return []
        if resp.status_code == 200:
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        return []


class AsyncLLMClient:
    """OpenAI 兼容 async 客户端 (httpx.AsyncClient, 守卫导入)。

    achat  -> (content, tool_calls, usage)
             tool_calls: List[{"id","name","arguments"(dict)}]
    astream-> AsyncIterator[dict]:
             {"type":"delta","content":str}         增量文本
             {"type":"final","content","tool_calls","usage"} 聚合结果
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        circuit_breaker: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        if httpx is None:
            raise RuntimeError("httpx 未安装,无法使用 AsyncLLMClient "
                               "(pip install httpx)")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "xxx")
        self.base_url = _normalize_base_url(
            base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self._client: Optional["httpx.AsyncClient"] = None
        if circuit_breaker is not None:
            self._circuit_breaker = circuit_breaker
        elif CircuitBreaker is not None:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=5, recovery_timeout=30.0
            )
        else:
            self._circuit_breaker = None
        self._metrics = metrics or (_get_obs_metrics() if _get_obs_metrics else None)

    def _get_client(self) -> "httpx.AsyncClient":
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _payload(self, messages, tools, stream=False, model=None,
                 max_tokens=4096, temperature=0.7) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
        return payload

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        idle_timeout: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(messages, tools, model=model,
                                max_tokens=max_tokens, temperature=temperature)
        idle_limit = float(idle_timeout if idle_timeout is not None else self.timeout)
        if idle_limit <= 0:
            raise ValueError("idle_timeout must be positive")

        # Circuit breaker gate
        if self._circuit_breaker is not None:
            if not self._circuit_breaker.allow_request():
                if self._metrics is not None:
                    self._metrics.increment_counter(
                        "llm_circuit_open", {"model": self.model})
                raise CircuitOpenError("Circuit breaker open for LLM upstream")

        call_start = time.monotonic()
        deadline = call_start + TOTAL_DEADLINE
        last_exc: Optional[Exception] = None
        client = self._get_client()
        for attempt in range(MAX_RETRIES + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                resp = await client.post(
                    url, json=payload, headers=self._headers(),
                    timeout=min(idle_limit, remaining))
                if resp.status_code in (429, 500, 502, 503, 504) \
                        and attempt < MAX_RETRIES:
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                    if self._metrics is not None:
                        self._metrics.increment_counter(
                            "llm_retries", {"model": self.model,
                                            "status": str(resp.status_code)})
                    await asyncio.sleep(min(_backoff_delay(attempt),
                                            max(0, deadline - time.monotonic())))
                    continue
                resp.raise_for_status()
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()
                if self._metrics is not None:
                    elapsed_ms = (time.monotonic() - call_start) * 1000
                    self._metrics.observe_histogram(
                        "llm_call_latency", elapsed_ms,
                        labels={"model": self.model, "status": "success"})
                    self._metrics.increment_counter(
                        "llm_calls", {"model": self.model, "status": "success"})
                data = _merge_reasoning(resp.json())
                return self._parse_completion(data)
            except CircuitOpenError:
                # Circuit breaker rejection must NOT be retried — fail immediately.
                raise
            except Exception as e:  # noqa: BLE001
                last_exc = e
                # Only retry on transient / network errors, not on HTTPError
                # from non-retryable status codes (400, 401, 403, etc.).
                _is_http_error = (
                    httpx is not None
                    and isinstance(e, httpx.HTTPStatusError)
                )
                _retryable_http = (
                    _is_http_error
                    and getattr(e, "response", None) is not None
                    and e.response.status_code in (429, 500, 502, 503, 504)
                )
                _is_transient = not _is_http_error or _retryable_http
                if not _is_transient:
                    break  # non-retryable — stop immediately
                if attempt < MAX_RETRIES and time.monotonic() < deadline:
                    await asyncio.sleep(min(_backoff_delay(attempt),
                                            max(0, deadline - time.monotonic())))
                    continue
                raise
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_failure()
        if self._metrics is not None:
            elapsed_ms = (time.monotonic() - call_start) * 1000
            self._metrics.observe_histogram(
                "llm_call_latency", elapsed_ms,
                labels={"model": self.model, "status": "error"})
            self._metrics.increment_counter(
                "llm_calls", {"model": self.model, "status": "error"})
        raise last_exc or RuntimeError("LLM 请求超出总 deadline")

    @staticmethod
    def _parse_completion(data: Dict[str, Any]
                          ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        choices = data.get("choices") or [{}]
        msg = choices[0].get("message", {}) or {}
        content = msg.get("content") or ""
        tool_calls: List[Dict[str, Any]] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "id": tc.get("id") or "",
                "name": fn.get("name") or "",
                "arguments": args if isinstance(args, dict) else {},
            })
        usage = data.get("usage") or {}
        return content, tool_calls, usage

    async def astream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        idle_timeout: Optional[float] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """OpenAI 兼容 SSE 流式解析: 增量 delta + 工具调用按 index 聚合。"""
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(messages, tools, stream=True, model=model,
                                max_tokens=max_tokens, temperature=temperature)
        client = self._get_client()
        idle_limit = float(idle_timeout if idle_timeout is not None else self.timeout)
        if idle_limit <= 0:
            raise ValueError("idle_timeout must be positive")

        content_parts: List[str] = []
        tool_acc: Dict[int, Dict[str, Any]] = {}
        usage: Dict[str, Any] = {}

        async with client.stream("POST", url, json=payload,
                                 headers=self._headers(),
                                 timeout=idle_limit) as resp:
            resp.raise_for_status()
            last_activity = time.monotonic()
            async for line in resp.aiter_lines():
                if time.monotonic() - last_activity > idle_limit:
                    raise TimeoutError(f"LLM 流式响应空闲超时（{idle_limit:g}秒）")
                if not line or not line.startswith("data:"):
                    continue
                last_activity = time.monotonic()
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                piece = delta.get("content") or ""
                if piece:
                    content_parts.append(piece)
                    yield {"type": "delta", "content": piece}
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    acc = tool_acc.setdefault(
                        idx, {"id": "", "name": "", "arguments_raw": ""})
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    fn = tc.get("function", {}) or {}
                    if fn.get("name"):
                        acc["name"] += fn["name"]
                    if fn.get("arguments"):
                        acc["arguments_raw"] += fn["arguments"]

        tool_calls: List[Dict[str, Any]] = []
        for idx in sorted(tool_acc):
            acc = tool_acc[idx]
            try:
                args = json.loads(acc["arguments_raw"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "id": acc["id"] or f"call_{idx}",
                "name": acc["name"],
                "arguments": args if isinstance(args, dict) else {},
            })
        yield {
            "type": "final",
            "content": "".join(content_parts),
            "tool_calls": tool_calls,
            "usage": usage,
        }

    async def alist_models(self) -> List[str]:
        url = f"{self.base_url}/models"
        try:
            client = self._get_client()
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return []
