# -*- coding: utf-8 -*-
"""OpenAI-compatible LLM client with bounded request and stream lifecycles."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional


class LLMRequestTimeout(TimeoutError):
    """Raised when the configured request deadline or stream idle deadline expires."""


class LLMRequestCancelled(RuntimeError):
    """Raised when a caller-owned cancellation event interrupts a request."""


class LLMClient:
    """OpenAI-compatible LLM client (works with mimo-v2.5)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout_seconds: Optional[float] = None,
        stream_idle_timeout_seconds: Optional[float] = None,
    ) -> None:
        env_path = Path(__file__).parent.parent.parent.parent / ".env"
        if env_path.exists():
            self._load_env(env_path)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.xiaomimimo.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.model = model or os.getenv("OPENAI_MODEL") or "mimo-v2.5"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = self._positive_timeout(
            timeout_seconds if timeout_seconds is not None else os.getenv("REQUEST_TIMEOUT_SECONDS", "120"),
            "timeout_seconds",
        )
        self.stream_idle_timeout_seconds = self._positive_timeout(
            stream_idle_timeout_seconds if stream_idle_timeout_seconds is not None else os.getenv("STREAM_IDLE_TIMEOUT_SECONDS", "30"),
            "stream_idle_timeout_seconds",
        )

    @staticmethod
    def _positive_timeout(value: object, name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive number") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive number")
        return parsed

    def _load_env(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    def _request_data(self, messages: List[dict], stream: bool, temperature: Optional[float], max_tokens: Optional[int]) -> tuple[str, dict, dict]:
        return (
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "stream": stream,
            },
        )

    @staticmethod
    def _ensure_not_cancelled(cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise LLMRequestCancelled("LLM 请求已取消")

    def chat(self, messages: List[dict], stream: bool = False, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None, cancel_event: Optional[threading.Event] = None) -> str:
        if stream:
            return "".join(self.chat_stream(messages, temperature, max_tokens, cancel_event=cancel_event))
        import requests

        self._ensure_not_cancelled(cancel_event)
        url, headers, payload = self._request_data(messages, False, temperature, max_tokens)
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise LLMRequestTimeout(f"LLM 请求超时（{self.timeout_seconds:g}秒）") from exc
        self._ensure_not_cancelled(cancel_event)
        msg = response.json()["choices"][0]["message"]
        return (msg.get("content") or "").strip()

    def chat_stream(self, messages: List[dict], temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None, cancel_event: Optional[threading.Event] = None) -> Iterator[str]:
        """Yield SSE content and close the HTTP response on cancellation or idle timeout."""
        import requests

        self._ensure_not_cancelled(cancel_event)
        url, headers, payload = self._request_data(messages, True, temperature, max_tokens)
        response = None
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout_seconds)
            response.raise_for_status()
            last_activity = time.monotonic()
            for line in response.iter_lines():
                self._ensure_not_cancelled(cancel_event)
                if time.monotonic() - last_activity > self.stream_idle_timeout_seconds:
                    raise LLMRequestTimeout(f"LLM 流式响应空闲超时（{self.stream_idle_timeout_seconds:g}秒）")
                if not line:
                    continue
                last_activity = time.monotonic()
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded.startswith("data: "):
                    continue
                data_text = decoded[6:]
                if data_text == "[DONE]":
                    return
                try:
                    content = json.loads(data_text)["choices"][0].get("delta", {}).get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if content:
                    yield content
        except requests.Timeout as exc:
            raise LLMRequestTimeout(f"LLM 请求超时（{self.timeout_seconds:g}秒）") from exc
        finally:
            if response is not None:
                response.close()

    def run_agent_loop(self, user_input: str, system_prompt: str = "You are a helpful assistant.",
                       history: Optional[List[dict]] = None) -> dict:
        messages = [{"role": "system", "content": system_prompt}, *(history or []), {"role": "user", "content": user_input}]
        response = self.chat(messages)
        new_history = list(history or [])
        new_history.extend(({"role": "user", "content": user_input}, {"role": "assistant", "content": response}))
        return {"response": response, "history": new_history}


__all__ = ["LLMClient", "LLMRequestCancelled", "LLMRequestTimeout"]
