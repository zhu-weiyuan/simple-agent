#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-compatible LLM client for mimo-v2.5

Usage:
    from my_agent.llm import LLMClient
    llm = LLMClient()  # Reads from .env or env vars
    response = llm.chat("你好")
"""

import json
import os
from pathlib import Path
from typing import List, Optional


class LLMClient:
    """OpenAI-compatible LLM client (works with mimo-v2.5)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,  # mimov2.5 needs higher tokens due to reasoning
    ):
        # Load from .env file if exists
        env_path = Path(__file__).parent.parent.parent.parent / ".env"
        if env_path.exists():
            self._load_env(env_path)

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.xiaomimimo.com/v1"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.model = model or os.getenv("OPENAI_MODEL") or "mimo-v2.5"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _load_env(self, path: Path):
        """Load .env file."""
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    def chat(
        self,
        messages: List[dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send chat completion request.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            stream: Whether to stream responses (returns generator if True)
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Assistant's response text.
        """
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        return (content + reasoning).strip() or reasoning.strip()

    def chat_stream(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Stream chat completion.

        Yields chunks of text as they arrive.
        """
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }

        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8").strip()
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]  # Remove "data: " prefix
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def run_agent_loop(
        self,
        user_input: str,
        system_prompt: str = "You are a helpful assistant.",
        history: Optional[List[dict]] = None,
    ) -> dict:
        """Run a single agent turn.

        Args:
            user_input: User's message
            system_prompt: System instruction
            history: Previous conversation history

        Returns:
            {"response": str, "history": List[dict]}
        """
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_input})

        response = self.chat(messages)

        # Update history
        new_history = history or []
        new_history.append({"role": "user", "content": user_input})
        new_history.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "history": new_history,
        }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    llm = LLMClient()
    print(f"LLM Client: {llm.model}")
    print(f"Base URL: {llm.base_url}")

    result = llm.run_agent_loop(
        user_input="你好，请介绍一下自己",
        system_prompt="你是一个全栈工程师助手。",
    )
    print(f"\nResponse: {result['response']}")
