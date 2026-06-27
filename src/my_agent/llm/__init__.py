# -*- coding: utf-8 -*-
"""
my_agent.llm — LLM 客户端层 (mimo-v2.5 compatible)

Handles reasoning_content → content merge for models that separate thinking from output.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


class LLMClient:
    """OpenAI 兼容 API 客户端（基于 requests）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "xxx")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8080")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "Qwen3.6-35B-A3B-APEX-I-Quality.gguf")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """调用 chat completions API"""
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Handle mimo-v2.5 reasoning_content → content merge
        if "choices" in data and len(data["choices"]) > 0:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            # If content is empty but reasoning exists, use reasoning as content
            if not content.strip() and reasoning:
                data["choices"][0]["message"]["content"] = reasoning
        
        return data

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """流式调用 chat completions API"""
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()

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
                # Also capture reasoning_content in streaming
                reasoning = delta.get("reasoning_content", "")
                if reasoning:
                    yield reasoning
                elif content:
                    yield content
            except json.JSONDecodeError:
                continue

    def list_models(self) -> List[str]:
        """列出可用模型"""
        url = f"{self.base_url}/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        return []
