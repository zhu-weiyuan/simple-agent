"""Token estimation and deterministic context-budget allocation."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Sequence


class TokenEstimator:
    """Best-effort local token estimates; never calls a model API.

    P6: \u4f30\u7b97\u903b\u8f91\u7edf\u4e00\u5e76\u5165 context_assembler.estimate_tokens
    (tiktoken \u4f18\u5148, \u56de\u9000 CJK\u00d70.7 + ASCII\u8bcd\u00d71.3 + \u5176\u4f59\u00d70.3)\u3002
    \u6b64\u7c7b\u4fdd\u7559\u4f5c\u4e3a\u517c\u5bb9\u5305\u88c5\u3002
    """

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt

    def estimate(self, text: str) -> int:
        from .context_assembler import estimate_tokens
        text = text or ""
        if not text:
            return 0
        return estimate_tokens(text)

    def estimate_message(self, msg: Any) -> int:
        if isinstance(msg, dict):
            content = str(msg.get("content") or "")
            role = str(msg.get("role") or "")
        else:
            content = str(getattr(msg, "content", "") or "")
            role = str(getattr(msg, "role", "") or "")
        return self.estimate(content) + self.estimate(role) + 4

    def estimate_system_prompt(self) -> int:
        return self.estimate(self.system_prompt)

    def estimate_rag_chunks(self, chunks: List[str]) -> int:
        return sum(self.estimate(chunk) for chunk in chunks)


class TokenBudgetManager:
    """Keeps system, memory, history and RAG context inside an input budget."""

    def __init__(self, total_context_window: int, reserved_for_output: int = 2048) -> None:
        if total_context_window <= 0:
            raise ValueError("total_context_window must be positive")
        self.total_context_window = total_context_window
        self.reserved_for_output = max(0, reserved_for_output)
        self.current_usage = 0
        self._logger = logging.getLogger(__name__)

    @property
    def input_budget(self) -> int:
        return max(0, self.total_context_window - self.reserved_for_output)

    def allocate(self, system_tokens: Any, rag_tokens: Any, memory_tokens: Any, history_tokens: Any) -> list:
        """Return ordered pieces that fit, accepting integers or token-bearing sequences.

        With integer inputs this returns included section names. With sequences, RAG/history
        are trimmed from oldest/distant (front) first and returned as their retained values.
        """
        pieces: List[tuple[str, Any, int]] = []
        for name, value in (("system", system_tokens), ("memory", memory_tokens),
                            ("history", history_tokens), ("rag", rag_tokens)):
            token_count = value if isinstance(value, int) else sum(
                len(str(item)) // 4 + 1 for item in (value if isinstance(value, Sequence) and not isinstance(value, str) else [value])
            )
            pieces.append((name, value, token_count))
        budget = self.input_budget
        # System and memory are highest priority. History is compacted before RAG is discarded.
        retained: List[tuple[str, Any, int]] = []
        used = 0
        for name, value, count in pieces[:2]:
            if used + count <= budget:
                retained.append((name, value, count)); used += count
            else:
                self._logger.warning("Token budget cannot fit %s context", name)
        for name, value, count in pieces[2:]:
            available = max(0, budget - used)
            if count <= available:
                retained.append((name, value, count)); used += count
            elif not isinstance(value, int) and isinstance(value, Sequence) and not isinstance(value, str):
                kept, kept_count = [], 0
                for item in reversed(value):  # retain newest history / closest RAG last
                    item_count = len(str(item)) // 4 + 1
                    if kept_count + item_count <= available:
                        kept.insert(0, item); kept_count += item_count
                if kept:
                    retained.append((name, kept, kept_count)); used += kept_count
        self.current_usage = used + self.reserved_for_output
        alerts = self.alerts_for_usage(self.usage_percentage())
        for alert in alerts:
            level = logging.ERROR if alert.startswith("ERROR") else logging.WARNING if alert.startswith("WARNING") else logging.INFO
            self._logger.log(level, alert)
        return [value if not isinstance(value, int) else name for name, value, _ in retained]

    def usage_percentage(self) -> float:
        return min(100.0, (self.current_usage / self.total_context_window) * 100)

    def alerts_for_usage(self, usage_pct: float) -> List[str]:
        if usage_pct > 80:
            return ["ERROR: Context usage exceeds 80% (Dumb Zone)"]
        if usage_pct > 60:
            return ["WARNING: Context Anxiety approaching (usage exceeds 60%)"]
        if usage_pct > 40:
            return ["INFO: Context usage exceeds 40%"]
        return []
