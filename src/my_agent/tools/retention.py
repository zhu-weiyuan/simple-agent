# -*- coding: utf-8 -*-
"""Bounded tool-result retention inspired by DSH output-retention.

The retainer owns only the budget fact: what was kept and what was omitted.
Tool-specific code remains responsible for rendering, exit codes, and recovery
instructions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RetainedItems:
    items: list[str]
    omitted: int

    @property
    def truncated(self) -> bool:
        return self.omitted > 0

    def notice(self, scope: str, recovery: str = "") -> str:
        if not self.truncated:
            return ""
        message = f"\n... {scope} 已截断，保留前 {len(self.items)} 项，省略 {self.omitted} 项。"
        if recovery:
            message += f" {recovery}"
        return message


class ItemRetainer:
    """Retain the first N logical items while counting every omitted item."""

    def __init__(self, max_items: int) -> None:
        if not isinstance(max_items, int) or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        self.max_items = max_items
        self._items: list[str] = []
        self._omitted = 0

    def push(self, item: object) -> None:
        if len(self._items) < self.max_items:
            self._items.append(str(item))
        else:
            self._omitted += 1

    def extend(self, items: Iterable[object]) -> None:
        for item in items:
            self.push(item)

    def finish(self) -> RetainedItems:
        return RetainedItems(list(self._items), self._omitted)


def retain_text(text: str, max_chars: int) -> tuple[str, int]:
    """Keep a bounded text prefix and return the exact omitted character count."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    value = text or ""
    return value[:max_chars], max(0, len(value) - max_chars)


def format_text_notice(scope: str, omitted: int, recovery: str = "") -> str:
    if omitted <= 0:
        return ""
    message = f"\n... {scope} 已截断，省略 {omitted} 个字符。"
    return message + (f" {recovery}" if recovery else "")


__all__ = ["ItemRetainer", "RetainedItems", "retain_text", "format_text_notice"]
