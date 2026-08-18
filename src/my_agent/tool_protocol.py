"""Stable, model-facing tool error classification and recovery hints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorCategory = Literal[
    "transient", "validation", "not_found", "permission", "conflict",
    "cancelled", "timeout", "internal",
]


@dataclass(frozen=True)
class ToolErrorInfo:
    """A normalized error classification; the original tool text is retained."""

    code: str
    category: ErrorCategory
    retryable: bool
    circuit_trackable: bool
    recovery_action: str
    message: str


_TRANSIENT_MARKERS = (
    "工具执行超时", "timeout", "timed out", "temporarily unavailable",
    "connection", "network", "sqlite", "database is locked", "resource busy",
)


def classify_tool_error(message: str) -> ToolErrorInfo:
    """Classify an already-rendered tool error without changing its text.

    The engine keeps sending the human-readable original string to the model for
    compatibility, while hooks/events/metrics receive these stable fields.
    """
    raw = str(message or "")
    text = raw.lower()

    if any(marker in text for marker in _TRANSIENT_MARKERS):
        code = "TOOL_TIMEOUT" if ("timeout" in text or "超时" in raw) else "UPSTREAM_TRANSIENT"
        return ToolErrorInfo(code, "timeout" if code == "TOOL_TIMEOUT" else "transient",
                             True, True, "retry_same_call", raw)
    if any(marker in raw for marker in ("权限", "拒绝", "安全限制", "未授权")) or "permission" in text:
        return ToolErrorInfo("PERMISSION_DENIED", "permission", False, False,
                             "request_permission_or_explain", raw)
    if any(marker in raw for marker in ("不存在", "不是文件", "不是 Git 仓库")) or "not found" in text:
        return ToolErrorInfo("NOT_FOUND", "not_found", False, False,
                             "list_or_search_then_retry_with_new_arguments", raw)
    if any(marker in raw for marker in ("参数", "未提供", "未知工具", "schema", "json 解析失败", "仅支持")):
        return ToolErrorInfo("INVALID_ARGUMENT", "validation", False, False,
                             "correct_arguments_or_choose_another_tool", raw)
    if any(marker in raw for marker in ("版本冲突", "已变化", "stale version", "conflict")):
        return ToolErrorInfo("STALE_STATE", "conflict", False, False,
                             "re_read_state_then_retry_with_new_arguments", raw)
    if any(marker in raw for marker in ("取消", "cancelled", "canceled")):
        return ToolErrorInfo("CANCELLED", "cancelled", False, False,
                             "stop_or_start_a_new_task", raw)
    return ToolErrorInfo("TOOL_INTERNAL_ERROR", "internal", False, False,
                         "explain_failure_or_try_a_safe_alternative", raw)


__all__ = ["ToolErrorInfo", "classify_tool_error"]
