# -*- coding: utf-8 -*-
"""
my_agent.types.message — 统一消息模型

参考 Claude Code message.ts 设计:
- Role 枚举映射 OpenAI API 角色
- Message 数据类支持 system / user / assistant / tool
- ToolCall 表示单次工具调用请求
- 双向序列化:内部格式 ↔ OpenAI Chat Completions 格式
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(Enum):
    """消息角色，对应 OpenAI Chat Completions 的 role 字段"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用请求（对应 OpenAI tool_calls 数组元素）"""
    id: str
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def from_openai(cls, tc: Any) -> ToolCall:
        """从 OpenAI response.choices[0].message.tool_calls 元素构建"""
        return cls(
            id=tc.id,
            name=tc.function.name,
            arguments=cls._parse_args(tc.function.arguments),
        )

    def to_openai(self) -> Dict[str, Any]:
        """序列化为 OpenAI tool_calls 格式"""
        import json
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @staticmethod
    def _parse_args(raw: Optional[str]) -> Dict[str, Any]:
        import json
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class Message:
    """
    Agent 内部消息模型。

    职责:
    - 统一表示所有角色消息
    - 提供 to_openai() / from_openai() 双向序列化
    - 工厂方法:system() / user() / assistant() / tool_result() / summary_boundary()
    """
    role: Role
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""  # 仅 TOOL 角色使用
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── 序列化 ───────────────────────────────────────────────

    def to_openai(self) -> Dict[str, Any]:
        """序列化为 OpenAI Chat Completions 消息格式"""
        msg: Dict[str, Any] = {"role": self.role.value, "content": self.content or None}

        if self.role == Role.ASSISTANT and self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]

        if self.role == Role.TOOL and self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        return msg

    def to_dict(self) -> Dict[str, Any]:
        """通用字典序列化"""
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [tc.__dict__ for tc in self.tool_calls] if self.tool_calls else [],
            "tool_call_id": self.tool_call_id,
            "metadata": self.metadata,
        }

    # ── 工厂方法 ─────────────────────────────────────────────

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> Message:
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
        )

    @classmethod
    def tool_result(
        cls, tool_call_id: str, content: str, is_error: bool = False
    ) -> Message:
        msg = cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)
        if is_error:
            msg.metadata["is_error"] = True
        return msg

    @classmethod
    def summary_boundary(cls, summary: str) -> Message:
        """历史压缩后的摘要标记消息"""
        return cls(
            role=Role.SYSTEM,
            content=f"[HISTORY SUMMARY]\n{summary}",
            metadata={"summary_boundary": True},
        )

    # ── 反序列化 ─────────────────────────────────────────────

    @classmethod
    def from_openai_choice(cls, message_attr: Any) -> Message:
        """从 OpenAI response.choices[0].message 构建"""
        tool_calls_raw = getattr(message_attr, "tool_calls", None) or []
        tool_calls = [ToolCall.from_openai(tc) for tc in tool_calls_raw]
        return cls(
            role=Role(getattr(message_attr, "role", "assistant")),
            content=getattr(message_attr, "content", "") or "",
            tool_calls=tool_calls,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """从 OpenAI 格式字典构建"""
        import json
        role = Role(data.get("role", "user"))
        content = data.get("content", "") or ""

        tool_calls: List[ToolCall] = []
        if "tool_calls" in data:
            for tc in data["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"].get("arguments", "{}")),
                ))

        tool_call_id = data.get("tool_call_id", "")
        is_error = bool(data.get("is_error", False))

        return cls(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            metadata={"is_error": True} if is_error else {},
        )


# ── Type Alias ──────────────────────────────────────────────

Messages = List[Message]
