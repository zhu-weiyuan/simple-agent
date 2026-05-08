# -*- coding: utf-8 -*-
"""
my_agent.types.agent — Agent 类型定义

参考 strands-agents/sdk-python 和 A2A 协议设计:
- AgentBase Protocol: 定义标准 agent 接口
- AgentResult: 结构化结果
- AgentState: 可序列化状态
- AgentCard: agent 元数据（A2A 兼容）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Protocol,
    TypeAlias,
    TYPE_CHECKING,
    runtime_checkable,
)

from .message import Message, Messages

# JSON 可序列化字典
JSONSerializableDict: TypeAlias = Dict[str, Any]


# ── Agent State ─────────────────────────────────────────────

AgentState = JSONSerializableDict


# ── Agent Input ─────────────────────────────────────────────

AgentInput: TypeAlias = str | Messages | None


# ── Stop Reason ─────────────────────────────────────────────

class StopReason(str, Enum):
    """Agent 停止原因"""
    COMPLETE = "complete"              # 正常完成
    TOOL_USE = "tool_use"              # 等待工具执行
    MAX_TOKENS = "max_tokens"          # 达到 token 上限
    CONTENT_FILTER = "content_filter"  # 内容过滤
    ERROR = "error"                    # 错误
    STOP_SEQUENCE = "stop_sequence"    # 停止序列


# ── Agent Metrics ───────────────────────────────────────────

@dataclass
class AgentMetrics:
    """Agent 执行指标"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0
    iterations: int = 0

    @property
    def context_size(self) -> int:
        """当前上下文大小"""
        return self.input_tokens

    @property
    def projected_context_size(self) -> int:
        """预测下次调用上下文大小"""
        return self.input_tokens + self.output_tokens


# ── Agent Result ────────────────────────────────────────────

@dataclass
class AgentResult:
    """Agent 执行结果（参考 strands-agents AgentResult）

    Attributes:
        stop_reason: 停止原因
        message: 最后一条消息
        metrics: 执行指标
        state: 额外状态
        structured_output: 结构化输出（如果有）
    """
    stop_reason: StopReason
    message: Message
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    state: AgentState = field(default_factory=dict)
    structured_output: Dict[str, Any] | None = None

    def text(self) -> str:
        """获取文本内容"""
        return self.message.content or ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "stop_reason": self.stop_reason.value,
            "message": self.message.to_dict(),
            "metrics": {
                "input_tokens": self.metrics.input_tokens,
                "output_tokens": self.metrics.output_tokens,
                "total_tokens": self.metrics.total_tokens,
                "tool_calls": self.metrics.tool_calls,
                "duration_ms": self.metrics.duration_ms,
                "iterations": self.metrics.iterations,
            },
            "state": self.state,
            "structured_output": self.structured_output,
        }

    def __str__(self) -> str:
        return self.text()


# ── Agent Card (A2A Compatible) ─────────────────────────────

@dataclass
class AgentCard:
    """Agent 元数据卡片（兼容 A2A 协议）

    用于 agent 发现和互操作，包含 agent 的能力描述。
    """
    name: str
    description: str = ""
    version: str = "1.0.0"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    url: str = ""
    default_input_languages: List[str] = field(default_factory=lambda: ["zh", "en"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "url": self.url,
            "default_input_languages": self.default_input_languages,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ── Agent Base Protocol ─────────────────────────────────────

@runtime_checkable
class AgentBase(Protocol):
    """Agent 基础协议（参考 strands-agents AgentBase）

    所有 agent 实现必须满足的最小接口。
    """

    name: str
    description: str

    def invoke(
        self,
        prompt: AgentInput = None,
        **kwargs: Any,
    ) -> AgentResult:
        """同步调用 agent

        Args:
            prompt: 输入提示
            **kwargs: 额外参数

        Returns:
            AgentResult 包含完整结果
        """
        ...

    def stream(
        self,
        prompt: AgentInput = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式调用 agent

        Args:
            prompt: 输入提示
            **kwargs: 额外参数

        Yields:
            文本片段
        """
        ...

    def card(self) -> AgentCard:
        """获取 agent 元数据卡片"""
        ...


# ── Interrupt ───────────────────────────────────────────────

@dataclass
class Interrupt:
    """执行中断（用户介入、外部事件等）"""
    reason: str
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Interrupt({self.reason})"
