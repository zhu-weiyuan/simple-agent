# -*- coding: utf-8 -*-
"""
my_agent.types.tool — 工具类型定义

参考 Claude Code Tool.ts 设计：
- ToolDefinition: 工具元数据（名称、描述、参数 schema、处理器、标签）
- ToolResult: 工具执行结果
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDefinition:
    """工具定义元数据"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Callable[[Dict[str, Any]], str]
    tags: List[str] = field(default_factory=list)
    permission_level: str = "ask"  # alwaysAllow | ask | deny

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                },
            },
        }


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    output: str
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_message(self, tool_call_id: str) -> Dict[str, Any]:
        """转换为 OpenAI tool 消息格式"""
        return {
            "role": "tool",
            "content": self.output,
            "tool_call_id": tool_call_id,
        }
