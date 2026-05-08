# -*- coding: utf-8 -*-
"""
my_agent.tools.registry — 工具注册表

参考 Claude Code ToolRegistry，支持:
- 声明式注册（装饰器风格）
- schema 自动生成
- 分组和标签
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDefinition:
    """工具定义元数据"""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], str]
    tags: List[str] = field(default_factory=list)
    permission_level: str = "ask"

    def to_openai_schema(self) -> Dict[str, Any]:
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


class ToolRegistry:
    """
    工具注册中心。

    用法:
        registry = ToolRegistry()

        @registry.register(
            name="get_time",
            description="获取当前时间",
            parameters={"type": "object", "properties": {}},
        )
        def get_time(params):
            return str(datetime.now())
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        permission_level: str = "ask",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """装饰器风格注册"""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}},
                handler=func,
                tags=tags or [],
                permission_level=permission_level,
            )
            return func
        return decorator

    def add(
        self,
        name: str,
        handler: Callable[[Dict[str, Any]], str],
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """直接添加工具"""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            handler=handler,
            tags=tags or [],
        )

    def get_handler(self, name: str) -> Optional[Callable[[Dict[str, Any]], str]]:
        tool = self._tools.get(name)
        return tool.handler if tool else None

    def all_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def all_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
