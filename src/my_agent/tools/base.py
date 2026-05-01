"""
my_agent.tools.base — 工具基类

参考 Claude Code Tool.ts 设计：
- BaseTool: 抽象基类，定义工具接口
- 子类实现 execute() 方法
- 自动注册到 ToolRegistry
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


class BaseTool(ABC):
    """
    工具基类。

    子类应实现：
    - name: 工具名称
    - description: 工具描述
    - parameters: JSON Schema 参数定义
    - execute(params): 工具执行逻辑

    使用方式：
        class MyTool(BaseTool):
            name = "my_tool"
            description = "Do something"
            parameters = {"type": "object", "properties": {"x": {"type": "string"}}}

            def execute(self, params: Dict[str, Any]) -> str:
                return f"Result: {params.get('x')}"

        tool = MyTool()
        tool.register(registry)
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    tags: List[str] = []
    permission_level: str = "ask"

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> str:
        """执��工具并返回结果字符串"""
        ...

    def register(self, registry: ToolRegistry) -> None:
        """注册到 ToolRegistry"""
        registry.add(
            name=self.name,
            handler=self.execute,
            description=self.description,
            parameters=self.parameters,
            tags=self.tags,
        )

    def to_openai_schema(self) -> Dict[str, Any]:
        """返回 OpenAI function calling schema"""
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
