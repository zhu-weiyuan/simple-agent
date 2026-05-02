# -*- coding: utf-8 -*-
"""
my_agent.tools.agent_tool — Agent-as-Tool 适配器

参考 strands-agents _AgentAsTool 设计：
- 将 Agent 包装为工具，供其他 agent 调用
- 支持上下文保持/重置
- 支持并行执行
"""
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import BaseTool
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentAsTool(BaseTool):
    """将 Agent 包装为工具（参考 strands-agents _AgentAsTool）

    允许一个 agent 作为工具被另一个 agent 调用，实现多 agent 协作。

    Example:
        ```python
        researcher = SimpleAgent(name="researcher", description="搜索信息")
        writer = SimpleAgent(name="writer", description="撰写内容")

        # 将 researcher 作为 writer 的工具
        research_tool = AgentAsTool(researcher, name="research")
        writer.add_tool(research_tool)

        writer.run("写一篇关于AI的文章")  # writer 会自动调用 researcher
        ```
    """

    def __init__(
        self,
        agent: Any,  # SimpleAgent or any AgentBase implementation
        name: str = "agent_tool",
        description: Optional[str] = None,
        preserve_context: bool = False,
    ):
        """初始化 Agent-as-Tool 适配器

        Args:
            agent: 要包装的 agent 实例
            name: 工具名称
            description: 工具描述
            preserve_context: 是否保持对话上下文（False=每次调用重置）
        """
        self._agent = agent
        self.name = name
        self.description = description or self._auto_description(agent, name)
        self.parameters: Dict[str, Any] = {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Input text for the agent"}},
            "required": ["input"],
        }
        self.tags: list[str] = ["agent"]
        self._preserve_context = preserve_context

        # 快照初始状态（用于非保持模式）
        self._initial_history = None
        if not preserve_context and hasattr(agent.engine, 'session'):
            self._initial_history = copy.deepcopy(agent.engine.session.messages)

        self._call_count = 0

    @staticmethod
    def _auto_description(agent: Any, name: str) -> str:
        desc = getattr(agent, 'description', None)
        if desc:
            return desc
        return f"Use the {name} agent to handle specialized tasks"

    def execute(self, params: Dict[str, Any]) -> str:
        """执行 agent 调用

        Args:
            params: 必须包含 'input' 键，为用户输入文本

        Returns:
            Agent 的回复文本
        """
        input_text = params.get("input", params.get("query", params.get("question", "")))
        if not input_text:
            return "错误：请提供输入文本（input 参数）"

        # 重置状态（如果不保持上下文）
        if not self._preserve_context and self._initial_history:
            self._agent.engine.session.messages = copy.deepcopy(self._initial_history)

        try:
            result = self._agent.run(input_text)
            self._call_count += 1
            return str(result)
        except Exception as e:
            logger.error(f"Agent tool '{self.name}' call failed: {e}")
            return f"Agent 调用失败：{type(e).__name__}: {e}"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "type": "agent_tool",
            "preserve_context": self._preserve_context,
            "call_count": self._call_count,
        }

    def register(self, registry: ToolRegistry) -> None:
        """注册到工具注册表"""
        registry.add(
            name=self.name,
            handler=self.execute,
            description=self.description,
            parameters=self.parameters,
        )


def create_agent_tool(
    agent: Any,
    name: str,
    description: Optional[str] = None,
    preserve_context: bool = False,
) -> AgentAsTool:
    """便捷函数：创建 Agent-as-Tool"""
    return AgentAsTool(
        agent=agent,
        name=name,
        description=description,
        preserve_context=preserve_context,
    )
