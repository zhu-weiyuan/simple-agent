"""
my_agent.graph.node — Node & Edge

参考 LangGraph 设计：
- Node: 图中的处理单元，接收状态返回新状态
- Edge: 连接节点的规则（普通边 / 条件边）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from .state import GraphState


# ── Node（节点）────────────────────────────────────

class BaseNode(ABC):
    """节点基类"""

    name: str = "unnamed"

    @abstractmethod
    def execute(self, state: GraphState) -> GraphState:
        """执行节点逻辑，返回更新后的状态"""
        ...


def node(node_name: str = ""):
    """装饰器：将函数转为 Node"""
    def decorator(fn: Callable[[GraphState], GraphState]) -> BaseNode:
        actual_name = node_name or fn.__name__

        class FunctionNode(BaseNode):
            name = actual_name  # type: ignore[assignment]

            def execute(self, state: GraphState) -> GraphState:
                state.current_node = self.name
                state.execution_path.append(self.name)
                state.iteration_count += 1
                return fn(state)

        return FunctionNode()

    return decorator


# ── Edge（边）───────────────────────────────────────

class Edge:
    """普通边：固定路由 A → B"""

    def __init__(self, from_node: str, to_node: str) -> None:
        self.from_node = from_node
        self.to_node = to_node


class ConditionalEdge:
    """条件边：根据状态决定路由 A → [B/C/D]

    condition_fn(state) 返回要路由到的节点名称
    """

    def __init__(
        self,
        from_node: str,
        condition_fn: Callable[[GraphState], str],
        options: List[str],
    ) -> None:
        self.from_node = from_node
        self.condition_fn = condition_fn
        self.options = options  # 所有可能的目标节点
