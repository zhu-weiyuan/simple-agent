# -*- coding: utf-8 -*-
"""
my_agent.graph.graph — Graph 类

参考 LangGraph 设计：
- 注册节点和边
- 构建执行图（邻接表）
- 带检查点的循环执行
- 支持错误处理和中止
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .state import GraphState
from .node import BaseNode, Edge, ConditionalEdge


class Graph:
    """图编排引擎

    用法：
    ```python
    graph = Graph(nodes={"start": start_node})
    graph.add_edge("start", "router")
    graph.add_conditional_edge(
        "router",
        lambda s: "llm" if s.get("need_llm") else "output",
        options=["llm", "output"]
    )
    state = GraphState()
    result = graph.run(state)
    ```
    """

    def __init__(self, nodes: Optional[Dict[str, BaseNode]] = None) -> None:
        self.nodes: Dict[str, BaseNode] = nodes or {}
        self.edges: List[Edge] = []
        self.conditional_edges: List[ConditionalEdge] = []
        self._checkpointer_enabled = False
        self._checkpoint_path = None

    def register_node(self, name: str, node: BaseNode) -> "Graph":
        """注册节点"""
        self.nodes[name] = node
        return self

    def add_edge(self, from_node: str, to_node: str) -> "Graph":
        """添加普通边"""
        self.edges.append(Edge(from_node, to_node))
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        condition_fn: Callable[[GraphState], str],
        options: List[str],
    ) -> "Graph":
        """添加条件边"""
        self.conditional_edges.append(ConditionalEdge(from_node, condition_fn, options))
        return self

    def enable_checkpointer(self, path: Optional[str] = None) -> "Graph":
        """启用检查点系统"""
        self._checkpointer_enabled = True
        self._checkpoint_path = path or "checkpoints"
        return self

    def run(self, initial_state: GraphState, entry_node: str = "start") -> GraphState:
        """执行图

        流程：
        1. 从 entry_node 开始
        2. 找到当前节点的所有出边
        3. 执行目标节点
        4. 重复直到没有出边或超过 max_iterations
        """
        state = initial_state
        current = entry_node

        # 检查点恢复
        if self._checkpointer_enabled:
            checkpoint = self._load_checkpoint(current)
            if checkpoint:
                state.restore(checkpoint)
                print(f"[Checkpoint] Restored from node '{current}'")

        while not state.should_stop():
            # 执行当前节点
            node_obj = self.nodes.get(current)
            if node_obj is None:
                raise KeyError(f"Node '{current}' not found. Available: {list(self.nodes.keys())}")

            try:
                state = node_obj.execute(state)
            except Exception as e:
                print(f"[Error] Node '{current}' failed: {e}")
                # 尝试错误恢复（未来扩展）
                break

            # 保存检查点
            if self._checkpointer_enabled:
                self._save_checkpoint(current, state)

            # 查找出边
            next_node = self._find_next_node(current, state)
            if next_node is None:
                print(f"[Graph] No outgoing edge from '{current}'. Stopping.")
                break

            current = next_node

        return state

    def _find_next_node(self, current: str, state: GraphState) -> Optional[str]:
        """找到下一个节点"""
        # 先检查条件边
        for cond_edge in self.conditional_edges:
            if cond_edge.from_node == current:
                target = cond_edge.condition_fn(state)
                if target in cond_edge.options:
                    print(f"[Edge] {current} → {target}")
                    return target

        # 再检查普通边
        for edge in self.edges:
            if edge.from_node == current:
                print(f"[Edge] {current} → {edge.to_node}")
                return edge.to_node

        return None

    def _save_checkpoint(self, node_name: str, state: GraphState) -> None:
        """保存检查点"""
        try:
            import os
            path = self._checkpoint_path or "checkpoints"
            os.makedirs(path, exist_ok=True)
            checkpoint_file = os.path.join(path, f"{node_name}_{state.run_id}.json")
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(state.snapshot(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Checkpoint] Save failed: {e}")

    def _load_checkpoint(self, node_name: str) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        try:
            import os
            path = self._checkpoint_path or "checkpoints"
            checkpoint_file = os.path.join(path, f"{node_name}_{self._nodes_list()[0]}.json")  # 简化
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[Checkpoint] Load failed: {e}")
        return None

    def _nodes_list(self) -> List[str]:
        return list(self.nodes.keys())


class GraphBuilder:
    """图构建器（链式 API）"""

    def __init__(self) -> None:
        self.graph = Graph()

    def entry_point(self, node_name: str) -> "GraphBuilder":
        self._entry_node = node_name
        return self

    def build(self) -> Graph:
        return self.graph


def define_graph():
    """图定义函数（子类覆盖）"""
    raise NotImplementedError("Subclasses must implement define_graph()")
