# -*- coding: utf-8 -*-
"""
my_agent.graph.state — Graph State

参考 LangGraph 的 State 设计:
- 用 TypedDict 定义图的全局状态
- 所有节点共享状态
- 支持 reducer 合并策略（最后一个值 / 追加 / 替换）
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphState:
    """图的全局状态"""

    # ── 消息历史 ────────────────────────────────
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # ── 工具调用结果 ────────────────────────────
    tool_results: Dict[str, str] = field(default_factory=dict)

    # ── 当前执行上下文 ──────────────────────────
    current_node: Optional[str] = None
    execution_path: List[str] = field(default_factory=list)

    # ── 运行时元数据 ────────────────────────────
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    iteration_count: int = 0
    max_iterations: int = 10

    # ── 自定义扩展字段 ──────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """追加消息"""
        self.messages.append({"role": role, "content": content})

    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """创建状态快照（用于检查点）"""
        import copy
        return {
            "messages": copy.deepcopy(self.messages),
            "tool_results": copy.deepcopy(self.tool_results),
            "current_node": self.current_node,
            "execution_path": list(self.execution_path),
            "run_id": self.run_id,
            "iteration_count": self.iteration_count,
            "metadata": copy.deepcopy(self.metadata),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """从快照恢复状态"""
        import copy
        self.messages = copy.deepcopy(snapshot["messages"])
        self.tool_results = copy.deepcopy(snapshot["tool_results"])
        self.current_node = snapshot["current_node"]
        self.execution_path = list(snapshot["execution_path"])
        self.run_id = snapshot["run_id"]
        self.iteration_count = snapshot["iteration_count"]
        self.metadata = copy.deepcopy(snapshot["metadata"])

    def should_stop(self) -> bool:
        """检查是否应该停止（超过最大迭代次数）"""
        return self.iteration_count >= self.max_iterations
