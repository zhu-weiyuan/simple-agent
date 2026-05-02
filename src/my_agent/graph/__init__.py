# -*- coding: utf-8 -*-
"""
my_agent.graph — LangGraph 风格图编排引擎

核心组件：
- GraphState: 图的全局状态
- BaseNode: 节点基类（通过 @node() 装饰器使用）
- Edge / ConditionalEdge: 边的规则
- Graph: 组装节点和边，驱动执行
"""
from .state import GraphState
from .node import BaseNode, node, Edge, ConditionalEdge
from .graph import Graph, GraphBuilder

__all__ = ["GraphState", "BaseNode", "node", "Edge", "ConditionalEdge", "Graph", "GraphBuilder"]
