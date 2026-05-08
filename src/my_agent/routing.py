# -*- coding: utf-8 -*-
"""
my_agent.routing — 向后兼容 shim

旧代码 from my_agent.routing import QueryComplexityClassifier 仍然可用。
实际实现已迁移到 enhanced/query_router.py。
"""
from .enhanced.query_router import (
    QueryComplexityClassifier,
    DynamicRouter,
    QueryTier,
    RetrievalStrategy,
    QueryAnalysis,
)

__all__ = [
    "QueryComplexityClassifier",
    "DynamicRouter",
    "QueryTier",
    "RetrievalStrategy",
    "QueryAnalysis",
]
