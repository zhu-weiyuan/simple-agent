# -*- coding: utf-8 -*-
"""
my_agent.memory.retrieval — 记忆检索器

基于 MemoryStore 提供更高阶的检索能力：
- 关键词匹配
- 上下文感知检索
- 记忆摘要生成
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .store import MemoryStore


class MemoryRetriever:
    """记忆检索器 — 在 MemoryStore 之上提供检索策略"""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve_for_prompt(
        self, query: str, include_lessons: bool = True, max_lessons: int = 5
    ) -> str:
        """为 prompt 构建记忆上下文"""
        return self.store.build_memory_prompt(query)

    def retrieve_facts(self, query: str) -> Dict:
        """检索与查询相关的结构化事实"""
        result = self.store.recall(query)
        return result.facts

    def retrieve_lessons(
        self, query: str, max_lessons: int = 5
    ) -> List[str]:
        """检索与查询相关的经验教训"""
        result = self.store.recall(query, max_lessons=max_lessons)
        return result.matched_lessons

    def retrieve_all(self) -> Dict:
        """检索全部记忆"""
        return {
            "facts": self.store.load_facts(),
            "lessons": self.store.load_lessons(),
        }
