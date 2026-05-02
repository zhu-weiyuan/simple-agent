# -*- coding: utf-8 -*-
"""
my_agent.memory — 向后兼容 shim

旧代码 from my_agent.memory import SimpleMemoryStore 仍然可用。
实际实现已迁移到 memory/store.py。
"""
from .memory.store import MemoryStore, MemoryRecallResult
from .memory.retrieval import MemoryRetriever

# 向后兼容别名
SimpleMemoryStore = MemoryStore

__all__ = ["SimpleMemoryStore", "MemoryStore", "MemoryRecallResult", "MemoryRetriever"]
