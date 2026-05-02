# -*- coding: utf-8 -*-
"""
my_agent — SimpleAgent 重构版

分层架构：
- types/    : 统一类型定义（Message, ToolDefinition, SessionConfig）
- core/     : 核心引擎（QueryEngine, HookRegistry）
- tools/    : 工具系统（ToolRegistry, BaseTool, builtins/）
- memory/   : 记忆系统（MemoryStore, MemoryRetriever）
- bridge/   : 桥接层（Bridge, LocalBridge, PermissionPolicy）
- cli/      : CLI 接口（repl）
- enhanced/ : 增强模块（query_router, persona, hallucination, citation）

入口：
    from my_agent import SimpleAgent
    agent = SimpleAgent()
    result = agent.run("你好")
"""
from .agent import SimpleAgent

__all__ = ["SimpleAgent"]
