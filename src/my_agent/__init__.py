# -*- coding: utf-8 -*-
"""
my_agent — SimpleAgent v2.0

融合最新 Agent 框架最佳实践:
- strands-agents: AgentBase Protocol, AgentResult, Agent-as-Tool
- A2A Protocol: Agent Card, 任务状态, 互操作
- LangGraph: 图状态管理
- AgentScope Runtime: 生产级特性

分层架构:
- types/    : 统一类型定义（Message, ToolDefinition, SessionConfig, AgentResult）
- core/     : 核心引擎（QueryEngine, HookRegistry）
- tools/    : 工具系统（ToolRegistry, BaseTool, builtins/, agent_tool, structured_output）
- memory/   : 记忆系统（MemoryStore, MemoryRetriever）
- bridge/   : 桥接层（Bridge, LocalBridge, PermissionPolicy）
- cli/      : CLI 接口（repl）
- enhanced/ : 增强模块（query_router, persona, hallucination, citation）
- a2a/      : A2A 协议（Agent-to-Agent 互操作）
- multiagent/: 多 agent 编排（Supervisor, Chain, Parallel）

入口:
    from my_agent import SimpleAgent
    agent = SimpleAgent()
    result = agent.run("你好")
"""
from .agent import SimpleAgent
from .types.agent import (
    AgentBase,
    AgentCard,
    AgentInput,
    AgentResult,
    AgentState,
    AgentMetrics,
    StopReason,
)
from .tools.agent_tool import AgentAsTool, create_agent_tool
from .tools.structured_output import StructuredOutputTool
from .a2a import A2AClient, A2AServer, AgentCard as A2AAgentCard
from .multiagent import SupervisorAgent, AgentChain, ParallelAgent, AgentRole

__all__ = [
    # Core
    "SimpleAgent",
    # Types
    "AgentBase",
    "AgentCard",
    "AgentInput",
    "AgentResult",
    "AgentState",
    "AgentMetrics",
    "StopReason",
    # Tools
    "AgentAsTool",
    "create_agent_tool",
    "StructuredOutputTool",
    # A2A
    "A2AClient",
    "A2AServer",
    "A2AAgentCard",
    # Multi-agent
    "SupervisorAgent",
    "AgentChain",
    "ParallelAgent",
    "AgentRole",
]
