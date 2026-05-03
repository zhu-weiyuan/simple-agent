# -*- coding: utf-8 -*-
"""
my_agent.multiagent — 多 Agent 编排

参考 strands-agents 和 LangGraph 设计：
- Supervisor 模式：一个 supervisor 调度多个 specialist agents
- Agent-as-Tool：agent 作为工具被其他 agent 调用
- 并行执行：多个 agent 同时处理不同任务
- 链式执行：agent 输出作为下一个 agent 的输入
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .types.agent import AgentResult, AgentState, StopReason, AgentMetrics
from .types.message import Message, Messages
from .tools.agent_tool import AgentAsTool, create_agent_tool

logger = logging.getLogger(__name__)


# ── Multi-Agent Patterns ────────────────────────────────────


@dataclass
class AgentRole:
    """Agent 角色定义"""
    name: str
    description: str
    agent: Any  # SimpleAgent instance
    system_prompt: Optional[str] = None
    priority: int = 0  # Higher = more likely to be selected


@dataclass
class MultiAgentResult:
    """多 Agent 执行结果"""
    final_response: str
    agent_responses: Dict[str, str] = field(default_factory=dict)
    supervisor_choice: Optional[str] = None
    metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    duration_ms: float = 0.0


class SupervisorAgent:
    """Supervisor 模式：一个 supervisor 调度多个 specialist agents

    参考 strands-agents 多 agent 架构：
    1. Supervisor 接收用户输入
    2. 决定哪个 specialist 处理（或并行调用多个）
    3. 汇总结果返回

    Example:
        ```python
        researcher = SimpleAgent(name="researcher", description="搜索和分析信息")
        writer = SimpleAgent(name="writer", description="撰写和编辑内容")
        coder = SimpleAgent(name="coder", description="编写和调试代码")

        supervisor = SupervisorAgent(
            name="supervisor",
            roles=[
                AgentRole("researcher", "Research and analysis", researcher),
                AgentRole("writer", "Writing and editing", writer),
                AgentRole("coder", "Coding and debugging", coder),
            ]
        )

        result = supervisor.run("帮我写一篇关于AI的文章并生成演示代码")
        ```
    """

    def __init__(
        self,
        name: str = "supervisor",
        roles: Optional[List[AgentRole]] = None,
        max_rounds: int = 5,
    ):
        self.name = name
        self.roles = roles or []
        self.max_rounds = max_rounds
        self._history: List[Dict[str, Any]] = []

    def add_role(self, role: AgentRole) -> None:
        """添加一个角色"""
        self.roles.append(role)

    def run(self, user_input: str) -> MultiAgentResult:
        """执行多 agent 编排

        Args:
            user_input: 用户输入

        Returns:
            MultiAgentResult 包含所有 agent 的响应
        """
        start_time = time.time()
        agent_responses = {}

        # Step 1: Supervisor decides which agent(s) to use
        choice = self._decide(user_input)
        logger.info(f"Supervisor chose: {choice}")

        # Step 2: Execute chosen agent(s)
        if isinstance(choice, list):
            # Parallel execution
            results = {}
            for role_name in choice:
                role = self._get_role(role_name)
                if role:
                    try:
                        result = role.agent.run(user_input)
                        results[role_name] = str(result)
                    except Exception as e:
                        results[role_name] = f"Error: {e}"
            final = self._synthesize(user_input, results)
        else:
            # Single agent
            role = self._get_role(choice)
            if role:
                try:
                    result = role.agent.run(user_input)
                    agent_responses[choice] = str(result)
                    final = str(result)
                except Exception as e:
                    final = f"Error calling {choice}: {e}"
            else:
                final = f"Unknown agent: {choice}"

        duration = (time.time() - start_time) * 1000

        self._history.append({
            "input": user_input,
            "choice": choice,
            "responses": agent_responses,
            "final": final,
        })

        return MultiAgentResult(
            final_response=final,
            agent_responses=agent_responses,
            supervisor_choice=choice if isinstance(choice, str) else ", ".join(choice),
            duration_ms=duration,
        )

    def _decide(self, user_input: str) -> str | List[str]:
        """决定使用哪个 agent

        简单实现：基于关键词匹配。
        生产环境应该用 LLM 来做路由决策。
        """
        if not self.roles:
            return ""

        # Simple keyword-based routing
        keywords = {
            "researcher": ["搜索", "查找", "研究", "分析", "信息", "资料", "research", "search", "analyze"],
            "writer": ["写", "文章", "博客", "报告", "总结", "撰写", "write", "article", "blog", "report"],
            "coder": ["代码", "编程", "函数", "脚本", "debug", "code", "program", "function", "script"],
            "calculator": ["计算", "数学", "算", "calculator", "math", "calculate"],
        }

        input_lower = user_input.lower()
        scores = {}

        for role in self.roles:
            role_name = role.name.lower()
            score = role.priority  # Base score from priority

            # Keyword matching
            for keyword in keywords.get(role_name, []):
                if keyword.lower() in input_lower:
                    score += 10

            # Description matching
            if role.description:
                for word in role.description.lower().split()[:5]:
                    if len(word) > 2 and word in input_lower:
                        score += 3

            scores[role_name] = score

        if not scores:
            return self.roles[0].name if self.roles else ""

        best = max(scores, key=scores.get)
        return best

    def _get_role(self, name: str) -> Optional[AgentRole]:
        """获取角色"""
        for role in self.roles:
            if role.name.lower() == name.lower():
                return role
        return None

    def _synthesize(self, user_input: str, results: Dict[str, str]) -> str:
        """综合多个 agent 的结果"""
        parts = []
        for name, result in results.items():
            parts.append(f"### {name} 的结果\n{result}")
        return "\n\n".join(parts)


class AgentChain:
    """链式 Agent：输出作为下一个的输入

    Example:
        ```python
        chain = AgentChain([
            ("researcher", researcher_agent),
            ("planner", planner_agent),
            ("writer", writer_agent),
        ])
        result = chain.run("写一篇关于量子计算的文章")
        ```
    """

    def __init__(self, agents: List[Tuple[str, Any]], separator: str = "\n\n---\n\n"):
        self.agents = agents  # List of (name, agent)
        self.separator = separator

    def run(self, user_input: str) -> MultiAgentResult:
        """执行链"""
        start_time = time.time()
        current_input = user_input
        responses = {}

        for name, agent in self.agents:
            try:
                result = agent.run(current_input)
                responses[name] = str(result)
                current_input = str(result)  # Pass output to next
            except Exception as e:
                responses[name] = f"Error: {e}"
                current_input = f"Previous step failed: {e}"

        final = self.separator.join(responses.values())
        duration = (time.time() - start_time) * 1000

        return MultiAgentResult(
            final_response=final,
            agent_responses=responses,
            duration_ms=duration,
        )


class ParallelAgent:
    """并行 Agent：同时调用多个 agent

    Example:
        ```python
        parallel = ParallelAgent([
            ("summary", summary_agent),
            ("keypoints", keypoints_agent),
            ("sentiment", sentiment_agent),
        ])
        result = parallel.run("分析这篇文章：...")
        ```
    """

    def __init__(self, agents: List[Tuple[str, Any]]):
        self.agents = agents

    def _run_agent(self, name: str, agent: Any, user_input: str) -> Tuple[str, str]:
        """Thread worker: run a single agent and return (name, result)."""
        try:
            result = agent.run(user_input)
            return (name, str(result))
        except Exception as e:
            return (name, f"Error: {type(e).__name__}: {e}")

    def run(self, user_input: str) -> MultiAgentResult:
        """真正并行执行（使用 ThreadPoolExecutor）"""
        start_time = time.time()
        responses: Dict[str, str] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(self._run_agent, name, agent, user_input): name
                for name, agent in self.agents
            }
            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                responses[name] = result

        final = "\n\n".join(
            f"### {name}\n{result}" for name, result in responses.items()
        )
        duration = (time.time() - start_time) * 1000

        return MultiAgentResult(
            final_response=final,
            agent_responses=responses,
            duration_ms=duration,
        )
