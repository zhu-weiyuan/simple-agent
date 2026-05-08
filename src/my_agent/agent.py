# -*- coding: utf-8 -*-
"""
my_agent.agent — SimpleAgent 主入口

v2.0 架构（融合最新 Agent 框架最佳实践）:

核心层（参考 strands-agents SDK）:
- QueryEngine: 核心循环（core/engine.py）
- AgentBase Protocol: 标准 agent 接口（types/agent.py）
- AgentResult: 结构化结果（types/agent.py）
- ToolRegistry: 工具注册表（tools/registry.py）
- BaseTool: 工具基类（tools/base.py）
- Message: 统一消息模型（types/message.py）
- SessionState: 会话管理（types/session.py）
- HookRegistry: 钩子系统（core/hooks.py）
- MemoryStore: 记忆存储（memory/store.py）
- Bridge: 桥接层（bridge/base.py）
- EnhancedPipeline: 增强流水线（enhanced/）

多 Agent 层（参考 A2A Protocol + LangGraph）:
- A2A: Agent-to-Agent 协议（a2a.py）
- AgentAsTool: Agent 作为工具（tools/agent_tool.py）
- SupervisorAgent: 多 agent 编排（multiagent.py）
- AgentChain: 链式执行（multiagent.py）
- ParallelAgent: 并行执行（multiagent.py）

结构化输出（参考 strands-agents structured_output）:
- StructuredOutputTool: 强制 JSON Schema 输出（tools/structured_output.py）

职责:
- 组装所有组件
- 提供 run() / run_stream() / invoke() 公共 API
- AgentBase Protocol 兼容
- A2A 协议支持
- 向后兼容原有 CLI 接口
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

# ── 内部模块 ────────────────────────────────────────────────

from .types.message import Message, Role
from .types.session import SessionConfig
from .types.agent import (
    AgentBase,
    AgentCard,
    AgentInput,
    AgentResult,
    AgentState,
    AgentMetrics,
    StopReason,
)
from .core.engine import QueryEngine
from .core.hooks import HookPoint, HookRegistry
from .tools.registry import ToolRegistry
from .tools.builtins import (
    CalculatorTool,
    GetTimeTool,
    ListFilesTool,
    PowerShellTool,
    ReadFileTool,
)
from .tools.agent_tool import AgentAsTool, create_agent_tool
from .tools.structured_output import StructuredOutputTool
from .memory.store import MemoryStore
from .memory.retrieval import MemoryRetriever
from .bridge.base import LocalBridge
from .a2a import A2AClient, A2AServer, TaskState as A2ATaskState, A2AMessage

# ── 可选依赖 ────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .llm import LLMClient

try:
    from .mcp_client import MCPClient

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ── 增强模块 ────────────────────────────────────────────────

try:
    from .enhanced import (
        CategoryRAG,
        CognitiveDomain,
        DeterministicCitation,
        DynamicRouter,
        HallucinationDetector,
        MultiIndexRetrieval,
        PersonaExtractor,
        PersonaMemory,
    )

    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


class _FakeResponse:
    """模拟 OpenAI SDK 的 Response 对象，适配 QueryEngine"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data
        self.choices = [_FakeChoice(c) for c in data.get("choices", [])]


class _FakeChoice:
    """模拟 OpenAI SDK 的 Choice 对象"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data
        self.finish_reason = data.get("finish_reason", "stop")
        msg = data.get("message", {})
        self.message = _FakeMessage(msg)


class _FakeMessage:
    """模拟 OpenAI SDK 的 Message 对象"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.role = data.get("role", "assistant")
        self.content = data.get("content")
        raw_tools = data.get("tool_calls", [])
        self.tool_calls = [_FakeToolCall(t) for t in raw_tools]


class _FakeFunction:
    """模拟 OpenAI SDK 的 Function 对象"""
    def __init__(self, data: Dict[str, Any]) -> None:
        self.name = data.get("name", "")
        self.arguments = data.get("arguments", "{}")


class _FakeToolCall:
    """模拟 OpenAI SDK 的 ToolCall 对象"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.id = data.get("id", "")
        self.type = data.get("type", "function")
        func_data = data.get("function", {})
        self.function = _FakeFunction(func_data)
        self.name = func_data.get("name", "")
        self.arguments_str = func_data.get("arguments", "{}")
        try:
            self.arguments = json.loads(self.arguments_str)
        except (json.JSONDecodeError, TypeError):
            self.arguments = {}


class SimpleAgent:
    """
    v2.0 增强型 Python Agent（兼容 AgentBase Protocol）。

    融合最新 Agent 框架最佳实践:
    - strands-agents: AgentBase Protocol, AgentResult, Agent-as-Tool
    - A2A Protocol: Agent Card, 任务状态, 互操作
    - LangGraph: 图状态管理
    - AgentScope Runtime: 生产级特性

    架构分层:
    ```
    SimpleAgent (编排层)
    ├── QueryEngine (核心循环)
    │   ├── SessionState (会话管理)
    │   ├── ToolRegistry (工具注册表)
    │   └── HookRegistry (钩子系统)
    ├── MemoryStore (记忆存储)
    ├── Bridge (桥接层)
    ├── EnhancedPipeline (增强流水线)
    ├── A2A (Agent-to-Agent 协议)
    ├── MultiAgent (多 agent 编排)
    └── StructuredOutput (结构化输出)
    ```
    """

    BASE_SYSTEM_PROMPT = (
        "你是一个教学型 Python Agent，运行在 Windows 系统上。"
        "当可以直接回答时就直接回答；当需要精确外部信息时再调用工具。"
        "你拥有以下能力:"
        "- execute_powershell: 执行 PowerShell 命令"
        "- read_file: 读取文件内容"
        "- list_files: 列出目录中的文件和文件夹"
        "- calculator: 执行数学计算"
        "- get_time: 获取当前时间"
        "调用工具后，请基于工具结果给出简洁、明确的中文回答。"
        "如果长期记忆里有相关信息，应优先利用。"
    )

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        enable_enhanced: bool = True,
        session_config: Optional[SessionConfig] = None,
        name: str = "SimpleAgent",
        description: str = "A lightweight Python AI agent",
        version: str = "2.0.0",
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[2]

        # ── AgentBase Protocol 属性 ──────────────────────────
        self.name = name
        self.description = description
        self.version = version

        # ── 记忆系统 ─────────────────────────────────────────
        self.memory_store = MemoryStore(self.project_root)
        self.memory = MemoryRetriever(self.memory_store)  # 向后兼容别名

        # ── 桥接层 ───────────────────────────────────────────
        self.bridge = LocalBridge()

        # ── 工具注册表 ───────────────────────────────────────
        self.tool_registry = ToolRegistry()
        self._register_builtin_tools()

        # ── 钩子系统 ─────────────────────────────────────────
        self.hooks = HookRegistry()

        # ── 核心引擎 ─────────────────────────────────────────
        prompt = system_prompt or self.BASE_SYSTEM_PROMPT
        self.engine = QueryEngine(
            system_prompt=prompt,
            tool_registry=self.tool_registry,
            hooks=self.hooks,
            session_config=session_config,
        )



        # ── MCP ──────────────────────────────────────────────
        self._mcp_client: Optional[Any] = None
        self._init_mcp()

        # ── 调试 ─────────────────────────────────────────────
        self.debug_enabled = os.getenv(
            "MY_AGENT_DEBUG", "0"
        ).strip().lower() not in {"0", "false", "off"}

        # ── LLM 客户端 ───────────────────────────────────────
        api_key = os.getenv("OPENAI_API_KEY", "xxx")
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8080")
        model = os.getenv("OPENAI_MODEL", "Qwen3.6-35B-A3B-APEX-I-Quality.gguf")
        self.llm = LLMClient(api_key=api_key, base_url=base_url, model=model)
        self._inject_llm()

        # ── 增强模块 ─────────────────────────────────────────
        self.enable_enhanced = enable_enhanced and ENHANCED_AVAILABLE
        self._router: Optional[Any] = None
        self._persona_memory: Optional[Any] = None
        self._persona_extractor: Optional[Any] = None
        self._category_rag: Optional[Any] = None
        self._hallucination_detector: Optional[Any] = None
        self._citation_system: Optional[Any] = None
        self._multi_index: Optional[Any] = None

        if self.enable_enhanced:
            self._init_enhanced_modules()

    # ── 工具注册 ─────────────────────────────────────────────

    def _register_builtin_tools(self) -> None:
        """注册所有内置工具"""
        for tool_cls in (
            GetTimeTool,
            CalculatorTool,
            PowerShellTool,
            ReadFileTool,
            ListFilesTool,
        ):
            tool_cls().register(self.tool_registry)

    # ── LLM 注入 ─────────────────────────────────────────────

    def _inject_llm(self) -> None:
        """注入 LLM 调用函数到 QueryEngine（使用 requests 客户端）"""

        def call_fn(
            messages: List[Dict[str, Any]],
            schemas: List[Dict[str, Any]],
        ) -> Any:
            # 返回一个兼容 OpenAI SDK 的模拟对象
            resp_json = self.llm.chat(messages, tools=schemas, tool_choice="auto")
            return _FakeResponse(resp_json)

        self.engine.set_llm(call_fn)
        self._debug(f"LLM 客户端已初始化: base_url={self.llm.base_url}, model={self.llm.model}")

    # ── MCP ──────────────────────────────────────────────────

    def _init_mcp(self) -> None:
        if not MCP_AVAILABLE:
            return

        mcp_command = os.getenv("MCP_WEATHER_COMMAND", "").strip()
        if not mcp_command:
            return

        try:
            command = mcp_command.split()
            self._mcp_client = MCPClient(command)
            self._mcp_client.start()

            mcp_tools = self._mcp_client.list_tools()
            for tool in mcp_tools:
                self.tool_registry.add(
                    name=tool["name"],
                    handler=self._make_mcp_tool_wrapper(tool["name"]),
                    description=tool.get("description", ""),
                    parameters=tool.get(
                        "inputSchema",
                        {"type": "object", "properties": {}},
                    ),
                )
            self._debug(f"已加载 {len(mcp_tools)} 个 MCP 工具")
        except Exception as e:
            self._debug(f"MCP 初始化失败:{e}")
            self._mcp_client = None

    def _make_mcp_tool_wrapper(self, tool_name: str):
        def wrapper(params: Dict[str, Any]) -> str:
            if self._mcp_client is None:
                return "错误:MCP 客户端未启动"
            try:
                result = self._mcp_client.call_tool(tool_name, params)
                return str(result)
            except Exception as e:
                return f"MCP 调用失败:{type(e).__name__}: {e}"

        return wrapper

    # ── 增强模块 ─────────────────────────────────────────────

    def _init_enhanced_modules(self) -> None:
        self._router = DynamicRouter()
        self._persona_memory = PersonaMemory()
        self._persona_extractor = PersonaExtractor()
        self._category_rag = CategoryRAG(self._persona_memory)
        self._hallucination_detector = HallucinationDetector()
        self._citation_system = DeterministicCitation()
        self._multi_index = MultiIndexRetrieval()
        self._load_enhanced_state()

    def _load_enhanced_state(self) -> None:
        state_path = self.project_root / "enhanced_state.json"
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if "persona_facts" in state:
                from .enhanced.persona_memory import PersonaFact

                for fact_data in state["persona_facts"]:
                    fact = PersonaFact(
                        domain=CognitiveDomain(fact_data["domain"]),
                        fact=fact_data["fact"],
                        confidence=fact_data["confidence"],
                        timestamp=fact_data["timestamp"],
                        source=fact_data["source"],
                    )
                    self._persona_memory.add_fact(fact)
            if "documents" in state:
                from .enhanced.multi_index_retrieval import Document

                for doc_data in state["documents"]:
                    doc = Document(
                        id=doc_data["id"],
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata", {}),
                        embedding=doc_data.get("embedding"),
                    )
                    self._multi_index.add_document(doc)
            if "citations" in state:
                from .enhanced.deterministic_citation import Citation

                for cit_data in state["citations"]:
                    cit = Citation(
                        source=cit_data["source"],
                        content=cit_data["content"],
                        confidence=cit_data["confidence"],
                        timestamp=cit_data["timestamp"],
                    )
                    self._citation_system.add_citation(cit)
        except Exception as e:
            self._debug(f"加载增强状态失败: {e}")

    def _save_enhanced_state(self) -> None:
        state_path = self.project_root / "enhanced_state.json"
        state: Dict[str, Any] = {}

        if self._persona_memory:
            state["persona_facts"] = [
                {
                    "domain": f.domain.value,
                    "fact": f.fact,
                    "confidence": f.confidence,
                    "timestamp": f.timestamp,
                    "source": f.source,
                }
                for f in self._persona_memory.get_all_facts()
            ]

        if self._multi_index:
            state["documents"] = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embedding": doc.embedding,
                }
                for doc in self._multi_index.keyword_index.documents
            ]

        if self._citation_system:
            state["citations"] = [
                {
                    "source": c.source,
                    "content": c.content,
                    "confidence": c.confidence,
                    "timestamp": c.timestamp,
                }
                for c in self._citation_system.citations
            ]

        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── 增强流水线执行 ───────────────────────────────────────

    def _run_enhanced_pipeline(self, user_input: str) -> tuple[str, Dict[str, Any]]:
        """执行增强推理流水线，返回 (回复, 上下文摘要)"""
        context: Dict[str, Any] = {
            "query_tier": None,
            "routing_strategy": None,
            "is_hallucination": False,
            "hallucination_type": "",
            "has_citations": False,
            "citations_count": 0,
        }

        # Phase 1: Query Routing
        if self._router:
            analysis = self._router.route_query(user_input)
            context["query_tier"] = analysis.tier.value
            context["routing_strategy"] = analysis.strategy.value

        # Phase 2: Multi-Index Retrieval
        retrieval_context = ""
        if self._multi_index and context["query_tier"] and not context["query_tier"].startswith("tier_1"):
            results = self._multi_index.search(user_input, top_k=3)
            if results:
                retrieval_context = (
                    "相关检索结果:\n"
                    + "\n".join(
                        f"  [{r.domain}] Score:{r.score:.3f} {r.document.content}"
                        for r in results
                    )
                )

        # Phase 3: Persona Matching
        persona_context = ""
        if self._persona_extractor and self._category_rag:
            facts = self._persona_extractor.extract_facts(user_input, "conversation")
            for fact in facts:
                self._persona_memory.add_fact(fact)

            domain_facts = self._category_rag.retrieve_all(user_input, top_k=3)
            if domain_facts:
                persona_context = (
                    "用户画像记忆:\n"
                    + "\n".join(
                        f"  [{f.domain.value}] {f.fact} ({f.confidence:.2f})"
                        for f in domain_facts
                    )
                )

        # Phase 4: Build enhanced system prompt
        memory_prompt = self.memory_store.build_memory_prompt(user_input)
        self.memory_store.mark_memory_injected(memory_prompt)

        extra_parts: List[str] = []
        if memory_prompt:
            extra_parts.append(
                "以下是与当前问题相关的长期记忆，请在回答时参考:\n" + memory_prompt
            )
        if context["query_tier"] and not context["query_tier"].startswith("tier_1"):
            extra_parts.append(
                f"[路由分析: {context['query_tier']}, "
                f"策略: {context['routing_strategy']}]"
            )
        if persona_context:
            extra_parts.append(persona_context)
        if retrieval_context:
            extra_parts.append(retrieval_context)

        extra_context = "\n\n".join(extra_parts) if extra_parts else ""
        self.engine.refresh_system_prompt(extra_context)

        # Phase 5: Core loop (generation)
        raw_output = self.engine.run(user_input)

        # Phase 6: Hallucination Detection
        hallucination_note = ""
        if self._hallucination_detector:
            detection = self._hallucination_detector.detect(raw_output)
            if detection.is_hallucination:
                context["is_hallucination"] = True
                context["hallucination_type"] = detection.hallucination_type
                hallucination_note = (
                    f"\n\n[⚠️ 幻觉检测提示] {detection.correction_suggestion}"
                )

        # Phase 7: Citation Verification
        citation_note = ""
        if self._citation_system:
            citation_result = self._citation_system.extract_citations(raw_output)
            if citation_result.has_citation:
                context["has_citations"] = True
                context["citations_count"] = len(citation_result.citations)
                for cit in citation_result.citations:
                    self._citation_system.add_citation(cit)
                citation_list = "\n".join(
                    f"• [{c.source}] {c.content}" for c in citation_result.citations
                )
                citation_note = f"\n\n[引用来源]\n{citation_list}"

        final_output = raw_output + hallucination_note + citation_note
        return final_output, context

    # ── public API ───────────────────────────────────────────

    def run(self, user_input: str) -> str:
        """
        处理用户输入，返回回复。

        流程:
        1. 基础记忆召回
        2. 增强流水线（路由 → 检索 → Persona → 生成 → 检测 → 引用）
        3. 记忆自动沉淀
        """
        # 基础记忆召回
        remembered = self.memory_store.remember_from_user_text(user_input)

        # 增强流水线
        result, context = self._run_enhanced_pipeline(user_input)

        # 记忆自动沉淀
        auto_captured = self.memory_store.auto_capture_lesson(user_input, result)
        memory_note = ""
        if remembered:
            memory_note += "\n\n[Memory] 已记住这条信息。"
        if auto_captured:
            self._debug("已自动沉淀一条 lesson")

        # 调试日志
        self._debug("增强流水线执行完成:")
        self._debug(f"  查询层级: {context['query_tier']}")
        self._debug(f"  路由策略: {context['routing_strategy']}")
        self._debug(
            f"  幻觉检测: {context['is_hallucination']} "
            f"({context['hallucination_type']})"
        )
        self._debug(
            f"  引用验证: {context['has_citations']} "
            f"({context['citations_count']}条)"
        )

        # 持久化增强状态
        if self.enable_enhanced:
            self._save_enhanced_state()

        # 清空已注入记忆
        self.memory_store.clear_injected_memory()

        return result + memory_note

    def run_stream(self, user_input: str) -> Generator[str, None, None]:
        """流式处理用户输入"""
        return self.engine.run_stream(user_input)

    # ── AgentBase Protocol 方法 ──────────────────────────────

    def invoke(
        self,
        prompt: AgentInput = None,
        **kwargs: Any,
    ) -> "AgentResult":
        """AgentBase Protocol: 同步调用，返回结构化结果

        Args:
            prompt: 输入提示（字符串或 Messages）
            **kwargs: 额外参数

        Returns:
            AgentResult 包含完整结果、指标和状态
        """
        if prompt is None:
            return AgentResult(
                stop_reason=StopReason.ERROR,
                message=Message.assistant("请输入内容"),
            )

        if isinstance(prompt, list):
            # Messages list
            text = " ".join(
                m.content for m in prompt if hasattr(m, 'content') and m.content
            )
        else:
            text = str(prompt)

        start_time = time.time()
        try:
            result_text = self.run(text)
            duration = (time.time() - start_time) * 1000

            metrics = AgentMetrics(
                tool_calls=self.engine.session.tool_call_count
                    if hasattr(self.engine.session, 'tool_call_count') else 0,
                duration_ms=duration,
                iterations=len(self.engine.session.messages),
            )

            return AgentResult(
                stop_reason=StopReason.COMPLETE,
                message=Message.assistant(result_text),
                metrics=metrics,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return AgentResult(
                stop_reason=StopReason.ERROR,
                message=Message.assistant(f"错误:{type(e).__name__}: {e}"),
                metrics=AgentMetrics(duration_ms=duration),
            )

    def stream(
        self,
        prompt: AgentInput = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """AgentBase Protocol: 流式调用

        Args:
            prompt: 输入提示
            **kwargs: 额外参数

        Yields:
            文本片段
        """
        if prompt is None:
            yield "请输入内容"
            return

        text = str(prompt) if not isinstance(prompt, list) else " ".join(
            m.content for m in prompt if hasattr(m, 'content') and m.content
        )

        yield from self.run_stream(text)

    def card(self) -> "AgentCard":
        """获取 Agent Card（A2A 协议兼容）"""
        tool_names = list(self.tool_registry._handlers.keys()) if hasattr(self.tool_registry, '_handlers') else []

        return AgentCard(
            name=self.name,
            description=self.description,
            version=self.version,
            capabilities={
                "tools": len(tool_names),
                "memory": True,
                "enhanced": self.enable_enhanced,
                "mcp": self._mcp_client is not None,
                "streaming": True,
            },
            tools=tool_names,
        )

    def as_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        preserve_context: bool = False,
    ) -> "AgentAsTool":
        """将当前 agent 包装为工具（Agent-as-Tool）

        Example:
            researcher = SimpleAgent(name="researcher", description="搜索信息")
            writer = SimpleAgent(name="writer", description="撰写内容")

            # 将 researcher 作为 writer 的工具
            writer.add_tool(researcher.as_tool())
        """
        return AgentAsTool(
            agent=self,
            name=name or self.name,
            description=description or self.description,
            preserve_context=preserve_context,
        )

    def add_tool(self, tool: Any) -> None:
        """添加工具到注册表

        Args:
            tool: BaseTool 实例、AgentAsTool、或可调用对象
        """
        if isinstance(tool, AgentAsTool):
            tool.register(self.tool_registry)
        elif hasattr(tool, 'register'):
            tool.register(self.tool_registry)
        elif callable(tool):
            self.tool_registry.add(
                name=tool.__name__,
                handler=lambda p, fn=tool: fn(p),
                description=getattr(tool, '__doc__', '') or '',
                parameters={"type": "object", "properties": {}},
            )
        self._debug(f"已添加工具: {getattr(tool, 'name', tool)}")

    # ── lifecycle ────────────────────────────────────────────

    def close(self) -> None:
        if self._mcp_client is not None:
            self._mcp_client.stop()
            self._mcp_client = None

    def __enter__(self) -> "SimpleAgent":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── debug ────────────────────────────────────────────────

    def _debug(self, text: str) -> None:
        if not self.debug_enabled:
            return
        try:
            print(f"[DEBUG] {text}")
        except UnicodeEncodeError:
            safe = str(text).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            sys.stdout.buffer.write(f"[DEBUG] {safe}\n".encode("utf-8"))
