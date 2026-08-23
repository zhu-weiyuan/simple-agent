# SimpleAgent

> **A production-ready Agent framework** — built from scratch, integrating best practices from strands-agents, A2A Protocol, LangGraph, AgentScope, and **DeepSeek Harness (DSH)**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What is this?

**SimpleAgent v2.1** is a lightweight yet production-grade AI Agent framework written in Python. It demonstrates how to build a **reliable, observable, and interoperable** agent system by combining:

| Inspiration | What we adopted |
|-------------|-----------------|
| **strands-agents** | AgentBase Protocol, `AgentResult`, Agent-as-Tool |
| **A2A Protocol** | Agent Card, Task state machine, HTTP/JSON interop |
| **LangGraph** | Graph-based state management, SessionState |
| **AgentScope Runtime** | Budget control, Circuit breaker, Audit logging |
| **DeepSeek Harness (DSH)** | **Event-driven state machine, structured compaction, long-task reliability** |

**Use cases**: Enterprise agent platforms, Multi-agent collaboration, Auditable LLM applications, Cost-controlled agent services, **Long-running autonomous tasks (code exploration, research, refactoring)**.

---

## ✨ Key Capabilities

### 🔄 **DSH-Style Event-Driven State Machine** *(NEW in v2.1)*
```
Phase: IDLE → RUNNING → MAINTENANCE
  Turn: 1, 2, 3...
    Step: 1, 2, 3...  (each step = 1 LLM call)
      Inbox: NEXT_TURN / NEXT_STEP dual queues
```
- **Phase/Turn/Step** layered state machine — explicit execution boundaries
- **Inbox dual-queue** — decoupled scheduling, resumable after interruption
- **Exploration progress tracking** — coverage-based completion (not model self-assessment)
- **Continuation injection as system message** — overrides conversational drift
- **Checkpoint/Recovery** — event sourcing for durable resumption

### 🧠 **Enhanced Reasoning Pipeline (7 Stages)**
```
Query Router → Multi-Index Retrieval → Persona Memory → 
Core Generation → Hallucination Detection → Citation Verification → Output
```
- **4-tier Query Router** (arXiv:2604.14222) — Simple → Multi-Fact → Cross-Ref → Synthesis
- **Hybrid Retrieval** — Vector + Keyword + Graph indexes with cross-validation
- **Real-time Hallucination Detection** — 5 types: factual, temporal, causal, overconfidence, fabrication
- **Deterministic Citations** — Every claim traceable to source with confidence scoring

### 🏗️ **Production-Grade Runtime**
| Component | Purpose |
|-----------|---------|
| **QueryEngine** | Async core loop with 4-layer guardrails (max tools, error circuit, progress detection, token budget) |
| **DSHAgentLoop** | DSH state machine bridge — per-request isolation, LLM injection, stream queue |
| **Job Manager** | Background task lifecycle (submit, poll, cancel, timeout, artifact storage) |
| **Artifact Store** | Large binary/blob persistence with deduplication |
| **Session Events** | Immutable event log for replay & audit |
| **Resilience Layer** | Circuit breaker, exponential backoff, error classification |

### 🗜️ **DSH-Style Context Compaction** *(NEW in v2.1)*
- **Head-anchored + priced tail** — retain_ratio=16% dynamic budget
- **Tool-pairing balanced boundaries** — never split tool_call ↔ tool_result
- **KV cache reuse** — replay prefix + compaction instruction as FINAL user message
- **Structured summary (8 sections)** — `<compacted-summary>` durable format
- **Dual-layer compaction**: L1 (SessionState, simple) + L2 (CompactionEngine, LLM)

### 📏 **Unified 128K Context Window** *(NEW in v2.1)*
| Component | Window | Input Budget (70%) |
|-----------|--------|-------------------|
| QueryEngine | 131,072 | ~91K |
| DSH StateMachine | 131,072 | ~91K |
| ContextAssembler | 128,000 | ~90K |
| fit_messages_to_budget | 131,072 | ~91K |
| CompactionEngine fallback | 131,072 | — |

### 🤝 **Multi-Agent Orchestration**
```python
# Agent-as-Tool (LLM decides when to call)
main.add_tool(sub_agent.as_tool(name="reviewer"))

# Supervisor (explicit routing)
SupervisorAgent(roles=[researcher, coder, reviewer])

# Chain / Parallel execution
AgentChain([("research", r), ("write", w)])
ParallelAgent([("summary", s), ("sentiment", s)])
```

### 🔗 **A2A Protocol (Full Implementation)**
- **Task-oriented HTTP API**: `POST /messages` → `GET /tasks/{id}` → `POST /tasks/{id}/cancel`
- **SQLite persistence** with fingerprint-based idempotency
- **True async support** (`arun` + cooperative cancellation)
- **Remote agent registry** via `A2A_AGENTS_JSON`

### 🛡️ **Security & Governance**
- PII redaction (regex + entity detection)
- Prompt injection guard (input/output scanning)
- Permission policy: `ask` / `allow` / `deny`
- System prompt confidentiality directive (anti-leakage)

### 📊 **Observability**
- Prometheus metrics (`/api/metrics`)
- Health probes: `/healthz` (liveness) / `/api/ready` (readiness) / `/api/health` (detail)
- Request tracing with correlation IDs
- Token budget estimation → real usage reconciliation

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- An OpenAI-compatible LLM endpoint (Ollama, LM Studio, vLLM, or cloud API)

### Installation
```bash
git clone https://github.com/zhu-weiyuan/simple-agent.git
cd simple-agent
pip install -e .
```

### Configuration
```bash
cp .env.example .env
# Edit .env with your LLM credentials:
# OPENAI_API_KEY=xxx
# OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama example
# OPENAI_MODEL=qwen2.5:7b
```

### Run
```bash
# CLI mode
my-agent "列出当前目录文件"

# Web mode (FastAPI + static UI)
uvicorn app_prod:app --host 0.0.0.0 --port 8000
# Then open http://localhost:8000 (chat) or http://localhost:8000/a2a.html (A2A console)
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Non-streaming: `{"message": "...", "session_id": "..."}` |
| `/api/chat` | POST | Streaming (SSE): `{"message": "...", "stream": true}` |
| `/api/health` | GET | Detailed health (system, LLM, DB, A2A) |
| `/api/ready` | GET | Kubernetes readiness probe |
| `/healthz` | GET | Kubernetes liveness probe |
| `/api/metrics` | GET | Prometheus text format |
| `/api/tools` | GET | List registered tools |
| `/api/card` | GET | Agent Card (A2A metadata) |
| `/api/conversations` | GET | Session management |
| `/a2a/messages` | POST | **A2A: Submit task** |
| `/a2a/tasks/{id}` | GET | **A2A: Query task status** |
| `/a2a/tasks/{id}/cancel` | POST | **A2A: Cancel task** |

---

## 📁 Project Structure

```
simple-agent/
├── src/my_agent/           # Core framework
│   ├── agent.py            # SimpleAgent main class
│   ├── dsh_state_machine.py  # DSH event-driven state machine (NEW)
│   ├── loop.py             # SimpleAgentLoop ← DSH bridge (NEW)
│   ├── compaction.py       # DSH compaction engine (NEW)
│   ├── core/               # QueryEngine, Hooks, ContextAssembler
│   ├── tools/              # ToolRegistry, Builtins, AgentAsTool
│   ├── memory/             # MemoryStore, Retrieval, SQLite
│   ├── enhanced/           # 7-stage pipeline modules
│   ├── multiagent.py       # Supervisor, Chain, Parallel, AgentAsTool
│   ├── a2a.py              # A2A Protocol (Server/Client/TaskStore)
│   ├── a2a_hub.py          # FastAPI route registration
│   ├── graph/              # Graph orchestration engine
│   ├── bridge/             # Permission policy, LocalBridge
│   ├── security/           # PII, Prompt Guard
│   ├── llm/                # LLMClient, AsyncLLMClient, Gateway
│   └── types/              # Message, Session, Tool, Agent types
├── web/                    # Static UI (chat, dashboard, A2A console)
├── app_prod.py             # Production FastAPI entrypoint
├── examples/               # Demo applications
│   └── code_review/        # AI code review assistant
├── evals/                  # Evaluation harness & datasets
├── tests/                  # Unit & integration tests
│   ├── test_dsh_long_task_regression.py  # Long-task regression (NEW)
│   ├── test_dsh_context_budget.py        # Context budget tests (NEW)
│   ├── test_dsh_stream_contract.py       # Stream contract (NEW)
│   └── test_llm_template_errors.py       # Template error regressions (NEW)
└── docs/                   # Architecture & operations guides
```

---

## 🧪 Testing & Evaluation

```bash
# Unit tests
pytest tests/

# Core regression suite (72 tests)
pytest tests/test_p6_pure.py tests/test_external_bug_regressions.py \
     tests/test_llm_template_errors.py tests/test_dsh_long_task_regression.py \
     tests/test_runtime_api.py tests/test_chat_idempotency.py tests/test_builtin_tools.py

# DSH long-task regression (mock LLM: 8 tool calls → completion)
pytest tests/test_dsh_long_task_regression.py -v

# Evaluation harness (offline + live)
python evals/run_eval.py
python evals/run_live_eval.py
```

**Evaluation suites**: Tool calling, Structured output, Skill routing, Safety, Performance, Production reliability (50-case stress test).

---

## 📖 Documentation

| Guide | Audience |
|-------|----------|
| [Learning Guide](docs/LEARNING_GUIDE.md) | Developers learning agent architecture |
| [Evaluation Guide](docs/EVALUATION_GUIDE.md) | QA / Researchers running benchmarks |
| [Operations Guide](docs/operations/README.md) | SREs deploying to production |
| [Source Code Map](src/my_agent/README.md) | Contributors navigating codebase |
| [Architecture Docs](docs/architecture/) | Architects reviewing design decisions |

---

## 🗣️ Interview Talking Points

> **Architecture**: "分层解耦——types 定义契约，core 跑循环，tools/memory/bridge 可插拔，enhanced pipeline 按需叠加，DSH 状态机托底长任务可靠性。"
>
> **Reliability**: "QueryEngine 四层护栏防止无限循环/成本失控/幻觉累积；DSH Phase/Turn/Step + Inbox 显式调度；Resilience 统一错误分类+熔断+重试。"
>
> **Long-Task Mastery**: "DSH 探索进度追踪 + 覆盖度完成判定 + 续行指令 system message 注入 + Checkpoint 恢复，彻底解决 '三四轮提前结束' 顽疾。"
>
> **Interop**: "完整落地 A2A 协议——任务状态机、指纹幂等、SQLite 断点恢复、真 async 取消。"
>
> **Multi-Agent**: "Agent-as-Tool 让 LLM 自主决策委托；Supervisor/Chain/Parallel 覆盖三大编排范式。"
>
> **Cost Control**: "Token 预算 estimate→reconcile 两阶段，BudgetPolicy 支持 warn/degrade/reject；128K 窗口 + 70% target_ratio + DSH 分级压缩。"

---

## 🛠️ Development

```bash
# Format & lint
ruff check --fix && ruff format

# Type check
mypy src/

# Pre-commit (configured)
pre-commit install
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [strands-agents](https://github.com/strands-agents) — AgentBase protocol design
- [A2A Protocol](https://github.com/google/A2A) — Agent interoperability standard
- [LangGraph](https://github.com/langchain-ai/langgraph) — Graph state management patterns
- [AgentScope](https://github.com/modelscope/agentscope) — Production runtime patterns
- [DeepSeek Harness](https://github.com/deepseek-ai/dsh) — Event-driven state machine & compaction engine