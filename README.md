# SimpleAgent

> **A production-ready Agent framework** — built from scratch, integrating best practices from strands-agents, A2A Protocol, LangGraph, and AgentScope.

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

**Use cases**: Enterprise agent platforms, Multi-agent collaboration, Auditable LLM applications, Cost-controlled agent services.

---

## ✨ Key Capabilities

### 🧠 Enhanced Reasoning Pipeline (7 Stages)
```
Query Router → Multi-Index Retrieval → Persona Memory → 
Core Generation → Hallucination Detection → Citation Verification → Output
```
- **4-tier Query Router** (arXiv:2604.14222) — Simple → Multi-Fact → Cross-Ref → Synthesis
- **Hybrid Retrieval** — Vector + Keyword + Graph indexes with cross-validation
- **Real-time Hallucination Detection** — 5 types: factual, temporal, causal, overconfidence, fabrication
- **Deterministic Citations** — Every claim traceable to source with confidence scoring

### 🏗️ Production-Grade Runtime
| Component | Purpose |
|-----------|---------|
| **QueryEngine** | Async core loop with 4-layer guardrails (max tools, error circuit, progress detection, token budget) |
| **Job Manager** | Background task lifecycle (submit, poll, cancel, timeout, artifact storage) |
| **Artifact Store** | Large binary/blob persistence with deduplication |
| **Session Events** | Immutable event log for replay & audit |
| **Resilience Layer** | Circuit breaker, exponential backoff, error classification |

### 🤝 Multi-Agent Orchestration
```python
# Agent-as-Tool (LLM decides when to call)
main.add_tool(sub_agent.as_tool(name="reviewer"))

# Supervisor (explicit routing)
SupervisorAgent(roles=[researcher, coder, reviewer])

# Chain / Parallel execution
AgentChain([("research", r), ("write", w)])
ParallelAgent([("summary", s), ("sentiment", s)])
```

### 🔗 A2A Protocol (Full Implementation)
- **Task-oriented HTTP API**: `POST /messages` → `GET /tasks/{id}` → `POST /tasks/{id}/cancel`
- **SQLite persistence** with fingerprint-based idempotency
- **True async support** (`arun` + cooperative cancellation)
- **Remote agent registry** via `A2A_AGENTS_JSON`

### 🛡️ Security & Governance
- PII redaction (regex + entity detection)
- Prompt injection guard (input/output scanning)
- Permission policy: `ask` / `allow` / `deny`
- System prompt confidentiality directive (anti-leakage)

### 📊 Observability
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
git clone https://github.com/your-org/simple-agent.git
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
└── docs/                   # Architecture & operations guides
```

---

## 🧪 Testing & Evaluation

```bash
# Unit tests
pytest tests/

# Integration tests (requires running LLM)
pytest tests/test_integration.py

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

> **Architecture**: "分层解耦——types 定义契约，core 跑循环，tools/memory/bridge 可插拔，enhanced pipeline 按需叠加。"
>
> **Reliability**: "QueryEngine 四层护栏防止无限循环/成本失控/幻觉累积；Resilience 统一错误分类+熔断+重试。"
>
> **Interop**: "完整落地 A2A 协议——任务状态机、指纹幂等、SQLite 断点恢复、真 async 取消。"
>
> **Multi-Agent**: "Agent-as-Tool 让 LLM 自主决策委托；Supervisor/Chain/Parallel 覆盖三大编排范式。"
>
> **Cost Control**: "Token 预算 estimate→reconcile 两阶段，BudgetPolicy 支持 warn/degrade/reject。"

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