# SimpleAgent

A Python agent framework combining ideas from strands-agents, A2A Protocol, LangGraph, AgentScope, and DeepSeek Harness (DSH).

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

SimpleAgent v2.1 is an AI agent framework written in Python. It puts together patterns from:

| Source | Adopted |
|--------|---------|
| strands-agents | AgentBase Protocol, `AgentResult`, Agent-as-Tool |
| A2A Protocol | Agent Card, Task state machine, HTTP/JSON interop |
| LangGraph | Graph-style state management, SessionState |
| AgentScope Runtime | Budget control, Circuit breaker, Audit logging |
| DeepSeek Harness (DSH) | Event-driven state machine, structured compaction, long-task reliability |

Typical uses: local coding agents, research assistants, multi-agent experiments, cost-controlled LLM workflows.

---

## Key Components

### DSH-Style Event-Driven State Machine (v2.1)
```
Phase: IDLE → RUNNING → MAINTENANCE
  Turn: 1, 2, 3...
    Step: 1, 2, 3...  (each step = 1 LLM call)
      Inbox: NEXT_TURN / NEXT_STEP dual queues
```
- Phase/Turn/Step layered state machine with explicit boundaries
- Inbox dual-queue for decoupled scheduling
- Exploration progress tracking with coverage-based completion (not model self-assessment)
- Continuation prompts injected as system messages to override conversational drift
- Checkpoint/Recovery via event sourcing

### 7-Stage Reasoning Pipeline
```
Query Router → Multi-Index Retrieval → Persona Memory → 
Core Generation → Hallucination Detection → Citation Verification → Output
```
- 4-tier Query Router (simple → multi-fact → cross-ref → synthesis)
- Hybrid retrieval: vector + keyword + graph with cross-validation
- Hallucination detection: factual, temporal, causal, overconfidence, fabrication
- Deterministic citations with confidence scoring

### Runtime Core
| Component | Role |
|-----------|------|
| QueryEngine | Legacy async loop with guardrails (max tools, error circuit, progress detection, token budget) |
| DSHAgentLoop | **Primary path** — DSH state machine bridge with per-request isolation, LLM injection, stream queue |
| Job Manager | Background task lifecycle (submit, poll, cancel, timeout, artifacts) |
| Artifact Store | Large binary/blob persistence with deduplication |
| Session Events | Immutable event log for replay & audit |
| Resilience Layer | Circuit breaker, exponential backoff, error classification |

### DSH-Style Context Compaction (v2.1)
- Head-anchored + priced tail (retain_ratio=16% dynamic budget)
- Tool-pairing balanced boundaries — never split tool_call ↔ tool_result
- KV cache reuse — replay prefix + compaction instruction as FINAL user message
- Structured summary (8 sections) in `<compacted-summary>` durable format
- Dual-layer: L1 (SessionState, simple) + L2 (CompactionEngine, LLM)

### Unified 128K Context Window (v2.1)
| Component | Window | Input Budget (70%) |
|-----------|--------|-------------------|
| QueryEngine | 131,072 | ~91K |
| DSH StateMachine | 131,072 | ~91K |
| ContextAssembler | 128,000 | ~90K |
| fit_messages_to_budget | 131,072 | ~91K |
| CompactionEngine fallback | 131,072 | — |

### Multi-Agent Orchestration
```python
# Agent-as-Tool (LLM decides when to call)
main.add_tool(sub_agent.as_tool(name="reviewer"))

# Supervisor (explicit routing)
SupervisorAgent(roles=[researcher, coder, reviewer])

# Chain / Parallel
AgentChain([("research", r), ("write", w)])
ParallelAgent([("summary", s), ("sentiment", s)])
```

### A2A Protocol
- Task-oriented HTTP API: `POST /messages` → `GET /tasks/{id}` → `POST /tasks/{id}/cancel`
- SQLite persistence with fingerprint-based idempotency
- Async support (`arun` + cooperative cancellation)
- Remote agent registry via `A2A_AGENTS_JSON`

### Security & Governance
- PII redaction (regex + entity detection)
- Prompt injection guard (input/output scanning)
- Permission policy: `ask` / `allow` / `deny`
- System prompt confidentiality directive

### Observability
- Prometheus metrics (`/api/metrics`)
- Health probes: `/healthz` / `/api/ready` / `/api/health`
- Request tracing with correlation IDs
- Token budget estimation → real usage reconciliation

---

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI-compatible LLM endpoint (Ollama, LM Studio, vLLM, or cloud API)

### Install
```bash
git clone https://github.com/zhu-weiyuan/simple-agent.git
cd simple-agent
pip install -e .
```

### Configure
```bash
cp .env.example .env
# Edit .env:
# OPENAI_API_KEY=xxx
# OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama example
# OPENAI_MODEL=qwen2.5:7b
```

### Run
```bash
# CLI
my-agent "list files in current directory"

# Web API
uvicorn app_prod:app --host 0.0.0.0 --port 8000
# http://localhost:8000 (chat) or http://localhost:8000/a2a.html (A2A console)
```

---

## API Reference

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
| `/a2a/messages` | POST | A2A: Submit task |
| `/a2a/tasks/{id}` | GET | A2A: Query task status |
| `/a2a/tasks/{id}/cancel` | POST | A2A: Cancel task |

---

## Project Structure

```
simple-agent/
├── src/my_agent/
│   ├── agent.py              # Main class
│   ├── dsh_state_machine.py  # DSH event-driven state machine
│   ├── loop.py               # SimpleAgentLoop ← DSH bridge
│   ├── compaction.py         # DSH compaction engine
│   ├── core/                 # QueryEngine, Hooks, ContextAssembler
│   ├── tools/                # ToolRegistry, Builtins, AgentAsTool
│   ├── memory/               # MemoryStore, Retrieval, SQLite
│   ├── enhanced/             # 7-stage pipeline modules
│   ├── multiagent.py         # Supervisor, Chain, Parallel
│   ├── a2a.py                # A2A Protocol
│   ├── a2a_hub.py            # FastAPI route registration
│   ├── graph/                # Graph orchestration
│   ├── bridge/               # Permission policy, LocalBridge
│   ├── security/             # PII, Prompt Guard
│   ├── llm/                  # LLMClient, AsyncLLMClient, Gateway
│   └── types/                # Message, Session, Tool, Agent types
├── web/                      # Static UI
├── app_prod.py               # FastAPI entrypoint
├── examples/
├── evals/                    # Evaluation harness & datasets
├── tests/
│   ├── test_dsh_long_task_regression.py
│   ├── test_dsh_context_budget.py
│   ├── test_dsh_stream_contract.py
│   └── test_llm_template_errors.py
└── docs/
```

---

## Testing

```bash
# Unit tests
pytest tests/

# Core regression (72 tests)
pytest tests/test_p6_pure.py tests/test_external_bug_regressions.py \
     tests/test_llm_template_errors.py tests/test_dsh_long_task_regression.py \
     tests/test_runtime_api.py tests/test_chat_idempotency.py tests/test_builtin_tools.py

# DSH long-task regression (mock LLM: 8 tool calls → completion)
pytest tests/test_dsh_long_task_regression.py -v
```

---

## Development

```bash
# Format & lint
ruff check --fix && ruff format

# Type check
mypy src/

# Pre-commit
pre-commit install
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgments

- [strands-agents](https://github.com/strands-agents) — AgentBase protocol
- [A2A Protocol](https://github.com/google/A2A) — Interoperability standard
- [LangGraph](https://github.com/langchain-ai/langgraph) — Graph state patterns
- [AgentScope](https://github.com/modelscope/agentscope) — Runtime patterns
- [DeepSeek Harness](https://github.com/deepseek-ai/dsh) — State machine & compaction