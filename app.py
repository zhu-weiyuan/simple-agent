#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleAgent v2.0 — Web Server (FastAPI)

启动: python app.py
访问: http://localhost:8000
"""
import os, sys, time, platform, uuid, psutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

os.environ.setdefault("OPENAI_API_KEY", "sk-ckmnbfew0gajnwb508q42tvbvyvcswtf9k2c6wfqwi991ksj")



sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, HTTPException, Request, Response
from my_agent.logging_config import setup_logging, logger
from my_agent.metrics import metrics
from my_agent.auth import auth_required
from my_agent.logging_config import setup_logging, logger
from my_agent.metrics import metrics
from my_agent.auth import auth_required
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import json

# Override default JSON response to use UTF-8 encoding
class UTF8JSONResponse(JSONResponse):
    def render(self, content):
        return super().render(content).decode("utf-8").encode("utf-8")

app = FastAPI(title="SimpleAgent v2.0", json_response_class=UTF8JSONResponse)

web_dir = Path(__file__).parent / "web"
web_dir.mkdir(exist_ok=True)

# ── Rate Limiter (sliding window) ───────────────────────────────
class RateLimiter:
    """Simple in-memory sliding-window rate limiter per IP.
    
    Configurable via env vars:
      RATE_LIMIT_REQUESTS  — max requests per window (default: 30)
      RATE_LIMIT_WINDOW    — window size in seconds (default: 60)
    """
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window
        # Prune old entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        now = time.time()
        cutoff = now - self.window
        current = [t for t in self._requests[key] if t > cutoff]
        return max(0, self.max_requests - len(current))

_rate_limiter = RateLimiter(
    max_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "30")),
    window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW", "60")),
)

# ── Request counter for health endpoint ────────────────────────
_request_counter = {"total": 0, "errors": 0}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only rate-limit API endpoints
    if not request.url.path.startswith("/api/"):
        response = await call_next(request)
        return response

    # ── Request correlation ID (P0: traceability) ────────────────
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id

    _request_counter["total"] += 1
    client_ip = request.client.host if request.client else "unknown"

    if not _rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": _rate_limiter.window},
        )

    # Request timing — measure latency per request
    start_time = time.time()
    try:
        response = await call_next(request)
        elapsed = time.time() - start_time
        response.headers["X-Response-Time"] = f"{elapsed*1000:.1f}ms"
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        _request_counter["errors"] += 1
        raise

from my_agent import SimpleAgent
from my_agent.security import redact as pii_redact, scan_and_log as pii_scan
from my_agent.security import scan_input as prompt_scan
from my_agent.sentiment import analyze as sentiment_analyze
from my_agent.summary import summarize as conversation_summary
from my_agent.observability import get_metrics as get_obs_metrics, get_alerts

agent = SimpleAgent()
obs_metrics = get_obs_metrics()
alert_service = get_alerts()
# Alert rule: trigger when average latency exceeds 5s
alert_service.add_rule("high_latency", "request_latency", threshold=5000)


class ChatRequest(BaseModel):
    message: str
    stream: bool = False


@app.post("/api/chat")
@auth_required
async def chat(request: Request, req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    # ── Security scan ──────────────────────────────────────
    pii_result = pii_redact(req.message)
    prompt_result = prompt_scan(req.message)

    # Use cleaned/redacted text for processing
    safe_message = pii_result.redacted_text
    if not prompt_result.is_safe:
        safe_message = prompt_result.cleaned

    # Log security findings
    if pii_result.found_pii:
        pii_scan(req.message)  # logs warning
    if not prompt_result.is_safe:
        logger.warning(f"Prompt injection detected: {prompt_result.threats}")

    try:
        start_time = time.time()
        if req.stream:
            return StreamingResponse(
                agent.run_stream(safe_message),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        reply = agent.run(safe_message)
        elapsed_ms = (time.time() - start_time) * 1000

        # Record observability metrics
        obs_metrics.increment_counter("chat_requests")
        obs_metrics.observe_histogram("request_latency", elapsed_ms)

        # Check alerts
        alert_service.check_and_alert()

        # Force UTF-8 encoding with ensure_ascii=False
        import json as j
        body = j.dumps({"reply": reply}, ensure_ascii=False).encode("utf-8")
        return Response(
            content=body,
            media_type="application/json; charset=utf-8",
        )
    except Exception as e:
        obs_metrics.increment_counter("chat_errors")
        import json as j
        body = j.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
        return Response(
            content=body,
            status_code=500,
            media_type="application/json; charset=utf-8",
        )


@app.get("/api/health")
async def health():
    """Enhanced health check with system metrics."""
    try:
        # System info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        boot_time = datetime.fromtimestamp(psutil.boot_time()).isoformat()
        uptime_seconds = int(time.time() - psutil.boot_time())
        
        # LLM connectivity check (non-blocking: skip if slow)
        llm_reachable = False
        llm_latency = None
        try:
            import urllib.request as ur
            start = time.time()
            resp = ur.urlopen(f"{agent.llm.base_url}/models", timeout=5)
            llm_reachable = resp.status == 200
            llm_latency = int((time.time() - start) * 1000)
        except Exception:
            pass
        
        # ── 动态版本: LLM 可达 + 0 错误 → 2.1.0, 否则 2.0.0 ──
        version_str = "2.1.0" if (llm_reachable and _request_counter["errors"] == 0) else "2.0.0"
        
        # sessions / inflight 各自容错，不炸全局
        try:
            active_sessions = agent.engine.get_active_session_count() if hasattr(agent.engine, 'get_active_session_count') else 0
        except Exception:
            active_sessions = 0
        try:
            inflight = obs_metrics.window_average('request_latency', 60) or 0
        except Exception:
            inflight = 0
        
        return {
            "ok": True,
            "agent": agent.name,
            "version": version_str,
            "uptime_seconds": uptime_seconds,
            "boot_time": boot_time,
            "platform": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_total_gb": round(mem.total / (1024**3), 1),
                "memory_used_gb": round(mem.used / (1024**3), 1),
                "memory_percent": mem.percent,
                "disk_total_gb": round(disk.total / (1024**3), 1),
                "disk_used_gb": round(disk.used / (1024**3), 1),
                "disk_percent": disk.percent,
            },
            "llm": {
                "reachable": llm_reachable,
                "base_url": agent.llm.base_url,
                "model": agent.llm.model,
                "latency_ms": llm_latency,
            },
            "sessions": {
                "active_sessions": active_sessions,
                "max_sessions": 100,
            },
            "inflight": inflight,
            "requests": {
                "total": _request_counter["total"],
                "errors": _request_counter["errors"],
            },
        }
    except Exception as e:
        return {"ok": True, "agent": agent.name, "version": "2.0.0", "health_error": str(e)}


@app.get("/api/ready")
async def ready():
    """Lightweight readiness probe for orchestrators."""
    return {"ready": True, "checks": {"process": True, "agent": agent is not None}}


@app.get("/api/tools")
async def tools_list():
    """List all registered tools with descriptions."""
    tools = []
    for name, tool in agent.tool_registry.tools.items():
        tools.append({
            "name": name,
            "description": getattr(tool, "description", "") or getattr(tool, "__doc__", ""),
        })
    return {"tools": tools}


@app.get("/api/costs")
async def costs_endpoint():
    """Cost summary by model (dashboard compatibility)."""
    from my_agent.cost_tracker import CostTracker
    if hasattr(agent, 'cost_tracker') and agent.cost_tracker:
        summary = agent.cost_tracker.get_summary()
        by_model = agent.cost_tracker.get_model_breakdown()
        total_cost = sum(m.get('total_cost', 0) for m in by_model.values())
        total_tokens = sum(m.get('total_tokens', 0) for m in by_model.values())
        req_count = sum(m.get('request_count', 0) for m in by_model.values())
        return {
            "by_model": by_model,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "request_count": req_count,
            "price_version": getattr(agent.cost_tracker, 'price_version', CostTracker.PRICE_VERSION),
            "currency": "¥",
        }
    # Fallback: return empty but valid structure (dashboard expects this shape)
    return {
        "by_model": {
            agent.llm.model: {
                "request_count": _request_counter["total"],
                "total_tokens": 0,
                "total_cost": 0,
            }
        },
        "total_cost": 0,
        "total_tokens": 0,
        "request_count": _request_counter["total"],
        "price_version": "2026-07",
        "currency": "¥",
    }


@app.get("/api/metrics")
async def metrics_endpoint():
    """Prometheus-style text metrics endpoint (combined legacy + observability)."""
    mem = psutil.virtual_memory()
    lines = [
        f"# HELP agent_uptime_seconds Agent uptime in seconds",
        f"# TYPE agent_uptime_seconds gauge",
        f"agent_uptime_seconds {int(time.time() - psutil.boot_time())}",
        f"# HELP agent_requests_total Total API requests",
        f"# TYPE agent_requests_total counter",
        f'agent_requests_total{{status="success"}} {_request_counter["total"] - _request_counter["errors"]}',
        f'agent_requests_total{{status="error"}} {_request_counter["errors"]}',
        f"# HELP system_memory_usage_percent System memory usage percentage",
        f"# TYPE system_memory_usage_percent gauge",
        f"system_memory_usage_percent {mem.percent}",
    ]
    # Add observability metrics
    lines.append(obs_metrics.get_prometheus_text())
    return "\n".join(lines) + "\n"


@app.get("/api/observability")
async def observability_dashboard():
    """Structured JSON observability data (not Prometheus format)."""
    return obs_metrics.get_metrics()

@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory store statistics."""
    try:
        stats = agent.memory_store.get_stats() if hasattr(agent.memory_store, 'get_stats') else {}
        return {"memory": stats}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/conversations")
async def list_conversations(limit: int = 20):
    """List recent conversation sessions from memory store.

    Returns a summary of recent conversations including message counts,
    timestamps, and last messages. Useful for session management dashboards.
    """
    try:
        results = []
        # Try to get sessions from the agent's memory store
        if hasattr(agent.memory_store, 'get_sessions'):
            sessions = agent.memory_store.get_sessions(limit=limit)
            for s in sessions:
                results.append({
                    "session_id": s.get("session_id", "unknown"),
                    "message_count": s.get("message_count", 0),
                    "created_at": s.get("created_at", ""),
                    "last_message": s.get("last_message", "")[:100] if s.get("last_message") else "",
                })
        # Fallback: read from facts.json if it exists
        elif Path("memory/facts.json").exists():
            with open("memory/facts.json", "r") as f:
                facts = json.load(f)
            if isinstance(facts, list):
                for item in facts[-limit:]:
                    results.append({
                        "session_id": item.get("session_id", "unknown"),
                        "fact": str(item.get("fact", ""))[:100],
                        "created_at": item.get("created_at", ""),
                    })
        return {
            "conversations": results,
            "total": len(results),
            "limit": limit,
        }
    except Exception as e:
        return {"error": str(e), "conversations": []}


@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a conversation session from memory.

    Removes all messages and facts associated with the given session ID.
    """
    try:
        deleted = False
        if hasattr(agent.memory_store, 'delete_session'):
            agent.memory_store.delete_session(session_id)
            deleted = True
        return {"deleted": deleted, "session_id": session_id}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── SQLite-backed Conversation Endpoints ───────────────────────

@app.get("/api/sqlite/sessions")
async def sqlite_list_sessions(limit: int = 20):
    """List sessions from SQLite store."""
    try:
        from my_agent.memory.sqlite_store import SqliteConversationStore
        store = SqliteConversationStore("conversations.db")
        sessions = store.list_sessions(limit=limit)
        store.close()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/sqlite/session/{session_id}")
async def sqlite_get_session(session_id: str):
    """Get a complete session from SQLite store."""
    try:
        from my_agent.memory.sqlite_store import SqliteConversationStore
        store = SqliteConversationStore("conversations.db")
        exported = store.export_session(session_id)
        store.close()
        return exported
    except ValueError:
        raise HTTPException(404, f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/sqlite/search")
async def sqlite_search_messages(query: str, limit: int = 20):
    """Search messages in SQLite store."""
    try:
        from my_agent.memory.sqlite_store import SqliteConversationStore
        store = SqliteConversationStore("conversations.db")
        results = store.search_messages(query, limit=limit)
        store.close()
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/sqlite/stats")
async def sqlite_stats():
    """Get SQLite store statistics."""
    try:
        from my_agent.memory.sqlite_store import SqliteConversationStore
        store = SqliteConversationStore("conversations.db")
        stats = store.get_stats()
        store.close()
        return stats
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/analytics")
async def analytics(limit: int = 50):
    """Conversation analytics endpoint.

    Returns aggregated statistics about conversations including:
    - Total message count per session
    - Average response length
    - Most active sessions
    - Token usage estimates (based on message lengths)
    """
    try:
        results = []
        total_messages = 0
        total_chars = 0

        # Try memory store
        if hasattr(agent.memory_store, 'get_sessions'):
            sessions = agent.memory_store.get_sessions(limit=limit)
            for s in sessions:
                msg_count = s.get("message_count", 0)
                total_messages += msg_count
                results.append({
                    "session_id": s.get("session_id", "unknown"),
                    "message_count": msg_count,
                    "created_at": s.get("created_at", ""),
                    "last_message": s.get("last_message", "")[:100] if s.get("last_message") else "",
                })

        # Fallback: read from facts.json
        elif Path("memory/facts.json").exists():
            with open("memory/facts.json", "r") as f:
                facts = json.load(f)
            if isinstance(facts, list):
                total_messages = len(facts)
                for item in facts[-limit:]:
                    fact_text = str(item.get("fact", ""))
                    total_chars += len(fact_text)
                    results.append({
                        "session_id": item.get("session_id", "unknown"),
                        "fact": fact_text[:100],
                        "created_at": item.get("created_at", ""),
                    })

        avg_chars = round(total_chars / max(total_messages, 1), 1)

        return {
            "total_sessions": len(results),
            "total_messages": total_messages,
            "avg_message_length": avg_chars,
            "sessions": results,
        }
    except Exception as e:
        return {"error": str(e), "total_sessions": 0, "total_messages": 0}


@app.post("/api/intent")
async def classify_intent(req: ChatRequest):
    """Classify the intent of a user message.

    Returns intent type, confidence score, and suggested follow-up actions.
    Useful for routing user messages to appropriate handlers.
    """
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    # Lightweight keyword-based intent classification (no LLM call needed)
    text = req.message.lower()
    intents = {
        "greeting": ["你好", "hello", "hi", "hey", "在吗", "早上好", "晚上好"],
        "question": ["怎么", "什么", "为什么", "如何", "请问", "哪里", "多少", "who", "what", "how", "why"],
        "command": ["帮我", "请", "执行", "run", "execute", "do this", "please"],
        "feedback": ["谢谢", "感谢", "好", "不错", "满意", "thanks", "good", "great"],
        "complaint": ["投诉", "不好", "差", "垃圾", "bug", "问题", "error", "broken"],
    }

    scores = {}
    for intent, keywords in intents.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[intent] = score

    if scores:
        top_intent = max(scores, key=scores.get)
        confidence = min(0.95, 0.5 + scores[top_intent] * 0.15)
    else:
        top_intent = "general"
        confidence = 0.3

    # Generate suggested follow-ups based on intent
    suggestions = {
        "greeting": ["有什么可以帮你的？", "想聊点什么？", "需要查询什么信息？"],
        "question": ["能详细说明一下吗？", "还有其他问题吗？", "需要我查资料吗？"],
        "command": ["确认执行？", "需要更多参数？", "要查看结果吗？"],
        "feedback": ["还有什么需要帮助的？", "有其他问题吗？", "感谢反馈！"],
        "complaint": ["能详细描述问题吗？", "什么时间出现的？", "有错误截图吗？"],
        "general": ["请详细描述您的需求", "需要什么帮助？"],
    }

    return {
        "intent": top_intent,
        "confidence": round(confidence, 2),
        "all_scores": {k: round(v / max(list(scores.values()) or [1]), 2) for k, v in scores.items()},
        "suggestions": suggestions.get(top_intent, []),
    }


class SecurityRequest(BaseModel):
    text: str


class SummaryRequest(BaseModel):
    messages: List[Dict[str, str]]


class SentimentRequest(BaseModel):
    text: str
    session_id: str = "anonymous"


@app.post("/api/security/scan")
async def security_scan(req: SecurityRequest):
    """Scan user input for PII and prompt injection.

    Returns redacted text, PII findings, and injection threats.
    """
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    # PII detection
    pii_result = pii_redact(req.text)
    has_pii = len(pii_result.found_pii) > 0

    # Prompt injection detection
    scan_result = prompt_scan(req.text)

    return {
        "is_safe": not has_pii and scan_result.is_safe,
        "pii": {
            "detected": has_pii,
            "count": len(pii_result.found_pii),
            "types": list(set(f.pii_type for f in pii_result.found_pii)),
            "redacted_text": pii_result.redacted_text,
        },
        "injection": {
            "detected": not scan_result.is_safe,
            "threats": scan_result.threats,
            "cleaned_text": scan_result.cleaned,
        },
    }


@app.post("/api/sentiment")
async def sentiment(req: SentimentRequest):
    """Analyze user message sentiment.

    Returns emotion type, intensity (1-5), trend, and escalation flag.
    """
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    result = sentiment_analyze(req.text, session_id=req.session_id)

    # Add tone adjustment suggestion
    from my_agent.sentiment import get_tone_adjustment
    tone = get_tone_adjustment(
        result.get("emotion", "neutral"),
        result.get("intensity", 1),
        result.get("trend", "unknown"),
    )
    result["tone_adjustment"] = tone

    return result


@app.post("/api/summary")
async def summary(req: SummaryRequest):
    """Generate a service ticket summary from conversation history.

    Returns structured ticket with subject, category, description,
    resolution, priority, and status.
    """
    if not req.messages:
        raise HTTPException(400, "Empty messages")

    # Try to get LLM client from agent
    llm_client = None
    if hasattr(agent, 'llm'):
        llm_client = agent.llm

    result = conversation_summary(req.messages, llm_client=llm_client)
    return result


@app.get("/api/card")
async def agent_card():
    """Get Agent Card (A2A protocol compatible).

    Returns structured agent metadata including name, version,
    capabilities, and supported tools.
    """
    try:
        # Prefer agent.card() method if available (newer interface)
        if hasattr(agent, 'card') and callable(agent.card):
            card = agent.card()
            return {
                "name": card.name,
                "description": card.description,
                "version": card.version,
                "capabilities": card.capabilities,
                "tools": card.tools,
            }
    except Exception:
        pass
    # Fallback: build card from agent attributes directly
    return {
        "name": agent.name,
        "description": getattr(agent, 'description', 'SimpleAgent'),
        "version": agent.version,
        "capabilities": [
            "chat",
            "streaming",
            "tool_use",
            "memory",
            "enhanced_pipeline",
        ],
        "tools": [name for name in agent.tool_registry.tools.keys()],
        "supported_protocols": ["http", "sse", "a2a"],
    }


# ── Auth login endpoint (for frontend login gate) ──
@app.post("/api/auth/login")
async def auth_login(request: Request):
    """Login or register a user. Returns JWT token (used by web frontend)."""
    from my_agent.auth import UserStore, issue_token, AuthError
    try:
        body = await request.json()
        username = (body.get("username") or "").strip()
        password = body.get("password") or None
        if not username:
            raise HTTPException(400, detail="username is required")
        store = UserStore()
        record = store.login_or_register(username, password)
        token = issue_token(record.user_id, record.username)
        return {
            "user_id": record.user_id,
            "username": record.username,
            "token": token,
            "access_token": token,
            "has_password": record.has_password,
        }
    except AuthError as e:
        raise HTTPException(401, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# Static files mounted AFTER API routes
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    print("SimpleAgent v2.0 — http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
