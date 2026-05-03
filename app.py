#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleAgent v2.0 — Web Server (FastAPI)

启动: python app.py
访问: http://localhost:8000
"""
import os, sys, time, platform, psutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

os.environ.setdefault("OPENAI_API_KEY", "your_key_here")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8080")
os.environ.setdefault("OPENAI_MODEL", "Qwen3.6-27B-IQ4_NL.gguf")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import json

web_dir = Path(__file__).parent / "web"
web_dir.mkdir(exist_ok=True)

app = FastAPI(title="SimpleAgent v2.0")

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

    _request_counter["total"] += 1
    client_ip = request.client.host if request.client else "unknown"

    if not _rate_limiter.is_allowed(client_ip):
        return HTTPException(
            status_code=429,
            detail={"error": "Rate limit exceeded", "retry_after": _rate_limiter.window}
        )

    # Request timing — measure latency per request
    start_time = time.time()
    try:
        response = await call_next(request)
        elapsed = time.time() - start_time
        response.headers["X-Response-Time"] = f"{elapsed*1000:.1f}ms"
        return response
    except Exception as e:
        _request_counter["errors"] += 1
        raise

from my_agent import SimpleAgent
agent = SimpleAgent()


class ChatRequest(BaseModel):
    message: str
    stream: bool = False


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    try:
        if req.stream:
            return StreamingResponse(
                agent.run_stream(req.message),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        reply = agent.run(req.message)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(500, str(e))


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
        
        # LLM connectivity check
        llm_reachable = False
        try:
            import urllib.request as ur
            resp = ur.urlopen(f"{agent.llm.base_url}/v1/models", timeout=3)
            llm_reachable = resp.status == 200
        except Exception:
            pass
        
        return {
            "ok": True,
            "agent": agent.name,
            "version": agent.version,
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
            },
            "requests": {
                "total": _request_counter["total"],
                "errors": _request_counter["errors"],
            },
        }
    except Exception as e:
        return {"ok": True, "agent": agent.name, "version": agent.version, "health_error": str(e)}


@app.get("/api/tools")
async def list_tools():
    """List all registered tools with descriptions."""
    tools = []
    for name, tool in agent.tool_registry.tools.items():
        tools.append({
            "name": name,
            "description": getattr(tool, "description", "") or getattr(tool, "__doc__", ""),
        })
    return {"tools": tools}


@app.get("/api/metrics")
async def metrics():
    """Prometheus-style text metrics endpoint."""
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
        f"# HELP agent_rate_limit_remaining Remaining rate limit requests",
        f"# TYPE agent_rate_limit_remaining gauge",
    ]
    return "\n".join(lines) + "\n"

@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory store statistics."""
    try:
        stats = agent.memory_store.get_stats() if hasattr(agent.memory_store, 'get_stats') else {}
        return {"memory": stats}
    except Exception as e:
        return {"error": str(e)}


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


# Static files mounted AFTER API routes
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    print("SimpleAgent v2.0 — http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
