#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleAgent v2.0 — Web Server (FastAPI)

启动: python app.py
访问: http://localhost:8000
"""
import os, sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "your_key_here")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8080")
os.environ.setdefault("OPENAI_MODEL", "Qwen3.6-27B-IQ4_NL.gguf")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import json

web_dir = Path(__file__).parent / "web"
web_dir.mkdir(exist_ok=True)

app = FastAPI(title="SimpleAgent v2.0")

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
    return {"ok": True, "agent": agent.name, "version": agent.version}


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


@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory store statistics."""
    try:
        stats = agent.memory_store.get_stats() if hasattr(agent.memory_store, 'get_stats') else {}
        return {"memory": stats}
    except Exception as e:
        return {"error": str(e)}


# Static files mounted AFTER API routes
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    print("SimpleAgent v2.0 — http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
