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
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

web_dir = Path(__file__).parent / "web"
web_dir.mkdir(exist_ok=True)

app = FastAPI(title="SimpleAgent v2.0")

from my_agent import SimpleAgent
agent = SimpleAgent()


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    try:
        reply = agent.run(req.message)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
async def health():
    return {"ok": True, "agent": agent.name, "version": agent.version}


# Static files mounted AFTER API routes
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    print("SimpleAgent v2.0 — http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
