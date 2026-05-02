# -*- coding: utf-8 -*-
"""直接启动 SimpleAgent Web 服务，不经过 OpenClaw embedded agent"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from app import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
