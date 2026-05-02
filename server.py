#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleAgent v2.0 — Web Server

启动: python server.py
访问: http://localhost:7890
"""
import os, sys, json, threading
from pathlib import Path

# Windows UTF-8
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors="replace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors="replace")
from http.server import HTTPServer, SimpleHTTPRequestHandler

# 环境配置
os.environ.setdefault("OPENAI_API_KEY", "your_key_here")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8080")
os.environ.setdefault("OPENAI_MODEL", "Qwen3.6-27B-IQ4_NL.gguf")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from my_agent import SimpleAgent

# 全局 agent 实例（保持对话上下文）
agent = SimpleAgent()


class ChatHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).resolve().parent / "web"), **kwargs)

    def do_GET(self):
        if self.path == '/api/health':
            self._json(200, {"ok": True, "agent": agent.name, "version": agent.version})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                message = data.get("message", "")
            except Exception:
                self._text(400, "Invalid JSON", "text/plain")
                return

            if not message:
                self._json(400, {"error": "Empty message"})
                return

            try:
                reply = agent.run(message)
                self._json(200, {"reply": reply})
            except Exception as e:
                self._json(500, {"error": str(e)})
        else:
            self._text(404, "Not Found", "text/plain")

    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, code, text, ct="text/plain"):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{ct}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[server] {args[0]}")


def main():
    port = int(os.getenv("SIMPLE_AGENT_PORT", "7890"))
    server = HTTPServer(("0.0.0.0", port), ChatHandler)
    print(f"🚀 SimpleAgent v2.0 Web Server")
    print(f"   访问: http://localhost:{port}")
    print(f"   Agent: {agent.name} ({agent.version})")
    print(f"   LLM: {os.environ.get('OPENAI_BASE_URL')} / {os.environ.get('OPENAI_MODEL')}")
    print(f"   按 Ctrl+C 停止\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 再见！")
        agent.close()
        server.server_close()


if __name__ == "__main__":
    main()
