#!/usr/bin/env python3
"""MCP (Model Context Protocol) 客户端，使用 stdio + JSON-RPC 2.0。"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPError(Exception):
    pass


class MCPClient:
    def __init__(self, command: List[str]):
        self.command = command
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending_requests: Dict[int, queue.Queue] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self.server_info: Dict[str, Any] = {}
        self.protocol_version = "2024-11-05"
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    def start(self) -> None:
        if self.process is not None:
            raise MCPError("MCP 客户端已启动")

        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        self._initialize()

    def stop(self) -> None:
        self._running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def _initialize(self) -> None:
        response = self._send_request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "my-agent-python", "version": "0.1.0"},
            },
        )
        self.server_info = response.get("serverInfo", {})
        capabilities = response.get("capabilities", {})
        if "tools" not in capabilities:
            raise MCPError("服务器不支持工具功能")

    def list_tools(self) -> List[Dict[str, Any]]:
        if self._tools_cache is not None:
            return self._tools_cache
        response = self._send_request("tools/list", {})
        tools = response.get("tools", [])
        self._tools_cache = tools
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        response = self._send_request("tools/call", {"name": name, "arguments": arguments})
        content = response.get("content", [])
        result = ""
        for item in content:
            if item.get("type") == "text":
                result += item.get("text", "")
            elif item.get("type") == "error":
                raise MCPError(f"工具错误：{item.get('text', '未知错误')}")
        return result or "（空结果）"

    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.process or not self.process.stdin:
            raise MCPError("MCP 客户端未启动")

        self._request_id += 1
        request_id = self._request_id
        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}

        response_queue: queue.Queue = queue.Queue()
        self._pending_requests[request_id] = response_queue

        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise MCPError(f"发送请求失败：{e}")

        try:
            response = response_queue.get(timeout=30)
        except queue.Empty:
            self._pending_requests.pop(request_id, None)
            raise MCPError(f"请求超时：{method}")

        self._pending_requests.pop(request_id, None)

        if "error" in response:
            error = response["error"]
            raise MCPError(f"RPC 错误 [{method}]: {error.get('message', '未知错误')}")

        return response.get("result", {})

    def _read_loop(self) -> None:
        while self._running and self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("无法解析 MCP 消息")
                    continue

                request_id = message.get("id")
                if request_id is not None and request_id in self._pending_requests:
                    self._pending_requests[request_id].put(message)
            except Exception as e:
                logger.error(f"读取 MCP 响应时出错：{e}")
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
