# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from my_agent.mcp_client import (
    HTTPMCPClient,
    MCPConfigError,
    MCPServerConfig,
    create_mcp_client,
    load_mcp_server_configs,
)


def _write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_mcp_servers_supports_stdio_http_and_environment_expansion(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "mcp.json",
        {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "${WORKSPACE}"],
                    "env": {"TOKEN": "${MCP_TOKEN}"},
                },
                "web-reader": {
                    "type": "http",
                    "url": "https://mcp.example.com/mcp",
                    "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
                    "timeoutSeconds": 12,
                },
            }
        },
    )

    configs = load_mcp_server_configs(path, {"WORKSPACE": "C:/allowed", "MCP_TOKEN": "secret"})

    assert [(item.name, item.transport) for item in configs] == [("filesystem", "stdio"), ("web-reader", "http")]
    assert configs[0].command == "npx"
    assert configs[0].args[-1] == "C:/allowed"
    assert configs[0].env == {"TOKEN": "secret"}
    assert configs[1].headers["Authorization"] == "Bearer secret"
    assert configs[1].timeout_seconds == 12
    assert isinstance(create_mcp_client(configs[0]).command, list)
    assert isinstance(create_mcp_client(configs[1]), HTTPMCPClient)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"mcpServers": {"bad": {"type": "ftp", "url": "ftp://example.com"}}}, "仅支持 stdio 或 http"),
        ({"mcpServers": {"bad": {"command": "node", "args": "not-a-list"}}}, "args 必须是字符串数组"),
        ({"mcpServers": {"bad": {"type": "http", "url": "file:///tmp/mcp"}}}, "必须使用 http 或 https"),
        ({"mcpServers": {"bad": {"command": "${MISSING}"}}}, "环境变量未设置"),
    ],
)
def test_load_mcp_servers_rejects_invalid_or_unsafe_entries(tmp_path: Path, payload: dict[str, Any], expected: str) -> None:
    path = _write_config(tmp_path / "mcp.json", payload)
    with pytest.raises(MCPConfigError, match=expected):
        load_mcp_server_configs(path, {})


def test_http_client_sends_json_rpc_preserves_headers_and_reuses_session(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class _Response:
        def __init__(self, payload: dict[str, Any], session: str = "session-1") -> None:
            self._body = json.dumps(payload).encode("utf-8")
            self.headers = {"Content-Type": "application/json", "Mcp-Session-Id": session}

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_urlopen(request, timeout: float):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append({"payload": payload, "headers": dict(request.header_items()), "timeout": timeout})
        method = payload["method"]
        if method == "initialize":
            result = {"serverInfo": {"name": "fake"}, "capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {"tools": [{"name": "read", "inputSchema": {"type": "object"}}]}
        else:
            result = {"content": [{"type": "text", "text": "ok"}]}
        return _Response({"jsonrpc": "2.0", "id": payload["id"], "result": result})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = HTTPMCPClient("https://mcp.example.com/mcp", headers={"Authorization": "Bearer test"}, timeout_seconds=7)
    client.start()
    assert client.list_tools()[0]["name"] == "read"
    assert client.call_tool("read", {"path": "README.md"}) == "ok"
    client.stop()

    assert [call["payload"]["method"] for call in calls] == ["initialize", "tools/list", "tools/call"]
    assert all(call["timeout"] == 7 for call in calls)
    assert calls[0]["headers"]["Authorization"] == "Bearer test"
    assert calls[1]["headers"]["Mcp-session-id"] == "session-1"


def test_http_client_accepts_sse_json_rpc_response() -> None:
    body = b'event: message\ndata: {"jsonrpc":"2.0","id":3,"result":{"ok":true}}\n\n'
    parsed = HTTPMCPClient._parse_response_body(body, "text/event-stream", 3)
    assert parsed["result"] == {"ok": True}


def test_namespaced_mcp_tool_names_are_stable_and_safe() -> None:
    from my_agent.agent import SimpleAgent

    assert SimpleAgent._mcp_registered_tool_name("web-reader", "read/page") == "mcp_web_reader_read_page"
    assert SimpleAgent._mcp_registered_tool_name("###", "***") == "mcp_server_tool"
