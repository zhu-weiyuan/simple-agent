#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP 客户端与 ``mcpServers`` 配置加载。

配置格式兼容主流 MCP 客户端：

.. code-block:: json

  {"mcpServers": {
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/workspace"]},
    "web-reader": {"type": "http", "url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${MCP_TOKEN}"}}
  }}

stdio 使用逐行 JSON-RPC；HTTP 使用 MCP Streamable HTTP 的 JSON 响应，并兼容服务端
以 SSE 返回单个 JSON-RPC 响应的情况。配置中的 ``${NAME}`` 从当前环境变量展开，
避免把令牌直接提交到仓库。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_TIMEOUT_SECONDS = 120.0
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SERVER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class MCPError(Exception):
    """MCP transport or protocol error."""


class MCPConfigError(MCPError):
    """The MCP configuration file is invalid or unsafe to load."""


def _expand_environment(value: str, environ: Mapping[str, str]) -> str:
    """Expand ``${NAME}`` without silently replacing a missing secret."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in environ:
            raise MCPConfigError(f"MCP 配置引用的环境变量未设置: {key}")
        return str(environ[key])

    return _ENV_REFERENCE.sub(replace, value)


def _string_mapping(value: Any, field_name: str, environ: Mapping[str, str]) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MCPConfigError(f"{field_name} 必须是对象")
    result: Dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise MCPConfigError(f"{field_name} 的键必须是非空字符串")
        if not isinstance(raw, str):
            raise MCPConfigError(f"{field_name}.{key} 必须是字符串")
        result[key] = _expand_environment(raw, environ)
    return result


def _timeout_from(raw: Mapping[str, Any]) -> float:
    value = raw.get("timeoutSeconds", raw.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise MCPConfigError("timeoutSeconds 必须是数字") from exc
    if not 1 <= timeout <= _MAX_TIMEOUT_SECONDS:
        raise MCPConfigError(f"timeoutSeconds 必须介于 1 和 {_MAX_TIMEOUT_SECONDS:g} 秒之间")
    return timeout


@dataclass(frozen=True)
class MCPServerConfig:
    """A validated external MCP server declaration."""

    name: str
    transport: str
    command: Optional[str] = None
    args: tuple[str, ...] = ()
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_mapping(
        cls,
        name: str,
        raw: Any,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "MCPServerConfig":
        if not isinstance(name, str) or not _SERVER_NAME.fullmatch(name):
            raise MCPConfigError("MCP 服务器名称只能包含字母、数字、下划线和连字符，且最长 64 位")
        if not isinstance(raw, Mapping):
            raise MCPConfigError(f"mcpServers.{name} 必须是对象")
        env = os.environ if environ is None else environ
        transport = str(raw.get("type", "stdio")).strip().lower() or "stdio"
        timeout_seconds = _timeout_from(raw)

        if transport == "stdio":
            command = raw.get("command")
            if not isinstance(command, str) or not command.strip():
                raise MCPConfigError(f"mcpServers.{name}.command 必须是非空字符串")
            raw_args = raw.get("args", [])
            if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
                raise MCPConfigError(f"mcpServers.{name}.args 必须是字符串数组")
            return cls(
                name=name,
                transport="stdio",
                command=_expand_environment(command.strip(), env),
                args=tuple(_expand_environment(item, env) for item in raw_args),
                env=_string_mapping(raw.get("env"), f"mcpServers.{name}.env", env),
                timeout_seconds=timeout_seconds,
            )

        if transport == "http":
            url = raw.get("url")
            if not isinstance(url, str) or not url.strip():
                raise MCPConfigError(f"mcpServers.{name}.url 必须是非空 HTTP(S) 地址")
            expanded_url = _expand_environment(url.strip(), env)
            parsed = urlparse(expanded_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise MCPConfigError(f"mcpServers.{name}.url 必须使用 http 或 https")
            return cls(
                name=name,
                transport="http",
                url=expanded_url,
                headers=_string_mapping(raw.get("headers"), f"mcpServers.{name}.headers", env),
                timeout_seconds=timeout_seconds,
            )

        raise MCPConfigError(f"mcpServers.{name}.type 仅支持 stdio 或 http")


def load_mcp_server_configs(
    path: Optional[Union[str, Path]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> List[MCPServerConfig]:
    """Load one JSON file in the common ``mcpServers`` format.

    The path is deliberately explicit: pass ``MCP_CONFIG_PATH`` (for example
    ``~/.astrcode/mcp.json``) rather than automatically launching processes
    configured for another desktop application.
    """

    env = os.environ if environ is None else environ
    raw_path = str(path or env.get("MCP_CONFIG_PATH", "")).strip()
    if not raw_path:
        return []
    config_path = Path(_expand_environment(raw_path, env)).expanduser()
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MCPConfigError(f"MCP 配置文件不存在: {config_path}") from exc
    except OSError as exc:
        raise MCPConfigError(f"无法读取 MCP 配置文件: {config_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MCPConfigError(f"MCP 配置 JSON 无效: 第 {exc.lineno} 行第 {exc.colno} 列") from exc

    if not isinstance(document, Mapping):
        raise MCPConfigError("MCP 配置根节点必须是对象")
    servers = document.get("mcpServers")
    if not isinstance(servers, Mapping):
        raise MCPConfigError("MCP 配置必须包含对象字段 mcpServers")
    return [MCPServerConfig.from_mapping(name, entry, env) for name, entry in servers.items()]


def legacy_stdio_config(command: str) -> MCPServerConfig:
    """Adapt the previous ``MCP_WEATHER_COMMAND`` setting without breaking it."""
    import shlex

    parts = shlex.split(command, posix=False)
    if not parts:
        raise MCPConfigError("MCP_WEATHER_COMMAND 不能为空")
    return MCPServerConfig(name="legacy", transport="stdio", command=parts[0], args=tuple(parts[1:]))


class _MCPClientBase:
    """Shared lifecycle, initialization and tool-result handling."""

    protocol_version = "2024-11-05"

    def __init__(self, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds
        self.server_info: Dict[str, Any] = {}
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._running = False
        self._state_lock = threading.RLock()

    def _initialize(self) -> None:
        response = self._send_request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "simple-agent", "version": "2.1.0"},
            },
        )
        self.server_info = response.get("serverInfo", {})
        capabilities = response.get("capabilities", {})
        if "tools" not in capabilities:
            raise MCPError("服务器不支持工具功能")

    def list_tools(self) -> List[Dict[str, Any]]:
        with self._state_lock:
            if self._tools_cache is not None:
                return list(self._tools_cache)
        response = self._send_request("tools/list", {})
        tools = response.get("tools", [])
        if not isinstance(tools, list):
            raise MCPError("tools/list 响应格式无效")
        with self._state_lock:
            self._tools_cache = [tool for tool in tools if isinstance(tool, dict)]
            return list(self._tools_cache)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        response = self._send_request("tools/call", {"name": name, "arguments": arguments})
        if response.get("isError"):
            raise MCPError("MCP 工具返回错误")
        content = response.get("content", [])
        if not isinstance(content, list):
            raise MCPError("tools/call 响应格式无效")
        parts: List[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "error":
                raise MCPError(f"工具错误:{item.get('text', '未知错误')}")
        return "".join(parts) or "（空结果）"

    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class MCPClient(_MCPClientBase):
    """JSON-RPC-over-stdio MCP client (backward-compatible public name)."""

    _STOPPED = object()

    def __init__(
        self,
        command: Sequence[str],
        env: Optional[Mapping[str, str]] = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.command = [str(part) for part in command]
        self.env = dict(env or {})
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending_requests: Dict[int, queue.Queue] = {}
        self._write_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        with self._state_lock:
            if self.process is not None:
                raise MCPError("MCP 客户端已启动")
            if not self.command or not self.command[0]:
                raise MCPError("MCP stdio command 不能为空")
            child_env = os.environ.copy()
            child_env.update(self.env)
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=child_env,
            )
            self._running = True
            self._tools_cache = None
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()
        try:
            self._initialize()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the subprocess and wake requests immediately."""
        with self._state_lock:
            self._running = False
            process = self.process
            self.process = None
            pending = list(self._pending_requests.values())
            self._pending_requests.clear()
            self._tools_cache = None
            reader = self._reader_thread

        for response_queue in pending:
            response_queue.put(self._STOPPED)
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug("停止 MCP 进程时出错: %s", exc)
        if reader and reader is not threading.current_thread():
            reader.join(timeout=1)

    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._state_lock:
            process = self.process
            if not self._running or not process or not process.stdin:
                raise MCPError("MCP 客户端未启动")
            self._request_id += 1
            request_id = self._request_id
            response_queue: queue.Queue = queue.Queue(maxsize=1)
            self._pending_requests[request_id] = response_queue

        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        try:
            with self._write_lock:
                process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                process.stdin.flush()
        except Exception as exc:
            with self._state_lock:
                self._pending_requests.pop(request_id, None)
            raise MCPError(f"发送请求失败:{exc}") from exc

        try:
            response = response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            with self._state_lock:
                self._pending_requests.pop(request_id, None)
            raise MCPError(f"请求超时:{method}") from exc
        finally:
            # The successful path also removes the waiter below; this makes
            # timeout/error cleanup explicit without holding the queue alive.
            pass

        with self._state_lock:
            self._pending_requests.pop(request_id, None)
        if response is self._STOPPED:
            raise MCPError("MCP 客户端已停止")
        if not isinstance(response, dict):
            raise MCPError("MCP 响应格式无效")
        if "error" in response:
            error = response["error"]
            message = error.get("message", "未知错误") if isinstance(error, Mapping) else str(error)
            raise MCPError(f"RPC 错误 [{method}]: {message}")
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise MCPError("MCP result 格式无效")
        return result

    def _read_loop(self) -> None:
        while True:
            with self._state_lock:
                running = self._running
                process = self.process
                stdout = process.stdout if process else None
            if not running or not stdout:
                break
            try:
                line = stdout.readline()
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
                request_id = message.get("id") if isinstance(message, Mapping) else None
                if request_id is not None:
                    with self._state_lock:
                        response_queue = self._pending_requests.get(request_id)
                    if response_queue is not None:
                        response_queue.put(message)
            except Exception as exc:
                with self._state_lock:
                    should_log = self._running
                if should_log:
                    logger.error("读取 MCP 响应时出错:%s", exc)
                break

    def __enter__(self) -> "MCPClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class HTTPMCPClient(_MCPClientBase):
    """Synchronous MCP Streamable HTTP client for request/response servers."""

    def __init__(
        self,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.url = url
        self.headers = dict(headers or {})
        self._request_id = 0
        self._session_id: Optional[str] = None

    def start(self) -> None:
        with self._state_lock:
            if self._running:
                raise MCPError("MCP 客户端已启动")
            self._running = True
            self._tools_cache = None
        try:
            self._initialize()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        with self._state_lock:
            self._running = False
            self._tools_cache = None
            self._session_id = None

    @staticmethod
    def _parse_response_body(raw: bytes, content_type: str, request_id: int) -> Dict[str, Any]:
        text = raw.decode("utf-8", errors="replace").strip()
        if "text/event-stream" in content_type.lower():
            payloads = []
            for line in text.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        payloads.append(data)
            for data in reversed(payloads):
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and parsed.get("id") == request_id:
                    return parsed
            raise MCPError("HTTP MCP SSE 响应中没有匹配的 JSON-RPC 结果")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MCPError("HTTP MCP 响应不是有效 JSON") from exc
        if not isinstance(parsed, dict):
            raise MCPError("HTTP MCP 响应格式无效")
        return parsed

    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._state_lock:
            if not self._running:
                raise MCPError("MCP 客户端未启动")
            self._request_id += 1
            request_id = self._request_id
            session_id = self._session_id
        body = json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol_version,
            **self.headers,
        }
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                returned_session = response.headers.get("Mcp-Session-Id")
                if returned_session:
                    with self._state_lock:
                        self._session_id = returned_session
                message = self._parse_response_body(
                    raw,
                    response.headers.get("Content-Type", ""),
                    request_id,
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise MCPError(f"HTTP MCP 请求失败 [{exc.code}]: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise MCPError(f"HTTP MCP 连接失败:{exc}") from exc

        if "error" in message:
            error = message["error"]
            detail = error.get("message", "未知错误") if isinstance(error, Mapping) else str(error)
            raise MCPError(f"RPC 错误 [{method}]: {detail}")
        if message.get("id") != request_id:
            raise MCPError("HTTP MCP 响应 id 不匹配")
        result = message.get("result", {})
        if not isinstance(result, dict):
            raise MCPError("MCP result 格式无效")
        return result

    def __enter__(self) -> "HTTPMCPClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def create_mcp_client(config: MCPServerConfig) -> Union[MCPClient, HTTPMCPClient]:
    """Create a concrete client without starting it."""
    if config.transport == "stdio":
        assert config.command is not None
        return MCPClient(
            [config.command, *config.args],
            env=config.env,
            timeout_seconds=config.timeout_seconds,
        )
    if config.transport == "http":
        assert config.url is not None
        return HTTPMCPClient(config.url, headers=config.headers, timeout_seconds=config.timeout_seconds)
    raise MCPConfigError(f"未知 MCP transport: {config.transport}")
