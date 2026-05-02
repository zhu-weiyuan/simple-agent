# -*- coding: utf-8 -*-
"""
my_agent.a2a — Agent-to-Agent (A2A) 协议实现

参考 a2aproject/A2A (⭐23545) 协议设计：
- AgentCard: agent 元数据和能力声明
- A2AClient: 远程 agent 客户端
- A2AServer: 本地 agent 服务器
- 任务状态管理
- 消息格式标准化

A2A 协议允许不同 agent 系统之间互操作，实现：
1. Agent 发现（通过 Agent Card）
2. 任务委托（一个 agent 委托任务给另一个）
3. 状态跟踪（任务进度、取消、取消）
4. 结果获取
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Task State ──────────────────────────────────────────────

class TaskState(str, Enum):
    """任务状态（A2A 协议）"""
    SUBMITTED = "submitted"          # 已提交
    WORKING = "working"              # 执行中
    COMPLETED = "completed"          # 已完成
    CANCELLED = "cancelled"          # 已取消
    FAILED = "failed"                # 失败


# ── A2A Message ─────────────────────────────────────────────

class MessageType(str, Enum):
    """消息类型"""
    PROMPT = "prompt"                # 用户提示
    RESULT = "result"                # 执行结果
    CANCEL = "cancel"                # 取消请求


@dataclass
class A2AMessage:
    """A2A 消息（协议标准格式）"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: Optional[str] = None
    type: MessageType = MessageType.PROMPT
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messageId": self.message_id,
            "taskId": self.task_id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        return cls(
            message_id=data.get("messageId", str(uuid.uuid4())),
            task_id=data.get("taskId"),
            type=MessageType(data.get("type", "prompt")),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
        )


# ── Task Status ─────────────────────────────────────────────

@dataclass
class TaskStatus:
    """任务状态（A2A 协议）"""
    task_id: str
    state: TaskState = TaskState.SUBMITTED
    message: Optional[A2AMessage] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskId": self.task_id,
            "state": self.state.value,
            "message": self.message.to_dict() if self.message else None,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "duration": self.duration,
            "metadata": self.metadata,
        }


# ── Agent Card ──────────────────────────────────────────────

@dataclass
class AgentCard:
    """Agent 元数据卡片（A2A 协议兼容）

    用于 agent 发现和互操作，声明 agent 的能力和接口。
    """
    name: str
    description: str = ""
    version: str = "1.0.0"
    url: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    default_input_languages: List[str] = field(default_factory=lambda: ["zh", "en"])
    preferred_transport: str = "http"  # http | sse | websocket

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "defaultInputLanguages": self.default_input_languages,
            "preferredTransport": self.preferred_transport,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            url=data.get("url", ""),
            capabilities=data.get("capabilities", {}),
            tools=data.get("tools", []),
            default_input_languages=data.get("defaultInputLanguages", ["zh", "en"]),
            preferred_transport=data.get("preferredTransport", "http"),
        )


# ── A2A Client ──────────────────────────────────────────────

class A2AClient:
    """A2A 客户端：调用远程 agent

    参考 a2aproject/A2A 客户端设计，支持：
    - 发送提示消息
    - 获取任务状态
    - 取消任务
    """

    def __init__(self, endpoint: str, timeout: int = 300):
        """
        Args:
            endpoint: 远程 agent 的 URL
            timeout: 超时时间（秒）
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._card: Optional[AgentCard] = None

    def get_card(self) -> Optional[AgentCard]:
        """获取远程 agent 的 Agent Card"""
        if self._card:
            return self._card

        try:
            import urllib.request
            url = f"{self.endpoint}/.well-known/agent.json"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self._card = AgentCard.from_dict(data)
                return self._card
        except Exception as e:
            logger.warning(f"Failed to fetch agent card from {self.endpoint}: {e}")
            return None

    def send_prompt(self, content: str, task_id: Optional[str] = None) -> TaskStatus:
        """发送提示消息到远程 agent

        Args:
            content: 提示内容
            task_id: 可选的任务 ID

        Returns:
            TaskStatus 任务状态
        """
        task_id = task_id or str(uuid.uuid4())
        message = A2AMessage(task_id=task_id, content=content)

        try:
            import urllib.request
            url = f"{self.endpoint}/messages"
            data = message.to_json().encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return TaskStatus(
                    task_id=task_id,
                    state=TaskState(result.get("state", "completed")),
                    message=A2AMessage.from_dict(result.get("message", {})),
                )
        except Exception as e:
            logger.error(f"A2A request failed: {e}")
            return TaskStatus(task_id=task_id, state=TaskState.FAILED)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """查询任务状态"""
        try:
            import urllib.request
            url = f"{self.endpoint}/tasks/{task_id}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return TaskStatus(
                    task_id=task_id,
                    state=TaskState(data.get("state", "unknown")),
                )
        except Exception as e:
            logger.warning(f"Failed to get task status: {e}")
            return None

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            import urllib.request
            url = f"{self.endpoint}/tasks/{task_id}/cancel"
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"Failed to cancel task: {e}")
            return False


# ── A2A Server ──────────────────────────────────────────────

class A2AServer:
    """A2A 服务器：暴露本地 agent 为 A2A 端点

    提供标准 A2A 接口：
    - GET /.well-known/agent.json → Agent Card
    - POST /messages → 发送消息
    - GET /tasks/{id} → 查询任务状态
    - POST /tasks/{id}/cancel → 取消任务
    """

    def __init__(self, agent: Any, card: AgentCard, host: str = "0.0.0.0", port: int = 8090):
        """
        Args:
            agent: 本地 agent 实例
            card: Agent Card 元数据
            host: 监听地址
            port: 监听端口
        """
        self.agent = agent
        self.card = card
        self.host = host
        self.port = port
        self._tasks: Dict[str, TaskStatus] = {}
        self._running = False

    def get_card(self) -> str:
        """返回 Agent Card JSON"""
        return self.card.to_json()

    def handle_message(self, message: A2AMessage) -> TaskStatus:
        """处理 incoming 消息"""
        task_id = message.task_id or str(uuid.uuid4())

        # 创建任务
        task = TaskStatus(task_id=task_id, state=TaskState.WORKING)
        self._tasks[task_id] = task

        try:
            # 调用本地 agent
            result = self.agent.run(message.content)

            # 更新任务状态
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()
            task.message = A2AMessage(
                task_id=task_id,
                type=MessageType.RESULT,
                content=str(result),
            )
        except Exception as e:
            task.state = TaskState.FAILED
            task.completed_at = time.time()
            task.message = A2AMessage(
                task_id=task_id,
                type=MessageType.RESULT,
                content=f"错误：{type(e).__name__}: {e}",
            )

        return task

    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self._tasks.get(task_id)
        if task and task.state in (TaskState.SUBMITTED, TaskState.WORKING):
            task.state = TaskState.CANCELLED
            task.completed_at = time.time()
            return True
        return False

    def start(self):
        """启动 HTTP 服务器"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading

            server = self._create_http_server()
            self._running = True
            logger.info(f"A2A Server started on {self.host}:{self.port}")

            try:
                server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        except ImportError:
            logger.error("A2A Server requires Python http.server")

    def stop(self):
        """停止服务器"""
        self._running = False
        logger.info("A2A Server stopped")

    def _create_http_server(self):
        """创建 HTTP 服务器"""
        server_instance = self

        class A2AHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/.well-known/agent.json":
                    self._respond(200, server_instance.get_card())
                elif self.path.startswith("/tasks/"):
                    task_id = self.path.split("/")[-1]
                    task = server_instance.get_task(task_id)
                    if task:
                        self._respond(200, json.dumps(task.to_dict()))
                    else:
                        self._respond(404, json.dumps({"error": "Task not found"}))
                else:
                    self._respond(404, json.dumps({"error": "Not found"}))

            def do_POST(self):
                if self.path == "/messages":
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    try:
                        data = json.loads(body)
                        message = A2AMessage.from_dict(data)
                        task = server_instance.handle_message(message)
                        self._respond(200, json.dumps(task.to_dict()))
                    except Exception as e:
                        self._respond(400, json.dumps({"error": str(e)}))
                elif self.path.startswith("/tasks/") and self.path.endswith("/cancel"):
                    task_id = self.path.split("/")[-2]
                    success = server_instance.cancel_task(task_id)
                    self._respond(200 if success else 404, json.dumps({"cancelled": success}))
                else:
                    self._respond(404, json.dumps({"error": "Not found"}))

            def _respond(self, status, body):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

            def log_message(self, format, *args):
                logger.debug(f"A2A: {format % args}")

        return HTTPServer((self.host, self.port), A2AHandler)
