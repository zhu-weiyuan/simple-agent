# -*- coding: utf-8 -*-
"""A small, resilient A2A implementation.

The A2A transport is deliberately task-oriented:

* POST /messages only submits work and returns a task id quickly.
* GET /tasks/{id} reads the durable-in-process task state.
* POST /tasks/{id}/cancel requests cooperative cancellation.

The implementation keeps the public data classes compatible with the previous
module, while preventing a slow agent from blocking the HTTP server.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


class TaskStore:
    """Best-effort SQLite persistence for A2A task records."""

    def __init__(self, db_path: str = "") -> None:
        self._db_path = db_path or os.getenv("A2A_TASKS_DB", "runtime/a2a_tasks.db")
        self._db_available = self._init_db()
        self._recover_inflight()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=2.0)

    def _init_db(self) -> bool:
        try:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = self._connect()
            try:
                conn.execute("""CREATE TABLE IF NOT EXISTS a2a_tasks (
                    task_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )""")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_a2a_tasks_created ON a2a_tasks(created_at DESC)")
                conn.commit()
            finally:
                conn.close()
            return True
        except (sqlite3.Error, OSError) as exc:
            logger.warning("A2A task persistence unavailable: %s", exc)
            return False

    def _recover_inflight(self) -> None:
        if not self._db_available:
            return
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT task_id, payload FROM a2a_tasks WHERE state IN ('submitted','working','cancel_requested')"
                ).fetchall()
                for task_id, payload in rows:
                    try:
                        data = json.loads(payload)
                        data["state"] = TaskState.TIMED_OUT.value
                        data["error"] = "interrupted by service restart"
                        data["completedAt"] = time.time()
                        data.setdefault("metadata", {})["retryable"] = True
                        conn.execute(
                            "UPDATE a2a_tasks SET state=?, payload=?, updated_at=? WHERE task_id=?",
                            (TaskState.TIMED_OUT.value, json.dumps(data, ensure_ascii=False),
                             time.time(), task_id),
                        )
                    except (ValueError, TypeError):
                        continue
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not recover A2A tasks: %s", exc)

    def upsert(self, task_id: str, fingerprint: str, status: "TaskStatus") -> None:
        if not self._db_available:
            return
        try:
            payload = json.dumps(status.to_dict(), ensure_ascii=False)
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO a2a_tasks (task_id, fingerprint, payload, state, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(task_id) DO UPDATE SET fingerprint=excluded.fingerprint,
                       payload=excluded.payload, state=excluded.state, updated_at=excluded.updated_at""",
                    (task_id, fingerprint, payload, status.state.value,
                     status.started_at, time.time()),
                )
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not persist A2A task %s: %s", task_id, exc)

    def list_tasks(self, state: Optional[str] = None, limit: int = 50,
                   offset: int = 0) -> List[Dict[str, Any]]:
        if not self._db_available:
            return []
        try:
            sql = "SELECT payload FROM a2a_tasks WHERE 1=1"
            params: List[Any] = []
            if state:
                sql += " AND state = ?"
                params.append(state)
            sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([max(0, int(limit)), max(0, int(offset))])
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
            return [json.loads(r[0]) for r in rows]
        except (sqlite3.Error, ValueError, TypeError) as exc:
            logger.warning("Could not list A2A tasks: %s", exc)
            return []

    def stats(self) -> Dict[str, int]:
        if not self._db_available:
            return {}
        try:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT state, COUNT(*) FROM a2a_tasks GROUP BY state").fetchall()
            finally:
                conn.close()
            return {state: count for state, count in rows}
        except sqlite3.Error as exc:
            logger.warning("Could not read A2A task stats: %s", exc)
            return {}

    def delete_old_terminal(self, keep: int = 200) -> int:
        """Keep the newest ``keep`` terminal rows and remove older terminal rows."""
        if not self._db_available:
            return 0
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM a2a_tasks WHERE state NOT IN ('submitted','working','cancel_requested')"
                ).fetchone()
                count = int(row[0]) if row else 0
                if count <= keep:
                    return 0
                deleted = conn.execute(
                    """DELETE FROM a2a_tasks WHERE task_id IN (
                        SELECT task_id FROM a2a_tasks
                        WHERE state NOT IN ('submitted','working','cancel_requested')
                        ORDER BY created_at DESC LIMIT -1 OFFSET ?
                    )""", (keep,),
                ).rowcount
                conn.commit()
                return int(deleted)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not prune A2A tasks: %s", exc)
            return 0


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    UNKNOWN = "unknown"  # client-side state when transport outcome is unknown


class MessageType(str, Enum):
    PROMPT = "prompt"
    RESULT = "result"
    CANCEL = "cancel"


class A2AConflictError(RuntimeError):
    """The same task id was submitted with a different message."""


class _TaskCancelled(Exception):
    pass


class _TaskTimedOut(Exception):
    pass


@dataclass
class A2AMessage:
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
            metadata=data.get("metadata", {}) or {},
        )


@dataclass
class TaskStatus:
    task_id: str
    state: TaskState = TaskState.SUBMITTED
    message: Optional[A2AMessage] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        end = self.completed_at or time.time()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskId": self.task_id,
            "state": self.state.value,
            "message": self.message.to_dict() if self.message else None,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "duration": self.duration,
            "metadata": dict(self.metadata),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskStatus":
        message_data = data.get("message")
        message = A2AMessage.from_dict(message_data) if message_data else None
        raw_state = data.get("state", TaskState.UNKNOWN.value)
        try:
            state = TaskState(raw_state)
        except ValueError:
            state = TaskState.UNKNOWN
        return cls(
            task_id=data.get("taskId", ""),
            state=state,
            message=message,
            started_at=float(data.get("startedAt") or time.time()),
            completed_at=data.get("completedAt"),
            metadata=data.get("metadata", {}) or {},
            error=data.get("error"),
        )


@dataclass
class AgentCard:
    name: str
    description: str = ""
    version: str = "1.0.0"
    url: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    default_input_languages: List[str] = field(default_factory=lambda: ["zh", "en"])
    preferred_transport: str = "http"

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
            capabilities=data.get("capabilities", {}) or {},
            tools=data.get("tools", []) or [],
            default_input_languages=data.get("defaultInputLanguages", ["zh", "en"]),
            preferred_transport=data.get("preferredTransport", "http"),
        )


class A2AClient:
    """Task-oriented A2A client.

    ``send_prompt`` only submits a task. Use ``wait_for_task`` when the caller
    wants a blocking convenience method. A transport timeout is reported as
    UNKNOWN rather than FAILED because the remote task may already be running.
    """

    def __init__(self, endpoint: str, timeout: float = 5.0,
                 poll_interval: float = 1.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = max(0.1, float(timeout))
        self.poll_interval = max(0.05, float(poll_interval))
        self._card: Optional[AgentCard] = None

    @staticmethod
    def _status_from_dict(data: Dict[str, Any]) -> TaskStatus:
        message_data = data.get("message")
        message = A2AMessage.from_dict(message_data) if message_data else None
        raw_state = data.get("state", TaskState.UNKNOWN.value)
        try:
            state = TaskState(raw_state)
        except ValueError:
            state = TaskState.UNKNOWN
        return TaskStatus(
            task_id=data.get("taskId", ""),
            state=state,
            message=message,
            started_at=float(data.get("startedAt") or time.time()),
            completed_at=data.get("completedAt"),
            metadata=data.get("metadata", {}) or {},
            error=data.get("error"),
        )

    def _request_json(self, url: str, *, data: Optional[bytes] = None,
                      method: str = "GET", timeout: Optional[float] = None) -> Dict[str, Any]:
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib_request.Request(url, data=data, headers=headers, method=method)
        with urllib_request.urlopen(req, timeout=timeout or self.timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}

    def get_card(self) -> Optional[AgentCard]:
        if self._card:
            return self._card
        try:
            data = self._request_json(
                f"{self.endpoint}/.well-known/agent.json", timeout=self.timeout)
            self._card = AgentCard.from_dict(data)
            return self._card
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch agent card from %s: %s", self.endpoint, exc)
            return None

    def send_prompt(self, content: str, task_id: Optional[str] = None) -> TaskStatus:
        task_id = task_id or str(uuid.uuid4())
        message = A2AMessage(task_id=task_id, content=content)
        data = message.to_json().encode("utf-8")
        try:
            result = self._request_json(
                f"{self.endpoint}/messages", data=data, method="POST")
            return self._status_from_dict(result)
        except urllib_error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception:  # noqa: BLE001
                payload = {}
            return TaskStatus(
                task_id=task_id,
                state=TaskState.FAILED,
                error=payload.get("error") or f"remote HTTP {exc.code}",
                metadata={"httpStatus": exc.code, "retryable": exc.code >= 500},
            )
        except (TimeoutError, urllib_error.URLError, OSError) as exc:
            logger.warning("A2A submission outcome is unknown: %s", exc)
            return TaskStatus(
                task_id=task_id,
                state=TaskState.UNKNOWN,
                error=str(exc),
                metadata={"retryable": True, "queryTaskBeforeResubmit": True},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("A2A request failed: %s", exc)
            return TaskStatus(
                task_id=task_id,
                state=TaskState.UNKNOWN,
                error=str(exc),
                metadata={"retryable": True},
            )

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        try:
            data = self._request_json(f"{self.endpoint}/tasks/{task_id}")
            return self._status_from_dict(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to get task status: %s", exc)
            return None

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskStatus:
        """Poll a submitted task until it reaches a terminal state."""
        deadline = time.monotonic() + (float(timeout) if timeout is not None else self.timeout)
        terminal = {
            TaskState.COMPLETED, TaskState.FAILED,
            TaskState.CANCELLED, TaskState.TIMED_OUT,
        }
        last = TaskStatus(task_id=task_id, state=TaskState.UNKNOWN)
        while time.monotonic() < deadline:
            status = self.get_task_status(task_id)
            if status is not None:
                last = status
                if status.state in terminal:
                    return status
            time.sleep(min(self.poll_interval, max(0.05, deadline - time.monotonic())))
        last.state = TaskState.UNKNOWN
        last.error = last.error or "task status polling timed out"
        last.metadata = {**last.metadata, "retryable": True}
        return last

    def cancel_task(self, task_id: str) -> bool:
        try:
            data = self._request_json(
                f"{self.endpoint}/tasks/{task_id}/cancel", data=b"{}", method="POST")
            return bool(data.get("cancelled"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to cancel task: %s", exc)
            return False

    def resolve_or_resubmit(self, content: str, task_id: Optional[str] = None) -> TaskStatus:
        """Submit idempotently when the remote state is unknown.

        Strategy: query the existing task id first. If the remote answers with
        a known task, return it unchanged instead of creating a second task.
        If the remote does not know the task (or is unreachable), submit with
        the same task id so a retry cannot accidentally fork into two tasks.
        """
        task_id = task_id or str(uuid.uuid4())
        try:
            existing = self.get_task_status(task_id)
        except Exception:  # noqa: BLE001
            existing = None
        if existing is not None and existing.state is not TaskState.UNKNOWN:
            return existing
        return self.send_prompt(content, task_id=task_id)


class A2AServer:
    """Threaded, task-oriented A2A server.

    Agent calls are never performed in the HTTP request handler. Async agents
    (``arun``) get real deadline/cancellation support. Legacy synchronous
    agents run in a bounded worker and support cooperative cancellation; their
    underlying blocking call cannot be force-killed by Python threads.
    """

    def __init__(self, agent: Any, card: AgentCard, host: str = "0.0.0.0",
                 port: int = 8090, task_timeout: float = 60.0,
                 max_workers: int = 4, db_path: str = "",
                 max_retained_tasks: int = 200,
                 on_terminal: Optional[Callable[[TaskStatus], None]] = None):
        self.agent = agent
        self.card = card
        self.host = host
        self.port = port
        self.task_timeout = max(0.1, float(task_timeout))
        self.max_retained_tasks = max(1, int(max_retained_tasks))
        self._on_terminal = on_terminal
        self.store = TaskStore(db_path)
        self._tasks: Dict[str, TaskStatus] = {}
        self._fingerprints: Dict[str, str] = {}
        self._futures: Dict[str, Future] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._async_handles: Dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Task]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="a2a-task")
        self._agent_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="a2a-agent")
        self._http_server: Optional[ThreadingHTTPServer] = None
        self._running = False

    def get_card(self) -> str:
        return self.card.to_json()

    @staticmethod
    def _fingerprint(message: A2AMessage) -> str:
        canonical = json.dumps(
            {"type": message.type.value, "content": message.content,
             "metadata": message.metadata},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def handle_message(self, message: A2AMessage) -> TaskStatus:
        if message.type is not MessageType.PROMPT:
            raise ValueError("only prompt messages can create tasks")
        if not isinstance(message.content, str) or not message.content.strip():
            raise ValueError("message content must not be empty")
        if len(message.content) > 100_000:
            raise ValueError("message content is too large")

        task_id = message.task_id or str(uuid.uuid4())
        fingerprint = self._fingerprint(message)
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing:
                if self._fingerprints.get(task_id) != fingerprint:
                    raise A2AConflictError("task id was already used with different content")
                return copy.deepcopy(existing)

            task = TaskStatus(
                task_id=task_id,
                state=TaskState.SUBMITTED,
                metadata={
                    "deadlineSeconds": self.task_timeout,
                    "retryable": True,
                    "lastHeartbeatAt": time.time(),
                },
            )
            self._tasks[task_id] = task
            self._fingerprints[task_id] = fingerprint
            cancel_event = threading.Event()
            self._cancel_events[task_id] = cancel_event
            future = self._executor.submit(self._run_task, task_id, message, cancel_event)
            self._futures[task_id] = future
            self.store.upsert(task_id, fingerprint, task)
            return copy.deepcopy(task)

    def _set_state(self, task_id: str, state: TaskState, *, error: Optional[str] = None,
                   message: Optional[A2AMessage] = None) -> None:
        terminal_task: Optional[TaskStatus] = None
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            previous_state = task.state
            task.state = state
            if error is not None:
                task.error = error
            if message is not None:
                task.message = message
            task.metadata["lastHeartbeatAt"] = time.time()
            if state in {
                TaskState.COMPLETED, TaskState.FAILED,
                TaskState.CANCELLED, TaskState.TIMED_OUT,
            }:
                task.completed_at = time.time()
            self.store.upsert(task.task_id, self._fingerprints.get(task.task_id, ""), task)
            terminal_states = {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED, TaskState.TIMED_OUT}
            if previous_state not in terminal_states and state in terminal_states:
                terminal_task = copy.deepcopy(task)
        if terminal_task is not None and self._on_terminal is not None:
            try:
                self._on_terminal(terminal_task)
            except Exception:  # noqa: BLE001
                logger.exception("A2A terminal callback failed for task %s", task_id)

    def _heartbeat(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.metadata["lastHeartbeatAt"] = time.time()

    def _run_task(self, task_id: str, message: A2AMessage,
                  cancel_event: threading.Event) -> None:
        self._set_state(task_id, TaskState.WORKING)
        deadline = time.monotonic() + self.task_timeout
        try:
            result = self._invoke_agent(task_id, message.content, cancel_event, deadline)
            if cancel_event.is_set():
                raise _TaskCancelled()
            if time.monotonic() >= deadline:
                raise _TaskTimedOut()
            result_content = result
            if isinstance(result, dict):
                stop_reason = str(result.get("stop_reason", "")).strip().lower()
                result_content = result.get("content", result)
                if stop_reason in {"error", "failed", "llm_error", "timed_out"}:
                    detail = result.get("stop_detail") or result.get("error") or result_content
                    self._set_state(
                        task_id,
                        TaskState.FAILED,
                        error=str(detail),
                        message=A2AMessage(
                            task_id=task_id,
                            type=MessageType.RESULT,
                            content=str(result_content),
                        ),
                    )
                    return
            self._set_state(
                task_id,
                TaskState.COMPLETED,
                message=A2AMessage(
                    task_id=task_id,
                    type=MessageType.RESULT,
                    content=str(result_content),
                ),
            )
        except (_TaskCancelled, asyncio.CancelledError):
            self._set_state(task_id, TaskState.CANCELLED, error="task cancelled")
        except _TaskTimedOut:
            self._set_state(task_id, TaskState.TIMED_OUT,
                            error=f"agent execution timed out after {self.task_timeout:.1f}s")
        except Exception as exc:  # noqa: BLE001
            logger.exception("A2A task %s failed", task_id)
            self._set_state(
                task_id,
                TaskState.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                message=A2AMessage(
                    task_id=task_id,
                    type=MessageType.RESULT,
                    content=f"error: {type(exc).__name__}: {exc}",
                ),
            )
        finally:
            with self._lock:
                self._async_handles.pop(task_id, None)
                self._cancel_events.pop(task_id, None)
            self.store.delete_old_terminal(keep=self.max_retained_tasks)

    def _invoke_agent(self, task_id: str, content: str,
                      cancel_event: threading.Event, deadline: float) -> Any:
        arun = getattr(self.agent, "arun", None)
        if callable(arun):
            return self._invoke_async_agent(task_id, arun, content, cancel_event, deadline)

        run = getattr(self.agent, "run", None)
        if not callable(run):
            raise TypeError("agent must provide arun() or run()")

        # The supervisor can stop waiting and mark the A2A task cancelled/timed
        # out. Python cannot force-kill an already running synchronous thread.
        future = self._agent_executor.submit(run, content)
        while True:
            self._heartbeat(task_id)
            if cancel_event.is_set():
                future.cancel()
                raise _TaskCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                raise _TaskTimedOut()
            try:
                return future.result(timeout=min(0.2, remaining))
            except FutureTimeoutError:
                continue

    def _invoke_async_agent(self, task_id: str, arun: Any, content: str,
                            cancel_event: threading.Event, deadline: float) -> Any:
        async def runner() -> Any:
            current = asyncio.current_task()
            loop = asyncio.get_running_loop()
            with self._lock:
                self._async_handles[task_id] = (loop, current)
            try:
                value = arun(content)
                if not inspect.isawaitable(value):
                    raise TypeError("agent.arun() must return an awaitable")
                if cancel_event.is_set():
                    raise _TaskCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _TaskTimedOut()
                try:
                    return await asyncio.wait_for(value, timeout=remaining)
                except asyncio.TimeoutError as exc:
                    raise _TaskTimedOut() from exc
            finally:
                with self._lock:
                    self._async_handles.pop(task_id, None)

        # Run one isolated event loop in the task worker. This lets cancel_task
        # call loop.call_soon_threadsafe(task.cancel) safely.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Use a dedicated task so it can be cancelled from another thread.
            async def start_runner():
                return await runner()
            task = loop.create_task(start_runner())
            with self._lock:
                self._async_handles[task_id] = (loop, task)
            return loop.run_until_complete(task)
        finally:
            with self._lock:
                self._async_handles.pop(task_id, None)
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return copy.deepcopy(task)
        # Fall back to durable history so tasks survive process restarts.
        try:
            rows = self.store.list_tasks(limit=1, offset=0)
        except Exception:  # noqa: BLE001
            rows = []
        # list_tasks has no id filter; scan a small page for the id.
        for row in self.store.list_tasks(limit=500, offset=0):
            if row.get("taskId") == task_id:
                return TaskStatus.from_dict(row)
        return None

    def list_tasks(self, state: Optional[str] = None, limit: int = 50,
                   offset: int = 0) -> List[Dict[str, Any]]:
        """Return task summaries ordered newest first."""
        limit = max(1, min(200, int(limit)))
        offset = max(0, int(offset))
        if self.store._db_available:
            return self.store.list_tasks(state=state, limit=limit, offset=offset)
        with self._lock:
            items = [copy.deepcopy(t).to_dict() for t in self._tasks.values()]
        if state:
            items = [i for i in items if i.get("state") == state]
        items.sort(key=lambda i: i.get("startedAt", 0.0), reverse=True)
        return items[offset:offset + limit]

    def stats(self) -> Dict[str, Any]:
        persisted = self.store.stats()
        if persisted:
            return {"persisted": persisted, "in_memory": len(self._tasks)}
        with self._lock:
            counts: Dict[str, int] = {}
            for task in self._tasks.values():
                counts[task.state.value] = counts.get(task.state.value, 0) + 1
            return {"in_memory": len(self._tasks), "by_state": counts}

    def cancel_task(self, task_id: str) -> bool:
        cancelled_before_execution = False
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.state in {
                TaskState.COMPLETED, TaskState.FAILED,
                TaskState.CANCELLED, TaskState.TIMED_OUT,
            }:
                return False
            event = self._cancel_events.get(task_id)
            if event:
                event.set()
            future = self._futures.get(task_id)
            if task.state is TaskState.SUBMITTED and future and future.cancel():
                cancelled_before_execution = True
            else:
                task.state = TaskState.CANCEL_REQUESTED
                task.metadata["cancelRequestedAt"] = time.time()
                handle = self._async_handles.get(task_id)
                if handle:
                    loop, async_task = handle
                    loop.call_soon_threadsafe(async_task.cancel)
        if cancelled_before_execution:
            self._set_state(task_id, TaskState.CANCELLED, error="task cancelled before execution")
        return True

    def start(self) -> None:
        server = self._create_http_server()
        self._http_server = server
        self._running = True
        logger.info("A2A Server started on %s:%s", self.host, self.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            server.server_close()

    def stop(self, drain_seconds: float = 5.0) -> None:
        """Stop accepting work and wait briefly for in-flight tasks to settle.

        Cancellation is cooperative: in-flight tasks receive the cancel event so
        their terminal state can be persisted before shutdown.
        """
        self._running = False
        server = self._http_server
        if server:
            server.shutdown()
            self._http_server = None
        with self._lock:
            for event in list(self._cancel_events.values()):
                event.set()
            futures = [f for f in self._futures.values() if not f.done()]
        deadline = time.monotonic() + max(0.0, float(drain_seconds))
        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                future.result(timeout=remaining)
            except Exception:  # noqa: BLE001 - task already recorded its state
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._agent_executor.shutdown(wait=False, cancel_futures=True)
        logger.info("A2A Server stopped")

    def _create_http_server(self) -> ThreadingHTTPServer:
        server_instance = self

        class A2AHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/.well-known/agent.json":
                    self._respond(200, server_instance.get_card())
                    return
                if path.startswith("/tasks/"):
                    task_id = path.split("/")[-1]
                    task = server_instance.get_task(task_id)
                    if task:
                        self._respond(200, json.dumps(task.to_dict(), ensure_ascii=False))
                    else:
                        self._respond(404, json.dumps({"error": "Task not found"}))
                    return
                self._respond(404, json.dumps({"error": "Not found"}))

            def do_POST(self) -> None:
                path = urlsplit(self.path).path
                if path == "/messages":
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if length <= 0 or length > 2_000_000:
                            raise ValueError("invalid request body")
                        data = json.loads(self.rfile.read(length).decode("utf-8"))
                        task = server_instance.handle_message(A2AMessage.from_dict(data))
                        status = 200 if task.state in {
                            TaskState.COMPLETED, TaskState.FAILED,
                            TaskState.CANCELLED, TaskState.TIMED_OUT,
                        } else 202
                        self._respond(status, json.dumps(task.to_dict(), ensure_ascii=False))
                    except A2AConflictError as exc:
                        self._respond(409, json.dumps({"error": str(exc)}))
                    except ValueError as exc:
                        self._respond(400, json.dumps({"error": str(exc)}))
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("A2A message handling failed")
                        self._respond(500, json.dumps({"error": str(exc)}))
                    return
                if path.startswith("/tasks/") and path.endswith("/cancel"):
                    task_id = path.split("/")[-2]
                    success = server_instance.cancel_task(task_id)
                    self._respond(200 if success else 409,
                                  json.dumps({"cancelled": success}))
                    return
                self._respond(404, json.dumps({"error": "Not found"}))

            def _respond(self, status: int, body: str) -> None:
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("A2A: %s", fmt % args)

        return ThreadingHTTPServer((self.host, self.port), A2AHandler)
