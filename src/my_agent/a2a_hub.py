# -*- coding: utf-8 -*-
"""A2A hub: local agent card/task server + remote A2A client registry.

The hub exposes the task-oriented A2A API on the FastAPI app (``app_prod.py``)
while keeping task lifecycle, timeouts, persistence, cancellation and idempotent
task ids inside :mod:`my_agent.a2a`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from my_agent.a2a import (
    A2AConflictError,
    A2AClient,
    A2AMessage,
    A2AServer,
    AgentCard,
    MessageType,
    TaskState,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class A2AMessageIn(BaseModel):
    """Pydantic body model for POST /a2a/messages."""

    message: str = Field(..., min_length=1, max_length=100_000,
                         description="prompt content for the remote agent")
    task_id: Optional[str] = None


class A2ACancelOut(BaseModel):
    cancelled: bool


class A2ALocalAgent:
    """Adapter that exposes the local engine as an A2A agent.

    ``arun`` is preferred when available so cancellation works; ``run`` is kept
    for callers that only know the synchronous interface.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._session = getattr(engine, "session", None)

    def arun(self, content: str) -> Any:
        return self.engine.arun(content, session=self._session)

    def run(self, content: str) -> str:
        result = self.engine.run(content)
        if isinstance(result, dict):
            return str(result.get("content", ""))
        return str(result)


class A2AHub:
    """Registry and dispatch layer used by the FastAPI routes."""

    def __init__(self, engine: Any, card: AgentCard, db_path: str = "",
                 task_timeout: float = 60.0, max_workers: int = 4,
                 remote_agents: Optional[Dict[str, str]] = None,
                 max_retained_tasks: int = 200) -> None:
        self.card = card
        self.task_timeout = max(0.1, float(task_timeout))
        self.remote_agents: Dict[str, A2AClient] = {}
        for name, endpoint in (remote_agents or {}).items():
            self.remote_agents[name] = A2AClient(endpoint, timeout=min(10.0, self.task_timeout))
        self.local = A2AServer(
            A2ALocalAgent(engine),
            card,
            task_timeout=self.task_timeout,
            max_workers=max_workers,
            db_path=db_path,
            max_retained_tasks=max_retained_tasks,
            on_terminal=self.record_terminal,
        )
        self._lock = threading.Lock()
        self._counters = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "timed_out": 0,
            "cancelled": 0,
            "conflicts": 0,
        }

    def _count(self, key: str) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1

    def list_agents(self) -> List[Dict[str, Any]]:
        agents = [{
            "name": "local",
            "endpoint": "/a2a",
            "card": self.card.to_dict(),
        }]
        for name, client in self.remote_agents.items():
            c = client.get_card()
            agents.append({
                "name": name,
                "endpoint": client.endpoint,
                "card": c.to_dict() if c else None,
                "unreachable": c is None,
            })
        return agents

    def submit(self, content: str, task_id: Optional[str] = None,
               agent: str = "local") -> TaskStatus:
        if agent == "local":
            self._count("submitted")
            return self.local.handle_message(
                A2AMessage(task_id=task_id, type=MessageType.PROMPT, content=content))
        client = self.remote_agents.get(agent)
        if client is None:
            raise A2AConflictError(f"unknown agent: {agent}")
        self._count("submitted")
        return client.send_prompt(content, task_id=task_id)

    def get_task(self, task_id: str, agent: str = "local") -> Optional[TaskStatus]:
        if agent == "local":
            return self.local.get_task(task_id)
        client = self.remote_agents.get(agent)
        if client is None:
            return None
        return client.get_task_status(task_id)

    def cancel(self, task_id: str, agent: str = "local") -> bool:
        if agent == "local":
            return self.local.cancel_task(task_id)
        client = self.remote_agents.get(agent)
        if client is None:
            return False
        return client.cancel_task(task_id)

    def list_tasks(self, agent: str = "local", state: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        if agent == "local":
            return self.local.list_tasks(state=state, limit=limit, offset=offset)
        client = self.remote_agents.get(agent)
        if client is None:
            return []
        # Remote listing is not part of the A2A protocol; report nothing.
        return []

    def stats(self) -> Dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "local": self.local.stats(),
            "agents": list(self.remote_agents.keys()),
        }

    def record_terminal(self, task: TaskStatus) -> None:
        if task.state is TaskState.COMPLETED:
            self._count("completed")
        elif task.state is TaskState.FAILED:
            self._count("failed")
        elif task.state is TaskState.TIMED_OUT:
            self._count("timed_out")
        elif task.state is TaskState.CANCELLED:
            self._count("cancelled")

    def shutdown(self) -> None:
        try:
            self.local.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("A2A hub shutdown: %s", exc)


def register_a2a_routes(app: Any, get_hub: Any) -> None:
    """Register task-oriented A2A routes on the FastAPI app.

    ``get_hub`` is a zero-argument callable returning the active A2AHub (or
    None before lifespan startup), so route handlers stay lifecycle-safe.
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    def _hub() -> A2AHub:
        hub = get_hub()
        if hub is None:
            raise HTTPException(503, "A2A is not ready")
        return hub

    def _to_json(status: TaskStatus) -> Dict[str, Any]:
        return status.to_dict()

    @app.get("/a2a/.well-known/agent.json", tags=["a2a"])
    async def a2a_agent_card():
        return _hub().card.to_dict()

    @app.get("/a2a/agents", tags=["a2a"])
    async def a2a_list_agents():
        return {"agents": _hub().list_agents()}

    @app.get("/a2a/tasks", tags=["a2a"])
    async def a2a_list_tasks(state: Optional[str] = None, limit: int = 50,
                             offset: int = 0, agent: str = "local"):
        hub = _hub()
        items = hub.list_tasks(agent=agent, state=state, limit=limit, offset=offset)
        return {"tasks": items, "count": len(items)}

    @app.post("/a2a/messages", tags=["a2a"])
    async def a2a_submit(payload: A2AMessageIn, agent: str = "local"):
        hub = _hub()
        try:
            status = hub.submit(payload.message, task_id=payload.task_id, agent=agent)
        except A2AConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"remote agent unavailable: {exc}") from exc
        return _to_json(status)

    @app.get("/a2a/tasks/{task_id}", tags=["a2a"])
    async def a2a_task_status(task_id: str, agent: str = "local"):
        hub = _hub()
        status = hub.get_task(task_id, agent=agent)
        if status is None:
            raise HTTPException(404, "task not found")
        return _to_json(status)

    @app.post("/a2a/tasks/{task_id}/cancel", tags=["a2a"])
    async def a2a_task_cancel(task_id: str, agent: str = "local"):
        hub = _hub()
        cancelled = hub.cancel(task_id, agent=agent)
        if not cancelled:
            raise HTTPException(409, "task is not cancellable or not found")
        return A2ACancelOut(cancelled=True)

    @app.get("/a2a", tags=["a2a"])
    async def a2a_demo_page():
        from pathlib import Path
        page = Path(__file__).resolve().parent.parent.parent / "web" / "a2a.html"
        if page.exists():
            return FileResponse(str(page), media_type="text/html")
        raise HTTPException(404, "a2a.html not found")

    @app.get("/a2a/stats", tags=["a2a"])
    async def a2a_stats():
        return _hub().stats()


def build_hub_from_env(engine: Any) -> A2AHub:
    """Construct the hub from environment configuration (A2A_*)."""
    db_path = os.environ.get("A2A_TASKS_DB", "runtime/a2a_tasks.db")
    task_timeout = float(os.environ.get("A2A_TASK_TIMEOUT_SECONDS", "60"))
    max_retained = int(os.environ.get("A2A_MAX_RETAINED_TASKS", "200"))
    remote_raw = os.environ.get("A2A_AGENTS_JSON", "")
    remote: Dict[str, str] = {}
    if remote_raw.strip():
        try:
            parsed = json.loads(remote_raw)
            if isinstance(parsed, dict):
                remote = {str(k): str(v) for k, v in parsed.items() if str(v).startswith("http")}
        except (ValueError, TypeError) as exc:
            logger.warning("A2A_AGENTS_JSON invalid: %s", exc)
    card = AgentCard(
        name="SimpleAgent",
        description="SimpleAgent local agent exposed over A2A",
        version="2.1.0",
        url="/a2a",
        capabilities={
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        tools=[],
        default_input_languages=["zh", "en"],
        preferred_transport="http",
    )
    return A2AHub(
        engine=engine,
        card=card,
        db_path=db_path,
        task_timeout=task_timeout,
        max_workers=int(os.environ.get("A2A_MAX_WORKERS", "4")),
        remote_agents=remote or None,
        max_retained_tasks=max_retained,
    )
