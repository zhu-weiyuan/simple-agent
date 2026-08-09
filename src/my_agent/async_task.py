"""Thread-pool execution for bounded long-running application tasks."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class _Task:
    name: str
    future: Future
    status: str = "PENDING"
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class AsyncTaskQueue:
    """Best-effort durable task metadata queue.

    Callables themselves cannot be safely serialized, so in-flight work from a
    previous process is recovered as FAILED rather than silently rerun.
    """
    def __init__(self, max_workers: int = 4, db_path: Optional[str] = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent-task")
        self._tasks: Dict[str, _Task] = {}
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        self._db_path = db_path or os.getenv("TASK_DB_PATH", "runtime/tasks.db")
        self._db_available = self._init_db()
        self._recover_tasks()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> bool:
        try:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, task_name TEXT NOT NULL, status TEXT NOT NULL,
                    created_at REAL NOT NULL, completed_at REAL, error_msg TEXT
                )""")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)")
                conn.commit()
            return True
        except (sqlite3.Error, OSError) as exc:
            self._logger.warning("Task persistence unavailable: %s", exc)
            return False

    def _recover_tasks(self) -> None:
        if not self._db_available:
            return
        try:
            with self._connect() as conn:
                conn.execute("""UPDATE tasks SET status='FAILED', completed_at=?, error_msg=?
                    WHERE status IN ('PENDING', 'RUNNING')""",
                    (time.time(), "Interrupted by process restart"))
                rows = conn.execute("SELECT task_id, task_name, status, created_at, completed_at, error_msg FROM tasks").fetchall()
                conn.commit()
            with self._lock:
                for task_id, name, status, created, completed, error in rows:
                    future = Future()
                    if status == "COMPLETED":
                        future.set_result(None)
                    elif status == "FAILED":
                        future.set_exception(RuntimeError(error or "Task failed"))
                    self._tasks[task_id] = _Task(name, future, status, error=error, created_at=created, completed_at=completed)
        except sqlite3.Error as exc:
            self._logger.warning("Could not recover task metadata: %s", exc)

    def _persist(self, task_id: str, task: _Task) -> None:
        if not self._db_available:
            return
        try:
            with self._connect() as conn:
                conn.execute("""INSERT INTO tasks (task_id, task_name, status, created_at, completed_at, error_msg)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET status=excluded.status,
                    completed_at=excluded.completed_at, error_msg=excluded.error_msg""",
                    (task_id, task.name, task.status, task.created_at, task.completed_at, task.error))
                conn.commit()
        except sqlite3.Error as exc:
            self._logger.warning("Could not persist task %s: %s", task_id, exc)

    def submit(self, task_name: str, func: Callable[..., Any], args: tuple = (), kwargs: Optional[dict] = None) -> str:
        if not callable(func):
            raise TypeError("func must be callable")
        task_id = uuid.uuid4().hex
        task = _Task(task_name, Future())
        # Persist before scheduling, so even an immediate worker completion has
        # a durable row to update.
        self._persist(task_id, task)

        def execute() -> Any:
            with task.lock:
                task.status = "RUNNING"
            self._persist(task_id, task)
            try:
                result = func(*args, **(kwargs or {}))
                with task.lock:
                    task.result, task.status, task.completed_at = result, "COMPLETED", time.time()
                self._persist(task_id, task)
                self._logger.info("Async task %s (%s) completed", task_id, task_name)
                return result
            except Exception as exc:
                with task.lock:
                    task.error, task.status, task.completed_at = f"{type(exc).__name__}: {exc}", "FAILED", time.time()
                self._persist(task_id, task)
                self._logger.exception("Async task %s (%s) failed", task_id, task_name)
                raise

        task.future = self._executor.submit(execute)
        with self._lock:
            self._tasks[task_id] = task
        return task_id

    def get_result(self, task_id: str) -> Any:
        task = self._get(task_id)
        with task.lock:
            return task.result if task.status == "COMPLETED" else None

    def get_status(self, task_id: str) -> str:
        return self._get(task_id).status

    def get_all_tasks(self, limit: int = 50) -> list[Dict[str, Any]]:
        """Return newest task metadata first, including recovered tasks."""
        limit = max(0, int(limit))
        if self._db_available:
            try:
                with self._connect() as conn:
                    rows = conn.execute("SELECT task_id, task_name, status, created_at, completed_at, error_msg FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
                return [dict(zip(("task_id", "task_name", "status", "created_at", "completed_at", "error_msg"), row)) for row in rows]
            except sqlite3.Error:
                pass
        with self._lock:
            tasks = list(self._tasks.items())
        return [{"task_id": ident, "task_name": task.name, "status": task.status, "created_at": task.created_at, "completed_at": task.completed_at, "error_msg": task.error} for ident, task in sorted(tasks, key=lambda item: item[1].created_at, reverse=True)[:limit]]

    def _get(self, task_id: str) -> _Task:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Unknown task_id: {task_id}")
            return self._tasks[task_id]

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
