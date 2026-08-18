"""Durable subprocess jobs with cancellation, hard deadlines, and bounded output reads."""
from __future__ import annotations

import json
from contextlib import contextmanager
import os
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO


_TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted"}


class JobManager:
    """Manage long-running subprocesses without replaying them after restart.

    A job has a persisted lifecycle, while its process remains owned by the current
    service process.  Cancellation and deadlines are enforced by the supervisor,
    not by reads from the child process: a silent child can therefore never bypass
    a deadline simply because it has not written a newline to stdout.
    """

    def __init__(self, root: str | Path = "runtime/jobs", max_output_bytes: int = 5_000_000) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "jobs.sqlite3"
        self.output_dir = self.root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_output_bytes = max(1_024, int(max_output_bytes))
        self._lock = threading.RLock()
        self._processes: Dict[str, subprocess.Popen[str]] = {}
        self._init_db()
        self._mark_interrupted()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Yield a transaction and always release the SQLite file handle.

        sqlite3.Connection as a context manager commits or rolls back, but does
        not close the connection. Explicit close is essential on Windows, where
        a lingering handle can keep jobs.sqlite3 locked during maintenance.
        """
        db = self._connect()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _init_db(self) -> None:
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, tool_name TEXT NOT NULL,
                command_json TEXT NOT NULL, cwd TEXT NOT NULL, status TEXT NOT NULL,
                created_at REAL NOT NULL, started_at REAL, finished_at REAL,
                returncode INTEGER, timeout_seconds REAL NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '', output_path TEXT NOT NULL,
                owner_pid INTEGER NOT NULL DEFAULT 0, process_pid INTEGER,
                finish_reason TEXT NOT NULL DEFAULT '',
                 output_bytes INTEGER NOT NULL DEFAULT 0,
                 output_truncated INTEGER NOT NULL DEFAULT 0
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
            # Keep existing job databases forward compatible.  Jobs that predate
            # owner_pid have no trustworthy active supervisor and are safely marked
            # interrupted below instead of being replayed.
            if "owner_pid" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN owner_pid INTEGER NOT NULL DEFAULT 0")
            if "process_pid" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN process_pid INTEGER")
            if "finish_reason" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN finish_reason TEXT NOT NULL DEFAULT ''")
            if "output_bytes" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN output_bytes INTEGER NOT NULL DEFAULT 0")
            if "output_truncated" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN output_truncated INTEGER NOT NULL DEFAULT 0")
            db.commit()

    @staticmethod
    def _is_process_alive(pid: Any) -> bool:
        """Return whether a supervisor PID is still alive without signalling it."""
        try:
            value = int(pid)
            if value <= 0:
                return False
            os.kill(value, 0)
            return True
        except (TypeError, ValueError, ProcessLookupError):
            return False
        except PermissionError:
            # A process exists but belongs to another account.
            return True
        except OSError:
            return False

    def _mark_interrupted(self) -> None:
        """Mark only jobs from a *dead* supervisor as interrupted.

        The application exposes both the main registry and legacy SimpleAgent
        registry in one Python process.  A second manager must therefore never
        mistake jobs supervised by the first manager for restart leftovers.
        """
        with self._db() as db:
            candidates = db.execute(
                "SELECT job_id, owner_pid FROM jobs "
                "WHERE status IN ('queued','running','cancel_requested')"
            ).fetchall()
            stale_ids = [row["job_id"] for row in candidates if not self._is_process_alive(row["owner_pid"])]
            if stale_ids:
                placeholders = ",".join("?" for _ in stale_ids)
                db.execute(
                    f"UPDATE jobs SET status='interrupted', finished_at=?, error=?, finish_reason=? "
                    f"WHERE job_id IN ({placeholders}) "
                    "AND status IN ('queued','running','cancel_requested')",
                    (time.time(), "进程在服务重启时被安全中断；不会自动重放", "interrupted_after_restart", *stale_ids),
                )
            db.commit()

    def submit_process(self, command: List[str], cwd: str | Path, session_id: str = "default",
                       tool_name: str = "job", timeout_seconds: float = 120) -> Dict[str, Any]:
        if not command:
            raise ValueError("command cannot be empty")
        workdir = Path(cwd).resolve()
        if not workdir.is_dir():
            raise FileNotFoundError(f"工作目录不存在: {workdir}")
        job_id = f"job-{uuid.uuid4().hex}"
        output_path = self.output_dir / f"{job_id}.log"
        row = {
            "job_id": job_id, "session_id": str(session_id or "default"),
            "tool_name": str(tool_name), "command_json": json.dumps([str(x) for x in command], ensure_ascii=False),
            "cwd": str(workdir), "status": "queued", "created_at": time.time(),
            "timeout_seconds": max(1.0, float(timeout_seconds)), "output_path": str(output_path),
            "owner_pid": os.getpid(),
        }
        with self._db() as db:
            db.execute("""INSERT INTO jobs(job_id,session_id,tool_name,command_json,cwd,status,created_at,timeout_seconds,output_path,owner_pid)
                         VALUES(?,?,?,?,?,?,?,?,?,?)""", tuple(row[k] for k in ("job_id", "session_id", "tool_name", "command_json", "cwd", "status", "created_at", "timeout_seconds", "output_path", "owner_pid")))
            db.commit()
        threading.Thread(target=self._run_process, args=(job_id,), daemon=True, name=f"simpleagent-{job_id}").start()
        return self.get(job_id, session_id=session_id) or row

    def _update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "status", "started_at", "finished_at", "returncode", "cancel_requested",
            "error", "process_pid", "finish_reason", "output_bytes", "output_truncated",
        }
        fields = {key: value for key, value in fields.items() if key in allowed}
        if not fields:
            return
        sql = ", ".join(f"{key}=?" for key in fields)
        with self._db() as db:
            db.execute(f"UPDATE jobs SET {sql} WHERE job_id=?", (*fields.values(), job_id))
            db.commit()

    @staticmethod
    def _copy_output(stream: TextIO, output_path: str, max_bytes: int, state: Dict[str, Any]) -> None:
        """Drain stdout without letting an unbounded child log fill the disk.

        The pipe is always drained, even after the retention budget is reached, so
        a noisy child cannot deadlock.  The persisted log contains a truthful
        marker and the job row records the exact retained byte count.
        """
        marker = "\n...[任务输出已达到保留上限，后续内容未保存。]\n"
        marker_bytes = len(marker.encode("utf-8"))
        content_budget = max(0, int(max_bytes) - marker_bytes)
        retained = 0
        truncated = False
        try:
            with open(output_path, "w", encoding="utf-8", newline="\n") as out:
                for line in iter(stream.readline, ""):
                    encoded = line.encode("utf-8", errors="replace")
                    if retained < content_budget:
                        chunk = encoded[:content_budget - retained].decode("utf-8", errors="ignore")
                        if chunk:
                            out.write(chunk)
                            out.flush()
                            retained += len(chunk.encode("utf-8"))
                        if len(encoded) > len(chunk.encode("utf-8")):
                            truncated = True
                    else:
                        truncated = True
                if truncated:
                    out.write(marker)
                    out.flush()
                    retained += marker_bytes
        finally:
            state["output_bytes"] = retained
            state["output_truncated"] = truncated
            try:
                stream.close()
            except OSError:
                pass

    def _run_process(self, job_id: str) -> None:
        row = self.get(job_id)
        if not row:
            return
        command = list(row["command"])
        if row.get("cancel_requested"):
            self._update(
                job_id,
                status="cancelled",
                finished_at=time.time(),
                error="任务在启动前已取消",
                finish_reason="cancelled_before_start",
            )
            return
        proc: Optional[subprocess.Popen[str]] = None
        reader: Optional[threading.Thread] = None
        output_state: Dict[str, Any] = {"output_bytes": 0, "output_truncated": False}
        started = time.time()
        started_monotonic = time.monotonic()
        try:
            popen_kwargs: Dict[str, Any] = {
                "cwd": row["cwd"],
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(command, **popen_kwargs)
            self._update(job_id, status="running", started_at=started, process_pid=proc.pid)
            with self._lock:
                self._processes[job_id] = proc

            assert proc.stdout is not None
            reader = threading.Thread(
                target=self._copy_output,
                args=(proc.stdout, row["output_path"], self.max_output_bytes, output_state),
                daemon=True,
                name=f"simpleagent-output-{job_id}",
            )
            reader.start()

            deadline = started_monotonic + float(row["timeout_seconds"])
            while proc.poll() is None:
                current = self.get(job_id) or {}
                if current.get("cancel_requested"):
                    self._terminate(proc)
                    break
                if time.monotonic() >= deadline:
                    self._update(
                        job_id,
                        status="timed_out",
                        error="后台任务执行超时",
                        finish_reason="deadline_exceeded",
                    )
                    self._terminate(proc)
                    break
                time.sleep(0.05)

            # _terminate waits for the process; this is a final safety net for a
            # platform-level race where the process exits between poll() and wait().
            try:
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._terminate(proc, grace_seconds=0)
                returncode = proc.wait(timeout=5)

            current = self.get(job_id) or {}
            if current.get("cancel_requested"):
                status, reason, error = "cancelled", "user_cancelled", "任务已取消"
            elif current.get("status") == "timed_out":
                status, reason, error = "timed_out", "deadline_exceeded", current.get("error") or "后台任务执行超时"
            elif returncode == 0:
                status, reason, error = "completed", "completed", ""
            else:
                status, reason, error = "failed", "execution_error", f"进程以退出代码 {returncode} 结束"
            self._update(
                job_id,
                status=status,
                returncode=returncode,
                finished_at=time.time(),
                finish_reason=reason,
                error=error,
            )
        except Exception as exc:  # pragma: no cover - platform/process errors
            self._update(
                job_id,
                status="failed",
                finished_at=time.time(),
                error=f"{type(exc).__name__}: {exc}",
                finish_reason="execution_error",
            )
        finally:
            if reader is not None:
                reader.join(timeout=1)
            self._update(
                job_id,
                output_bytes=int(output_state.get("output_bytes", 0)),
                output_truncated=int(bool(output_state.get("output_truncated", False))),
            )
            with self._lock:
                self._processes.pop(job_id, None)

    @staticmethod
    def _terminate(proc: subprocess.Popen[str], grace_seconds: float = 0.5) -> None:
        """Terminate a job idempotently, including children created by the job.

        Windows does not provide POSIX process groups, so taskkill /T is used to
        terminate the complete tree immediately.  On POSIX the child is started in
        its own session and gets SIGTERM first, followed by SIGKILL only if it
        ignores the short grace period.
        """
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=max(2.0, grace_seconds + 2.0),
                )
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    return
                try:
                    proc.wait(timeout=max(0.0, grace_seconds))
                    return
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            pass
        finally:
            try:
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def get(self, job_id: str, session_id: str = "") -> Optional[Dict[str, Any]]:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        if session_id and data["session_id"] != str(session_id):
            return None
        data["command"] = json.loads(data.pop("command_json"))
        data["cancel_requested"] = bool(data["cancel_requested"])
        return data

    def list(self, session_id: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        limit = min(max(1, int(limit)), 200)
        with self._db() as db:
            if session_id:
                rows = db.execute("SELECT * FROM jobs WHERE session_id=? ORDER BY created_at DESC LIMIT ?", (str(session_id), limit)).fetchall()
            else:
                rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_dict(row) for row in rows]

    @staticmethod
    def _row_dict(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["command"] = json.loads(data.pop("command_json"))
        data["cancel_requested"] = bool(data["cancel_requested"])
        return data

    def read_output(self, job_id: str, session_id: str = "", offset: int = 0, limit: int = 12000) -> Dict[str, Any]:
        row = self.get(job_id, session_id=session_id)
        if row is None:
            raise KeyError("job 不存在或不属于当前会话")
        path = Path(row["output_path"])
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        offset = max(0, int(offset))
        limit = min(max(1, int(limit)), 100_000)
        chunk = text[offset:offset + limit]
        return {
            "job_id": job_id,
            "status": row["status"],
            "offset": offset,
            "limit": limit,
            "total_chars": len(text),
            "has_more": offset + len(chunk) < len(text),
            "output_bytes": int(row.get("output_bytes") or 0),
            "output_truncated": bool(row.get("output_truncated")),
            "finish_reason": row.get("finish_reason") or "",
            "content": chunk,
        }

    def cancel(self, job_id: str, session_id: str = "") -> Dict[str, Any]:
        row = self.get(job_id, session_id=session_id)
        if row is None:
            raise KeyError("job 不存在或不属于当前会话")
        if row["status"] in _TERMINAL:
            return row
        if row["status"] == "queued":
            self._update(
                job_id,
                status="cancelled",
                cancel_requested=1,
                finished_at=time.time(),
                error="任务在启动前已取消",
                finish_reason="cancelled_before_start",
            )
            return self.get(job_id, session_id=session_id) or row
        self._update(job_id, status="cancel_requested", cancel_requested=1)
        with self._lock:
            proc = self._processes.get(job_id)
        if proc is not None:
            self._terminate(proc)
        return self.get(job_id, session_id=session_id) or row

    def stats(self) -> Dict[str, Any]:
        with self._db() as db:
            rows = db.execute(
                "SELECT status, COUNT(*) AS count, COALESCE(SUM(output_bytes), 0) AS output_bytes "
                "FROM jobs GROUP BY status"
            ).fetchall()
        states = {str(row["status"]): int(row["count"]) for row in rows}
        return {
            "total": sum(states.values()),
            "states": states,
            "output_bytes": sum(max(0, int(row["output_bytes"] or 0)) for row in rows),
        }

    def cleanup(self, max_age_seconds: float = 0, max_total_output_bytes: int = 0) -> Dict[str, Any]:
        """Purge only terminal jobs, protecting live subprocess state from retention."""
        max_age_seconds = max(0.0, float(max_age_seconds))
        max_total_output_bytes = max(0, int(max_total_output_bytes))
        now = time.time()
        with self._lock:
            with self._db() as db:
                rows = db.execute(
                    "SELECT job_id, output_path, output_bytes, finished_at FROM jobs "
                    "WHERE status IN ('completed','failed','cancelled','timed_out','interrupted') "
                    "ORDER BY COALESCE(finished_at, created_at) ASC, job_id ASC"
                ).fetchall()
                selected: dict[str, str] = {}
                if max_age_seconds:
                    cutoff = now - max_age_seconds
                    for row in rows:
                        if float(row["finished_at"] or 0) < cutoff:
                            selected[str(row["job_id"])] = "expired"
                remaining_bytes = sum(
                    max(0, int(row["output_bytes"] or 0)) for row in rows
                    if str(row["job_id"]) not in selected
                )
                if max_total_output_bytes and remaining_bytes > max_total_output_bytes:
                    for row in rows:
                        job_id = str(row["job_id"])
                        if job_id in selected:
                            continue
                        selected[job_id] = "capacity"
                        remaining_bytes -= max(0, int(row["output_bytes"] or 0))
                        if remaining_bytes <= max_total_output_bytes:
                            break

                removed = removed_bytes = 0
                reasons = {"expired": 0, "capacity": 0}
                for row in rows:
                    job_id = str(row["job_id"])
                    reason = selected.get(job_id)
                    if not reason:
                        continue
                    output_path = Path(str(row["output_path"])).resolve()
                    try:
                        if self.output_dir not in output_path.parents:
                            raise OSError("job output path escapes store")
                        output_path.unlink(missing_ok=True)
                        db.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
                    except OSError:
                        continue
                    removed += 1
                    removed_bytes += max(0, int(row["output_bytes"] or 0))
                    reasons[reason] += 1
        return {
            "removed": removed,
            "removed_bytes": removed_bytes,
            "expired_removed": reasons["expired"],
            "capacity_removed": reasons["capacity"],
            "remaining": self.stats(),
        }


__all__ = ["JobManager"]






