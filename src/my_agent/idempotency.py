# -*- coding: utf-8 -*-
"""Durable request idempotency records backed by SQLite.

The store intentionally keeps only a request fingerprint and the final response;
raw user messages are never written to this table.  SQLite's transactional
unique key makes the claim operation safe across threads and worker processes.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class IdempotencyResult:
    status: str  # claimed, replay, conflict, in_progress
    token: Optional[str] = None
    status_code: int = 200
    content_type: str = "application/json"
    response_body: Optional[str] = None


class IdempotencyStore:
    """SQLite-backed idempotency key store.

    ``scope`` should contain the authenticated tenant and user.  A pending
    claim is leased so a process killed mid-request cannot block a key forever.
    """

    def __init__(self, db_path: str, pending_ttl_seconds: float = 300.0) -> None:
        self.db_path_str = db_path
        self.pending_ttl_seconds = max(1.0, float(pending_ttl_seconds))
        self.db_path = Path(db_path) if db_path != ":memory:" else None
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._persistent_conn = self._connect()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path_str, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _conn(self) -> sqlite3.Connection:
        return self._persistent_conn or self._connect()

    def _ensure_schema(self) -> None:
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_records (
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'completed')),
                    status_code INTEGER NOT NULL DEFAULT 200,
                    content_type TEXT NOT NULL DEFAULT 'application/json',
                    response_body TEXT,
                    owner_token TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (scope, key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_idempotency_updated "
                "ON idempotency_records(updated_at)"
            )
            conn.commit()

    @staticmethod
    def fingerprint(payload: Any) -> str:
        """Return a stable SHA-256 fingerprint without retaining the payload."""
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def claim(self, scope: str, key: str, request_hash: str) -> IdempotencyResult:
        now = time.time()
        token = secrets.token_urlsafe(24)
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM idempotency_records WHERE scope=? AND key=?",
                (scope, key),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO idempotency_records
                    (scope, key, request_hash, status, owner_token, created_at, updated_at)
                    VALUES (?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (scope, key, request_hash, token, now, now),
                )
                conn.commit()
                return IdempotencyResult(status="claimed", token=token)

            if row["request_hash"] != request_hash:
                conn.commit()
                return IdempotencyResult(status="conflict")

            if row["status"] == "completed":
                conn.commit()
                return IdempotencyResult(
                    status="replay",
                    status_code=int(row["status_code"]),
                    content_type=row["content_type"],
                    response_body=row["response_body"] or "",
                )

            # Reclaim an abandoned lease, but make finalize/release token-bound
            # so an old worker cannot overwrite the newer request's response.
            if now - float(row["updated_at"]) > self.pending_ttl_seconds:
                conn.execute(
                    "UPDATE idempotency_records SET owner_token=?, updated_at=? "
                    "WHERE scope=? AND key=? AND status='pending'",
                    (token, now, scope, key),
                )
                conn.commit()
                return IdempotencyResult(status="claimed", token=token)

            conn.commit()
            return IdempotencyResult(status="in_progress")
        except Exception:
            conn.rollback()
            raise
        finally:
            if self._persistent_conn is None:
                conn.close()

    def finalize(
        self,
        scope: str,
        key: str,
        token: str,
        response: Any,
        status_code: int = 200,
        content_type: str = "application/json",
    ) -> bool:
        body = response if isinstance(response, str) else json.dumps(
            response, ensure_ascii=False, separators=(",", ":"), default=str
        )
        conn = self._conn()
        try:
            cur = conn.execute(
                """
                UPDATE idempotency_records
                   SET status='completed', status_code=?, content_type=?,
                       response_body=?, owner_token=NULL, updated_at=?
                 WHERE scope=? AND key=? AND status='pending' AND owner_token=?
                """,
                (int(status_code), content_type, body, time.time(), scope, key, token),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            if self._persistent_conn is None:
                conn.close()

    def release(self, scope: str, key: str, token: str) -> bool:
        conn = self._conn()
        try:
            cur = conn.execute(
                "DELETE FROM idempotency_records WHERE scope=? AND key=? "
                "AND status='pending' AND owner_token=?",
                (scope, key, token),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            if self._persistent_conn is None:
                conn.close()

    def purge(self, completed_ttl_seconds: float = 86400.0) -> int:
        """Remove old completed records; safe to call from a maintenance task."""
        cutoff = time.time() - max(1.0, float(completed_ttl_seconds))
        conn = self._conn()
        try:
            cur = conn.execute(
                "DELETE FROM idempotency_records WHERE status='completed' AND updated_at<?",
                (cutoff,),
            )
            conn.commit()
            return cur.rowcount
        finally:
            if self._persistent_conn is None:
                conn.close()

    def close(self) -> None:
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None


__all__ = ["IdempotencyResult", "IdempotencyStore"]
