# -*- coding: utf-8 -*-
"""
my_agent.session_manager — Per-session state isolation with LRU + SQLite persistence

SessionManager owns all in-memory SessionState objects, keyed by session_id:

- ``get_or_create(session_id)`` returns an isolated SessionState per session;
  the QueryEngine no longer holds a global session.
- LRU cap (default 500). When exceeded, the least-recently-used session is
  evicted and its full message history is persisted to SqliteConversationStore.
- Thread-safe (RLock). Safe under asyncio too: all critical sections are
  short, synchronous, non-blocking operations (except the SQLite writes,
  which use WAL + busy_timeout and are fast; call ``persist_all`` from a
  thread/`asyncio.to_thread` on shutdown if needed).
- ``restore_recent(n)`` reloads the most recently updated sessions at startup.
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .types.message import Message
from .types.session import SessionConfig, SessionState

logger = logging.getLogger(__name__)


def new_session_id() -> str:
    """Generate a server-side session id."""
    return f"sess-{uuid.uuid4().hex[:24]}"


class SessionManager:
    """LRU-bounded registry of SessionState objects with SQLite persistence."""

    def __init__(
        self,
        system_prompt: str,
        store: Optional[Any] = None,  # SqliteConversationStore-compatible
        max_sessions: int = 500,
        session_config: Optional[SessionConfig] = None,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self.system_prompt = system_prompt
        self.store = store
        self.max_sessions = max_sessions
        self.session_config = session_config
        self._sessions: "OrderedDict[str, SessionState]" = OrderedDict()
        self._lock = threading.RLock()

    # ── public API ───────────────────────────────────────────

    def get_or_create(
        self, session_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> "tuple[str, SessionState]":
        """Return (session_id, SessionState); creates or restores as needed.

        A missing/empty session_id gets a server-generated one.
        Marks the session as most-recently-used.
        当提供 user_id 时, 记录会话归属 (user_id → session_id) 到 store。
        """
        sid = session_id or new_session_id()
        with self._lock:
            state = self._sessions.get(sid)
            if state is not None:
                self._sessions.move_to_end(sid)
            else:
                state = self._load_from_store(sid)
                if state is None:
                    state = SessionState.create(self.system_prompt, self.session_config)
                self._sessions[sid] = state
                self._evict_if_needed()
        if user_id:
            self._record_owner(sid, user_id)
        return sid, state

    def _record_owner(self, session_id: str, user_id: str) -> None:
        if self.store is None:
            return
        try:
            self.store.set_session_user(session_id, user_id)
        except Exception as e:  # noqa: BLE001 - 归属记录失败不阻塞对话
            logger.warning("failed to record session owner %s/%s: %s",
                           user_id, session_id, e)

    def list_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """返回某 user 的历史会话 (store 支持时)。"""
        if self.store is None:
            return []
        try:
            return self.store.list_user_sessions(user_id, limit=limit)
        except Exception as e:  # noqa: BLE001
            logger.warning("list_user_sessions failed for %s: %s", user_id, e)
            return []

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is not None:
                self._sessions.move_to_end(session_id)
            return state

    def drop(self, session_id: str, persist: bool = True) -> bool:
        """Remove a session from memory (optionally persisting it first)."""
        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return False
        if persist:
            self._persist(session_id, state)
        return True

    def persist_session(self, session_id: str) -> bool:
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            return False
        self._persist(session_id, state)
        return True

    def persist_all(self) -> int:
        """Persist every in-memory session (e.g., on graceful shutdown)."""
        with self._lock:
            items = list(self._sessions.items())
        count = 0
        for sid, state in items:
            if self._persist(sid, state):
                count += 1
        return count

    def restore_recent(self, n: int = 50) -> int:
        """Load the n most recently updated sessions from the store at startup."""
        if self.store is None:
            return 0
        try:
            sessions = self.store.list_sessions(limit=min(n, self.max_sessions))
        except Exception as e:
            logger.warning("session restore failed: %s", e)
            return 0
        restored = 0
        # list_sessions returns newest first; insert oldest first so that the
        # newest end up most-recently-used in the LRU.
        for info in reversed(sessions):
            sid = info.get("id")
            if not sid:
                continue
            with self._lock:
                if sid in self._sessions:
                    continue
                state = self._load_from_store(sid)
                if state is None:
                    continue
                self._sessions[sid] = state
                self._evict_if_needed()
            restored += 1
        return restored

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "session_ids": list(self._sessions.keys())[-10:],
            }

    # ── internals ────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        """Must be called with the lock held."""
        while len(self._sessions) > self.max_sessions:
            sid, state = self._sessions.popitem(last=False)  # least recently used
            logger.info("Evicting LRU session %s (persisting to store)", sid)
            self._persist(sid, state)

    def _persist(self, session_id: str, state: SessionState) -> bool:
        if self.store is None:
            return False
        try:
            snapshot: List[Dict[str, Any]] = []
            for m in state.messages:
                openai_form = m.to_openai()
                snapshot.append({
                    "role": m.role.value,
                    "content": m.content or "",
                    "metadata": {"openai": openai_form, **(m.metadata or {})},
                })
            self.store.replace_session_messages(
                session_id, snapshot, metadata={"turn_count": state.turn_count}
            )
            return True
        except Exception as e:
            logger.error("Failed to persist session %s: %s", session_id, e)
            return False

    def _load_from_store(self, session_id: str) -> Optional[SessionState]:
        if self.store is None:
            return None
        try:
            rows = self.store.load_session_messages(session_id)
        except Exception as e:
            logger.warning("Failed to load session %s: %s", session_id, e)
            return None
        if not rows:
            return None
        messages: List[Message] = []
        for row in rows:
            meta = row.get("metadata") or {}
            raw = meta.get("openai")
            try:
                if isinstance(raw, dict):
                    msg = Message.from_dict({**raw, "content": raw.get("content") or ""})
                else:
                    msg = Message.from_dict(
                        {"role": row.get("role", "user"), "content": row.get("content", "")}
                    )
                messages.append(msg)
            except Exception:
                continue
        if not messages:
            return None
        state = SessionState(messages=messages)
        if self.session_config is not None:
            state.config = self.session_config
        info = None
        try:
            info = self.store.get_session_info(session_id)
        except Exception:
            pass
        if info and isinstance(info.get("metadata"), dict):
            state.turn_count = int(info["metadata"].get("turn_count", 0) or 0)
        # Ensure the first message is a system prompt
        if not state.messages or state.messages[0].role.value != "system":
            state.messages.insert(0, Message.system(self.system_prompt))
        return state
