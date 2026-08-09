# -*- coding: utf-8 -*-
"""
my_agent.memory.sqlite_store — SQLite-backed conversation persistence

Provides durable storage for conversation history with:
- Automatic table creation
- Session-based organization
- Full-text search support
- User profile tracking
- Conversation export

Usage:
    store = SqliteConversationStore("conversations.db")
    store.add_message(session_id, role, content)
    messages = store.get_conversation(session_id)
"""
from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


class SqliteConversationStore:
    """SQLite-backed conversation history store.

    Args:
        db_path: SQLite database path. ':memory:' for in-memory.
        retention_days: If set, sessions older than this many days are
            eligible for cleanup via ``purge_expired()``.  ``None`` (default)
            disables automatic expiry — all data is retained indefinitely.
    """

    def __init__(self, db_path: str = "conversations.db",
                 retention_days: Optional[int] = None) -> None:
        self.db_path_str = db_path
        self.db_path = Path(db_path) if db_path != ":memory:" else None
        self._persistent_conn: Optional[sqlite3.Connection] = None
        self.retention_days = retention_days
        
        # For in-memory databases, keep a persistent connection
        if db_path == ":memory:":
            self._persistent_conn = self._create_connection()
        
        self._ensure_db()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection."""
        path = self.db_path_str if self.db_path_str else str(self.db_path)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection (persistent for in-memory, new for file-based)."""
        if self._persistent_conn is not None:
            return self._persistent_conn
        return self._create_connection()

    def _ensure_db(self) -> None:
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            # Sessions table (user_id 关联会话归属; 详见 _migrate_columns)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT DEFAULT 'anonymous',
                    title TEXT DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    message_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            # Session snapshot 表 (SessionManager 全量落盘/恢复用;
            # 不受 messages 表 role CHECK 约束, 可存 tool/summary 角色)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_snapshots (
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (session_id, seq)
                )
            """)

            # Messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            # User profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    preferences TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id, created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created
                ON sessions(created_at DESC)
            """)

            conn.commit()
        self._migrate_columns()

    def _migrate_columns(self) -> None:
        """幂等迁移: 老库缺 user_id 列时补上 (SQLite ALTER 无 IF NOT EXISTS);
        并处理老库 messages 表的历史结构差异 (role CHECK 缺 'tool' / sequence_no 列)。"""
        with self._get_conn() as conn:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
            if "user_id" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT 'anonymous'")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC)"
            )
            conn.commit()
            # 老库 messages 表可能带 role CHECK 缺 'tool' → 需重建允许 tool 角色。
            self._migrate_role_check(conn)
            # 老库 messages 表可能带 sequence_no 列(含 UNIQUE(session_id, sequence_no)
            # 约束)。新代码不依赖它, 但 INSERT 时必须显式编号, 否则默认值 0 互相
            # 撞唯一约束 → 第二条消息即崩溃 (线上事故)。这里探测一次并缓存。
            mcols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
            self._has_sequence_no = "sequence_no" in mcols

    def _ensure_session_user_fk_target(
        self, conn: sqlite3.Connection, user_id: str, tenant_id: str = "default"
    ) -> None:
        """Provision an external identity when a legacy DB enforces users(id).

        Some existing installations have a multi-tenant ``sessions`` table with
        a foreign key from ``sessions.user_id`` to ``users.id``.  The current
        auth store also supports a simpler ``user_profiles``-only schema, so
        this is intentionally a no-op unless that legacy foreign key exists.
        Without this compatibility path, a first request from an external
        ``X-User-Id``/JWT identity logs a foreign-key warning and loses session
        ownership even though the chat itself succeeds.
        """
        fk_rows = conn.execute("PRAGMA foreign_key_list(sessions)").fetchall()
        fk = next(
            (row for row in fk_rows
             if row[2] == "users" and row[3] == "user_id" and row[4] in {"id", "user_id"}),
            None,
        )
        if fk is None or not user_id:
            return

        target_column = fk[4]
        if conn.execute(
            f"SELECT 1 FROM users WHERE {target_column} = ? LIMIT 1",
            (user_id,),
        ).fetchone():
            return

        table_columns = {row[1]: row for row in conn.execute("PRAGMA table_info(users)")}
        if "tenant_id" in table_columns:
            has_tenants = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tenants'"
            ).fetchone()
            if has_tenants and not conn.execute(
                "SELECT 1 FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchone():
                conn.execute(
                    "INSERT INTO tenants (id, name) VALUES (?, ?)",
                    (tenant_id, tenant_id),
                )

        values: Dict[str, Any] = {target_column: user_id}
        for column, value in (
            ("id", user_id),
            ("user_id", user_id),
            ("username", user_id),
            ("display_name", user_id),
            ("tenant_id", tenant_id),
        ):
            if column in table_columns:
                values[column] = value

        # Avoid attempting an insert if an installation has an unrelated
        # required column that cannot be derived safely.
        for name, row in table_columns.items():
            if row[3] and row[5] == 0 and row[4] is None and name not in values:
                return

        names = list(values)
        placeholders = ", ".join("?" for _ in names)
        conn.execute(
            f"INSERT OR IGNORE INTO users ({', '.join(names)}) VALUES ({placeholders})",
            [values[name] for name in names],
        )

    def _next_seq(self, conn: sqlite3.Connection, session_id: str) -> int:
        """老库带 sequence_no 时, 计算该会话下一个递增序号 (从 0 起)。"""
        row = conn.execute(
            "SELECT COALESCE(MAX(sequence_no), -1) + 1 FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0])

    def _migrate_role_check(self, conn: sqlite3.Connection) -> None:
        """若 messages 表的 CHECK 约束不含 'tool', 重建表以放开 (保留数据)。"""
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        sql = row["sql"] if row else None
        if not sql or "'tool'" in sql:
            return
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            # 保留老库可能存在的 sequence_no 列, 避免重建后丢约束/丢数据结构。
            mcols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
            has_seq = "sequence_no" in mcols
            seq_col = "sequence_no INTEGER NOT NULL DEFAULT 0," if has_seq else ""
            conn.execute("DROP TABLE IF EXISTS messages_new")
            conn.execute(f"""
                CREATE TABLE messages_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    metadata TEXT DEFAULT '{{}}',
                    {seq_col}
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            if has_seq:
                conn.execute(
                    "INSERT INTO messages_new (id, session_id, role, content, created_at, metadata, sequence_no) "
                    "SELECT id, session_id, role, content, created_at, metadata, sequence_no FROM messages"
                )
            else:
                conn.execute(
                    "INSERT INTO messages_new (id, session_id, role, content, created_at, metadata) "
                    "SELECT id, session_id, role, content, created_at, metadata FROM messages"
                )
            conn.execute("DROP TABLE messages")
            conn.execute("ALTER TABLE messages_new RENAME TO messages")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)"
            )
            if has_seq:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_seq ON messages(session_id, sequence_no)"
                )
            conn.commit()
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

    def create_session(self, session_id: str, title: str = "", metadata: Dict[str, Any] | None = None) -> str:
        """Create a new conversation session."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (id, title, message_count, metadata) VALUES (?, ?, 0, ?)",
                (session_id, title, json.dumps(metadata or {})),
            )
            conn.commit()
        return session_id

    _VALID_ROLES = {"user", "assistant", "system", "tool"}

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] | None = None) -> int:
        """Add a message to a conversation session."""
        if role not in self._VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'; expected one of {sorted(self._VALID_ROLES)}")
        with self._get_conn() as conn:
            # Ensure session exists
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id) VALUES (?)",
                (session_id,),
            )

            # Add message — 老库带 sequence_no UNIQUE 约束时必须显式编号,
            # 否则默认值 0 互撞唯一约束 (线上事故)。
            if getattr(self, "_has_sequence_no", False):
                cursor = conn.execute(
                    "INSERT INTO messages (session_id, role, content, metadata, sequence_no) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, json.dumps(metadata or {}),
                     self._next_seq(conn, session_id)),
                )
            else:
                cursor = conn.execute(
                    "INSERT INTO messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                    (session_id, role, content, json.dumps(metadata or {})),
                )

            # Update session stats
            conn.execute(
                """UPDATE sessions 
                   SET message_count = message_count + 1, 
                       updated_at = datetime('now') 
                   WHERE id = ?""",
                (session_id,),
            )

            conn.commit()
            return cursor.lastrowid

    def get_conversation(self, session_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get messages from a conversation session."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT role, content, created_at, metadata 
                   FROM messages 
                   WHERE session_id = ? 
                   ORDER BY created_at DESC 
                   LIMIT ? OFFSET ?""",
                (session_id, limit, offset),
            ).fetchall()

        return [dict(row) for row in reversed(rows)]

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

        if not row:
            return None

        result = dict(row)
        if result.get("metadata"):
            try:
                result["metadata"] = json.loads(result["metadata"])
            except json.JSONDecodeError:
                pass
        return result

    def list_sessions(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """List recent sessions."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_expired_sessions(self) -> List[Dict[str, Any]]:
        """Return sessions older than ``retention_days`` (if configured).

        Returns an empty list when ``retention_days`` is ``None``.
        Each dict contains ``session_id``, ``updated_at``, and ``message_count``.
        """
        if self.retention_days is None:
            return []
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id AS session_id, updated_at, message_count
                   FROM sessions
                   WHERE updated_at < datetime('now', ? || ' days')""",
                (-self.retention_days,),
            ).fetchall()
        return [dict(r) for r in rows]

    def purge_expired(self) -> int:
        """Delete sessions older than ``retention_days``.

        Returns the number of sessions deleted.  No-op when
        ``retention_days`` is ``None``.
        """
        if self.retention_days is None:
            return 0
        expired = self.get_expired_sessions()
        if not expired:
            return 0
        with self._get_conn() as conn:
            cursor = conn.execute(
                """DELETE FROM sessions
                   WHERE updated_at < datetime('now', ? || ' days')""",
                (-self.retention_days,),
            )
            conn.commit()
            return cursor.rowcount

    def search_messages(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search messages by content (simple LIKE search)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT m.*, s.title as session_title 
                   FROM messages m 
                   JOIN sessions s ON m.session_id = s.id 
                   WHERE m.content LIKE ? 
                   ORDER BY m.created_at DESC 
                   LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export a complete session as JSON-serializable dict."""
        session = self.get_session_info(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        messages = self.get_conversation(session_id, limit=10000)

        return {
            "session_id": session_id,
            "title": session.get("title", ""),
            "created_at": session.get("created_at", ""),
            "updated_at": session.get("updated_at", ""),
            "message_count": len(messages),
            "messages": messages,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_conn() as conn:
            total_sessions = conn.execute("SELECT COUNT(*) as count FROM sessions").fetchone()["count"]
            total_messages = conn.execute("SELECT COUNT(*) as count FROM messages").fetchone()["count"]

            # Average messages per session
            avg_row = conn.execute(
                "SELECT AVG(message_count) as avg_count FROM sessions"
            ).fetchone()
            avg_messages = avg_row["avg_count"] or 0

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "average_messages_per_session": round(avg_messages, 1),
            "db_path": str(self.db_path),
        }

    def update_user_profile(
        self, user_id: str, name: str = "", preferences: Dict[str, Any] | None = None
    ) -> None:
        """Update or create a user profile."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO user_profiles (user_id, name, preferences, updated_at) 
                   VALUES (?, ?, ?, datetime('now'))
                   ON CONFLICT(user_id) DO UPDATE SET 
                       name = COALESCE(?, name),
                       preferences = COALESCE(?, preferences),
                       updated_at = datetime('now')""",
                (user_id, name, json.dumps(preferences or {}), name, json.dumps(preferences or {})),
            )
            conn.commit()

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user profile."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if not row:
            return None

        result = dict(row)
        if result.get("preferences"):
            try:
                result["preferences"] = json.loads(result["preferences"])
            except json.JSONDecodeError:
                pass
        return result

    # ── 用户 ↔ 会话归属 ─────────────────────────────────────
    def set_session_user(self, session_id: str, user_id: str, title: str = "") -> bool:
        """Associate a new or unowned session with a user without overwriting ownership."""
        with self._get_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,))
            session_row = conn.execute(
                "SELECT user_id, tenant_id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            tenant_id = (session_row["tenant_id"] if session_row and "tenant_id" in session_row.keys()
                         else "default") or "default"
            self._ensure_session_user_fk_target(conn, user_id, tenant_id)
            row = conn.execute("SELECT user_id FROM sessions WHERE id = ?", (session_id,)).fetchone()
            owner = row["user_id"] if row else None
            if owner and owner != "anonymous" and owner != user_id:
                return False
            if title:
                conn.execute(
                    "UPDATE sessions SET user_id = ?, title = ?, updated_at = datetime('now') WHERE id = ?",
                    (user_id, title, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET user_id = ?, updated_at = datetime('now') WHERE id = ?",
                    (user_id, session_id),
                )
            conn.commit()
            return True

    def list_user_sessions(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """返回某 user 的历史会话 (newest first)。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, user_id, title, created_at, updated_at, message_count
                   FROM sessions WHERE user_id = ?
                   ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                (user_id, limit, offset),
            ).fetchall()
        return [{
            "session_id": r["id"],
            "user_id": r["user_id"],
            "title": r["title"] or "",
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "message_count": r["message_count"],
        } for r in rows]

    # ── 全量会话快照 (SessionManager 落盘/恢复) ──────────────
    def replace_session_messages(
        self, session_id: str, messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """整段替换会话快照 (role/content/metadata)。保留 sessions 行的 user_id。"""
        with self._get_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,))
            meta_json = json.dumps(metadata or {})
            conn.execute(
                "UPDATE sessions SET message_count = ?, metadata = ?, updated_at = datetime('now') WHERE id = ?",
                (len(messages), meta_json, session_id),
            )
            conn.execute("DELETE FROM session_snapshots WHERE session_id = ?", (session_id,))
            for seq, m in enumerate(messages):
                conn.execute(
                    "INSERT INTO session_snapshots (session_id, seq, role, content, metadata) VALUES (?, ?, ?, ?, ?)",
                    (session_id, seq, m.get("role", "user"), m.get("content", "") or "",
                     json.dumps(m.get("metadata", {}) or {})),
                )
            conn.commit()

    def load_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """加载会话快照 (seq 升序)。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata FROM session_snapshots WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = {}
            try:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
            except json.JSONDecodeError:
                meta = {}
            out.append({"role": r["role"], "content": r["content"], "metadata": meta})
        return out

    def close(self) -> None:
        """Close persistent connection and checkpoint WAL.
        
        Call this before deleting the database file (e.g., in tests).
        """
        try:
            if self._persistent_conn:
                self._persistent_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._persistent_conn.commit()
                self._persistent_conn.close()
                self._persistent_conn = None
            else:
                # For file-based, do a final checkpoint
                conn = self._get_conn()
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.commit()
                finally:
                    conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass


class PromptStore:
    """SQLite-backed prompt version storage.

    Provides durable storage for versioned prompts with:
    - Automatic table creation
    - JSON variable schemas
    - Default-version tracking
    - In-memory fallback if SQLite fails

    This is the store used by PromptRegistry for versioned prompt
    lifecycle management.
    """

    def __init__(self, db_path: str = "runtime/prompts.db") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            # In-memory fallback when SQLite path is unavailable
            self._conn = sqlite3.connect(":memory:")
            self._conn.row_factory = sqlite3.Row
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version_no TEXT NOT NULL,
                    content TEXT NOT NULL,
                    variables TEXT NOT NULL DEFAULT '{}',
                    is_default INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(name, version_no)
                )
            """)
            # Migrate old schema: add any missing columns
            cols = {
                r["name"]
                for r in self._conn.execute("PRAGMA table_info(prompts)").fetchall()
            }
            if "variables" not in cols and "variables_schema" in cols:
                self._conn.execute(
                    "ALTER TABLE prompts RENAME COLUMN variables_schema TO variables"
                )
            if "updated_at" not in cols:
                self._conn.execute(
                    "ALTER TABLE prompts ADD COLUMN updated_at TEXT"
                    " NOT NULL DEFAULT (datetime('now'))"
                )
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompts_name
                ON prompts(name, is_default)
            """)

    def add_prompt(
        self,
        name: str,
        version_no: str,
        content: str,
        variables: Optional[Dict[str, Any]] = None,
        is_default: bool = True,
    ) -> None:
        """Register a prompt version.

        If is_default is True, any previously default version for the
        same name is unmarked.
        """
        with self._conn:
            if is_default:
                self._conn.execute(
                    "UPDATE prompts SET is_default = 0 WHERE name = ? AND is_default = 1",
                    (name,),
                )
            self._conn.execute(
                """INSERT OR REPLACE INTO prompts
                   (name, version_no, content, variables, is_default, updated_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (name, version_no, content, json.dumps(variables or {}),
                 1 if is_default else 0),
            )

    def get_prompt(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt version.

        Args:
            name: Prompt name.
            version: Specific version. If None, returns the current default.

        Returns:
            Dict with keys 'name', 'version_no', 'content', 'is_default' or None.
        """
        if version:
            row = self._conn.execute(
                "SELECT name, version_no, content, is_default FROM prompts "
                "WHERE name = ? AND version_no = ?",
                (name, version),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT name, version_no, content, is_default FROM prompts "
                "WHERE name = ? AND is_default = 1",
                (name,),
            ).fetchone()
        return dict(row) if row else None

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all prompt versions ordered by name and version."""
        rows = self._conn.execute(
            "SELECT name, version_no, content, is_default FROM prompts "
            "ORDER BY name, version_no"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
