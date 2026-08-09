# -*- coding: utf-8 -*-
"""
Tests for SQLite conversation store.

Run: pytest test_sqlite_store.py -v
"""
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _get_store():
    """Create an in-memory SQLite store for testing."""
    from my_agent.memory.sqlite_store import SqliteConversationStore
    # Use :memory: to avoid Windows file lock issues
    return SqliteConversationStore(":memory:"), None


def _cleanup(store, path: str | None) -> None:
    """Close store and delete temp DB file if on disk."""
    store.close()
    if path:
        Path(path).unlink(missing_ok=True)
        Path(path + "-wal").unlink(missing_ok=True)
        Path(path + "-shm").unlink(missing_ok=True)


def test_create_session():
    """Creating a session should work."""
    store, path = _get_store()
    try:
        sid = store.create_session("test-001", "Test Session")
        assert sid == "test-001"

        info = store.get_session_info(sid)
        assert info is not None
        assert info["title"] == "Test Session"
        assert info["message_count"] == 0
    finally:
        _cleanup(store, path)


def test_add_and_get_messages():
    """Adding messages should persist and be retrievable."""
    store, path = _get_store()
    try:
        sid = store.create_session("test-002")

        # Add messages
        store.add_message(sid, "user", "Hello!")
        store.add_message(sid, "assistant", "Hi there!")
        store.add_message(sid, "user", "How are you?")

        # Get conversation
        msgs = store.get_conversation(sid)
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello!"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["content"] == "How are you?"

        # Check session stats
        info = store.get_session_info(sid)
        assert info["message_count"] == 3
    finally:
        _cleanup(store, path)


def test_list_sessions():
    """Listing sessions should return them in order."""
    store, path = _get_store()
    try:
        store.create_session("s1", "First")
        store.create_session("s2", "Second")
        store.add_message("s1", "user", "msg")

        sessions = store.list_sessions(limit=10)
        assert len(sessions) == 2
        # Most recently updated first
        assert sessions[0]["id"] == "s1"
    finally:
        _cleanup(store, path)


def test_delete_session():
    """Deleting a session should remove it and its messages."""
    store, path = _get_store()
    try:
        store.create_session("del-001")
        store.add_message("del-001", "user", "Hello")

        assert store.delete_session("del-001") is True
        assert store.get_session_info("del-001") is None
        assert store.get_conversation("del-001") == []

        # Deleting non-existent session returns False
        assert store.delete_session("nonexistent") is False
    finally:
        _cleanup(store, path)


def test_search_messages():
    """Searching messages should find matching content."""
    store, path = _get_store()
    try:
        store.create_session("search-001")
        store.add_message("search-001", "user", "I love Python programming")
        store.add_message("search-001", "assistant", "Python is great!")
        store.add_message("search-001", "user", "Tell me about JavaScript")

        results = store.search_messages("Python")
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)

        # Case insensitive search (LIKE is case-insensitive on SQLite)
        results_lower = store.search_messages("python")
        assert len(results_lower) >= 1
    finally:
        _cleanup(store, path)


def test_export_session():
    """Exporting a session should return complete data."""
    store, path = _get_store()
    try:
        sid = "export-001"
        store.create_session(sid, "Export Test")
        store.add_message(sid, "user", "Question 1")
        store.add_message(sid, "assistant", "Answer 1")

        exported = store.export_session(sid)
        assert exported["session_id"] == sid
        assert exported["title"] == "Export Test"
        assert exported["message_count"] == 2
        assert len(exported["messages"]) == 2
    finally:
        _cleanup(store, path)


def test_get_stats():
    """Stats should reflect database state."""
    store, path = _get_store()
    try:
        stats = store.get_stats()
        assert "total_sessions" in stats
        assert "total_messages" in stats
        assert "average_messages_per_session" in stats
        assert stats["total_sessions"] == 0

        store.create_session("stats-001")
        store.add_message("stats-001", "user", "Hello")

        stats = store.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["total_messages"] == 1
    finally:
        _cleanup(store, path)


def test_user_profile():
    """User profile CRUD should work."""
    store, path = _get_store()
    try:
        # Create/update profile
        store.update_user_profile("user-001", "Alice", {"lang": "zh"})

        profile = store.get_user_profile("user-001")
        assert profile is not None
        assert profile["name"] == "Alice"
        assert profile["preferences"]["lang"] == "zh"

        # Non-existent profile returns None
        assert store.get_user_profile("nonexistent") is None

        # Update with partial data
        store.update_user_profile("user-001", name="Alice Updated")
        profile = store.get_user_profile("user-001")
        assert profile["name"] == "Alice Updated"
    finally:
        _cleanup(store, path)


def test_metadata_storage():
    """Message metadata should be preserved."""
    store, path = _get_store()
    try:
        sid = "meta-001"
        store.create_session(sid)
        store.add_message(
            sid, "user", "Test message",
            metadata={"source": "web", "ip": "127.0.0.1"}
        )

        msgs = store.get_conversation(sid)
        assert len(msgs) == 1
        meta = json.loads(msgs[0]["metadata"])
        assert meta["source"] == "web"
    finally:
        _cleanup(store, path)


def test_pagination():
    """Conversation retrieval should support pagination."""
    store, path = _get_store()
    try:
        sid = "page-001"
        store.create_session(sid)

        # Add 10 messages
        for i in range(10):
            store.add_message(sid, "user", f"Message {i}")

        # Get first page (5 most recent)
        page1 = store.get_conversation(sid, limit=5, offset=0)
        assert len(page1) == 5

        # Get second page
        page2 = store.get_conversation(sid, limit=5, offset=5)
        assert len(page2) == 5
    finally:
        _cleanup(store, path)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


def test_legacy_multitenant_schema_provisions_external_session_owner(tmp_path):
    """Legacy sessions.user_id -> users.id schemas must keep ownership durable."""
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO tenants (id, name) VALUES ('default', 'Default tenant');
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                display_name TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                user_id TEXT,
                title TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                message_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            );
            """
        )

    from my_agent.memory.sqlite_store import SqliteConversationStore

    store = SqliteConversationStore(str(db_path))
    try:
        assert store.set_session_user("sess-legacy", "external-user") is True
        assert store.get_session_info("sess-legacy")["user_id"] == "external-user"
        assert store.list_user_sessions("external-user")[0]["session_id"] == "sess-legacy"
    finally:
        store.close()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id FROM users WHERE id = 'external-user'").fetchone()


def test_retention_days_none_means_no_expiry():
    """When retention_days is None, get_expired_sessions returns empty."""
    from my_agent.memory.sqlite_store import SqliteConversationStore

    store = SqliteConversationStore(":memory:", retention_days=None)
    try:
        store.create_session("ret-001")
        store.add_message("ret-001", "user", "Hello")
        assert store.get_expired_sessions() == []
        assert store.purge_expired() == 0
    finally:
        store.close()


def test_purge_expired_deletes_old_sessions():
    """Sessions older than retention_days should be purged."""
    import time
    from my_agent.memory.sqlite_store import SqliteConversationStore

    store = SqliteConversationStore(":memory:", retention_days=7)
    try:
        # Create a session
        store.create_session("ret-old")
        store.add_message("ret-old", "user", "Old message")

        # Backdate the session's updated_at to 10 days ago
        with store._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = datetime('now', '-10 days') WHERE id = 'ret-old'"
            )
            conn.commit()

        expired = store.get_expired_sessions()
        assert len(expired) == 1
        assert expired[0]["session_id"] == "ret-old"

        deleted = store.purge_expired()
        assert deleted == 1
        assert store.get_session_info("ret-old") is None
    finally:
        store.close()


def test_purge_expired_keeps_recent_sessions():
    """Sessions within retention window should survive purge."""
    from my_agent.memory.sqlite_store import SqliteConversationStore

    store = SqliteConversationStore(":memory:", retention_days=30)
    try:
        store.create_session("ret-new")
        store.add_message("ret-new", "user", "Recent message")

        expired = store.get_expired_sessions()
        assert len(expired) == 0

        deleted = store.purge_expired()
        assert deleted == 0
        assert store.get_session_info("ret-new") is not None
    finally:
        store.close()


def test_purge_expired_returns_count():
    """purge_expired should return the number of deleted sessions."""
    from my_agent.memory.sqlite_store import SqliteConversationStore

    store = SqliteConversationStore(":memory:", retention_days=1)
    try:
        # Create 3 sessions, backdate all
        for i in range(3):
            store.create_session(f"purge-{i}")
            store.add_message(f"purge-{i}", "user", f"Msg {i}")

        with store._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = datetime('now', '-5 days')"
            )
            conn.commit()

        deleted = store.purge_expired()
        assert deleted == 3
        assert store.get_stats()["total_sessions"] == 0
    finally:
        store.close()
