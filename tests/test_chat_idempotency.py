from types import SimpleNamespace

import pytest
import threading
import time
from fastapi.testclient import TestClient

import app_prod
from my_agent.idempotency import IdempotencyStore
from my_agent.memory.sqlite_store import SqliteConversationStore
from my_agent.session_manager import SessionManager


@pytest.fixture
def chat_context(monkeypatch, tmp_path):
    """Set up test environment without patching engine methods."""
    monkeypatch.setenv("DEV_AUTH_BYPASS", "1")

    class FakeAsyncLLM:
        async def aclose(self):
            return None

    monkeypatch.setattr(app_prod, "async_llm", FakeAsyncLLM())
    monkeypatch.setattr(
        app_prod,
        "user_memory",
        SimpleNamespace(build_background_block=lambda user_id, message: ""),
    )
    monkeypatch.setattr(app_prod, "_schedule_memory_extraction", lambda *args: None)

    db_path = str(tmp_path / "chat.db")
    monkeypatch.setattr(app_prod, "idempotency_store", IdempotencyStore(db_path))
    session_store = SqliteConversationStore(db_path)
    manager = SessionManager(
        system_prompt=app_prod.compose_system_prompt(app_prod.SYSTEM_PROMPT),
        store=session_store,
        max_sessions=20,
    )
    monkeypatch.setattr(app_prod, "session_manager", manager)

    calls = {"arun": 0, "stream": 0}
    return calls


@pytest.fixture
def client(chat_context, monkeypatch):
    """Create TestClient and patch engine methods after initialization."""
    with TestClient(app_prod.app) as test_client:
        # Patch instance methods after TestClient creation to avoid CI issues
        async def mock_arun_with_dsh(message, *, session, ctx, max_tool_calls=50, session_id="", llm_idle_timeout=None):
            chat_context["arun"] += 1
            return {
                "content": f"mock:{message}",
                "stop_reason": "stop",
                "usage": {"total_tokens": 1},
                "model": "test-model",
            }

        async def mock_arun_stream_with_dsh(message, *, session, ctx, max_tool_calls=50, session_id="", llm_idle_timeout=None):
            chat_context["stream"] += 1
            yield {"delta": f"mock:{message}"}

        monkeypatch.setattr(app_prod.engine, "arun_with_dsh", mock_arun_with_dsh)
        monkeypatch.setattr(app_prod.engine, "arun_stream_with_dsh", mock_arun_stream_with_dsh)
        yield test_client


def test_same_key_replays_without_second_engine_call(client, chat_context):
    headers = {"Idempotency-Key": "chat-1"}
    payload = {"message": "hello", "user_id": "user-a"}
    first = client.post("/api/chat", json=payload, headers=headers)
    second = client.post("/api/chat", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers["X-Idempotency-Replayed"] == "true"
    assert second.headers["content-type"].startswith("application/json")
    assert second.json() == first.json()
    assert chat_context["arun"] == 1


def test_same_key_with_different_payload_is_conflict(client, chat_context):
    headers = {"Idempotency-Key": "chat-conflict"}
    assert client.post("/api/chat", json={"message": "one"}, headers=headers).status_code == 200
    response = client.post("/api/chat", json={"message": "two"}, headers=headers)

    assert response.status_code == 409
    assert chat_context["arun"] == 1


def test_in_progress_returns_retry_after(client, chat_context):
    req = app_prod.ChatRequest(message="pending", user_id="user-pending")
    digest = app_prod._idempotency_request_hash(req, "pending")
    store = app_prod.idempotency_store
    claimed = store.claim("default:user-pending", "pending-key", digest)
    assert claimed.status == "claimed"

    response = client.post(
        "/api/chat",
        json={"message": "pending", "user_id": "user-pending"},
        headers={"Idempotency-Key": "pending-key"},
    )
    assert response.status_code == 409
    assert response.headers["Retry-After"] == "1"
    assert chat_context["arun"] == 0


def test_memory_recall_timeout_does_not_block_chat(client, chat_context, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_memory_recall(user_id, message):
        started.set()
        release.wait(timeout=2)
        return "This late result must not affect the request"

    monkeypatch.setattr(
        app_prod,
        "user_memory",
        SimpleNamespace(build_background_block=slow_memory_recall),
    )
    monkeypatch.setattr(app_prod, "MEMORY_RECALL_TIMEOUT_SECONDS", 0.05)

    started_at = time.monotonic()
    response = client.post("/api/chat", json={"message": "continue without memory"})
    elapsed = time.monotonic() - started_at
    release.set()

    assert started.wait(timeout=0.5)
    assert response.status_code == 200
    assert response.json()["reply"] == "mock:continue without memory"
    assert chat_context["arun"] == 1
    assert elapsed < 0.8


def test_stream_success_is_replayable(client, chat_context):
    headers = {"Idempotency-Key": "stream-1"}
    payload = {"message": "stream me", "stream": True, "user_id": "user-stream"}
    first = client.post("/api/chat", json=payload, headers=headers)
    second = client.post("/api/chat", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert "text/event-stream" in first.headers["content-type"]
    assert "text/event-stream" in second.headers["content-type"]
    assert second.headers["X-Idempotency-Replayed"] == "true"
    assert second.text == first.text
    assert "[DONE]" in first.text
    assert chat_context["stream"] == 1


def test_stream_done_is_not_blocked_by_slow_persistence(client, chat_context, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def blocked_persist(session_id):
        started.set()
        release.wait(timeout=2)
        return True

    monkeypatch.setattr(app_prod.session_manager, "persist_session", blocked_persist)
    monkeypatch.setattr(app_prod, "STREAM_PERSIST_TIMEOUT_SECONDS", 0.2)

    started_at = time.monotonic()
    response = client.post(
        "/api/chat",
        json={"message": "stream despite db", "stream": True},
    )
    elapsed = time.monotonic() - started_at

    assert response.status_code == 200
    assert "[DONE]" in response.text
    # The client receives the terminal frame without waiting for persistence.
    assert elapsed < 1.0
    release.set()
    started.wait(timeout=1)


def test_stream_failure_releases_key_for_retry(client, chat_context, monkeypatch):
    calls = {"count": 0}

    async def failing_stream(message, *, session, ctx, max_tool_calls=50, session_id="", llm_idle_timeout=None):
        calls["count"] += 1
        raise RuntimeError("stream failed")
        yield  # pragma: no cover

    monkeypatch.setattr(app_prod.engine, "arun_stream_with_dsh", failing_stream)
    headers = {"Idempotency-Key": "stream-fail"}
    payload = {"message": "retry stream", "stream": True}
    first = client.post("/api/chat", json=payload, headers=headers)
    second = client.post("/api/chat", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert '"stop_reason": "error"' in first.text
    assert calls["count"] == 2


def test_engine_failure_releases_key_for_retry(client, chat_context, monkeypatch):
    calls = {"count": 0}

    async def failing_arun(message, *, session, ctx, max_tool_calls=50, session_id="", llm_idle_timeout=None):
        calls["count"] += 1
        raise RuntimeError("engine failed")

    monkeypatch.setattr(app_prod.engine, "arun_with_dsh", failing_arun)
    headers = {"Idempotency-Key": "engine-fail"}
    payload = {"message": "retry me"}
    first = client.post("/api/chat", json=payload, headers=headers)
    second = client.post("/api/chat", json=payload, headers=headers)

    assert first.status_code == 500
    assert second.status_code == 500
    assert calls["count"] == 2


def test_long_key_is_rejected(client, chat_context):
    response = client.post(
        "/api/chat",
        json={"message": "hello"},
        headers={"Idempotency-Key": "x" * (app_prod.IDEMPOTENCY_KEY_MAX_LENGTH + 1)},
    )
    assert response.status_code == 400
    assert chat_context["arun"] == 0


def test_same_key_isolated_by_user_scope(client, chat_context):
    headers = {"Idempotency-Key": "same-key"}
    first = client.post("/api/chat", json={"message": "hello", "user_id": "a"}, headers=headers)
    second = client.post("/api/chat", json={"message": "hello", "user_id": "b"}, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert "X-Idempotency-Replayed" not in second.headers
    assert chat_context["arun"] == 2


def test_session_persistence_failure_releases_key(client, chat_context, monkeypatch):
    calls = {"count": 0}

    async def arun(message, *, session, ctx, max_tool_calls=50, session_id="", llm_idle_timeout=None):
        calls["count"] += 1
        return {"content": "ok", "stop_reason": "stop", "usage": {}, "model": "test"}

    monkeypatch.setattr(app_prod.engine, "arun_with_dsh", arun)
    monkeypatch.setattr(
        app_prod.session_manager,
        "persist_session",
        lambda session_id: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    headers = {"Idempotency-Key": "persist-fail"}
    payload = {"message": "retry persist"}
    first = client.post("/api/chat", json=payload, headers=headers)
    second = client.post("/api/chat", json=payload, headers=headers)

    assert first.status_code == 500
    assert second.status_code == 500
    assert calls["count"] == 2

