# -*- coding: utf-8 -*-
"""HTTP contract tests for server-enforced structured output."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app_prod
from my_agent.idempotency import IdempotencyStore
from my_agent.memory.sqlite_store import SqliteConversationStore
from my_agent.session_manager import SessionManager


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "count"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "failed"]},
        "count": {"type": "integer", "minimum": 0},
    },
}


@pytest.fixture
def structured_context(monkeypatch, tmp_path):
    monkeypatch.setenv("DEV_AUTH_BYPASS", "1")
    class FakeAsyncLLM:
        async def aclose(self):
            return None

    monkeypatch.setattr(app_prod, "async_llm", FakeAsyncLLM())
    monkeypatch.setattr(app_prod, "user_memory", SimpleNamespace(build_background_block=lambda *_: ""))
    monkeypatch.setattr(app_prod, "_schedule_memory_extraction", lambda *_: None)
    db_path = str(tmp_path / "structured.db")
    monkeypatch.setattr(app_prod, "idempotency_store", IdempotencyStore(db_path))
    monkeypatch.setattr(app_prod, "session_manager", SessionManager(
        system_prompt=app_prod.compose_system_prompt(app_prod.SYSTEM_PROMPT),
        store=SqliteConversationStore(db_path), max_sessions=10,
    ))
    captured = {}

    async def arun(message, *, session, ctx, max_tool_calls=200, session_id="", llm_idle_timeout=None):
        captured["session"] = session
        captured["ctx"] = ctx
        return {"content": "{\"status\":\"ok\",\"count\":2}", "stop_reason": "completed", "usage": {}, "model": "mock"}

    monkeypatch.setattr(app_prod.engine, "arun_with_dsh", arun)
    return captured


@pytest.fixture
def client(structured_context):
    with TestClient(app_prod.app) as test_client:
        yield test_client


def test_valid_output_returns_decoded_structured_value(client, structured_context):
    response = client.post("/api/chat", json={"message": "status", "response_schema": SCHEMA})
    assert response.status_code == 200
    body = response.json()
    assert body["structured_output"] == {"status": "ok", "count": 2}
    assert structured_context["ctx"].metadata["tools_enabled"] is False
    assert len(structured_context["session"].messages) == 1  # mock arun does not append; isolation is checked through the fresh system prompt
    assert "Do not call tools" in structured_context["session"].messages[0].content


def test_invalid_model_json_is_not_reported_as_success(client, monkeypatch):
    async def arun(message, *, session, ctx, max_tool_calls=200, session_id="", llm_idle_timeout=None):
        return {"content": "not json", "stop_reason": "completed", "usage": {}, "model": "mock"}

    monkeypatch.setattr(app_prod.engine, "arun_with_dsh", arun)
    response = client.post("/api/chat", json={"message": "status", "response_schema": SCHEMA})
    assert response.status_code == 422
    assert response.json()["error"] == "structured_output_invalid"


def test_schema_mismatch_is_not_reported_as_success(client, monkeypatch):
    async def arun(message, *, session, ctx, max_tool_calls=200, session_id="", llm_idle_timeout=None):
        return {"content": "{\"status\":\"unknown\"}", "stop_reason": "completed", "usage": {}, "model": "mock"}

    monkeypatch.setattr(app_prod.engine, "arun_with_dsh", arun)
    response = client.post("/api/chat", json={"message": "status", "response_schema": SCHEMA})
    assert response.status_code == 422
    assert "schema validation failed" in response.json()["detail"]


def test_invalid_contract_is_rejected_before_model_call(client, structured_context):
    response = client.post("/api/chat", json={"message": "status", "response_schema": {"type": "not-a-type"}})
    assert response.status_code == 422
    assert "invalid response_schema" in response.json()["detail"]
    assert structured_context == {}


def test_stream_plus_contract_is_rejected(client):
    response = client.post("/api/chat", json={"message": "status", "stream": True, "response_schema": SCHEMA})
    assert response.status_code == 400


def test_idempotency_hash_includes_response_schema():
    one = app_prod.ChatRequest(message="same", response_schema={"type": "object"})
    two = app_prod.ChatRequest(message="same", response_schema={"type": "array"})
    assert app_prod._idempotency_request_hash(one, "same") != app_prod._idempotency_request_hash(two, "same")
