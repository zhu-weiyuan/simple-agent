# -*- coding: utf-8 -*-
"""HTTP integration coverage for the runtime artifact/event/job endpoints."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app_prod
from my_agent.artifact_store import ArtifactStore
from my_agent.jobs import JobManager
from my_agent.memory.sqlite_store import SqliteConversationStore
from my_agent.session_events import SessionEventLog


def _wait_for_status(manager: JobManager, job_id: str, expected: set[str], timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = manager.get(job_id)
        if row and row["status"] in expected:
            return row
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach {expected}")


@pytest.fixture
def runtime_api(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DEV_AUTH_BYPASS", "1")
    class FakeAsyncLLM:
        async def aclose(self):
            return None

    monkeypatch.setattr(app_prod, "async_llm", FakeAsyncLLM())
    monkeypatch.setattr(app_prod, "_schedule_memory_extraction", lambda *args: None)

    conversation_store = SqliteConversationStore(str(tmp_path / "conversations.db"))
    artifact_store = ArtifactStore(tmp_path / "artifacts", max_inline_bytes=512)
    event_log = SessionEventLog(tmp_path / "events")
    job_manager = JobManager(tmp_path / "jobs")
    monkeypatch.setattr(app_prod, "_sqlite_store", conversation_store)
    monkeypatch.setattr(app_prod, "artifact_store", artifact_store)
    monkeypatch.setattr(app_prod, "session_event_log", event_log)
    monkeypatch.setattr(app_prod, "job_manager", job_manager)

    session_id = "api-session-a"
    conversation_store.set_session_user(session_id, "user-a")
    event_log.append(session_id, "query.started", {"message": "hello"}, trace_id="trace-a")
    _, record = artifact_store.retain("A" * 600 + "\nneedle from runtime API\n" + "B" * 600, session_id, "list_files")
    assert record is not None

    with TestClient(app_prod.app) as client:
        yield client, job_manager, session_id, record.artifact_id

    conversation_store.close()


def test_runtime_artifact_and_event_endpoints_are_session_scoped(runtime_api):
    client, _manager, session_id, artifact_id = runtime_api
    owner_headers = {"X-User-Id": "user-a"}

    metadata = client.get(
        f"/api/artifacts/{artifact_id}",
        params={"session_id": session_id},
        headers=owner_headers,
    )
    assert metadata.status_code == 200
    assert metadata.json()["artifact_id"] == artifact_id

    content = client.get(
        f"/api/artifacts/{artifact_id}/content",
        params={"session_id": session_id, "offset": 0, "limit": 20},
        headers=owner_headers,
    )
    assert content.status_code == 200
    assert content.json()["content"]
    assert content.json()["has_more"] is True

    search = client.get(
        f"/api/artifacts/{artifact_id}/search",
        params={"session_id": session_id, "query": "needle"},
        headers=owner_headers,
    )
    assert search.status_code == 200
    assert search.json()["count"] == 1

    events = client.get(
        f"/api/sessions/{session_id}/events",
        headers=owner_headers,
    )
    assert events.status_code == 200
    assert events.json()["events"][-1]["type"] == "query.started"

    forbidden = client.get(
        f"/api/artifacts/{artifact_id}",
        params={"session_id": session_id},
        headers={"X-User-Id": "user-b"},
    )
    assert forbidden.status_code == 403

    missing = client.get(
        "/api/artifacts/art-00000000000000000000000000000000",
        params={"session_id": session_id},
        headers=owner_headers,
    )
    assert missing.status_code == 404


def test_runtime_job_endpoints_read_output_and_cancel(runtime_api):
    client, manager, session_id, _artifact_id = runtime_api
    owner_headers = {"X-User-Id": "user-a"}
    completed = manager.submit_process(
        [sys.executable, "-c", "print('api-job-ok')"],
        Path.cwd(),
        session_id=session_id,
        tool_name="test",
        timeout_seconds=5,
    )
    job_id = completed["job_id"]
    _wait_for_status(manager, job_id, {"completed", "failed"})

    listed = client.get(
        "/api/jobs", params={"session_id": session_id}, headers=owner_headers
    )
    assert listed.status_code == 200
    assert any(row["job_id"] == job_id for row in listed.json()["jobs"])

    detail = client.get(
        f"/api/jobs/{job_id}", params={"session_id": session_id}, headers=owner_headers
    )
    assert detail.status_code == 200
    assert detail.json()["status"] == "completed"

    output = client.get(
        f"/api/jobs/{job_id}/output",
        params={"session_id": session_id},
        headers=owner_headers,
    )
    assert output.status_code == 200
    assert "api-job-ok" in output.json()["content"]

    running = manager.submit_process(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
        Path.cwd(),
        session_id=session_id,
        tool_name="test",
        timeout_seconds=20,
    )
    running_id = running["job_id"]
    _wait_for_status(manager, running_id, {"running", "completed"})
    cancelled = client.post(
        f"/api/jobs/{running_id}/cancel",
        params={"session_id": session_id},
        headers=owner_headers,
    )
    assert cancelled.status_code == 200
    _wait_for_status(manager, running_id, {"cancelled", "failed", "timed_out"})

    forbidden = client.get(
        f"/api/jobs/{job_id}",
        params={"session_id": session_id},
        headers={"X-User-Id": "user-b"},
    )
    assert forbidden.status_code == 403

    missing = client.get(
        "/api/jobs/job-00000000000000000000000000000000",
        params={"session_id": session_id},
        headers=owner_headers,
    )
    assert missing.status_code == 404
