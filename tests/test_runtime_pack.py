# -*- coding: utf-8 -*-
"""Regression coverage for retained tool output, events and cancellable subprocess jobs."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from my_agent.artifact_store import ArtifactStore
from my_agent.core.engine import QueryEngine
from my_agent.jobs import JobManager
from my_agent.session_events import SessionEventLog
from my_agent.tool_protocol import classify_tool_error
from my_agent.tools.registry import ToolRegistry
from my_agent.types.message import ToolCall


def test_artifact_store_retains_large_result_and_enforces_session_boundary(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts", max_inline_bytes=120, max_artifact_bytes=1000)
    original = "A" * 300 + "needle" + "B" * 300
    preview, record = store.retain(original, "session-a", "list_files")
    assert record is not None
    assert record.artifact_id in preview
    assert len(preview.encode("utf-8")) <= store.max_inline_bytes
    assert store.read(record.artifact_id, session_id="session-a")["content"] == original
    assert store.search(record.artifact_id, "needle", session_id="session-a")["count"] == 1
    try:
        store.read(record.artifact_id, session_id="session-b")
    except PermissionError:
        pass
    else:
        raise AssertionError("cross-session artifact access must be denied")


def test_engine_spills_large_tool_result_and_records_structured_events(tmp_path: Path):
    registry = ToolRegistry()
    registry.add("large_read", lambda _params: "x" * 1500, parameters={"type": "object", "properties": {}})
    store = ArtifactStore(tmp_path / "artifacts", max_inline_bytes=100)
    events = SessionEventLog(tmp_path / "events")
    engine = QueryEngine("test", tool_registry=registry, artifact_store=store, event_log=events)
    result = engine._execute_tool(ToolCall(id="call-1", name="large_read", arguments={}), session_id="s1", trace_id="trace-1")
    assert "artifact_id" not in result  # identifier is rendered as a value, not a JSON contract
    assert "art-" in result
    records = events.list("s1")
    assert records[-1]["type"] == "tool.completed"
    assert any(event["type"] == "tool.result_spilled" for event in records)


def test_tool_error_protocol_keeps_deterministic_errors_out_of_retry_circuit_logic():
    validation = classify_tool_error("错误:参数验证失败 [read_file]: path is required")
    assert validation.code == "INVALID_ARGUMENT"
    assert not validation.retryable and not validation.circuit_trackable
    timeout = classify_tool_error("错误:工具执行超时 [read_file]: 超过 1.0 秒")
    assert timeout.code == "TOOL_TIMEOUT"
    assert timeout.retryable and timeout.circuit_trackable


def _wait_for(manager: JobManager, job_id: str, expected: set[str], seconds: float = 5) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        row = manager.get(job_id)
        if row and row["status"] in expected:
            return row
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach {expected}")


def test_second_job_manager_keeps_jobs_owned_by_a_live_supervisor(tmp_path: Path):
    root = tmp_path / "jobs"
    primary = JobManager(root)
    job = primary.submit_process(
        [sys.executable, "-c", "import time; time.sleep(0.5)"],
        tmp_path,
        session_id="s1",
        tool_name="test",
        timeout_seconds=5,
    )

    # The app may create another registry (for legacy endpoints) in the same
    # process.  That registry must not turn an actively supervised job into a
    # false "interrupted" restart remnant.
    observer = JobManager(root)
    initial = observer.get(job["job_id"], "s1")
    assert initial is not None
    assert initial["status"] in {"queued", "running"}
    assert initial["owner_pid"] == os.getpid()

    completed = _wait_for(primary, job["job_id"], {"completed", "failed"})
    assert completed["status"] == "completed"


def test_job_manager_collects_output_and_cancels_process(tmp_path: Path):
    manager = JobManager(tmp_path / "jobs")
    completed = manager.submit_process(
        [sys.executable, "-c", "print('runtime-pack-ok')"], tmp_path, session_id="s1", tool_name="test", timeout_seconds=5,
    )
    row = _wait_for(manager, completed["job_id"], {"completed", "failed"})
    assert row["status"] == "completed"
    assert "runtime-pack-ok" in manager.read_output(completed["job_id"], "s1")["content"]

    running = manager.submit_process(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"], tmp_path, session_id="s1", tool_name="test", timeout_seconds=20,
    )
    _wait_for(manager, running["job_id"], {"running", "completed"})
    manager.cancel(running["job_id"], "s1")
    row = _wait_for(manager, running["job_id"], {"cancelled", "failed", "timed_out"})
    assert row["status"] == "cancelled"



def test_job_timeout_stops_a_silent_process_without_waiting_for_stdout(tmp_path: Path):
    manager = JobManager(tmp_path / "jobs")
    started = time.monotonic()
    job = manager.submit_process(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path,
        session_id="s1",
        tool_name="test",
        timeout_seconds=1,
    )
    row = _wait_for(manager, job["job_id"], {"timed_out", "failed"}, seconds=4)
    elapsed = time.monotonic() - started
    assert row["status"] == "timed_out"
    assert row["finish_reason"] == "deadline_exceeded"
    # A blocked stdout.readline() used to make this wait for the full 10 seconds.
    assert elapsed < 3.5


def test_job_records_distinct_completion_and_cancellation_reasons(tmp_path: Path):
    manager = JobManager(tmp_path / "jobs")
    completed = manager.submit_process(
        [sys.executable, "-c", "print('done')"],
        tmp_path,
        session_id="s1",
        tool_name="test",
        timeout_seconds=5,
    )
    complete_row = _wait_for(manager, completed["job_id"], {"completed", "failed"})
    assert complete_row["status"] == "completed"
    assert complete_row["finish_reason"] == "completed"

    running = manager.submit_process(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path,
        session_id="s1",
        tool_name="test",
        timeout_seconds=20,
    )
    _wait_for(manager, running["job_id"], {"running", "completed"})
    manager.cancel(running["job_id"], "s1")
    cancelled = _wait_for(manager, running["job_id"], {"cancelled", "failed", "timed_out"})
    assert cancelled["status"] == "cancelled"
    assert cancelled["finish_reason"] == "user_cancelled"



def test_windows_termination_requests_tree_cleanup_without_starting_real_processes(monkeypatch):
    """Verify the Windows command shape without executing taskkill in the test host."""
    import my_agent.jobs as jobs_module

    calls = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(timeout=None):
            return 1

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(jobs_module.os, "name", "nt")
    monkeypatch.setattr(jobs_module.subprocess, "run", fake_run)
    JobManager._terminate(FakeProcess())

    assert calls
    assert calls[0][0] == ["taskkill", "/PID", "4321", "/T", "/F"]
    assert calls[0][1]["check"] is False



def test_job_output_is_bounded_but_pipe_is_still_drained (tmp_path: Path):
    manager = JobManager(tmp_path / "jobs", max_output_bytes=1_024)
    job = manager.submit_process(
        [sys.executable, "-c", "print('x' * 20_000)"],
        tmp_path,
        session_id="s-output",
        tool_name="test",
        timeout_seconds=5,
    )
    row = _wait_for(manager, job["job_id"], {"completed", "failed"})
    assert row["status"] == "completed"
    assert row["output_truncated"] == 1
    assert 0 < row["output_bytes"] <= 1_024
    output = manager.read_output(job["job_id"], "s-output")
    assert "输出已达到保留上限" in output["content"]
    assert output["output_truncated"] is True
    assert output["output_bytes"] == row["output_bytes"]
    assert output["finish_reason"] == "completed"
