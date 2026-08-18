from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from my_agent.artifact_store import ArtifactStore
from my_agent.jobs import JobManager
from my_agent.session_events import SessionEventLog


def test_artifact_retention_removes_expired_and_reports_stats(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts", max_inline_bytes=10)
    old = store.save_text("old-content", "session-old", "tool")
    fresh = store.save_text("fresh-content", "session-fresh", "tool")
    index = Path(store._index_dir) / f"{old.artifact_id}.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["created_at"] = time.time() - 3600
    index.write_text(json.dumps(payload), encoding="utf-8")

    report = store.cleanup(max_age_seconds=60)

    assert report["expired_removed"] == 1
    assert report["removed_bytes"] == old.size_bytes
    assert store.describe(fresh.artifact_id, "session-fresh")["artifact_id"] == fresh.artifact_id
    assert report["remaining"]["count"] == 1


def test_event_log_pagination_corruption_and_age_cleanup(tmp_path: Path):
    log = SessionEventLog(tmp_path / "events")
    for i in range(4):
        log.append("session", "step", {"i": i}, trace_id=f"trace-{i}")
    path = tmp_path / "events" / "session.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
    page = log.list_page("session", limit=2)
    assert [row["payload"]["i"] for row in page["events"]] == [3, 2]
    assert page["next_before_seq"] == 3
    assert page["malformed_lines"] == 1

    old_path = tmp_path / "events" / "old-session.jsonl"
    log.append("old-session", "old", {})
    old_path = tmp_path / "events" / "old-session.jsonl"
    os.utime(old_path, (time.time() - 3600, time.time() - 3600))
    report = log.cleanup(max_age_seconds=60)
    assert report["expired_removed"] == 1
    assert not old_path.exists()


def test_job_retention_only_removes_terminal_rows(tmp_path: Path):
    manager = JobManager(tmp_path / "jobs")
    now = time.time()
    output = manager.output_dir / "job-old.out"
    output.write_text("old output", encoding="utf-8")
    with manager._db() as db:
        db.execute(
            """INSERT INTO jobs(
                job_id, session_id, tool_name, command_json, cwd, status,
                created_at, started_at, finished_at, returncode, timeout_seconds,
                cancel_requested, error, output_path, owner_pid, process_pid,
                finish_reason, output_bytes, output_truncated
            ) VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?, 0, 5, 0, '', ?, 0, NULL, '', ?, 0)""",
            ("job-old", "session", "test", "[]", str(tmp_path), now - 3600,
             now - 3600, now - 3600, str(output), len("old output")),
        )
        db.execute(
            """INSERT INTO jobs(
                job_id, session_id, tool_name, command_json, cwd, status,
                created_at, timeout_seconds, cancel_requested, error, output_path,
                owner_pid, finish_reason, output_bytes, output_truncated
            ) VALUES (?, ?, ?, ?, ?, 'running', ?, 5, 0, '', ?, 0, '', 0, 0)""",
            ("job-live", "session", "test", "[]", str(tmp_path), now, str(manager.output_dir / "job-live.out")),
        )

    report = manager.cleanup(max_age_seconds=60)
    assert report["expired_removed"] == 1
    assert manager.get("job-old") is None
    assert manager.get("job-live")["status"] == "running"
    assert not output.exists()

