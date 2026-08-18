from pathlib import Path

from my_agent.jobs import JobManager


def test_queued_job_cancel_is_terminal(tmp_path):
    manager = JobManager(tmp_path / "jobs")
    row = manager.submit_process(["python", "-c", "print('ok')"], Path.cwd(), session_id="s", timeout_seconds=5)
    cancelled = manager.cancel(row["job_id"], session_id="s")
    assert cancelled["status"] in {"cancelled", "cancel_requested", "completed"}
