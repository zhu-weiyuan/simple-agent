import tempfile, os, time
import pytest
from my_agent.a2a import A2AClient, A2AMessage, A2AServer, AgentCard, TaskState


class FastAsyncAgent:
    async def arun(self, content):
        await asyncio_sleep(0.02)
        return f"done:{content}"


class SlowSyncAgent:
    def run(self, content):
        time.sleep(0.3)
        return "sync-done"


def asyncio_sleep(seconds):
    import asyncio
    return asyncio.sleep(seconds)


def card():
    return AgentCard(name="test-agent", version="2.1.0")


def wait_state(server, task_id, expected, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = server.get_task(task_id)
        if task and task.state is expected:
            return task
        time.sleep(0.01)
    raise AssertionError(f"expected {expected}, got {server.get_task(task_id)}")


def test_task_store_persists_and_recovers_terminated_tasks():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "a2a.db")
        server = A2AServer(FastAsyncAgent(), card(), task_timeout=1, db_path=db)
        try:
            server.handle_message(A2AMessage(task_id="p1", content="hello"))
            wait_state(server, "p1", TaskState.COMPLETED)
            server2 = A2AServer(FastAsyncAgent(), card(), task_timeout=1, db_path=db)
            try:
                restored = server2.get_task("p1")
                assert restored is not None
                assert restored.state is TaskState.COMPLETED
                assert restored.message.content == "done:hello"
            finally:
                server2.stop()
        finally:
            server.stop()


def test_idempotency_survives_restart_and_conflict_is_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "a2a.db")
        first = A2AServer(FastAsyncAgent(), card(), task_timeout=1, db_path=db)
        try:
            first.handle_message(A2AMessage(task_id="restart-idem", content="hello"))
            wait_state(first, "restart-idem", TaskState.COMPLETED)
        finally:
            first.stop()

        second = A2AServer(FastAsyncAgent(), card(), task_timeout=1, db_path=db)
        try:
            replay = second.handle_message(A2AMessage(task_id="restart-idem", content="hello"))
            assert replay.state is TaskState.COMPLETED
            assert replay.message.content == "done:hello"
            with pytest.raises(Exception, match="different content"):
                second.handle_message(A2AMessage(task_id="restart-idem", content="different"))
        finally:
            second.stop()


def test_memory_store_is_shared_across_its_short_lived_connections():
    server = A2AServer(FastAsyncAgent(), card(), task_timeout=1, db_path=":memory:")
    try:
        server.handle_message(A2AMessage(task_id="memory-persist", content="hello"))
        wait_state(server, "memory-persist", TaskState.COMPLETED)
        restored = server.store.get_task("memory-persist")
        assert restored is not None
        assert restored["state"] == TaskState.COMPLETED.value
    finally:
        server.stop()


def test_task_store_marks_inflight_as_timed_out_after_restart():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "a2a.db")
        server = A2AServer(SlowSyncAgent(), card(), task_timeout=5, db_path=db)
        try:
            server.handle_message(A2AMessage(task_id="inflight", content="hello"))
            wait_state(server, "inflight", TaskState.WORKING)
            # Simulate crash: drop in-memory state without completing the task.
            server._tasks.pop("inflight", None)
            server2 = A2AServer(FastAsyncAgent(), card(), task_timeout=1, db_path=db)
            try:
                restored = server2.get_task("inflight")
                assert restored is not None
                assert restored.state is TaskState.TIMED_OUT
                assert "restart" in (restored.error or "")
            finally:
                server2.stop()
        finally:
            server.stop()


def test_list_tasks_and_prune_terminal():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "a2a.db")
        server = A2AServer(FastAsyncAgent(), card(), task_timeout=1,
                           db_path=db, max_retained_tasks=3)
        try:
            for i in range(5):
                server.handle_message(A2AMessage(task_id=f"t{i}", content=f"m{i}"))
            for i in range(5):
                wait_state(server, f"t{i}", TaskState.COMPLETED)
            # The last task's finally block prunes asynchronously; wait for it.
            deadline = time.monotonic() + 2
            items = server.list_tasks(limit=10)
            while len(items) > 3 and time.monotonic() < deadline:
                time.sleep(0.02)
                items = server.list_tasks(limit=10)
            assert len(items) == 3  # pruned down to retention limit
            assert items[0]["taskId"] == "t4"  # newest first
            stats = server.stats()
            assert stats["persisted"]["completed"] == 3
        finally:
            server.stop()


def test_client_resolve_or_resubmit_reuses_existing_task():
    from my_agent.a2a import A2AClient, A2AMessage, A2AServer, AgentCard, TaskState
    import threading
    server = A2AServer(FastAsyncAgent(), card(), host="127.0.0.1", port=0,
                       task_timeout=1, db_path=":memory:")
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while server._http_server is None and time.monotonic() < deadline:
            time.sleep(0.01)
        client = A2AClient(f"http://127.0.0.1:{server._http_server.server_port}",
                           timeout=1, poll_interval=0.01)
        first = client.send_prompt("hello", task_id="dup")
        client.wait_for_task("dup", timeout=2)
        second = client.resolve_or_resubmit("hello", task_id="dup")
        assert second.task_id == "dup"
        assert second.state is TaskState.COMPLETED
        # Unknown task id path: remote doesn't know it, so submit.
        fresh = client.resolve_or_resubmit("new", task_id="brand-new")
        assert fresh.task_id == "brand-new"
        assert fresh.state in {TaskState.SUBMITTED, TaskState.WORKING}
    finally:
        server.stop()
        thread.join(timeout=2)


def test_terminal_callback_runs_once_for_completed_task():
    recorded = []
    server = A2AServer(
        FastAsyncAgent(),
        card(),
        task_timeout=1,
        db_path=":memory:",
        on_terminal=lambda task: recorded.append((task.task_id, task.state)),
    )
    try:
        server.handle_message(A2AMessage(task_id="terminal-callback", content="hello"))
        wait_state(server, "terminal-callback", TaskState.COMPLETED)
        assert recorded == [("terminal-callback", TaskState.COMPLETED)]
    finally:
        server.stop()


class ErrorResultAgent:
    async def arun(self, content):
        return {
            "content": "LLM request failed",
            "stop_reason": "llm_error",
            "stop_detail": "HTTP 400",
        }


def test_llm_error_result_is_failed_not_completed():
    server = A2AServer(ErrorResultAgent(), card(), task_timeout=1, db_path=":memory:")
    try:
        server.handle_message(A2AMessage(task_id="llm-error-result", content="hello"))
        failed = wait_state(server, "llm-error-result", TaskState.FAILED)
        assert failed.error == "HTTP 400"
        assert failed.message.content == "LLM request failed"
    finally:
        server.stop()


def test_legacy_llm_error_record_is_repaired_to_failed():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "a2a.db")
        import json
        payload = {
            "taskId": "legacy-llm-error",
            "state": "completed",
            "message": {
                "messageId": "m1",
                "taskId": "legacy-llm-error",
                "type": "result",
                "content": str({
                    "content": "LLM ??????????????",
                    "stop_reason": "llm_error",
                    "stop_detail": "HTTP 400",
                }),
                "metadata": {},
            },
            "startedAt": time.time() - 1,
            "completedAt": time.time(),
            "metadata": {},
            "error": None,
        }
        import sqlite3
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE a2a_tasks (task_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
        con.execute("INSERT INTO a2a_tasks VALUES (?, ?, ?, ?, ?, ?)", (
            "legacy-llm-error", "fingerprint", json.dumps(payload, ensure_ascii=False),
            "completed", payload["startedAt"], time.time(),
        ))
        con.commit(); con.close()

        server = A2AServer(FastAsyncAgent(), card(), task_timeout=1, db_path=db)
        try:
            restored = server.get_task("legacy-llm-error")
            assert restored is not None
            assert restored.state is TaskState.FAILED
            assert restored.error == "HTTP 400"
            listed = server.list_tasks(limit=10)
            assert listed[0]["state"] == TaskState.FAILED.value
            assert server.store.stats()[TaskState.FAILED.value] == 1
        finally:
            server.stop()


class BudgetStopAgent:
    async def arun(self, content):
        return {"content": "???", "stop_reason": "budget_exceeded", "stop_detail": "????"}


def test_non_completed_engine_stop_reason_is_failed():
    server = A2AServer(BudgetStopAgent(), card(), task_timeout=1, db_path=":memory:")
    try:
        server.handle_message(A2AMessage(task_id="budget-stop", content="hello"))
        failed = wait_state(server, "budget-stop", TaskState.FAILED)
        assert failed.error == "????"
        assert failed.message.content == "???"
    finally:
        server.stop()


def test_async_agent_generator_is_closed_before_worker_loop_is_disposed():
    cleaned = []

    class GeneratorAgent:
        async def arun(self, content):
            async def provider_stream():
                try:
                    yield "first"
                finally:
                    cleaned.append("closed")

            stream = provider_stream()
            await stream.__anext__()
            # Deliberately leave the provider stream open.  A2A owns the
            # isolated loop and must drain async-generator finalizers.
            return "done"

    server = A2AServer(GeneratorAgent(), card(), task_timeout=1, db_path=":memory:")
    try:
        server.handle_message(A2AMessage(task_id="async-generator-cleanup", content="hello"))
        wait_state(server, "async-generator-cleanup", TaskState.COMPLETED)
        assert cleaned == ["closed"]
    finally:
        server.stop()
