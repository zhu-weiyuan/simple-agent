import asyncio
import threading
import time

import pytest

from my_agent.a2a import (
    A2AClient,
    A2AConflictError,
    A2AMessage,
    A2AServer,
    AgentCard,
    TaskState,
)


class FastAsyncAgent:
    async def arun(self, content):
        await asyncio.sleep(0.03)
        return f"done:{content}"


class SlowAsyncAgent:
    async def arun(self, content):
        await asyncio.sleep(10)
        return "late"


class FastSyncAgent:
    def run(self, content):
        return f"sync:{content}"


def card():
    return AgentCard(name="test-agent", version="2.1.0")


def wait_state(server, task_id, expected, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = server.get_task(task_id)
        if task and task.state is expected:
            return task
        time.sleep(0.01)
    task = server.get_task(task_id)
    raise AssertionError(f"expected {expected}, got {task}")


def test_a2a_submission_is_non_blocking_and_completes():
    server = A2AServer(FastAsyncAgent(), card(), task_timeout=1)
    started = time.monotonic()
    task = server.handle_message(A2AMessage(task_id="task-fast", content="hello"))
    elapsed = time.monotonic() - started
    try:
        # 0.5s: generous enough for a busy CI runner, still proves the submit
        # path is non-blocking (the old sync implementation blocked for the
        # agent's full runtime, e.g. 10s).
        assert elapsed < 0.5
        assert task.state is TaskState.SUBMITTED
        completed = wait_state(server, "task-fast", TaskState.COMPLETED)
        assert completed.message.content == "done:hello"
    finally:
        server.stop()


def test_a2a_task_id_is_idempotent_and_conflict_is_rejected():
    server = A2AServer(FastSyncAgent(), card(), task_timeout=1)
    try:
        first = server.handle_message(A2AMessage(task_id="same", content="one"))
        replay = server.handle_message(A2AMessage(task_id="same", content="one"))
        assert replay.task_id == first.task_id
        wait_state(server, "same", TaskState.COMPLETED)
        with pytest.raises(A2AConflictError):
            server.handle_message(A2AMessage(task_id="same", content="two"))
    finally:
        server.stop()


def test_a2a_timeout_marks_task_without_waiting_forever():
    server = A2AServer(SlowAsyncAgent(), card(), task_timeout=0.05)
    try:
        server.handle_message(A2AMessage(task_id="task-timeout", content="hello"))
        timed_out = wait_state(server, "task-timeout", TaskState.TIMED_OUT)
        assert "timed out" in (timed_out.error or "")
    finally:
        server.stop()


def test_a2a_cancel_stops_async_agent():
    server = A2AServer(SlowAsyncAgent(), card(), task_timeout=5)
    try:
        server.handle_message(A2AMessage(task_id="task-cancel", content="hello"))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            task = server.get_task("task-cancel")
            if task and task.state is TaskState.WORKING:
                break
            time.sleep(0.01)
        assert server.cancel_task("task-cancel") is True
        cancelled = wait_state(server, "task-cancel", TaskState.CANCELLED)
        assert cancelled.error == "task cancelled"
        assert server.cancel_task("task-cancel") is False
    finally:
        server.stop()


def test_a2a_http_submission_returns_before_agent_finishes():
    server = A2AServer(FastAsyncAgent(), card(), host="127.0.0.1", port=0, task_timeout=1)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while server._http_server is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server._http_server is not None
        endpoint = f"http://127.0.0.1:{server._http_server.server_port}"
        client = A2AClient(endpoint, timeout=1, poll_interval=0.01)
        submitted = client.send_prompt("hello", task_id="http-task")
        assert submitted.task_id == "http-task"
        assert submitted.state in {TaskState.SUBMITTED, TaskState.WORKING}
        completed = client.wait_for_task("http-task", timeout=2)
        assert completed.state is TaskState.COMPLETED
        assert completed.message.content == "done:hello"
    finally:
        server.stop()
        thread.join(timeout=2)
