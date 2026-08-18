"""Focused regression tests for externally reported correctness and safety bugs."""
from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from my_agent.core.context_assembler import _message_tokens, fit_messages_to_budget
from my_agent.core.engine import _explicit_workspace_file
from my_agent.graph.graph import Graph
from my_agent.graph.node import node
from my_agent.graph.state import GraphState
from my_agent.mcp_client import MCPClient, MCPError
from my_agent.memory.store import MemoryStore
from my_agent.tools.builtins.shell import PowerShellTool
from my_agent.tools.registry import ToolRegistry
from my_agent.types.message import Message


@node("start")
def _checkpoint_test_node(state: GraphState) -> GraphState:
    state.metadata["node_ran"] = True
    return state


def test_checkpoint_restores_matching_run_id_only(tmp_path: Path) -> None:
    graph = Graph({"start": _checkpoint_test_node}).enable_checkpointer(str(tmp_path))
    original = GraphState(run_id="run-abc123")
    original.metadata["checkpoint_marker"] = "saved-for-this-run"
    graph._save_checkpoint("start", original)

    resumed = GraphState(run_id="run-abc123")
    result = graph.run(resumed)
    assert result.metadata["checkpoint_marker"] == "saved-for-this-run"
    assert result.metadata["node_ran"] is True

    other_run = GraphState(run_id="run-other")
    graph.run(other_run)
    assert "checkpoint_marker" not in other_run.metadata


def test_checkpoint_rejects_unsafe_run_id(tmp_path: Path) -> None:
    graph = Graph({"start": _checkpoint_test_node}).enable_checkpointer(str(tmp_path))
    assert graph._load_checkpoint("start", "../escape") is None


def test_memory_store_concurrent_duplicate_append_is_single_record(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: store.append_lesson("并发唯一经验"), range(40)))

    assert sum(results) == 1
    assert store.load_lessons().count("并发唯一经验") == 1


def test_memory_store_concurrent_distinct_append_and_missing_final_newline(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.lessons_path.write_text("# lessons.md", encoding="utf-8")
    values = [f"经验-{index}" for index in range(24)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        assert all(pool.map(store.append_lesson, values))

    text = store.lessons_path.read_text(encoding="utf-8")
    assert text.startswith("# lessons.md\n- 经验-0")
    assert text.endswith("\n")
    assert set(values).issubset(set(store.load_lessons()))


class _QueueStdout:
    def __init__(self) -> None:
        self.lines: queue.Queue[object] = queue.Queue()

    def readline(self) -> str:
        item = self.lines.get(timeout=5)
        return "" if item is None else str(item)

    def close(self) -> None:
        self.lines.put(None)


class _RespondingStdin:
    def __init__(self, stdout: _QueueStdout) -> None:
        self.stdout = stdout
        self.writes: list[str] = []
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        request = json.loads(text)
        with self._lock:
            self.writes.append(text)
        self.stdout.lines.put(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"id": request["id"]}}) + "\n")
        return len(text)

    def flush(self) -> None:
        return None


class _SilentStdin:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, stdin, stdout: _QueueStdout | None = None) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        if self.stdout:
            self.stdout.close()

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.terminate()


def test_mcp_client_concurrent_requests_get_unique_ids_and_whole_json_lines() -> None:
    stdout = _QueueStdout()
    stdin = _RespondingStdin(stdout)
    client = MCPClient(["fake"])
    client.process = _FakeProcess(stdin, stdout)  # type: ignore[assignment]
    client._running = True
    client._reader_thread = threading.Thread(target=client._read_loop, daemon=True)
    client._reader_thread.start()

    with ThreadPoolExecutor(max_workers=12) as pool:
        responses = list(pool.map(lambda _: client._send_request("echo", {}), range(30)))

    client.stop()
    ids = [json.loads(line)["id"] for line in stdin.writes]
    assert sorted(ids) == list(range(1, 31))
    assert all(response["id"] in ids for response in responses)
    assert client._pending_requests == {}


def test_mcp_client_stop_wakes_pending_request_without_waiting_for_timeout() -> None:
    client = MCPClient(["fake"])
    client.process = _FakeProcess(_SilentStdin())  # type: ignore[assignment]
    client._running = True
    outcome: list[BaseException] = []

    def request() -> None:
        try:
            client._send_request("will-block", {})
        except BaseException as exc:  # assertion below checks exact public error type
            outcome.append(exc)

    worker = threading.Thread(target=request)
    worker.start()
    deadline = time.monotonic() + 1
    while not client._pending_requests and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client._pending_requests

    started = time.monotonic()
    client.stop()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert time.monotonic() - started < 1
    assert len(outcome) == 1 and isinstance(outcome[0], MCPError)
    assert "已停止" in str(outcome[0])


def test_registry_keeps_explicit_dynamic_properties_opt_in() -> None:
    registry = ToolRegistry()
    registry.add(
        name="dynamic",
        handler=lambda _: "ok",
        parameters={"type": "object", "additionalProperties": True},
    )
    definition = registry.get_definition("dynamic")
    assert definition is not None
    assert definition.parameters["additionalProperties"] is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("请读取 src/my_agent/tools/builtins/file.py", "src/my_agent/tools/builtins/file.py"),
        ("see https://example.com/path/to/file.py", None),
        ("see https://example.com/path/to/file.py?raw=1", None),
        ("读取 ../secrets.txt", None),
    ],
)
def test_explicit_workspace_file_rejects_url_and_traversal(text: str, expected: str | None) -> None:
    assert _explicit_workspace_file(text) == expected


def test_shell_tool_rejects_bypass_variants_without_starting_subprocess(monkeypatch) -> None:
    tool = PowerShellTool()
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsafe command reached subprocess")

    monkeypatch.setattr("my_agent.tools.builtins.shell.subprocess.run", should_not_run)
    for command in (
        "Remove-Item victim.txt",
        "Remove -Item victim.txt",
        "ri victim.txt",
        "Get-Process; Remove-Item victim.txt",
        "Get-Date | Invoke-Expression",
        "Get-Content .env",
    ):
        assert "拒绝" in tool.execute({"command": command})
    assert called is False


def test_shell_tool_allows_only_safe_diagnostics(monkeypatch) -> None:
    tool = PowerShellTool()
    monkeypatch.setattr(
        "my_agent.tools.builtins.shell.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=b"safe output", stderr=b""),
    )
    assert tool.execute({"command": "Get-Date"}) == "safe output"


def test_fit_messages_drops_oversized_summary_before_current_turn() -> None:
    messages = [
        Message.system("canonical safety policy"),
        Message.summary_boundary("history " * 2000),
        Message.user("current question"),
    ]
    fitted = fit_messages_to_budget(messages, context_window=1000, target_ratio=0.2)
    assert fitted[0].content == "canonical safety policy"
    assert fitted[-1].content == "current question"
    assert not any(message.metadata.get("summary_boundary") for message in fitted)
    assert sum(_message_tokens(message) for message in fitted) <= 200


def test_enhanced_initialization_rolls_back_all_attributes(monkeypatch) -> None:
    import my_agent.agent as agent_module

    agent = agent_module.SimpleAgent.__new__(agent_module.SimpleAgent)
    agent.enable_enhanced = True
    agent._router = agent._persona_memory = agent._persona_extractor = None
    agent._category_rag = agent._hallucination_detector = None
    agent._citation_system = agent._multi_index = None
    agent._debug = lambda _message: None

    monkeypatch.setattr(agent_module, "DynamicRouter", lambda: object())
    monkeypatch.setattr(agent_module, "PersonaMemory", lambda: object())
    monkeypatch.setattr(agent_module, "PersonaExtractor", lambda: object())
    monkeypatch.setattr(agent_module, "CategoryRAG", lambda _memory: object())
    monkeypatch.setattr(agent_module, "HallucinationDetector", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    agent._init_enhanced_modules()

    assert agent.enable_enhanced is False
    assert all(
        getattr(agent, name) is None
        for name in (
            "_router", "_persona_memory", "_persona_extractor", "_category_rag",
            "_hallucination_detector", "_citation_system", "_multi_index",
        )
    )
