# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

from my_agent.tools.builtins.file import (
    FileInfoTool, GitDiffTool, GitStatusTool, ListFilesTool,
    ReadFileRangeTool, ReadFileTool, ReadJsonTool, RunTestsTool, SearchFilesTool, SearchTextTool,
)

def test_workspace_boundary_and_sensitive_file(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "ok.txt").write_text("hello\nworld\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=bad\n", encoding="utf-8")
    assert ReadFileTool().execute({"path": "ok.txt"}) == "hello\nworld\n"
    assert "\u5b89\u5168\u9650\u5236" in ReadFileTool().execute({"path": ".env"})
    assert "\u5b89\u5168\u9650\u5236" in ReadFileTool().execute({"path": str(tmp_path.parent / "outside.txt")})

def test_search_and_range_and_info(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.py").write_text("x", encoding="utf-8")
    assert "a.py" in SearchFilesTool().execute({"pattern": "*.py"})
    assert "2: two" in ReadFileRangeTool().execute({"path": "a.py", "start_line": 2, "end_line": 2})
    info = json.loads(FileInfoTool().execute({"path": "a.py"}))
    assert info["type"] == "file" and info["size_bytes"] > 0

def test_list_hides_sensitive_file_names(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / '.env.example').write_text('TOKEN=placeholder\n', encoding='utf-8')
    (tmp_path / "id_ed25519").write_text("private-key", encoding="utf-8")
    (tmp_path / "README.md").write_text("safe", encoding="utf-8")

    result = ListFilesTool().execute({"path": "."})

    assert "README.md" in result
    assert ".env" not in result
    assert '.env.example' not in result
    assert "id_ed25519" not in result


def test_list_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    for i in range(205):
        (tmp_path / f"{i:03}.txt").write_text("x", encoding="utf-8")
    result = ListFilesTool().execute({"path": "."})
    assert "\u5df2\u622a\u65ad" in result

def test_git_tools_and_test_runner(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(root))
    status = GitStatusTool().execute({"path": "."})
    assert "##" in status or "\u5206\u652f" in status
    assert isinstance(GitDiffTool().execute({"path": "."}), str)
    result = RunTestsTool().execute({"path": ".", "runner": "pytest", "target": "tests/test_eval_datasets.py", "timeout_seconds": 30})
    assert "pytest" in result


def test_search_text_and_read_json(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "notes.txt").write_text("Alpha\nneedle value\n", encoding="utf-8")
    (tmp_path / "data.json").write_text("{\"name\": \"\u6d4b\u8bd5\", \"ok\": true}", encoding="utf-8")
    (tmp_path / "data.jsonl").write_text("{\"n\": 1}\n{\"n\": 2}\n", encoding="utf-8")
    assert "notes.txt:2" in SearchTextTool().execute({"query": "needle"})
    rendered = ReadJsonTool().execute({"path": "data.json"})
    assert "\u6d4b\u8bd5" in rendered and '"ok": true' in rendered
    assert "\"n\": 2" in ReadJsonTool().execute({"path": "data.jsonl"})
    assert "JSON \u89e3\u6790\u5931\u8d25" in ReadJsonTool().execute({"path": "notes.txt"})


def test_read_json_handles_utf8_bom(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "bom.jsonl").write_bytes(b"\xef\xbb\xbf{\"ok\": true}\n")
    rendered = ReadJsonTool().execute({"path": "bom.jsonl"})
    assert '"ok": true' in rendered
    assert "JSON " + "\u89e3\u6790\u5931\u8d25" not in rendered


def test_run_tests_ruff_uses_check_subcommand(monkeypatch, tmp_path):
    from my_agent.tools.builtins.file import RunTestsTool

    observed = {}

    class _Result:
        returncode = 0
        stdout = "All checks passed!"
        stderr = ""

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        return _Result()

    monkeypatch.setattr("my_agent.tools.builtins.file._resolve_path", lambda value: (tmp_path, None))
    monkeypatch.setattr("my_agent.tools.builtins.file.subprocess.run", fake_run)
    result = RunTestsTool().execute({"path": ".", "runner": "ruff", "target": "src/example.py", "timeout_seconds": 30})

    assert observed["command"][-3:-1] == ["ruff", "check"]
    assert observed["command"][-1].replace("\\", "/") == "src/example.py"
    assert result.startswith("ruff ")

