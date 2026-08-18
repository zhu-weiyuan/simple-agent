from pathlib import Path

from my_agent.bridge.permissions import PermissionLevel, PermissionPolicy
from my_agent.file_observation import FileObservationStore
from my_agent.tools.builtins.file import EditFileTool, ReadFileTool, WriteFileTool


def test_edit_requires_same_session_observation(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    target = tmp_path / "note.txt"
    target.write_text("before", encoding="utf-8")
    store = FileObservationStore()
    editor = EditFileTool(store)
    blocked = editor.execute({"path": "note.txt", "old_string": "before", "new_string": "after", "_session_id": "s1"})
    assert "必须在当前会话中先读取" in blocked

    reader = ReadFileTool(store)
    assert reader.execute({"path": "note.txt", "_session_id": "s1"}) == "before"
    assert "已编辑文件" in editor.execute({"path": "note.txt", "old_string": "before", "new_string": "after", "_session_id": "s1"})
    assert target.read_text(encoding="utf-8") == "after"


def test_edit_rejects_stale_observation(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    target = tmp_path / "note.txt"
    target.write_text("before", encoding="utf-8")
    store = FileObservationStore()
    ReadFileTool(store).execute({"path": "note.txt", "_session_id": "s1"})
    target.write_text("external change", encoding="utf-8")
    result = EditFileTool(store).execute({"path": "note.txt", "old_string": "before", "new_string": "after", "_session_id": "s1"})
    assert "已变化" in result


def test_write_allows_explicit_create_only(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    writer = WriteFileTool(FileObservationStore())
    assert "必须在当前会话中先读取" in writer.execute({"path": "new.txt", "content": "x", "_session_id": "s1"})
    result = writer.execute({"path": "new.txt", "content": "x", "create_if_absent": True, "_session_id": "s1"})
    assert "已原子写入文件" in result
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "x"


def test_permission_policy_override_and_pattern():
    policy = PermissionPolicy()
    assert policy.decision("read_file", "allow")[0]
    assert not policy.decision("write_file", "ask")[0]
    policy.allow_patterns.append("write_*")
    assert policy.decision("write_file", "ask")[0]
    policy.set("write_file", PermissionLevel.DENY)
    assert not policy.decision("write_file", "ask")[0]
