# -*- coding: utf-8 -*-
"""my_agent.tools.builtins.file — 安全的文件系统只读工具。"""
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from ..base import BaseTool
from ..retention import ItemRetainer, format_text_notice, retain_text
from ...jobs import JobManager
from ...file_observation import FileObservationStore, atomic_write_text

_MAX_LIST_ENTRIES = 200
_MAX_SEARCH_RESULTS = 100
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
_SENSITIVE_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def _workspace_root() -> Optional[Path]:
    """返回可选的工作区边界；未配置时保持旧版兼容行为。"""
    raw = os.environ.get("AGENT_WORKSPACE_ROOT", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _resolve_path(raw: Any, default: str = ".") -> Tuple[Optional[Path], Optional[str]]:
    value = str(raw if raw is not None else default).strip() or default
    root = _workspace_root()
    try:
        candidate = Path(value).expanduser()
        if root and not candidate.is_absolute():
            candidate = root / candidate
        path = candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"\u8def\u5f84\u65e0\u6548:{type(exc).__name__}: {exc}"
    if root and path != root and root not in path.parents:
        return None, f"\u5b89\u5168\u9650\u5236:\u8def\u5f84\u5fc5\u987b\u4f4d\u4e8e\u5de5\u4f5c\u533a {root} \u5185"
    return path, None


def _is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    return name in _SENSITIVE_NAMES or name.startswith(".env.") or name.endswith(_SENSITIVE_SUFFIXES)


def _safe_text(path: Path, max_bytes: int = 50_000) -> Tuple[Optional[str], Optional[str]]:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return None, f"文件过大（{size} bytes），超过{max_bytes // 1000}KB限制"
        return path.read_text(encoding="utf-8", errors="replace"), None
    except OSError as exc:
        return None, f"读取文件失败:{type(exc).__name__}: {exc}"


class ReadFileTool(BaseTool):
    """读取工作区内的小型文本文件。"""

    name = "read_file"
    description = 'Read an already known text file inside the workspace. When the exact path is known, use this tool directly rather than searching first. Sensitive files are denied.'
    parameters = {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}}, "required": ["path"]}
    tags = ["file", "read"]
    permission_level = "ask"

    def __init__(self, observations: Optional[FileObservationStore] = None) -> None:
        self.observations = observations

    def execute(self, params: Dict[str, Any]) -> str:
        path_str = str(params.get("path", "")).strip()
        if not path_str:
            return "错误:未提供文件路径"
        p, error = _resolve_path(path_str)
        if error:
            return error
        assert p is not None
        if _is_sensitive(p):
            return "安全限制:拒绝读取凭据、私钥或环境变量文件"
        if not p.exists():
            return f"文件不存在:{path_str}"
        if not p.is_file():
            return f"不是文件:{path_str}"
        content, error = _safe_text(p)
        if error:
            return error
        assert content is not None
        if self.observations is not None:
            self.observations.observe(str(params.get("_session_id", "default")), p)
        retained, omitted = retain_text(content, 10_000)
        return retained + format_text_notice(
            "文件内容", omitted, "请使用 read_file_range 读取所需行范围。"
        )


class ListFilesTool(BaseTool):
    """列出目录中的文件和文件夹。"""

    name = "list_files"
    description = 'List the direct files and folders in one directory; this is not recursive. Use search_files only for recursive filename matching.'
    parameters = {"type": "object", "properties": {"path": {"type": "string", "description": "目录路径，默认为当前目录"}}}
    tags = ["file", "list"]
    permission_level = "allow"

    def execute(self, params: Dict[str, Any]) -> str:
        path_str = str(params.get("path", ".")).strip() or "."
        p, error = _resolve_path(path_str)
        if error:
            return error
        assert p is not None
        if not p.exists():
            return f"路径不存在:{path_str}"
        if not p.is_dir():
            return f"不是目录:{path_str}"
        try:
            items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            retained = ItemRetainer(_MAX_LIST_ENTRIES)
            for item in items:
                # Listing a credential filename leaks useful reconnaissance
                # information even when ReadFileTool later refuses its body.
                # Keep directory browsing consistent with read/search rules.
                if _is_sensitive(item):
                    continue
                kind = "[DIR]" if item.is_dir() else "[FILE]"
                retained.push(f"{kind} {item.name}")
            result = retained.finish()
            body = "\n".join(result.items) if result.items else "目录为空"
            return f"目录: {p}\n" + body + result.notice(
                "目录结果", "请指定更具体的目录。"
            )
        except OSError as exc:
            return f"列出目录失败:{type(exc).__name__}: {exc}"


class SearchFilesTool(BaseTool):
    """按文件名或通配符在工作区内查找文件。"""

    name = "search_files"
    description = 'Recursively find file names by glob pattern only when the target path is unknown. For a known exact path, use read_file, file_info, or read_json directly.'
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "搜索起始目录，默认当前目录"},
        "pattern": {"type": "string", "description": "文件名通配符，如 *.py"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
    }, "required": ["pattern"]}
    tags = ["file", "search"]
    permission_level = "allow"

    def execute(self, params: Dict[str, Any]) -> str:
        pattern = str(params.get("pattern", "")).strip()
        if not pattern:
            return "错误:未提供搜索模式"
        root, error = _resolve_path(params.get("path", "."))
        if error:
            return error
        assert root is not None
        if not root.is_dir():
            return f"不是目录:{root}"
        try:
            limit = min(max(int(params.get("max_results", _MAX_SEARCH_RESULTS)), 1), _MAX_SEARCH_RESULTS)
        except (TypeError, ValueError):
            limit = _MAX_SEARCH_RESULTS
        found = []
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in files:
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name.lower(), pattern.lower()):
                    found.append(str((Path(current) / name).relative_to(root)))
        retained = ItemRetainer(limit)
        retained.extend(sorted(found))
        result = retained.finish()
        body = "\n".join(result.items) if result.items else "未找到匹配文件"
        return f"搜索结果（保留前{limit}项）:\n" + body + result.notice(
            "文件搜索结果", "请缩小 pattern 或 path。"
        )


class ReadFileRangeTool(BaseTool):
    """按行读取文件，避免把大文件全部送进上下文。"""

    name = "read_file_range"
    description = 'Read a known text-file line range. Use it only when the user explicitly requests line numbers or a small slice of a large file.'
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1},
    }, "required": ["path", "start_line"]}
    tags = ["file", "read", "range"]
    permission_level = "ask"

    def execute(self, params: Dict[str, Any]) -> str:
        p, error = _resolve_path(params.get("path", ""), default="")
        if error:
            return error
        assert p is not None
        if _is_sensitive(p):
            return "安全限制:拒绝读取凭据、私钥或环境变量文件"
        if not p.is_file():
            return f"不是文件:{p}"
        try:
            start = max(1, int(params.get("start_line", 1)))
            end = int(params.get("end_line", start + 199))
        except (TypeError, ValueError):
            return "错误:行号必须是整数"
        if end < start:
            return "错误:end_line 不能小于 start_line"
        end = min(end, start + 299)
        text, error = _safe_text(p, max_bytes=200_000)
        if error:
            return error
        assert text is not None
        lines = text.splitlines()
        selected = lines[start - 1:end]
        if not selected:
            return f"行范围超出文件长度（共{len(lines)}行）"
        return "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start))





class SearchTextTool(BaseTool):
    """在工作区文本文件中搜索内容，只返回行号和截断片段。"""

    name = "search_text"
    description = 'Search inside file contents, not filenames. Use it for keyword lookup; do not search first when the target path is already known.'
    parameters = {"type": "object", "properties": {
        "query": {"type": "string", "description": "要搜索的文本"},
        "path": {"type": "string", "description": "搜索起始目录，默认当前目录"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
        "case_sensitive": {"type": "boolean"},
    }, "required": ["query"]}
    tags = ["file", "search", "text"]
    permission_level = "allow"

    def execute(self, params: Dict[str, Any]) -> str:
        query = str(params.get("query", ""))
        if not query:
            return "错误:未提供搜索文本"
        root, error = _resolve_path(params.get("path", "."))
        if error:
            return error
        assert root is not None
        if not root.is_dir():
            return f"不是目录:{root}"
        try:
            limit = min(max(int(params.get("max_results", _MAX_SEARCH_RESULTS)), 1), _MAX_SEARCH_RESULTS)
        except (TypeError, ValueError):
            limit = _MAX_SEARCH_RESULTS
        case_sensitive = bool(params.get("case_sensitive", False))
        needle = query if case_sensitive else query.casefold()
        matches: list[str] = []
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in files:
                file_path = Path(current) / name
                if _is_sensitive(file_path):
                    continue
                try:
                    if file_path.stat().st_size > 200_000:
                        continue
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for line_no, line in enumerate(text.splitlines(), 1):
                    haystack = line if case_sensitive else line.casefold()
                    if needle in haystack:
                        rel = file_path.relative_to(root)
                        matches.append(f"{rel}:{line_no}: {line[:240]}")
        retained = ItemRetainer(limit)
        retained.extend(matches)
        result = retained.finish()
        body = "\n".join(result.items) if result.items else "未找到匹配内容"
        return f"文本搜索结果（保留前{limit}项）:\n" + body + result.notice(
            "文本搜索结果", "请缩小 query 或 path。"
        )


class ReadJsonTool(BaseTool):
    """读取并格式化工作区内的 JSON 和 JSONL 文件。"""

    name = "read_json"
    description = 'Parse a known JSON or JSONL file as data. Use read_file if the user needs raw text instead.'
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    tags = ["file", "read", "json"]
    permission_level = "ask"

    def execute(self, params: Dict[str, Any]) -> str:
        path_str = str(params.get("path", "")).strip()
        if not path_str:
            return "错误:未提供文件路径"
        path, error = _resolve_path(path_str)
        if error:
            return error
        assert path is not None
        if _is_sensitive(path):
            return "安全限制:拒绝读取凭据、私钥或环境变量文件"
        if not path.is_file():
            return f"不是文件:{path_str}"
        text, error = _safe_text(path, max_bytes=200_000)
        if error:
            return error
        assert text is not None
        # UTF-8 JSONL exports may include a BOM on the first row; Python JSON decoding rejects it.
        if text.startswith("\ufeff"):
            text = text[1:]
        try:
            if path.suffix.lower() == ".jsonl":
                value = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                value = json.loads(text)
        except json.JSONDecodeError as exc:
            return f"JSON 解析失败:第{exc.lineno}行第{exc.colno}列"
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
        retained, omitted = retain_text(rendered, 20_000)
        return retained + format_text_notice(
            "JSON 内容", omitted, "请改用更小的 JSON 文件或增加查询过滤。"
        )


class FileInfoTool(BaseTool):
    """获取文件或目录的非敏感元数据。"""

    name = "file_info"
    description = 'Return metadata only for a known file or directory (size, time, type). Do not read contents or search first.'
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    tags = ["file", "metadata"]
    permission_level = "allow"

    def execute(self, params: Dict[str, Any]) -> str:
        p, error = _resolve_path(params.get("path", ""), default="")
        if error:
            return error
        assert p is not None
        if not p.exists():
            return f"路径不存在:{p}"
        try:
            stat = p.stat()
            data = {"path": str(p), "type": "directory" if p.is_dir() else "file", "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime, "sensitive": _is_sensitive(p)}
            return json.dumps(data, ensure_ascii=False)
        except OSError as exc:
            return f"获取文件信息失败:{type(exc).__name__}: {exc}"


class GitStatusTool(BaseTool):
    """查看工作区 Git 状态。"""

    name = "git_status"
    description = 'Return Git working-tree and branch status only. For a status request, use this directly and do not call git_diff.'
    parameters = {"type": "object", "properties": {"path": {"type": "string", "description": "代码仓库目录"}}}
    tags = ["git", "status"]
    permission_level = "allow"

    def execute(self, params: Dict[str, Any]) -> str:
        cwd, error = _resolve_path(params.get("path", "."))
        if error:
            return error
        assert cwd is not None
        if not (cwd / ".git").exists():
            return f"不是 Git 仓库根目录或未发现 .git:{cwd}"
        try:
            result = subprocess.run(["git", "status", "--short", "--branch"], cwd=str(cwd), capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=10)
            output = (result.stdout or result.stderr).strip()
            retained, omitted = retain_text(output, 8_000)
            return (retained or "工作区干净") + format_text_notice(
                "Git 状态", omitted, "请在终端中查看完整输出。"
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Git 状态获取失败:{type(exc).__name__}: {exc}"


class GitDiffTool(BaseTool):
    """查看工作区差异。"""

    name = "git_diff"
    description = 'Return Git uncommitted diff only. Use directly for a diff request; optionally accept staged and a repository-local file. Do not call git_status first.'
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "代码仓库目录"},
        "staged": {"type": "boolean"},
        "file": {"type": "string", "description": "可选的仓库内相对文件路径"},
    }}
    tags = ["git", "diff"]
    permission_level = "allow"

    def execute(self, params: Dict[str, Any]) -> str:
        cwd, error = _resolve_path(params.get("path", "."))
        if error:
            return error
        assert cwd is not None
        if not (cwd / ".git").exists():
            return f"不是 Git 仓库根目录或未发现 .git:{cwd}"
        args = ["git", "diff"]
        if bool(params.get("staged", False)):
            args.append("--cached")
        file_arg = str(params.get("file", "")).strip()
        if file_arg:
            candidate = (cwd / file_arg).resolve()
            if candidate != cwd and cwd not in candidate.parents:
                return "安全限制:diff 文件必须位于仓库内"
            args.extend(["--", str(candidate.relative_to(cwd))])
        try:
            result = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
            output = (result.stdout or result.stderr).strip()
            retained, omitted = retain_text(output, 12_000)
            return (retained or "没有差异") + format_text_notice(
                "Git diff", omitted, "请按文件或更小的变更范围查看。"
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Git diff 获取失败:{type(exc).__name__}: {exc}"


class RunTestsTool(BaseTool):
    """运行受限的项目测试命令。"""

    name = "run_tests"
    description = 'Run exactly one project pytest or ruff check. Preserve user-specified path, target, runner, and timeout. After a safety rejection, do not vary paths to retry.'
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "项目目录"},
        "runner": {"type": "string", "enum": ["pytest", "ruff"]},
        "target": {"type": "string", "description": "可选的项目内测试路径或模块"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
        "background": {"type": "boolean", "default": False, "description": "提交为后台任务并立即返回 job_id"},
    }}
    tags = ["test", "quality", "job"]
    permission_level = "ask"

    def __init__(self, job_manager: Optional[JobManager] = None) -> None:
        self.job_manager = job_manager

    def execute(self, params: Dict[str, Any]) -> str:
        cwd, error = _resolve_path(params.get("path", "."))
        if error:
            return error
        assert cwd is not None
        runner = str(params.get("runner", "pytest")).strip().lower() or "pytest"
        if runner not in {"pytest", "ruff"}:
            return "错误:runner 仅支持 pytest 或 ruff"
        target = str(params.get("target", "")).strip()
        if target:
            candidate = (cwd / target).resolve()
            if candidate != cwd and cwd not in candidate.parents:
                return "安全限制:target 必须位于项目目录内"
            target = str(candidate.relative_to(cwd))
        try:
            timeout = min(max(int(params.get("timeout_seconds", 60)), 1), 120)
        except (TypeError, ValueError):
            timeout = 60
        # A background run is explicit and opt-in; normal runs retain the old synchronous behavior.
        if bool(params.get("background", False)):
            if self.job_manager is None:
                return "错误:后台任务服务不可用"
            command = [sys.executable, "-m", "pytest"] + ([target] if target else [])
            if runner == "ruff":
                command = [sys.executable, "-m", "ruff", "check"] + ([target] if target else [])
            try:
                job = self.job_manager.submit_process(
                    command, cwd, session_id=str(params.get("_session_id", "default")),
                    tool_name=self.name, timeout_seconds=timeout,
                )
                return json.dumps({"job_id": job["job_id"], "status": job["status"],
                                   "message": "后台任务已提交，可使用 job_list/job_output 查询"}, ensure_ascii=False)
            except (OSError, ValueError) as exc:
                return f"工具执行失败 [run_tests]:{type(exc).__name__}: {exc}"
        # Ruff requires an explicit subcommand (``ruff check``); invoking
        # ``python -m ruff <target>`` is parsed as an invalid subcommand.
        command = [sys.executable, "-m", "pytest"] + ([target] if target else [])
        if runner == "ruff":
            command = [sys.executable, "-m", "ruff", "check"] + ([target] if target else [])
        try:
            result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
            output = ((result.stdout or "") + ("\n" + result.stderr if result.stderr else "")).strip()
            status = "通过" if result.returncode == 0 else f"失败（退出码{result.returncode}）"
            retained, omitted = retain_text(output, 12_000)
            rendered = f"{runner} {status}\n{retained}" if retained else f"{runner} {status}"
            return rendered + format_text_notice(
                f"{runner} 输出", omitted, "请缩小 target 或作为后台任务运行后读取完整日志。"
            )
        except subprocess.TimeoutExpired:
            return f"错误:{runner} 执行超时（{timeout}秒）"
        except OSError as exc:
            return f"运行测试失败:{type(exc).__name__}: {exc}"


class WriteFileTool(BaseTool):
    """Create or replace a text file after a current-session observation."""

    name = "write_file"
    description = "Atomically write a known UTF-8 text file. The file must have been observed in this session before replacement."
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"},
        "create_if_absent": {"type": "boolean"},
    }, "required": ["path", "content"]}
    tags = ["file", "write"]
    permission_level = "ask"

    def __init__(self, observations: Optional[FileObservationStore] = None) -> None:
        self.observations = observations or FileObservationStore()

    def execute(self, params: Dict[str, Any]) -> str:
        path, error = _resolve_path(params.get("path", ""), default="")
        if error:
            return error
        assert path is not None
        if _is_sensitive(path):
            return "安全限制:拒绝写入凭据、私钥或环境变量文件"
        if not str(params.get("path", "")).strip():
            return "错误:未提供文件路径"
        session_id = str(params.get("_session_id", "default"))
        guard = self.observations.require_current(
            session_id, path, allow_create=bool(params.get("create_if_absent", False))
        )
        if guard:
            return guard
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, str(params.get("content", "")))
            self.observations.observe(session_id, path)
            return f"已原子写入文件: {path}"
        except OSError as exc:
            return f"写入文件失败:{type(exc).__name__}: {exc}"


class EditFileTool(BaseTool):
    """Atomically apply one literal edit under an observed version."""

    name = "edit_file"
    description = "Atomically replace one literal string in a known UTF-8 file. Read the file first; stale observations are rejected."
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"}, "old_string": {"type": "string"},
        "new_string": {"type": "string"}, "replace_all": {"type": "boolean"},
    }, "required": ["path", "old_string", "new_string"]}
    tags = ["file", "edit"]
    permission_level = "ask"

    def __init__(self, observations: Optional[FileObservationStore] = None) -> None:
        self.observations = observations or FileObservationStore()

    def execute(self, params: Dict[str, Any]) -> str:
        path, error = _resolve_path(params.get("path", ""), default="")
        if error:
            return error
        assert path is not None
        if _is_sensitive(path):
            return "安全限制:拒绝编辑凭据、私钥或环境变量文件"
        session_id = str(params.get("_session_id", "default"))
        guard = self.observations.require_current(session_id, path)
        if guard:
            return guard
        old = str(params.get("old_string", ""))
        new = str(params.get("new_string", ""))
        if not old:
            return "错误:old_string 不能为空"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"读取文件失败:{type(exc).__name__}: {exc}"
        count = text.count(old)
        if count == 0:
            return "编辑失败:未找到 old_string"
        if count > 1 and not bool(params.get("replace_all", False)):
            return f"编辑失败:old_string 匹配 {count} 处，请提供更精确文本或设置 replace_all=true"
        updated = text.replace(old, new) if bool(params.get("replace_all", False)) else text.replace(old, new, 1)
        try:
            atomic_write_text(path, updated)
            self.observations.observe(session_id, path)
            return f"已编辑文件: {path}（替换 {count if bool(params.get('replace_all', False)) else 1} 处）"
        except OSError as exc:
            return f"编辑文件失败:{type(exc).__name__}: {exc}"



