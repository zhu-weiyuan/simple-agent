# -*- coding: utf-8 -*-
"""
my_agent.tools.builtins.file — 文件系统工具
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..base import BaseTool


class ReadFileTool(BaseTool):
    """读取文件内容"""

    name = "read_file"
    description = "读取文件内容。适用于查看配置文件、代码、日志等文本文件。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径，如 'C:\\Users\\test.txt' 或 './config.json'",
            }
        },
        "required": ["path"],
    }
    tags = ["file", "read"]

    def execute(self, params: Dict[str, Any]) -> str:
        path_str = str(params.get("path", "")).strip()
        if not path_str:
            return "错误：未提供文件路径"

        try:
            p = Path(path_str).resolve()
            if not p.exists():
                return f"文件不存在：{path_str}"
            if not p.is_file():
                return f"不是文件：{path_str}"
            if p.stat().st_size > 50000:
                return (
                    f"文件过大（{p.stat().st_size} bytes），超过50KB限制。"
                    "请指定只读取部分内容。"
                )

            content = p.read_text(encoding="utf-8", errors="replace")
            if len(content) > 10000:
                content = content[:9800] + "\n...（内容截断，超过10000字符）"
            return content
        except Exception as e:
            return f"读取文件失败：{type(e).__name__}: {e}"


class ListFilesTool(BaseTool):
    """列出目录中的文件和文件夹"""

    name = "list_files"
    description = "列出目录中的文件和文件夹。可用于浏览文件系统。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目录路径，默认为当前目录",
            }
        },
    }
    tags = ["file", "list"]

    def execute(self, params: Dict[str, Any]) -> str:
        path_str = str(params.get("path", ".")).strip()
        try:
            p = Path(path_str).resolve()
            if not p.exists():
                return f"路径不存在：{path_str}"
            if not p.is_dir():
                return f"不是目录：{path_str}"

            entries = []
            for item in p.iterdir():
                kind = "[DIR]" if item.is_dir() else "[FILE]"
                entries.append(f"{kind} {item.name}")

            if not entries:
                return f"目录为空：{path_str}"

            entries.sort()
            return f"目录: {p}\n" + "\n".join(entries)
        except Exception as e:
            return f"列出目录失败：{type(e).__name__}: {e}"
