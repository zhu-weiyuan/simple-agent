"""
my_agent.bridge.base — Bridge 基类

Agent 与外部环境的适配层，参考 Claude Code bridge.ts 设计。
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class Bridge(ABC):
    """Bridge 基类"""

    @abstractmethod
    def execute_command(self, command: str, timeout_ms: int = 15000) -> str: ...

    @abstractmethod
    def read_file(self, path: str, max_bytes: int = 50000) -> str: ...

    @abstractmethod
    def list_directory(self, path: str) -> str: ...


class LocalBridge(Bridge):
    """本地环境 Bridge"""

    def execute_command(self, command: str, timeout_ms: int = 15000) -> str:
        timeout_sec = max(timeout_ms / 1000, 1)
        ps_cmd = (
            "$OutputEncoding=[System.Text.UTF8Encoding]::new($true);"
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($true); "
            + command
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoLogo", "-Command", ps_cmd],
                capture_output=True,
                timeout=timeout_sec,
            )
            raw = result.stdout or result.stderr
            try:
                output = raw.decode("utf-8", errors="replace")
            except Exception:
                output = raw.decode("gbk", errors="replace")

            cleaned = []
            for c in output:
                cp = ord(c)
                if cp < 32 and c not in "\n\r\t":
                    continue
                if cp == 127:
                    continue
                if (
                    0x4e00 <= cp <= 0x9fff
                    or 0x3000 <= cp <= 0x303f
                    or 0xFF01 <= cp <= 0xFF60
                    or 32 <= cp <= 126
                ):
                    cleaned.append(c)
                else:
                    cleaned.append("?")

            text = "".join(cleaned).strip()
            if len(text) > 4000:
                text = text[:3900] + "\n...（输出截断）"
            return text or "命令执行成功，无输出。"

        except subprocess.TimeoutExpired:
            return f"错误：命令执行超时（{timeout_sec:.0f}秒限制）"
        except Exception as e:
            return f"命令执行失败：{type(e).__name__}: {e}"

    def read_file(self, path: str, max_bytes: int = 50000) -> str:
        p = Path(path).resolve()
        if not p.exists():
            return f"文件不存在：{path}"
        if not p.is_file():
            return f"不是文件：{path}"
        if p.stat().st_size > max_bytes:
            return f"文件过大（{p.stat().st_size} bytes），超过 {max_bytes} 限制"

        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 10000:
            content = content[:9800] + "\n...（内容截断）"
        return content

    def list_directory(self, path: str) -> str:
        p = Path(path).resolve()
        if not p.exists():
            return f"路径不存在：{path}"
        if not p.is_dir():
            return f"不是目录：{path}"

        entries = []
        for item in sorted(p.iterdir()):
            kind = "[DIR]" if item.is_dir() else "[FILE]"
            entries.append(f"{kind} {item.name}")

        if not entries:
            return f"目录为空：{path}"
        return f"目录: {p}\n" + "\n".join(entries)
