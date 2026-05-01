"""
my_agent.tools.builtins.shell — PowerShell 执行工具
"""
from __future__ import annotations

import subprocess
from typing import Any, Dict

from ..base import BaseTool


class PowerShellTool(BaseTool):
    """执行 PowerShell 命令"""

    name = "execute_powershell"
    description = (
        "执行 PowerShell 命令。可用于：查看系统信息、管理文件、"
        "运行程序、检查网络、安装软件等。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 PowerShell 命令",
            }
        },
        "required": ["command"],
    }
    tags = ["system", "shell"]

    # 危险命令模式
    _BLOCK_PATTERNS = ["rm ", "del ", "format-*", "invoke-expression"]

    def execute(self, params: Dict[str, Any]) -> str:
        command = str(params.get("command", "")).strip()
        if not command:
            return "错误：未提供 PowerShell 命令"

        cmd_lower = command.lower()
        for pattern in self._BLOCK_PATTERNS:
            if pattern.lower() in cmd_lower:
                return (
                    f"错误：拒绝执行潜在危险的命令: '{command}'。"
                    "如需删除文件，请明确说明并获得确认。"
                )

        try:
            ps_cmd = (
                "$OutputEncoding=[System.Text.UTF8Encoding]::new($true);"
                "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($true); "
                + command
            )
            result = subprocess.run(
                ["powershell", "-NoLogo", "-Command", ps_cmd],
                capture_output=True,
                timeout=15,
            )
            raw = result.stdout if result.stdout else result.stderr
            try:
                output = raw.decode("utf-8", errors="replace")
            except Exception:
                output = raw.decode("gbk", errors="replace")

            # 清理不可打印字符
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
                text = text[:3900] + "\n...（输出截断，超过4000字符）"
            return text or "命令执行成功，无输出。"

        except subprocess.TimeoutExpired:
            return "错误：命令执行超时（15秒限制）"
        except Exception as e:
            return f"PowerShell 执行失败：{type(e).__name__}: {e}"
