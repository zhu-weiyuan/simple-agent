# -*- coding: utf-8 -*-
"""
my_agent.tools.builtins.shell — PowerShell 执行工具
"""
from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, Optional

from ..base import BaseTool
from ..retention import format_text_notice, retain_text


class PowerShellTool(BaseTool):
    """执行 PowerShell 命令"""

    name = "execute_powershell"
    description = (
        "执行 PowerShell 命令。可用于:查看系统信息、管理文件、"
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
    permission_level = "deny"  # arbitrary shell execution requires explicit override

    # Arbitrary PowerShell is intentionally unavailable.  Even though this tool
    # is permission_level="deny", execute() can be called directly by code or a
    # future permission override; a substring blacklist is not a security boundary.
    # Keep only non-mutating, no-composition diagnostics.
    _SAFE_COMMANDS = frozenset({"get-date", "get-location", "get-process", "get-service"})
    _SAFE_COMMAND_RE = re.compile(
        r"^\s*(get-date|get-location|get-process|get-service)\s*$", re.IGNORECASE
    )

    @classmethod
    def _validate_command(cls, command: str) -> Optional[str]:
        """Return a reason when *command* is outside the safe diagnostic subset."""
        if not command:
            return "未提供 PowerShell 命令"
        # Reject multiline input and every PowerShell composition/interpolation
        # primitive.  No aliases or command arguments are accepted, so an
        # allowlisted command cannot be redirected to a sensitive path.
        if any(token in command for token in ("\r", "\n", ";", "|", "&", "`", "$", "(", ")", "<", ">")):
            return "命令包含组合、重定向或动态执行语法"
        match = cls._SAFE_COMMAND_RE.fullmatch(command)
        if not match or match.group(1).casefold() not in cls._SAFE_COMMANDS:
            return "仅允许只读诊断命令: Get-Date、Get-Location、Get-Process、Get-Service"
        return None

    def execute(self, params: Dict[str, Any]) -> str:
        command = str(params.get("command", "")).strip()
        rejection = self._validate_command(command)
        if rejection:
            return f"错误:拒绝执行 PowerShell 命令（{rejection}）。"

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
            retained, omitted = retain_text(text, 4000)
            return (retained or "命令执行成功，无输出。") + format_text_notice(
                "PowerShell 输出", omitted, "请缩小命令输出范围。"
            )

        except subprocess.TimeoutExpired:
            return "错误:命令执行超时（15秒限制）"
        except Exception as e:
            return f"PowerShell 执行失败:{type(e).__name__}: {e}"
