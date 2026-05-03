# -*- coding: utf-8 -*-
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

    # 危险命令模式（按类别分组）
    _BLOCK_PATTERNS = {
        # 文件系统破坏
        "rm ": "删除文件",
        "del ": "删除文件",
        "Remove-Item": "删除文件/目录",
        "format-*": "格式化磁盘",
        "Clear-Content": "清空文件内容",
        "Truncate-Content": "截断文件内容",
        # 远程代码执行
        "invoke-expression": "动态代码执行",
        "iex": "动态代码执行(IEX别名)",
        "invoke-restmethod": "HTTP请求(可能被用于下载恶意脚本)",
        "irm": "HTTP请求(IRM别名)",
        "invoke-webrequest": "HTTP请求",
        "iwrm": "HTTP请求(IWRM别名)",
        "download-string": "下载字符串",
        "invoke-command": "远程命令执行",
        # 系统配置修改
        "set-executionpolicy": "修改执行策略",
        "reg add": "修改注册表",
        "reg delete": "修改注册表",
        "net user": "用户管理",
        "net localgroup": "用户组管理",
        "shutdown": "关机/重启",
        "restart-computer": "重启计算机",
        "stop-computer": "关闭计算机",
        # 数据泄露
        "certutil -encode": "编码文件(可能用于隐藏数据)",
        "certutil -decode": "解码文件(可能用于执行隐藏代码)",
    }

    def execute(self, params: Dict[str, Any]) -> str:
        command = str(params.get("command", "")).strip()
        if not command:
            return "错误：未提供 PowerShell 命令"

        cmd_lower = command.lower()
        for pattern, reason in self._BLOCK_PATTERNS.items():
            if pattern.lower() in cmd_lower:
                return (
                    f"错误：拒绝执行危险命令 '{pattern}'（{reason}）。"
                    "如需此操作，请明确说明并获得确认。"
                )

        # 检查命令链接 + 可疑操作（避免 $ 字符误杀，PowerShell 变量极常见）
        if any(ch in command for ch in [';', '&', '`']) and any(
            p in cmd_lower for p in ['http', 'download', 'invoke', 'comobj']
        ):
            return "错误：拒绝执行包含可疑模式的复合命令。"

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
