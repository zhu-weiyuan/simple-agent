# -*- coding: utf-8 -*-
"""
my_agent.bridge.permissions — 工具权限控制

参考 Claude Code 三级权限:alwaysAllow / ask / deny
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from typing import Dict, Optional


class PermissionLevel(Enum):
    ALWAYS_ALLOW = "alwaysAllow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class ToolPermission:
    name: str
    level: PermissionLevel = PermissionLevel.ASK
    description: str = ""


@dataclass
class PermissionPolicy:
    """工具权限策略"""

    default_level: PermissionLevel = PermissionLevel.ASK
    overrides: Dict[str, PermissionLevel] = field(default_factory=dict)
    allow_patterns: list[str] = field(default_factory=list)

    def check(self, tool_name: str, **kwargs) -> bool:
        """Return whether a tool is allowed in a non-interactive execution path.

        Overrides take precedence; explicit allow patterns may grant only tools
        whose names match (for example ``read_*``). ``ask`` remains deny-by-
        default unless the engine receives an approving hook decision.
        """
        level = self.overrides.get(tool_name, self.default_level)
        if level == PermissionLevel.ALWAYS_ALLOW:
            return True
        if level == PermissionLevel.DENY:
            return False
        return any(fnmatchcase(tool_name, pattern) for pattern in self.allow_patterns)

    def decision(self, tool_name: str, declared_level: str) -> tuple[bool, str]:
        """Combine the tool's declaration with explicit runtime policy."""
        if declared_level == "deny":
            return False, "工具声明为 deny"
        if tool_name in self.overrides:
            allowed = self.check(tool_name)
            return allowed, "策略 override 允许" if allowed else "策略 override 拒绝"
        if any(fnmatchcase(tool_name, pattern) for pattern in self.allow_patterns):
            return True, "策略 allow_patterns 允许"
        if declared_level == "allow":
            return True, "工具声明为 allow"
        return False, "工具需要显式审批"

    def set(self, tool_name: str, level: PermissionLevel) -> None:
        self.overrides[tool_name] = level
