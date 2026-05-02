# -*- coding: utf-8 -*-
"""
my_agent.bridge.permissions — 工具权限控制

参考 Claude Code 三级权限：alwaysAllow / ask / deny
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
        level = self.overrides.get(tool_name, self.default_level)
        if level == PermissionLevel.ALWAYS_ALLOW:
            return True
        if level == PermissionLevel.DENY:
            return False
        # ASK: 非交互模式拒绝
        return False

    def set(self, tool_name: str, level: PermissionLevel) -> None:
        self.overrides[tool_name] = level
