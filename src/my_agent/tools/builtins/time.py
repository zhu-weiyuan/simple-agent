# -*- coding: utf-8 -*-
"""
my_agent.tools.builtins.time — 时间工具
"""
from __future__ import annotations

import datetime
from typing import Any, Dict

from ..base import BaseTool


class GetTimeTool(BaseTool):
    """获取当前时间"""

    name = "get_time"
    description = "获取当前的日期和时间"
    parameters = {
        "type": "object",
        "properties": {},
    }
    tags = ["utility", "time"]

    def execute(self, params: Dict[str, Any]) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
