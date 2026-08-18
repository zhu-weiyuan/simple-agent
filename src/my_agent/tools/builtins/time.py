# -*- coding: utf-8 -*-
"""my_agent.tools.builtins.time — 时区明确的当前时间工具。"""
from __future__ import annotations

import datetime
import os
from typing import Any, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..base import BaseTool

_DEFAULT_TIMEZONE = "Asia/Shanghai"


class GetTimeTool(BaseTool):
    """获取当前时间（由 ``AGENT_TIMEZONE`` 明确指定时区）。"""

    name = "get_time"
    description = "获取当前的日期和时间"
    parameters = {"type": "object", "properties": {}}
    tags = ["utility", "time"]
    permission_level = "allow"

    def execute(self, params: Dict[str, Any]) -> str:
        timezone_name = os.getenv("AGENT_TIMEZONE", _DEFAULT_TIMEZONE).strip() or _DEFAULT_TIMEZONE
        try:
            now = datetime.datetime.now(ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError:
            # Do not silently return an arbitrary naive local time if an
            # operator made a typo. The formatted message keeps the tool
            # useful while making the configuration problem actionable.
            return f"错误:无效时区 {timezone_name}"
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
