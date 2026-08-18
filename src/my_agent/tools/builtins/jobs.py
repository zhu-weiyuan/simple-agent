# -*- coding: utf-8 -*-
"""Read/control tools for background test and quality jobs."""
from __future__ import annotations

import json
from typing import Any, Dict

from ...jobs import JobManager
from ..base import BaseTool


class _JobTool(BaseTool):
    def __init__(self, manager: JobManager) -> None:
        self.manager = manager

    @staticmethod
    def _session(params: Dict[str, Any]) -> str:
        return str(params.get("_session_id", "default"))


class JobListTool(_JobTool):
    name = "job_list"
    description = "列出当前会话的后台任务及状态。"
    parameters = {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50}}, "additionalProperties": False}
    tags = ["job", "read"]
    permission_level = "allow"

    def execute(self, params: Dict[str, Any]) -> str:
        return json.dumps({"jobs": self.manager.list(self._session(params), int(params.get("limit", 50)))}, ensure_ascii=False)


class JobOutputTool(_JobTool):
    name = "job_output"
    description = "读取当前会话后台任务的输出日志，可通过 offset 分页。"
    parameters = {"type": "object", "properties": {
        "job_id": {"type": "string"}, "offset": {"type": "integer", "minimum": 0, "default": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 12000},
    }, "required": ["job_id"], "additionalProperties": False}
    tags = ["job", "read"]
    permission_level = "allow"

    def execute(self, params: Dict[str, Any]) -> str:
        try:
            return json.dumps(self.manager.read_output(str(params.get("job_id", "")), self._session(params), int(params.get("offset", 0)), int(params.get("limit", 12000))), ensure_ascii=False)
        except (KeyError, TypeError, ValueError) as exc:
            return f"错误:参数验证失败 [job_output]: {exc}"


class JobCancelTool(_JobTool):
    name = "job_cancel"
    description = "请求取消当前会话中正在执行的后台任务。此操作会终止对应进程。"
    parameters = {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"], "additionalProperties": False}
    tags = ["job", "control"]
    permission_level = "ask"

    def execute(self, params: Dict[str, Any]) -> str:
        try:
            return json.dumps(self.manager.cancel(str(params.get("job_id", "")), self._session(params)), ensure_ascii=False)
        except KeyError as exc:
            return f"错误:参数验证失败 [job_cancel]: {exc}"


__all__ = ["JobListTool", "JobOutputTool", "JobCancelTool"]
