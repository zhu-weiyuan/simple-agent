# -*- coding: utf-8 -*-
"""Read-only tools for session-scoped oversized tool-result artifacts."""
from __future__ import annotations

import json
from typing import Any, Dict

from ...artifact_store import ArtifactStore
from ..base import BaseTool


class ReadArtifactTool(BaseTool):
    name = "read_artifact"
    description = "读取已归档的大型工具输出的一段内容。先使用工具输出中的 artifact_id，再按 offset 分页读取。"
    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "description": "工具输出中返回的 artifact ID"},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 12000},
        },
        "required": ["artifact_id"],
        "additionalProperties": False,
    }
    tags = ["artifact", "filesystem", "read"]
    permission_level = "allow"

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def execute(self, params: Dict[str, Any]) -> str:
        try:
            data = self.store.read(
                str(params.get("artifact_id", "")),
                offset=int(params.get("offset", 0)),
                limit=int(params.get("limit", 12000)),
                session_id=str(params.get("_session_id", "")),
            )
            return json.dumps(data, ensure_ascii=False)
        except PermissionError as exc:
            return f"工具被拒绝(权限级别: session): read_artifact - {exc}"
        except (KeyError, TypeError, ValueError) as exc:
            return f"错误:参数验证失败 [read_artifact]: {exc}"
        except OSError as exc:
            return f"工具执行失败 [read_artifact]:{type(exc).__name__}: {exc}"


class SearchArtifactTool(BaseTool):
    name = "search_artifact"
    description = "在已归档的大型工具输出中搜索关键词，返回匹配的行号和摘要。"
    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "description": "工具输出中返回的 artifact ID"},
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
        },
        "required": ["artifact_id", "query"],
        "additionalProperties": False,
    }
    tags = ["artifact", "filesystem", "read", "search"]
    permission_level = "allow"

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def execute(self, params: Dict[str, Any]) -> str:
        try:
            data = self.store.search(
                str(params.get("artifact_id", "")),
                str(params.get("query", "")),
                session_id=str(params.get("_session_id", "")),
                limit=int(params.get("limit", 50)),
            )
            return json.dumps(data, ensure_ascii=False)
        except PermissionError as exc:
            return f"工具被拒绝(权限级别: session): search_artifact - {exc}"
        except (KeyError, TypeError, ValueError) as exc:
            return f"错误:参数验证失败 [search_artifact]: {exc}"
        except OSError as exc:
            return f"工具执行失败 [search_artifact]:{type(exc).__name__}: {exc}"


__all__ = ["ReadArtifactTool", "SearchArtifactTool"]
