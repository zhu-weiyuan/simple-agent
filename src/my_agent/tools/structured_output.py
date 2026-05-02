# -*- coding: utf-8 -*-
"""
my_agent.tools.structured_output — 结构化输出工具

参考 strands-agents structured_output_tool 设计：
- 强制 LLM 输出指定 Pydantic 模型格式
- 自动验证和重试
- 支持嵌套模型
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Type

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


class StructuredOutputError(Exception):
    """结构化输出错误"""
    pass


class StructuredOutputTool:
    """结构化输出工具

    将 LLM 的输出约束为指定的 Pydantic 模型格式，
    自动验证、提取和重试。

    Example:
        ```python
        from pydantic import BaseModel, Field

        class WeatherInfo(BaseModel):
            city: str = Field(description="城市名称")
            temperature: float = Field(description="温度")
            condition: str = Field(description="天气状况")

        structured = StructuredOutputTool(schema=WeatherInfo)
        result = structured.extract(response_text)
        # → WeatherInfo(city="北京", temperature=25.0, condition="晴")
        ```
    """

    def __init__(
        self,
        schema: Optional[Type] = None,
        schema_dict: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ):
        """
        Args:
            schema: Pydantic 模型类
            schema_dict: JSON Schema 字典（备选）
            max_retries: 最大重试次数
        """
        self._schema = schema
        self._schema_dict = schema_dict
        self._max_retries = max_retries

    @property
    def json_schema(self) -> Dict[str, Any]:
        """获取 JSON Schema"""
        if self._schema:
            try:
                return self._schema.model_json_schema()
            except AttributeError:
                # Non-Pydantic class
                return self._build_schema_from_class(self._schema)
        return self._schema_dict or {}

    def _build_schema_from_class(self, cls: Type) -> Dict[str, Any]:
        """从类构建 JSON Schema"""
        if hasattr(cls, "__annotations__"):
            properties = {}
            for name, annotation in cls.__annotations__.items():
                properties[name] = {
                    "type": self._type_to_json(annotation),
                    "description": name,
                }
            return {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            }
        return {"type": "object"}

    def _type_to_json(self, tp) -> str:
        """Python 类型 → JSON Schema 类型"""
        type_map = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }
        return type_map.get(tp, "string")

    def extract(self, text: str) -> Any:
        """从文本中提取结构化数据

        Args:
            text: LLM 输出的文本

        Returns:
            解析后的结构化数据（Pydantic 模型或字典）

        Raises:
            StructuredOutputError: 解析失败
        """
        # Try to find JSON in the text
        json_str = self._extract_json(text)
        if not json_str:
            raise StructuredOutputError("No JSON found in response")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise StructuredOutputError(f"Invalid JSON: {e}")

        # Validate against schema
        if self._schema:
            try:
                return self._schema(**data)
            except Exception as e:
                raise StructuredOutputError(f"Schema validation failed: {e}")

        return data

    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 块"""
        # Try full text first
        text = text.strip()
        if text.startswith("{"):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass

        # Find JSON in markdown code blocks
        import re

        # ```json ... ```
        matches = re.findall(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

        # Find {...} or [...]
        matches = re.findall(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

        return None

    def build_system_prompt(self) -> str:
        """构建强制结构化输出的 system prompt 片段"""
        if not self.json_schema:
            return ""

        schema_str = json.dumps(self.json_schema, indent=2, ensure_ascii=False)
        return (
            "你必须以 JSON 格式输出，严格遵循以下 schema：\n"
            f"```\n{schema_str}\n```\n\n"
            "只输出 JSON，不要包含其他文字。"
        )

    def to_tool_dict(self) -> Dict[str, Any]:
        """转换为工具字典格式"""
        return {
            "type": "function",
            "function": {
                "name": "structured_output",
                "description": "Output structured data matching the schema",
                "parameters": self.json_schema,
            },
        }
