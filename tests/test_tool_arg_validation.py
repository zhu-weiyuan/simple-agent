# -*- coding: utf-8 -*-
"""
Tests for tool call argument schema validation (engine._validate_tool_args).
"""
import json
import pytest
from unittest.mock import MagicMock

from my_agent.core.engine import QueryEngine
from my_agent.tools.registry import ToolRegistry, ToolDefinition


# ── helpers ──────────────────────────────────────────────────

_DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "file path"},
        "count": {"type": "integer", "description": "item count"},
    },
    "required": ["path"],
}
_NO_SCHEMA = object()  # distinct sentinel for "no schema at all"


def _make_definition(name="test_tool", parameters=_NO_SCHEMA):
    """Create a ToolDefinition with an optional JSON schema.

    Pass parameters=None (explicit) to create a definition with no schema.
    Omit or use _NO_SCHEMA to get the default test schema.
    """
    defn = MagicMock(spec=ToolDefinition)
    defn.name = name
    if parameters is _NO_SCHEMA:
        defn.parameters = _DEFAULT_SCHEMA
    else:
        defn.parameters = parameters
    defn.permission_level = "allow"
    return defn


# ── valid arguments ──────────────────────────────────────────

class TestValidArgs:
    def test_valid_args_pass(self):
        defn = _make_definition()
        result = QueryEngine._validate_tool_args(
            "test_tool", {"path": "/tmp/file.txt", "count": 5}, defn
        )
        assert result is None

    def test_valid_args_missing_optional_pass(self):
        defn = _make_definition()
        result = QueryEngine._validate_tool_args(
            "test_tool", {"path": "/tmp/file.txt"}, defn
        )
        assert result is None

    def test_no_schema_passes(self):
        defn = _make_definition(parameters=None)
        result = QueryEngine._validate_tool_args("test_tool", {"anything": 1}, defn)
        assert result is None

    def test_no_definition_passes(self):
        result = QueryEngine._validate_tool_args("test_tool", {"path": "x"}, None)
        assert result is None


# ── invalid arguments ────────────────────────────────────────

class TestInvalidArgs:
    def test_wrong_type_rejected(self):
        defn = _make_definition()
        result = QueryEngine._validate_tool_args(
            "test_tool", {"path": 123}, defn  # path should be string
        )
        assert result is not None
        assert "参数验证失败" in result
        assert "test_tool" in result

    def test_missing_required_rejected(self):
        defn = _make_definition()
        result = QueryEngine._validate_tool_args(
            "test_tool", {"count": 5}, defn  # missing required "path"
        )
        assert result is not None
        assert "参数验证失败" in result

    def test_extra_properties_pass_by_default(self):
        """jsonschema by default allows additional properties."""
        defn = _make_definition()
        result = QueryEngine._validate_tool_args(
            "test_tool", {"path": "x", "unexpected": True}, defn
        )
        assert result is None  # extra props allowed by default

    def test_empty_dict_missing_required(self):
        defn = _make_definition()
        result = QueryEngine._validate_tool_args("test_tool", {}, defn)
        assert result is not None
        assert "参数验证失败" in result


# ── edge cases ───────────────────────────────────────────────

class TestEdgeCases:
    def test_non_dict_params_treated_as_empty(self):
        """Engine converts non-dict args to {} before calling handler,
        so validation should see {}."""
        defn = _make_definition()
        result = QueryEngine._validate_tool_args("test_tool", {}, defn)
        # {} is missing required "path", so it should fail
        assert result is not None

    def test_invalid_jsonschema_graceful_degradation(self):
        """Malformed schema should not crash validation."""
        defn = _make_definition(parameters={"not": "a real schema"})
        result = QueryEngine._validate_tool_args("test_tool", {"path": "x"}, defn)
        # Should not raise; returns None on unexpected errors
        assert result is None

    def test_error_message_includes_field_path(self):
        """Error message should include the field path for debugging."""
        schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                }
            },
            "required": ["config"],
        }
        defn = _make_definition(parameters=schema)
        result = QueryEngine._validate_tool_args(
            "test_tool", {"config": {"name": 123}}, defn  # name should be string
        )
        assert result is not None
        assert "config.name" in result
