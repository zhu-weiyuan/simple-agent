"""
my_agent.tools.builtins.calculator — 计算器工具
"""
from __future__ import annotations

import ast
from typing import Any, Dict

from ..base import BaseTool


class CalculatorTool(BaseTool):
    """执行数学计算"""

    name = "calculator"
    description = "执行基本数学运算，支持加减乘除、括号和幂运算 (^)。"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4'",
            }
        },
        "required": ["expression"],
    }
    tags = ["math", "utility"]

    _ALLOWED_CHARS = set("0123456789+-*/.()^ ")
    _ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
        ast.USub, ast.UAdd, ast.Constant, ast.Load,
    )

    def execute(self, params: Dict[str, Any]) -> str:
        expr = str(params.get("expression", "")).strip()
        if not expr:
            return "错误：未提供表达式"

        if not all(c in self._ALLOWED_CHARS for c in expr):
            return "错误：表达式包含非法字符"

        try:
            normalized = expr.replace("^", "**")
            node = ast.parse(normalized, mode="eval")
            self._validate_ast(node)
            result = eval(
                compile(node, "<calculator>", "eval"),
                {"__builtins__": {}},
                {},
            )
            return f"计算结果：{expr} = {result}"
        except ZeroDivisionError:
            return "错误：除数不能为零"
        except Exception as e:
            return f"计算错误：{type(e).__name__}: {e}"

    def _validate_ast(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if not isinstance(child, self._ALLOWED_NODES):
                raise ValueError(f"不支持的表达式节点：{type(child).__name__}")
