"""
my_agent.cli — 向后兼容的 CLI 入口

重定向到 cli/repl.py，保持 pyproject.toml 的 entry_points 兼容。
"""
from .cli.repl import main, print_help

__all__ = ["main", "print_help"]
