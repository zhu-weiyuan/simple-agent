# -*- coding: utf-8 -*-
"""
my_agent.cli.repl — CLI 入口

向后兼容：CLI 接口与原来完全一致。
"""
from __future__ import annotations

import sys
from typing import Optional

from ..agent import SimpleAgent


def print_help() -> None:
    help_text = """
My Agent — 你的终端 AI 助手

用法:
  my-agent [选项] [任务描述]

快速命令:
  --version, -v    显示版本号
  --help           显示此帮助信息

示例:
  my-agent "现在几点了"
  my-agent "计算 12*34"
  my-agent "北京的天气怎么样？"
"""
    print(help_text.strip())


def main() -> int:
    args = sys.argv[1:]

    help_flags = {"--help", "-h", "-help"}
    version_flags = {"--version", "-v"}

    if any(flag in args for flag in version_flags):
        print("0.1.0 (My Agent)")
        return 0

    if any(flag in args for flag in help_flags):
        print_help()
        return 0

    user_input = " ".join(args).strip()
    if not user_input:
        print(
            '错误：未提供任务描述。用法: my-agent "你的任务"',
            file=sys.stderr,
        )
        return 1

    agent = SimpleAgent()
    try:
        result = agent.run(user_input)
        print(result)
        return 0
    except ImportError as e:
        print(f"导入错误: {e}", file=sys.stderr)
        print(
            "请确保已安装依赖: pip install openai python-dotenv requests",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"运行时错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
