#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleAgent v2.0 — 快速启动

用法:
    python run.py                    # 交互式对话
    python run.py "你好，你是谁？"     # 单次问答
"""
import os
import sys
from pathlib import Path

# ── 环境配置（按需修改）───────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "your_key_here")          # llama.cpp 通常不校验 key
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8080")
os.environ.setdefault("OPENAI_MODEL", "Qwen3.6-27B-IQ4_NL.gguf")

# 把 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from my_agent import SimpleAgent


def interactive():
    """交互式对话模式"""
    agent = SimpleAgent()
    print("=" * 60)
    print("  SimpleAgent v2.0 — 输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        try:
            result = agent.run(user_input)
            print(f"\nAgent: {result}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def single(question):
    """单次问答模式"""
    agent = SimpleAgent()
    try:
        result = agent.run(question)
        print(result)
    finally:
        agent.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        single(" ".join(sys.argv[1:]))
    else:
        interactive()
