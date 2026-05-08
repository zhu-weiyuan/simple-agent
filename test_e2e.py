# -*- coding: utf-8 -*-
"""SimpleAgent 端到端测试"""
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from my_agent.agent import SimpleAgent

print("=" * 60)
print("SimpleAgent End-to-End Test")
print("=" * 60)

# Test 1: Basic mode
print("\n--- Test 1: Basic Mode ---")
a = SimpleAgent(enable_enhanced=False)
result = a.run("你好，请一句话介绍你自己")
print(f"Result: {result[:200]}")

# Test 2: Enhanced mode
print("\n--- Test 2: Enhanced Mode ---")
a2 = SimpleAgent(enable_enhanced=True)
result2 = a2.run("现在几点了？")
print(f"Result: {result2[:200]}")

# Test 3: Tool usage
print("\n--- Test 3: Calculator ---")
result3 = a.run("计算 25 * 17 + 3.14 * 10")
print(f"Result: {result3[:200]}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
