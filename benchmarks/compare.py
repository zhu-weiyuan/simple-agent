#!/usr/bin/env python3
"""
Benchmark comparison: SimpleAgent vs LangChain vs AutoGen

Measures cold start time, tool call latency, and memory usage.

Usage:
    python benchmarks/compare.py
"""

import time
import tracemalloc
import subprocess
import sys


def measure_cold_start(framework_name, import_code):
    """Measure framework cold start time."""
    code = f"""
import time
start = time.time()
{import_code}
end = time.time()
print(end - start)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        print(f"  Error: {result.stderr[:200]}")
        return None


def measure_memory(code):
    """Measure memory usage during code execution."""
    tracemalloc.start()
    try:
        exec(code)
        current, peak = tracemalloc.get_traced_memory()
        return peak / 1024 / 1024  # MB
    finally:
        tracemalloc.stop()


def run_benchmarks():
    print("=" * 60)
    print("SimpleAgent Benchmark Comparison")
    print("=" * 60)

    # 1. Cold start times
    print("\n--- Cold Start Time ---")
    frameworks = {
        "SimpleAgent": "from simple_agent import Agent",
        "LangChain": "from langchain_core.language_models import BaseChatModel",
    }

    results = {}
    for name, import_code in frameworks.items():
        times = []
        for _ in range(3):
            t = measure_cold_start(name, import_code)
            if t is not None:
                times.append(t)
        if times:
            avg = sum(times) / len(times)
            results[name] = avg
            print(f"  {name:20s}: {avg:.3f}s (avg of {len(times)} runs)")

    # 2. Memory usage for SimpleAgent
    print("\n--- Memory Usage (SimpleAgent) ---")
    mem_code = """
from simple_agent import Agent
agent = Agent()
"""
    try:
        mem = measure_memory(mem_code)
        print(f"  SimpleAgent base: {mem:.1f}MB peak")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. Tool call latency
    print("\n--- Tool Call Latency (SimpleAgent) ---")
    latencies = []
    for _ in range(5):
        start = time.time()
        try:
            from simple_agent import Agent
            agent = Agent()
            agent.register_tool("test", lambda x: x)
            agent.run("test")
            latencies.append(time.time() - start)
        except Exception as e:
            print(f"  Error: {e}")
            break

    if latencies:
        avg_latency = sum(latencies) / len(latencies) * 1000  # ms
        print(f"  Avg tool call: {avg_latency:.1f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    for name, avg in results.items():
        print(f"  {name} cold start: {avg:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
