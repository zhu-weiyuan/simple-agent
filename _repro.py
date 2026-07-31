import sys, os
sys.stdout.reconfigure(encoding='utf-8')

from src.my_agent.core.engine import QueryEngine, ToolCall

# Force a tool that raises -> hit the broken exception path
engine = QueryEngine("__system__")

class _FakeRegistry:
    def get_handler(self, name):
        if name == "boom":
            def boom(params): raise RuntimeError("intentional")
            return boom
        return None

engine.tool_registry = _FakeRegistry()

tc = ToolCall(id="t1", name="boom", arguments={})
try:
    result = engine._execute_tool(tc)
    print(f"OK: {result!r}")
except NameError as e:
    print(f"BUG REPRODUCED: NameError on tool exception path -> {e}")
