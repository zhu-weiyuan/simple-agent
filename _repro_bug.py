import sys, os
sys.stdout.reconfigure(encoding='utf-8')

# Test 1: hook_data undefined -> NameError on tool exception
from src.my_agent.core.engine import QueryEngine
from src.my_agent.types.message import ToolCall

qe = QueryEngine("__system__")

class _FakeRegistry:
    def get_handler(self, name):
        if name == "boom":
            def boom(params): raise ValueError("intentional error")
            return boom
        return None

qe.tool_registry = _FakeRegistry()
tc = ToolCall(id='x1', name="boom", arguments={})

try:
    qe._execute_tool(tc)
    print("FAIL: no NameError raised for hook_data bug")
except NameError as e:
    print(f"REPRODUCED BUG (hook_data): {e}")
except Exception as e:
    print(f"UNEXPECTED: {type(e).__name__}: {e}")
