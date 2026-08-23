import asyncio
from types import SimpleNamespace

from my_agent.core.engine import QueryEngine
from my_agent.tools.registry import ToolRegistry


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.state = "list_dirs"  # list_dirs -> read_readme -> read_core1 -> read_core2 -> partial -> complete

    def __call__(self, messages, tools):
        self.calls += 1
        
        # 模拟探索流程：需要 2 个目录，每个目录需要 README + 2 个核心文件
        if self.calls == 1:
            # 第 1 次：列出目录
            content = ""
            tool_calls = [{
                "id": f"call-{self.calls}",
                "function": {"name": "list_files", "arguments": '{"path": "."}'},
            }]
        elif self.calls == 2:
            # 第 2 次：读取 core/README.md
            content = ""
            tool_calls = [{
                "id": f"call-{self.calls}",
                "function": {"name": "read_file", "arguments": '{"path": "core/README.md"}'},
            }]
        elif self.calls == 3:
            # 第 3 次：读取 core/engine.py
            content = ""
            tool_calls = [{
                "id": f"call-{self.calls}",
                "function": {"name": "read_file", "arguments": '{"path": "core/engine.py"}'},
            }]
        elif self.calls == 4:
            # 第 4 次：读取 core/types.py
            content = ""
            tool_calls = [{
                "id": f"call-{self.calls}",
                "function": {"name": "read_file", "arguments": '{"path": "core/types.py"}'},
            }]
        elif self.calls == 5:
            # 第 5 次：读取 types/README.md
            content = ""
            tool_calls = [{
                "id": f"call-{self.calls}",
                "function": {"name": "read_file", "arguments": '{"path": "types/README.md"}'},
            }]
        elif self.calls == 6:
            # 第 6 次：读取 types/message.py
            content = ""
            tool_calls = [{
                "id": f"call-{self.calls}",
                "function": {"name": "read_file", "arguments": '{"path": "types/message.py"}'},
            }]
        elif self.calls == 7:
            # 第 7 次：读取 types/session.py
            content = ""
            tool_calls = [{
                "id": f"call-{self.calls}",
                "function": {"name": "read_file", "arguments": '{"path": "types/session.py"}'},
            }]
        elif self.calls == 8:
            # 第 8 次：阶段性总结（不应结束）
            content = "partial summary. Let me continue."
            tool_calls = []
        else:
            # 第 9 次：最终完成 - 必须提及所有必需目录
            content = "task complete; final architecture summary for core and types."
            tool_calls = []
        
        message = SimpleNamespace(content=content, tool_calls=[
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(**tc["function"]),
            ) for tc in tool_calls
        ])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def test_dsh_long_exploration_does_not_stop_after_partial_text():
    registry = ToolRegistry()
    executed = []
    
    def list_files(args):
        executed.append(("list_files", args))
        path = args.get("path", ".")
        if path == ".":
            return "[DIR] core\n[DIR] types"
        return ""
    
    def read_file(args):
        executed.append(("read_file", args))
        path = args.get("path", "")
        if "README.md" in path:
            return f"# {path}\nModule documentation."
        if path.endswith(".py"):
            return f"# {path}\nCore implementation."
        return ""
    
    registry.add("list_files", list_files)
    registry.add("read_file", read_file)
    
    engine = QueryEngine("system", tool_registry=registry)
    llm = FakeLLM()
    engine.set_llm(llm)

    result = asyncio.run(engine.arun_with_dsh(
        "explore the workspace directory and read every file",
        session=engine.session,
        max_tool_calls=20,
        session_id="regression",
    ))

    # 验证：8 次工具调用 + 1 次阶段性总结 + 1 次最终完成 = 9 次 LLM 调用
    assert llm.calls == 9, f"Expected 9 calls, got {llm.calls}"
    # 验证工具调用次数
    assert len(executed) == 7, f"Expected 7 tool calls, got {len(executed)}"
    # 验证最终结果
    assert result["stop_reason"] == "completed"
    assert result["content"] == "task complete; final architecture summary for core and types."