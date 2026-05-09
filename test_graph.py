# -*- coding: utf-8 -*-
"""LangGraph 风格图编排引擎 - 精简测试"""
import sys
import io
# Only wrap stdout when running directly (not under pytest)
if not sys.modules.get('_pytest') and 'pytest' not in sys.argv:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from my_agent.graph.state import GraphState
from my_agent.graph.node import node
from my_agent.graph.graph import Graph

@node("greeting")
def greet_node(state):
    state.add_message("assistant", "Hi!")
    return state

@node("llm_call")
def llm_node(state):
    state.add_message("assistant", "Hello! How can I help you?")
    return state

@node("final_output")
def output_node(state):
    state.set_metadata("final_output", "Complete")
    return state

def test_basic_graph():
    """测试1:基本图执行"""
    graph = (Graph()
        .register_node("greeting", greet_node)
        .register_node("llm_call", llm_node)
        .register_node("final_output", output_node))

    graph.add_edge("greeting", "llm_call")
    graph.add_edge("llm_call", "final_output")

    state = GraphState()
    result = graph.run(state, entry_node="greeting")

    print(f"Messages: {len(result.messages)}")
    print(f"Execution path: {' -> '.join(result.execution_path)}")

    assert len(result.messages) >= 2
    assert "final_output" in result.execution_path
    print("Test 1 PASSED")
    return True

def test_conditional_edge():
    """测试2:条件边路由"""
    def route_condition(state):
        last_msg = state.messages[-1].get("content", "").lower() if state.messages else ""
        return "llm_call" if "hi" in last_msg else "final_output"

    graph = (Graph()
        .register_node("greeting", greet_node)
        .register_node("router", greet_node)  # 简化测试
        .register_node("llm_call", llm_node)
        .register_node("final_output", output_node))

    graph.add_edge("greeting", "router")
    graph.add_conditional_edge("router", route_condition, ["llm_call", "final_output"])
    graph.add_edge("llm_call", "final_output")

    # 测试1:包含 "hi" -> 走 llm_call
    state1 = GraphState()
    state1.add_message("user", "Hi! Tell me about yourself.")
    result1 = graph.run(state1, entry_node="greeting")
    assert "llm_call" in result1.execution_path

    # 测试2:不包含 "hi" -> 走 final_output
    state2 = GraphState()
    state2.add_message("user", "Calculate 5 * 3")
    result2 = graph.run(state2, entry_node="greeting")
    assert "final_output" in result2.execution_path

    print("Test 2 PASSED")
    return True

def test_checkpoint():
    """测试3:检查点系统"""
    graph = (Graph()
        .register_node("greeting", greet_node)
        .register_node("llm_call", llm_node))
    graph.add_edge("greeting", "llm_call")
    graph.enable_checkpointer("checkpoints")

    state = GraphState()
    result = graph.run(state, entry_node="greeting")
    print("Test 3 PASSED")
    return True

def test_graph_builder():
    """测试4:Builder 模式"""
    from my_agent.graph.graph import GraphBuilder

    graph = GraphBuilder().entry_point("greeting").build()
    graph.register_node("greeting", greet_node)
    graph.add_edge("greeting", "llm_call")
    graph.register_node("llm_call", llm_node)

    state = GraphState()
    result = graph.run(state, entry_node="greeting")

    assert len(result.messages) >= 1
    print("Test 4 PASSED")
    return True

def run_all_tests():
    """运行所有测试"""
    tests = [
        ("Basic Graph", test_basic_graph),
        ("Conditional Edge", test_conditional_edge),
        ("Checkpoint", test_checkpoint),
        ("Builder", test_graph_builder),
    ]

    print("Running " + str(len(tests)) + " tests...")
    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print("FAILED: " + name + " - " + str(e))

    print(str(passed) + "/" + str(total) + " tests passed")
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
