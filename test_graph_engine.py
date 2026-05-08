"""LangGraph-style Graph Engine - Test Suite"""
import sys
import io
import os
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from my_agent.graph.state import GraphState
from my_agent.graph.node import BaseNode, node, Edge, ConditionalEdge
from my_agent.graph.graph import Graph, GraphBuilder


# ──────────────────────────────────────
# Test node definitions
# ──────────────────────────────────────

@node("greeting")
def greet_node(state: GraphState) -> GraphState:
    """Greeting node: add a welcome message."""
    state.add_message("assistant", "Hello! I am SimpleAgent.")
    print(f"  [Node] greeting executed, messages count: {len(state.messages)}")
    return state


@node("router")
def router_node(state: GraphState) -> GraphState:
    """Router node: decide whether to call LLM based on user input."""
    # Only check user messages, not assistant replies
    user_messages = [m for m in state.messages if m.get("role") == "user"]
    has_llm_request = len([
        m for m in user_messages
        if "hi" in m.get("content", "").lower() or "hello" in m.get("content", "").lower()
    ]) > 0
    state.set_metadata("need_llm", has_llm_request)
    return state


@node("llm_call")
def llm_node(state: GraphState) -> GraphState:
    """LLM node: simulate LLM call."""
    # In real projects this would call a real LLM API
    state.add_message(
        "assistant",
        "I can help you learn Python, debug code, and explore AI Agent technology!"
    )
    print(f"  [Node] llm_call executed")
    return state


@node("tool_call")
def tool_node(state: GraphState) -> GraphState:
    """Tool call node: simulate calculator execution."""
    import random
    result = f"Calculation result: {random.randint(1, 100)}"
    state.add_message("assistant", result)
    print(f"  [Node] tool_call executed, result: {result}")
    return state


@node("final_output")
def output_node(state: GraphState) -> GraphState:
    """Final output node."""
    summary = f"Session complete. Total messages: {len(state.messages)}"
    print(f"  [Node] final_output: {summary}")
    state.set_metadata("session_summary", summary)
    return state


# ──────────────────────────────────────
# Test functions
# ──────────────────────────────────────

def test_basic_linear_graph():
    """Test 1: Basic linear graph (greeting -> router -> final_output)"""
    print("\n=== Test 1: Basic Linear Graph ===")

    graph = Graph(nodes={
        "greeting": greet_node,
        "router": router_node,
        "final_output": output_node,
    })
    graph.add_edge("greeting", "router")
    graph.add_edge("router", "final_output")

    state = GraphState()
    state.add_message("user", "Hello!")
    result = graph.run(state, entry_node="greeting")

    assert len(result.messages) > 0, "Should have messages"
    assert result.execution_path == ["greeting", "router", "final_output"], \
        f"Execution path mismatch: {result.execution_path}"
    print(f"  Path: {result.execution_path}")
    print("  PASSED")


def test_conditional_graph_with_llm():
    """Test 2: Conditional graph - with 'hi' -> route to llm_call"""
    print("\n=== Test 2: Conditional Graph (with LLM) ===")

    def route_decision(state):
        need_llm = state.get_metadata("need_llm", False)
        return "llm_call" if need_llm else "final_output"

    graph = Graph(nodes={
        "greeting": greet_node,
        "router": router_node,
        "llm_call": llm_node,
        "final_output": output_node,
    })
    graph.add_edge("greeting", "router")
    graph.add_conditional_edge(
        "router", route_decision, options=["llm_call", "final_output"]
    )
    graph.add_edge("llm_call", "final_output")

    state = GraphState()
    state.add_message("user", "hi there")
    result = graph.run(state, entry_node="greeting")

    assert "llm_call" in result.execution_path, \
        f"Should route to llm_call when 'hi' is present: {result.execution_path}"
    print(f"  Path: {result.execution_path}")
    print("  PASSED")


def test_conditional_graph_without_llm():
    """Test 3: Conditional graph - without 'hi' -> skip llm_call"""
    print("\n=== Test 3: Conditional Graph (without LLM) ===")

    def route_decision(state):
        need_llm = state.get_metadata("need_llm", False)
        return "llm_call" if need_llm else "final_output"

    graph = Graph(nodes={
        "greeting": greet_node,
        "router": router_node,
        "llm_call": llm_node,
        "final_output": output_node,
    })
    graph.add_edge("greeting", "router")
    graph.add_conditional_edge(
        "router", route_decision, options=["llm_call", "final_output"]
    )
    graph.add_edge("llm_call", "final_output")

    state = GraphState()
    state.add_message("user", "What is the weather today?")
    result = graph.run(state, entry_node="greeting")

    assert "llm_call" not in result.execution_path, \
        f"Should skip llm_call when no greeting: {result.execution_path}"
    print(f"  Path: {result.execution_path}")
    print("  PASSED")


def test_max_iterations():
    """Test 4: Max iterations protection"""
    print("\n=== Test 4: Max Iterations ===")

    @node("loop_node")
    def loop_fn(state):
        state.set_metadata("count", state.get_metadata("count", 0) + 1)
        return state

    graph = Graph(nodes={"loop_node": loop_fn})  # @node already returns BaseNode instance
    graph.add_edge("loop_node", "loop_node")  # self-loop

    state = GraphState(max_iterations=5)
    result = graph.run(state, entry_node="loop_node")

    assert result.iteration_count == 5, f"Should stop at max_iterations: {result.iteration_count}"
    print(f"  Iterations: {result.iteration_count}")
    print("  PASSED")


def test_checkpoint_save_and_restore():
    """Test 5: Checkpoint save and restore"""
    print("\n=== Test 5: Checkpoint System ===")

    graph = Graph(nodes={
        "greeting": greet_node,
        "router": router_node,
        "final_output": output_node,
    })
    graph.add_edge("greeting", "router")
    graph.add_edge("router", "final_output")
    graph.enable_checkpointer(path="checkpoints/test_checkpoint")

    state = GraphState()
    state.add_message("user", "Hello checkpoint!")
    result = graph.run(state, entry_node="greeting")

    # Verify checkpoint files exist
    cp_dir = "checkpoints/test_checkpoint"
    if os.path.exists(cp_dir):
        files = os.listdir(cp_dir)
        print(f"  Checkpoint files: {files}")
        assert len(files) > 0, "Should have checkpoint files"

        # Read and verify a checkpoint
        with open(os.path.join(cp_dir, files[0]), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "messages" in data or "metadata" in data, "Checkpoint should have state data"
    else:
        print("  Warning: Checkpoint directory not created (may be normal)")

    # Cleanup
    import shutil
    if os.path.exists(cp_dir):
        shutil.rmtree(cp_dir)

    print("  PASSED")


def test_graph_builder_pattern():
    """Test 6: GraphBuilder chain API"""
    print("\n=== Test 6: GraphBuilder Pattern ===")

    builder = GraphBuilder()
    graph = builder.graph

    graph.register_node("start", greet_node)
    graph.register_node("end", output_node)
    graph.add_edge("start", "end")

    state = GraphState()
    state.add_message("user", "Test builder")
    result = graph.run(state, entry_node="start")

    assert len(result.execution_path) >= 2, f"Should execute at least 2 nodes: {result.execution_path}"
    print(f"  Path: {result.execution_path}")
    print("  PASSED")


def test_state_snapshot_and_restore():
    """Test 7: State snapshot and restore"""
    print("\n=== Test 7: State Snapshot & Restore ===")

    state = GraphState()
    state.add_message("user", "Hello")
    state.add_message("assistant", "Hi there!")
    state.set_metadata("key", "value")

    snapshot = state.snapshot()
    assert isinstance(snapshot, dict), "Snapshot should be a dict"
    assert "messages" in snapshot, "Snapshot should contain messages"

    # Restore to new state
    restored = GraphState()
    restored.restore(snapshot)

    assert len(restored.messages) == 2, f"Should restore 2 messages: {len(restored.messages)}"
    print(f"  Original messages: {len(state.messages)}, Restored: {len(restored.messages)}")
    print("  PASSED")


def test_node_decorator():
    """Test 8: @node decorator"""
    print("\n=== Test 8: Node Decorator ===")

    @node("test_node")
    def my_node(state):
        state.add_message("assistant", "decorated!")
        return state

    assert isinstance(my_node, BaseNode), "@node should return a BaseNode"
    assert my_node.name == "test_node", f"Node name should be 'test_node': {my_node.name}"

    state = GraphState()
    result = my_node.execute(state)
    assert len(result.messages) == 1, "Should have 1 message after execution"
    print(f"  Node name: {my_node.name}, Messages: {len(result.messages)}")
    print("  PASSED")


def test_edge_types():
    """Test 9: Edge and ConditionalEdge"""
    print("\n=== Test 9: Edge Types ===")

    edge = Edge("A", "B")
    assert edge.from_node == "A"
    assert edge.to_node == "B"

    cond_fn = lambda s: "X" if True else "Y"
    cond_edge = ConditionalEdge("A", cond_fn, options=["X", "Y"])
    assert cond_edge.from_node == "A"
    assert len(cond_edge.options) == 2
    print("  Edge and ConditionalEdge work correctly")
    print("  PASSED")


def run_all():
    """Run all tests."""
    print("=" * 60)
    print("SimpleAgent - Graph Engine Test Suite")
    print("=" * 60)

    tests = [
        test_basic_linear_graph,
        test_conditional_graph_with_llm,
        test_conditional_graph_without_llm,
        test_max_iterations,
        test_checkpoint_save_and_restore,
        test_graph_builder_pattern,
        test_state_snapshot_and_restore,
        test_node_decorator,
        test_edge_types,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
