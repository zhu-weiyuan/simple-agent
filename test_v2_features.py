# -*- coding: utf-8 -*-
"""
test_v2_features.py — SimpleAgent v2.0 新功能测试

测试内容:
1. AgentBase Protocol (invoke, stream, card)
2. Agent-as-Tool (as_tool, add_tool)
3. A2A Protocol (AgentCard, A2AMessage, TaskStatus)
4. StructuredOutput (JSON extraction, schema validation)
5. Multi-agent (SupervisorAgent, AgentChain, ParallelAgent)
"""
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import unittest
from unittest.mock import MagicMock, patch


class TestAgentTypes(unittest.TestCase):
    """测试 types/agent.py 新类型"""

    def test_stop_reason_enum(self):
        from my_agent.types.agent import StopReason
        self.assertEqual(StopReason.COMPLETE.value, "complete")
        self.assertEqual(StopReason.ERROR.value, "error")
        self.assertEqual(StopReason.MAX_TOKENS.value, "max_tokens")

    def test_agent_metrics(self):
        from my_agent.types.agent import AgentMetrics
        m = AgentMetrics(input_tokens=100, output_tokens=50, tool_calls=3)
        self.assertEqual(m.context_size, 100)
        self.assertEqual(m.projected_context_size, 150)

    def test_agent_result(self):
        from my_agent.types.agent import AgentResult, AgentMetrics, StopReason
        from my_agent.types.message import Message

        result = AgentResult(
            stop_reason=StopReason.COMPLETE,
            message=Message.assistant("Hello!"),
            metrics=AgentMetrics(input_tokens=10, output_tokens=5),
        )
        self.assertEqual(result.text(), "Hello!")
        d = result.to_dict()
        self.assertEqual(d["stop_reason"], "complete")
        self.assertEqual(d["message"]["role"], "assistant")

    def test_agent_card(self):
        from my_agent.types.agent import AgentCard

        card = AgentCard(
            name="TestAgent",
            description="A test agent",
            version="2.0.0",
            tools=["tool1", "tool2"],
        )
        self.assertEqual(card.name, "TestAgent")
        d = card.to_dict()
        self.assertIn("tool1", d["tools"])
        json_str = card.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["version"], "2.0.0")


class TestA2AProtocol(unittest.TestCase):
    """测试 A2A 协议实现"""

    def test_a2a_message(self):
        from my_agent.a2a import A2AMessage, MessageType

        msg = A2AMessage(content="Hello", type=MessageType.PROMPT)
        self.assertIsNotNone(msg.message_id)
        d = msg.to_dict()
        self.assertEqual(d["type"], "prompt")
        self.assertEqual(d["content"], "Hello")

        # Round-trip
        msg2 = A2AMessage.from_dict(d)
        self.assertEqual(msg2.content, "Hello")

    def test_task_status(self):
        from my_agent.a2a import TaskStatus, TaskState

        task = TaskStatus(task_id="test-1", state=TaskState.WORKING)
        self.assertEqual(task.state, TaskState.WORKING)
        self.assertGreaterEqual(task.duration, 0)
        d = task.to_dict()
        self.assertEqual(d["state"], "working")

    def test_a2a_agent_card(self):
        from my_agent.a2a import AgentCard

        card = AgentCard(
            name="RemoteAgent",
            description="A remote agent",
            url="http://example.com/agent",
        )
        self.assertEqual(card.preferred_transport, "http")
        d = card.to_dict()
        self.assertEqual(d["url"], "http://example.com/agent")

        # from_dict
        card2 = AgentCard.from_dict(d)
        self.assertEqual(card2.name, "RemoteAgent")


class TestStructuredOutput(unittest.TestCase):
    """测试结构化输出工具"""

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "my_agent", "tools"))
        from structured_output import StructuredOutputTool
        self.tool = StructuredOutputTool()

    def test_extract_json_plain(self):
        text = '{"name": "Alice", "age": 30}'
        result = self.tool.extract(text)
        self.assertEqual(result["name"], "Alice")
        self.assertEqual(result["age"], 30)

    def test_extract_json_markdown_block(self):
        text = "Here is the result:\n```json\n{\"city\": \"Beijing\", \"temp\": 25}\n```"
        result = self.tool.extract(text)
        self.assertEqual(result["city"], "Beijing")

    def test_extract_json_braces(self):
        text = "The data is {\"x\": 1, \"y\": 2} and that's it."
        result = self.tool.extract(text)
        self.assertEqual(result["x"], 1)

    def test_build_system_prompt(self):
        class WeatherInfo:
            __annotations__ = {
                "city": str,
                "temperature": float,
                "condition": str,
            }

        tool = self.tool.__class__(schema=WeatherInfo)
        prompt = tool.build_system_prompt()
        self.assertIn("JSON", prompt)
        self.assertIn("city", prompt)

    def test_schema_from_class(self):
        class Person:
            __annotations__ = {"name": str, "age": int}

        tool = self.tool.__class__(schema=Person)
        schema = tool.json_schema
        self.assertEqual(schema["type"], "object")
        self.assertIn("name", schema["properties"])
        self.assertEqual(schema["properties"]["name"]["type"], "string")


class TestAgentAsTool(unittest.TestCase):
    """测试 Agent-as-Tool 适配器"""

    def test_agent_as_tool_creation(self):
        from my_agent.tools.agent_tool import AgentAsTool

        mock_agent = MagicMock()
        mock_agent.run.return_value = "Research result"
        mock_agent.engine.session.messages = []

        tool = AgentAsTool(mock_agent, name="research", description="Do research")
        self.assertEqual(tool.name, "research")
        self.assertEqual(tool.description, "Do research")

    def test_agent_as_tool_execute(self):
        from my_agent.tools.agent_tool import AgentAsTool

        mock_agent = MagicMock()
        mock_agent.run.return_value = "Research: AI is great"
        mock_agent.engine.session.messages = []

        tool = AgentAsTool(mock_agent, name="research")
        result = tool.execute({"input": "Tell me about AI"})
        self.assertIn("AI", result)
        mock_agent.run.assert_called_once_with("Tell me about AI")

    def test_create_agent_tool_helper(self):
        from my_agent.tools.agent_tool import create_agent_tool, AgentAsTool

        mock_agent = MagicMock()
        mock_agent.engine.session.messages = []

        tool = create_agent_tool(mock_agent, name="test", description="Test tool")
        self.assertIsInstance(tool, AgentAsTool)
        self.assertEqual(tool.name, "test")


class TestMultiAgent(unittest.TestCase):
    """测试多 agent 编排"""

    def test_supervisor_creation(self):
        from my_agent.multiagent import SupervisorAgent, AgentRole

        mock_agent = MagicMock()
        supervisor = SupervisorAgent(
            name="test-supervisor",
            roles=[
                AgentRole("researcher", "Research tasks", mock_agent),
            ]
        )
        self.assertEqual(len(supervisor.roles), 1)
        self.assertEqual(supervisor.roles[0].name, "researcher")

    def test_supervisor_add_role(self):
        from my_agent.multiagent import SupervisorAgent, AgentRole

        supervisor = SupervisorAgent()
        mock_agent = MagicMock()
        supervisor.add_role(AgentRole("coder", "Code tasks", mock_agent))
        self.assertEqual(len(supervisor.roles), 1)

    def test_supervisor_decide_keywords(self):
        from my_agent.multiagent import SupervisorAgent, AgentRole

        mock_researcher = MagicMock()
        mock_writer = MagicMock()

        supervisor = SupervisorAgent(roles=[
            AgentRole("researcher", "Research and analysis", mock_researcher),
            AgentRole("writer", "Writing and editing", mock_writer),
        ])

        # Test keyword routing
        choice = supervisor._decide("帮我搜索一下AI的最新发展")
        self.assertEqual(choice, "researcher")

    def test_agent_chain(self):
        from my_agent.multiagent import AgentChain

        mock_a = MagicMock()
        mock_a.run.return_value = "Step A result"
        mock_b = MagicMock()
        mock_b.run.return_value = "Step B result"

        chain = AgentChain([("step_a", mock_a), ("step_b", mock_b)])
        result = chain.run("Initial input")

        self.assertIn("Step A", result.final_response)
        self.assertIn("Step B", result.final_response)

    def test_parallel_agent(self):
        from my_agent.multiagent import ParallelAgent

        mock_a = MagicMock()
        mock_a.run.return_value = "Analysis A"
        mock_b = MagicMock()
        mock_b.run.return_value = "Analysis B"

        parallel = ParallelAgent([("summary", mock_a), ("keypoints", mock_b)])
        result = parallel.run("Analyze this")

        self.assertIn("Analysis A", result.final_response)
        self.assertIn("Analysis B", result.final_response)


class TestSimpleAgentV2(unittest.TestCase):
    """测试 SimpleAgent v2 新方法（不需要 LLM）"""

    def test_agent_card(self):
        from my_agent.agent import SimpleAgent
        agent = SimpleAgent.__new__(SimpleAgent)
        agent.name = "TestAgent"
        agent.description = "A test agent"
        agent.version = "2.0.0"
        agent.enable_enhanced = False
        agent._mcp_client = None
        agent.tool_registry = MagicMock()
        agent.tool_registry._handlers = {"calc": None, "time": None}

        card = agent.card()
        self.assertEqual(card.name, "TestAgent")
        self.assertIn("calc", card.tools)

    def test_as_tool(self):
        from my_agent.agent import SimpleAgent
        from my_agent.tools.agent_tool import AgentAsTool

        agent = SimpleAgent.__new__(SimpleAgent)
        agent.name = "MyAgent"
        agent.description = "Does stuff"
        agent.engine = MagicMock()
        agent.engine.session.messages = []

        tool = agent.as_tool()
        self.assertIsInstance(tool, AgentAsTool)
        self.assertEqual(tool.name, "MyAgent")

    def test_invoke_empty(self):
        from my_agent.agent import SimpleAgent
        from my_agent.types.agent import StopReason

        agent = SimpleAgent.__new__(SimpleAgent)
        result = agent.invoke(None)
        self.assertEqual(result.stop_reason, StopReason.ERROR)

    def test_stream_empty(self):
        from my_agent.agent import SimpleAgent

        agent = SimpleAgent.__new__(SimpleAgent)
        result = list(agent.stream(None))
        self.assertEqual(len(result), 1)

    def test_add_tool_callable(self):
        from my_agent.agent import SimpleAgent
        from my_agent.tools.registry import ToolRegistry

        agent = SimpleAgent.__new__(SimpleAgent)
        agent.tool_registry = ToolRegistry()
        agent.debug_enabled = False

        def my_tool(params):
            return "hello"
        my_tool.__doc__ = "My custom tool"

        agent.add_tool(my_tool)
        handler = agent.tool_registry.get_handler("my_tool")
        self.assertIsNotNone(handler)


if __name__ == "__main__":
    unittest.main(verbosity=2)
