import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors="replace")

os.environ["OPENAI_API_KEY"] = "your_key_here"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8080"
os.environ["OPENAI_MODEL"] = "Qwen3.6-27B-IQ4_NL.gguf"

from my_agent import SimpleAgent, SupervisorAgent, AgentChain, ParallelAgent, StructuredOutputTool, AgentAsTool

print("=" * 60)
print("SimpleAgent v2.0 — Demo")
print("=" * 60)

# 1. Basic chat
print("\n[1] Basic Chat")
agent = SimpleAgent()
r = agent.run("你好，请简单介绍一下你自己")
print(f"  Reply ({len(r)} chars): {r[:120]}...")

# 2. Tool use - time
print("\n[2] Tool Use (get_time)")
r = agent.run("现在几点了？")
print(f"  Reply: {r[:120]}...")

# 3. Tool use - calculator
print("\n[3] Tool Use (calculator)")
r = agent.run("计算 123 * 456")
print(f"  Reply: {r.strip()}")

# 4. Agent Card
print("\n[4] Agent Card (A2A Protocol)")
card = agent.card()
print(f"  Name: {card.name}")
print(f"  Version: {card.version}")
print(f"  Tools: {len(card.tools)} registered")
print(f"  Capabilities: {card.capabilities}")

# 5. Structured Output
print("\n[5] StructuredOutputTool")
tool = StructuredOutputTool()
result = tool.extract('{"city": "Beijing", "temp": 25, "condition": "sunny"}')
print(f"  Parsed: city={result['city']}, temp={result['temp']}")

# 6. Agent-as-Tool
print("\n[6] Agent-as-Tool")
researcher = SimpleAgent(name="researcher", description="Research specialist")
tool_wrapper = researcher.as_tool(name="research")
print(f"  Tool name: {tool_wrapper.name}")
print(f"  Tool type: agent_tool")

# 7. Multi-Agent (Supervisor)
print("\n[7] SupervisorAgent (Multi-Agent Routing)")
supervisor = SupervisorAgent()
mock_a = SimpleAgent.__new__(SimpleAgent)
from my_agent.multiagent import AgentRole
supervisor.add_role(AgentRole("researcher", "Research tasks", mock_a))
supervisor.add_role(AgentRole("writer", "Writing tasks", mock_a))
choice = supervisor._decide("帮我搜索一下AI的最新发展")
print(f"  Query: '搜索AI最新发展' → routed to: {choice}")

# 8. Stats
print("\n[8] Project Stats")
import glob
py_files = glob.glob(str(Path(__file__).resolve().parent / "src" / "my_agent" / "**" / "*.py"), recursive=True)
total_lines = sum(len(open(f).readlines()) for f in py_files)
print(f"  Python files: {len(py_files)}")
print(f"  Total lines: ~{total_lines}")
print(f"  Modules: types, core, tools, memory, bridge, enhanced, graph, llm, a2a, multiagent")

agent.close()
researcher.close()

print("\n" + "=" * 60)
print("All demos completed successfully!")
print("=" * 60)
