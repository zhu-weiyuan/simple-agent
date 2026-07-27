"""Regression tests for Agent Card metadata."""

from my_agent.agent import SimpleAgent


def test_agent_card_reports_registered_tools():
    agent = SimpleAgent(enable_enhanced=False)
    try:
        card = agent.card()
        expected_tools = agent.tool_registry.all_names()

        assert card.tools == expected_tools
        assert card.capabilities["tools"] == len(expected_tools)
        assert expected_tools
    finally:
        agent.close()
