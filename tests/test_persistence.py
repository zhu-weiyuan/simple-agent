import pytest
from my_agent.prompt_store import PromptStore

def test_prompt_store():
    store = PromptStore()
    store.add_version("test prompt", 100.0)
    assert store.get_current_prompt() == "test prompt"