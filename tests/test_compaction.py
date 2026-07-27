import pytest
from my_agent.context_assembler import ContextAssembler

def test_summarize():
    assembler = ContextAssembler(max_tokens=20)
    for i in range(10):
        assembler.add_message("user", f"Message {i}")
    
    context = assembler.get_context()
    assert len(context) <= 6  # 1 summary + 5 recent

def test_prune():
    assembler = ContextAssembler(max_tokens=100)
    assembler.add_message("user", "High priority", priority=2)
    assembler.add_message("user", "Low priority", priority=1)
    
    context = assembler.prune_low_priority(assembler.context)
    assert len(context) == 1
    assert context[0]['content'] == "High priority"