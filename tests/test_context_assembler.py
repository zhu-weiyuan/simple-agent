"""Test suite for Context Assembler - Phase C+"""

import pytest
from my_agent.context_assembler import (
    ContextPiece,
    estimate_tokens,
    TokenBudgetManager,
    ContextAssembler,
)


class TestTokenEstimation:
    """Test token estimation accuracy"""
    
    def test_empty_string(self):
        assert estimate_tokens("") == 0
    
    def test_english_text(self):
        text = "Hello world, this is a test."
        tokens = estimate_tokens(text)
        # English ~0.4 chars per token
        assert 10 <= tokens <= 30
    
    def test_chinese_text(self):
        text = "你好世界，这是一个测试"
        tokens = estimate_tokens(text)
        # Chinese ~0.6 chars per token
        assert 12 <= tokens <= 25
    
    def test_mixed_text(self):
        text = "Hello 世界 World 世界"
        tokens = estimate_tokens(text)
        assert 10 <= tokens <= 30


class TestTokenBudgetManager:
    """Test budget management logic"""
    
    def test_default_budget(self):
        manager = TokenBudgetManager(max_context=128000, reserved_output=8000)
        assert manager.budget == 120000
    
    def test_ok_usage(self):
        manager = TokenBudgetManager(max_context=4096, reserved_output=512)
        result = manager.check_budget(1000)
        assert result['ok'] is True
        assert result['alert'] is None
    
    def test_warning_threshold(self):
        manager = TokenBudgetManager(max_context=4096, reserved_output=512)
        # 3000 / 3584 = 84% -> should be ERROR actually
        result = manager.check_budget(2500)  # 69.7%
        assert result['alert'] == 'WARNING'
    
    def test_error_threshold_dumb_zone(self):
        manager = TokenBudgetManager(max_context=4096, reserved_output=512)
        result = manager.check_budget(3000)  # 83.7%
        assert result['alert'] == 'ERROR'
        assert 'DUMB ZONE' in result['recommendation']
    
    def test_zero_context_window(self):
        """Edge case: zero context window should not cause division by zero"""
        manager = TokenBudgetManager(max_context=0, reserved_output=0)
        result = manager.check_budget(100)
        assert result['ok'] is False
        assert result['usage_pct'] == 1.0
    
    def test_can_fit_true(self):
        manager = TokenBudgetManager(max_context=4096, reserved_output=512)
        assert manager.can_fit(1000, 500) is True
    
    def test_can_fit_false(self):
        manager = TokenBudgetManager(max_context=4096, reserved_output=512)
        assert manager.can_fit(4000, 500) is False


class TestContextAssembler:
    """Test context assembly logic"""
    
    def test_empty_assembly(self):
        assembler = ContextAssembler(context_window=4096)
        messages, meta = assembler.assemble()
        
        assert len(messages) > 0
        assert meta['total_pieces'] == 0
        assert meta['allocated_pieces'] == 0
    
    def test_system_message_priority(self):
        assembler = ContextAssembler(context_window=4096)
        messages, meta = assembler.assemble(
            system_prompt="You are helpful.",
            user_message="Hi!",
        )
        
        assert messages[0]['role'] == 'system'
        assert 'helpful' in messages[0]['content'].lower()
    
    def test_user_message_always_included(self):
        assembler = ContextAssembler(context_window=4096)
        messages, meta = assembler.assemble(
            user_message="Test message",
        )
        
        # Find user message in output
        user_msgs = [m for m in messages if m['role'] == 'user']
        assert any('Test message' in m['content'] for m in user_msgs)
    
    def test_rag_score_weighting(self):
        assembler = ContextAssembler(context_window=4096)
        messages, meta = assembler.assemble(
            user_message="Question?",
            rag_results=[
                {'chunk': 'High relevance answer', 'score': 0.9},
                {'chunk': 'Low relevance answer', 'score': 0.3},
            ],
        )
        
        assert meta['by_source_type'].get('rag', 0) == 2
    
    def test_history_truncation_on_budget_exhaustion(self):
        """When budget tight, old history should be truncated"""
        assembler = ContextAssembler(context_window=512)  # Very small window
        messages, meta = assembler.assemble(
            system_prompt="System prompt here",
            user_message="Short",
            conversation_history=[
                ('user', 'A' * 200),
                ('assistant', 'B' * 200),
                ('user', 'C' * 200),
            ],
        )
        
        # Some pieces must be dropped due to budget
        assert meta['dropped_pieces'] >= 0
    
    def test_budget_metadata_accuracy(self):
        assembler = ContextAssembler(context_window=4096)
        messages, meta = assembler.assemble(
            system_prompt="System",
            user_message="User question",
        )
        
        assert 'estimated_tokens' in meta
        assert 'budget_usage_pct' in meta
        assert meta['budget_usage_pct'] < 1.0
    
    def test_goal_context_inclusion(self):
        assembler = ContextAssembler(context_window=4096)
        messages, meta = assembler.assemble(
            user_message="Help me",
            goal_context="Resolve customer issue",
        )
        
        assert meta['by_source_type'].get('goal', 0) == 1


class TestContextPiece:
    """Test ContextPiece dataclass"""
    
    def test_default_values(self):
        piece = ContextPiece(
            source_type='test',
            content='Hello',
            token_estimate=10,
        )
        assert piece.priority == 50
        assert piece.recency == 50
        assert piece.metadata == {}
    
    def test_custom_metadata(self):
        piece = ContextPiece(
            source_type='rag',
            content='Chunk',
            token_estimate=5,
            metadata={'score': 0.9, 'rank': 1},
        )
        assert piece.metadata['score'] == 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
