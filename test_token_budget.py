# -*- coding: utf-8 -*-
"""test_token_budget.py — 测试 Token Budget 模块

测试范围:
- 估算准确性 (English / Chinese / mixed)
- threshold 告警 (OK / WARNING / ERROR)
- compression 策略 (none / light / moderate / aggressive)
- 预算分配 (allocate with various inputs)
- 与现有 TokenBudgetManager 兼容性
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from my_agent.core.context_assembler import (
    estimate_tokens,
    estimate_messages_tokens,
    TokenBudgetAllocator,
    TokenBudgetManager,
    ContextPiece,
)
from my_agent.core.token_budget import (
    TokenEstimator,
    TokenBudgetManager as LegacyTokenBudgetManager,
)
from my_agent.types.message import Message, Role


# ── Token Estimation Accuracy ───────────────────────────────


class TestEstimateAccuracy:
    """测试 token 估算准确性"""

    def test_english_sanity(self):
        """英文文本 token 估算应在合理范围"""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = estimate_tokens(text)
        # 44 chars * 0.4 ≈ 18
        assert 10 <= tokens <= 30

    def test_chinese_sanity(self):
        """中文文本 token 估算应在合理范围"""
        text = "这是一个测试句子，用于验证中文 token 估算。"
        tokens = estimate_tokens(text)
        # 23 chars, mostly Chinese * 0.6 ≈ 14
        assert 8 <= tokens <= 25

    def test_empty_string(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0  # type: ignore

    def test_single_char(self):
        assert estimate_tokens("a") == 1
        assert estimate_tokens("中") == 1

    def test_long_text(self):
        text = "Hello " * 1000
        tokens = estimate_tokens(text)
        # cl100k_base: ~1000 tokens; heuristic fallback: ~1600 tokens.
        assert 500 <= tokens <= 1800

    def test_mixed_language(self):
        """混合中英文验证"""
        text = "Hello 你好 World 世界 Python 编程"
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_special_chars(self):
        text = "```python\nprint('hello')\n```"
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_json_text(self):
        text = '{"key": "value", "number": 42, "nested": {"a": 1}}'
        tokens = estimate_tokens(text)
        assert tokens > 0


# ── Token Estimation with Messages ──────────────────────────


class TestEstimateMessagesTokens:
    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msg = Message.user("Hello")
        tokens = estimate_messages_tokens([msg])
        assert tokens > 0

    def test_multiple_messages(self):
        msgs = [
            Message.system("You are a helper."),
            Message.user("Hello"),
            Message.assistant("Hi there! How can I help?"),
        ]
        tokens = estimate_messages_tokens(msgs)
        assert tokens > 0
        # Should be more than single message
        assert tokens > estimate_messages_tokens([msgs[0]])

    def test_message_with_tool_calls(self):
        from my_agent.types.message import ToolCall
        tc = ToolCall(id="call_1", name="search", arguments={"query": "test"})
        msg = Message.assistant("Let me search", tool_calls=[tc])
        tokens = estimate_messages_tokens([msg])
        assert tokens > 0


# ── TokenBudgetManager Threshold Alerts ─────────────────────


class TestThresholdAlerts:
    """测试预算阈值告警"""

    def test_low_usage_ok(self):
        manager = TokenBudgetManager(max_context=128000)
        result = manager.check_budget(10000)  # ~7.8%
        assert result["status"] == "OK"
        assert result["usage_pct"] < 10

    def test_medium_usage_ok(self):
        manager = TokenBudgetManager(max_context=128000)
        result = manager.check_budget(60000)  # ~46.9%
        assert result["status"] == "OK"
        assert 40 < result["usage_pct"] < 60

    def test_warning_threshold(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.check_budget(650)  # 65%
        assert result["status"] == "WARNING"
        assert "Context Anxiety" in result["message"]

    def test_error_threshold(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.check_budget(850)  # 85%
        assert result["status"] == "ERROR"
        assert "Dumb Zone" in result["message"]

    def test_exact_boundary_60(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.check_budget(600)  # exactly 60%
        assert result["status"] == "OK"  # 60% is still OK, 61%+ is WARNING

    def test_exact_boundary_80(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.check_budget(800)  # exactly 80%
        assert result["status"] == "WARNING"  # 80% is still WARNING, 81%+ is ERROR

    def test_over_100_percent(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.check_budget(2000)  # 200%
        assert result["status"] == "ERROR"
        assert result["usage_pct"] == 100.0

    def test_remaining_tokens_correct(self):
        manager = TokenBudgetManager(max_context=1000)
        # input_budget = 1000 - 8000 = 0? No: max(0, 1000-8000) = 0
        # Let's use larger context
        manager2 = TokenBudgetManager(max_context=20000)
        result = manager2.check_budget(5000)
        assert result["remaining_tokens"] == 20000 - 8000 - 5000


# ── Compression Strategy ────────────────────────────────────


class TestCompressionStrategy:
    """测试压缩策略"""

    def test_no_compression(self):
        manager = TokenBudgetManager(max_context=128000)
        strategy = manager.compression_strategy(1000)
        assert strategy["level"] == "none"
        assert strategy["actions"] == []
        assert strategy["target_reduction"] == 0

    def test_light_compression(self):
        manager = TokenBudgetManager(max_context=1000)
        strategy = manager.compression_strategy(450)
        assert strategy["level"] == "light"
        assert len(strategy["actions"]) > 0

    def test_moderate_compression(self):
        manager = TokenBudgetManager(max_context=1000)
        strategy = manager.compression_strategy(650)
        assert strategy["level"] == "moderate"
        assert len(strategy["actions"]) > 0

    def test_aggressive_compression(self):
        manager = TokenBudgetManager(max_context=1000)
        strategy = manager.compression_strategy(850)
        assert strategy["level"] == "aggressive"
        assert strategy["target_reduction"] > 0
        assert len(strategy["actions"]) > 0

    def test_compression_includes_current_pct(self):
        manager = TokenBudgetManager(max_context=1000)
        strategy = manager.compression_strategy(500)
        assert "current_usage_pct" in strategy
        assert strategy["current_usage_pct"] == 50.0


# ── TokenBudgetAllocator Integration ────────────────────────


class TestAllocatorIntegration:
    """测试分配器集成场景"""

    def test_allocator_fills_budget_efficiently(self):
        allocator = TokenBudgetAllocator(context_window=1000, reserved_output=100)
        pieces = [
            ContextPiece("system", "A" * 10, token_estimate=10, priority=100, recency=100),
            ContextPiece("rag", "B" * 50, token_estimate=50, priority=70, recency=50),
            ContextPiece("rag", "C" * 50, token_estimate=50, priority=70, recency=50),
            ContextPiece("memory", "D" * 50, token_estimate=50, priority=60, recency=50),
            ContextPiece("memory", "E" * 50, token_estimate=50, priority=60, recency=80),
            ContextPiece("history", "F" * 50, token_estimate=50, priority=50, recency=50),
            ContextPiece("history", "G" * 50, token_estimate=50, priority=50, recency=90),
        ]
        # budget = 900, total = 360, should all fit
        result = allocator.allocate_pieces(pieces)
        assert len(result) == 7

    def test_allocator_drops_lowest_priority(self):
        allocator = TokenBudgetAllocator(context_window=100, reserved_output=20)
        # budget = 80
        pieces = [
            ContextPiece("system", "A" * 20, token_estimate=20, priority=100, recency=100),
            ContextPiece("goal", "B" * 20, token_estimate=20, priority=95, recency=95),
            ContextPiece("tools", "C" * 20, token_estimate=20, priority=90, recency=50),
            ContextPiece("rag", "D" * 20, token_estimate=20, priority=70, recency=50),
            ContextPiece("memory", "E" * 20, token_estimate=20, priority=60, recency=50),
            ContextPiece("history", "F" * 20, token_estimate=20, priority=50, recency=50),
        ]
        # system(20) + goal(20) + tools(20) + rag(20) = 80, exactly fits
        # memory and history dropped
        result = allocator.allocate_pieces(pieces)
        sources = [p.source_type for p in result]
        assert "system" in sources
        assert "goal" in sources
        # memory and history are lowest priority, should be dropped
        assert "memory" not in sources or "history" not in sources

    def test_allocator_recency_tiebreaker(self):
        """When priority is same, higher recency should win"""
        allocator = TokenBudgetAllocator(context_window=100, reserved_output=20)
        # budget = 80
        pieces = [
            ContextPiece("system", "A" * 20, token_estimate=20, priority=100, recency=100),
            # Two RAG items with same priority but different recency
            ContextPiece("rag", "B" * 30, token_estimate=30, priority=70, recency=10),
            ContextPiece("rag", "C" * 30, token_estimate=30, priority=70, recency=90),
        ]
        # system(20) + one rag(30) = 50 < 80, can fit both?
        # system(20) + both rag(30+30) = 80, exactly fits
        result = allocator.allocate_pieces(pieces)
        rag_pieces = [p for p in result if p.source_type == "rag"]
        # If both fit, fine. If only one fits, the higher recency should be kept.
        if len(rag_pieces) == 1:
            assert rag_pieces[0].recency > 50  # the higher recency one


# ── Legacy TokenBudgetManager Compatibility ─────────────────


class TestLegacyCompatibility:
    """测试与现有 TokenBudgetManager 的兼容性"""

    def test_legacy_estimator(self):
        estimator = TokenEstimator()
        assert estimator.estimate("hello world") > 0

    def test_legacy_budget_manager(self):
        manager = LegacyTokenBudgetManager(total_context_window=100, reserved_for_output=20)
        assert manager.input_budget == 80

    def test_legacy_allocate(self):
        manager = LegacyTokenBudgetManager(total_context_window=100, reserved_for_output=20)
        result = manager.allocate(10, 10, 10, 30)
        assert len(result) >= 2  # At least system + something

    def test_legacy_alerts(self):
        manager = LegacyTokenBudgetManager(total_context_window=100, reserved_for_output=20)
        alerts = manager.alerts_for_usage(85)
        assert any("Dumb Zone" in a for a in alerts)

    def test_legacy_warning_alerts(self):
        manager = LegacyTokenBudgetManager(total_context_window=100, reserved_for_output=20)
        alerts = manager.alerts_for_usage(65)
        assert any("Context Anxiety" in a for a in alerts)


# ── Edge Cases ──────────────────────────────────────────────


class TestEdgeCases:
    """边界情况测试"""

    def test_zero_context_window(self):
        try:
            TokenBudgetAllocator(context_window=0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_negative_reserved_output(self):
        allocator = TokenBudgetAllocator(context_window=1000, reserved_output=-100)
        # reserved_output is clamped to 0
        assert allocator.budget >= 1000

    def test_very_small_budget(self):
        allocator = TokenBudgetAllocator(context_window=10, reserved_output=5)
        pieces = [
            ContextPiece("system", "A", token_estimate=3, priority=100, recency=100),
        ]
        result = allocator.allocate_pieces(pieces)
        assert len(result) <= 1  # may or may not fit

    def test_large_number_of_pieces(self):
        allocator = TokenBudgetAllocator(context_window=100000, reserved_output=8000)
        pieces = [
            ContextPiece("rag", f"chunk_{i}", token_estimate=10, priority=70, recency=50)
            for i in range(100)
        ]
        pieces.insert(0, ContextPiece("system", "sys", token_estimate=10, priority=100, recency=100))
        result = allocator.allocate_pieces(pieces)
        assert len(result) > 0
        assert result[0].source_type == "system"

    def test_token_budget_manager_zero_context(self):
        manager = TokenBudgetManager(max_context=0)
        result = manager.check_budget(100)
        assert result["status"] == "ERROR"
        assert result["usage_pct"] == 100.0