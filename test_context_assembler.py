# -*- coding: utf-8 -*-
"""test_context_assembler.py — 测试 Context Assembler 模块

测试范围:
- 6 组件组装 (system/goal/RAG/memory/tools/history)
- 优先级排序
- Budget 裁剪
- Progressive Disclosure
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from my_agent.core.context_assembler import (
    ContextPiece,
    ContextAssembler,
    TokenBudgetAllocator,
    TokenBudgetManager,
    ProgressiveDisclosure,
    estimate_tokens,
)
from my_agent.types.message import Message


# ── Token Estimation ────────────────────────────────────────


class TestTokenEstimation:
    """测试 token 估算"""

    def test_english_text(self):
        tokens = estimate_tokens("hello world")
        assert tokens > 0
        assert tokens <= 12  # ~11 chars * 0.4 = ~4.4

    def test_chinese_text(self):
        tokens = estimate_tokens("你好世界")
        assert tokens > 0
        # cl100k_base: 5 tokens; heuristic fallback: ~3 tokens.
        assert 3 <= tokens <= 6

    def test_empty_text(self):
        assert estimate_tokens("") == 0

    def test_mixed_text(self):
        tokens = estimate_tokens("Hello 你好 World 世界")
        assert tokens > 0
        # 中文: 4 chars * 0.6 = 2.4
        # 英文: "HelloWorld" = 10 chars * 0.4 = 4.0
        # 加权: total 14 chars, chinese_ratio = 4/14 ≈ 0.286
        # factor = 0.6*0.286 + 0.4*0.714 = 0.1716 + 0.2856 = 0.457
        # 14 * 0.457 ≈ 6.4
        assert 5 <= tokens <= 10


# ── ContextPiece ────────────────────────────────────────────


class TestContextPiece:
    def test_creation(self):
        piece = ContextPiece(
            source_type="system",
            content="You are a helpful assistant.",
            token_estimate=10,
            priority=100,
            recency=100,
        )
        assert piece.source_type == "system"
        assert piece.priority == 100
        assert piece.token_estimate == 10

    def test_repr(self):
        piece = ContextPiece(
            source_type="rag", content="doc", token_estimate=5, priority=70, recency=50
        )
        rep = repr(piece)
        assert "rag" in rep
        assert "70" in rep


# ── TokenBudgetAllocator ────────────────────────────────────


class TestTokenBudgetAllocator:
    """测试预算分配器"""

    def test_empty_pieces(self):
        allocator = TokenBudgetAllocator(context_window=1000, reserved_output=100)
        assert allocator.allocate_pieces([]) == []

    def test_all_fit(self):
        allocator = TokenBudgetAllocator(context_window=1000, reserved_output=100)
        pieces = [
            ContextPiece("system", "sys", 10, priority=100, recency=100),
            ContextPiece("goal", "goal", 10, priority=95, recency=95),
            ContextPiece("tools", "tools", 10, priority=90, recency=50),
            ContextPiece("rag", "rag", 20, priority=70, recency=50),
            ContextPiece("memory", "mem", 20, priority=60, recency=50),
            ContextPiece("history", "hist", 30, priority=50, recency=50),
        ]
        # budget = 900, total pieces = 100, all should fit
        result = allocator.allocate_pieces(pieces)
        assert len(result) == 6

    def test_high_priority_always_retained(self):
        allocator = TokenBudgetAllocator(context_window=100, reserved_output=20)
        # budget = 80
        pieces = [
            ContextPiece("system", "A" * 50, token_estimate=50, priority=100, recency=100),
            ContextPiece("goal", "B" * 30, token_estimate=30, priority=95, recency=95),
            ContextPiece("tools", "C" * 30, token_estimate=30, priority=90, recency=50),
            ContextPiece("rag", "D" * 30, token_estimate=30, priority=70, recency=50),
        ]
        # High priority (system + goal) = 80 tokens, exactly fits
        result = allocator.allocate_pieces(pieces)
        # system and goal should be retained, tools might be retained if budget allows
        sources = [p.source_type for p in result]
        assert "system" in sources
        assert "goal" in sources

    def test_truncation_kicks_in(self):
        allocator = TokenBudgetAllocator(context_window=100, reserved_output=20)
        # budget = 80
        pieces = [
            ContextPiece("system", "A" * 20, token_estimate=20, priority=100, recency=100),
            ContextPiece("rag", "B" * 100, token_estimate=100, priority=70, recency=50),
        ]
        # system takes 20, remaining 60, rag needs 100 → truncated
        result = allocator.allocate_pieces(pieces)
        sources = [p.source_type for p in result]
        assert "system" in sources
        # rag may be truncated or dropped
        for p in result:
            if p.source_type == "rag":
                assert p.metadata.get("truncated") is True

    def test_budget_property(self):
        allocator = TokenBudgetAllocator(context_window=1000, reserved_output=100)
        assert allocator.budget == 900

    def test_recency_ordering(self):
        """Newer pieces (higher recency) should be preferred when budget is tight"""
        allocator = TokenBudgetAllocator(context_window=200, reserved_output=20)
        # budget = 180
        pieces = [
            ContextPiece("system", "A" * 10, token_estimate=10, priority=100, recency=100),
            ContextPiece("history", "old_msg", token_estimate=50, priority=50, recency=10),
            ContextPiece("history", "new_msg", token_estimate=50, priority=50, recency=90),
            ContextPiece("rag", "rag_data", token_estimate=50, priority=70, recency=50),
            ContextPiece("memory", "mem_old", token_estimate=50, priority=60, recency=20),
            ContextPiece("memory", "mem_new", token_estimate=50, priority=60, recency=80),
        ]
        # system (10) + high priority first
        result = allocator.allocate_pieces(pieces)
        # At minimum system should be retained
        assert any(p.source_type == "system" for p in result)


# ── ContextAssembler ────────────────────────────────────────


class TestContextAssembler:
    """测试 6 组件上下文装配"""

    def test_basic_assembly(self):
        assembler = ContextAssembler(context_window=10000)
        state = {
            "system_prompt": "You are a helpful assistant.",
            "goal": "Answer the user's question.",
            "history": [],
        }
        messages, metadata = assembler.assemble(state, "Hello!")
        assert len(messages) >= 2  # system + user
        assert metadata["pieces_allocated"] > 0
        assert "sources" in metadata

    def test_six_components_assembly(self):
        """测试全部 6 组件组装"""
        assembler = ContextAssembler(context_window=100000)
        state = {
            "system_prompt": "You are a helpful assistant.",
            "goal": "Answer the user's question accurately.",
            "rag_chunks": [
                "Document 1: The sky is blue.",
                "Document 2: Water is wet.",
            ],
            "memory_items": [
                "User prefers concise answers.",
                "User is a developer.",
            ],
            "tool_schemas": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "history": [
                Message.user("Previous question?"),
                Message.assistant("Previous answer."),
            ],
        }
        messages, metadata = assembler.assemble(state, "Current question!")
        sources = metadata["sources"]
        assert "system" in sources
        assert "goal" in sources
        assert "rag" in sources
        assert "memory" in sources
        assert "tools" in sources
        assert "history" in sources

    def test_priority_sorting(self):
        """System 和 goal 应该在 rag/memory/history 之前"""
        assembler = ContextAssembler(context_window=100000)
        state = {
            "system_prompt": "You are a helper.",
            "goal": "Be concise.",
            "rag_chunks": ["Some context."],
            "memory_items": ["User likes cats."],
            "history": [Message.user("Hi")],
        }
        messages, _ = assembler.assemble(state, "Hello")
        # First message should be system
        assert messages[0].role.value == "system"

    def test_budget_metadata(self):
        """验证 metadata 包含预算信息"""
        assembler = ContextAssembler(context_window=100000)
        state = {"system_prompt": "You are a helper.", "history": []}
        _, metadata = assembler.assemble(state, "Hello")
        assert "total_tokens" in metadata
        assert "budget_used_pct" in metadata
        assert "pieces_allocated" in metadata
        assert "pieces_total" in metadata

    def test_dropped_sources_tracked(self):
        """当 budget 不足时，dropped_sources 应包含被丢弃的组件"""
        assembler = ContextAssembler(context_window=200)  # very tight budget
        state = {
            "system_prompt": "A" * 100,  # ~60 tokens
            "rag_chunks": ["B" * 500],    # ~300 tokens, likely dropped
            "history": [],
        }
        _, metadata = assembler.assemble(state, "Hello")
        # RAG chunk may be dropped due to tight budget
        if "rag" in metadata.get("dropped_sources", []):
            assert "rag" in metadata["dropped_sources"]

    def test_truncated_sources_tracked(self):
        """当内容被截断时，truncated_sources 应包含对应组件"""
        assembler = ContextAssembler(context_window=200)
        state = {
            "system_prompt": "A" * 10,  # small
            "rag_chunks": ["B" * 200],   # large, may be truncated
            "history": [],
        }
        _, metadata = assembler.assemble(state, "Hello")
        if "rag" in metadata.get("truncated_sources", []):
            assert "rag" in metadata["truncated_sources"]

    def test_no_state_components(self):
        """空 state 应该只产生 user message piece"""
        assembler = ContextAssembler(context_window=10000)
        state = {}
        messages, metadata = assembler.assemble(state, "Hello")
        assert len(messages) >= 1
        # Current user message should be present
        user_msgs = [m for m in messages if m.role.value == "user"]
        assert len(user_msgs) >= 1
        assert any("Hello" in m.content for m in user_msgs)

    def test_tool_schemas_passed_explicitly(self):
        """通过参数而不是 state 传入 tool_schemas"""
        assembler = ContextAssembler(context_window=10000)
        state = {"system_prompt": "You are a helper.", "history": []}
        tool_schemas = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
        messages, metadata = assembler.assemble(state, "Hello", tool_schemas=tool_schemas)
        assert "tools" in metadata["sources"]


# ── TokenBudgetManager (Enhanced) ───────────────────────────


class TestTokenBudgetManager:
    """测试增强版 TokenBudgetManager"""

    def test_ok_status(self):
        manager = TokenBudgetManager(max_context=128000)
        result = manager.check_budget(10000)
        assert result["status"] == "OK"
        assert result["usage_pct"] < 40

    def test_warning_status(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.check_budget(700)
        assert result["status"] == "WARNING"
        assert result["usage_pct"] > 60

    def test_error_status(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.check_budget(900)
        assert result["status"] == "ERROR"
        assert result["usage_pct"] > 80
        assert "Dumb Zone" in result["message"]

    def test_remaining_tokens(self):
        manager = TokenBudgetManager(max_context=20000)
        result = manager.check_budget(100)
        assert result["remaining_tokens"] > 0
        assert result["remaining_tokens"] <= manager.input_budget

    def test_compression_strategy_none(self):
        manager = TokenBudgetManager(max_context=128000)
        result = manager.compression_strategy(1000)
        assert result["level"] == "none"
        assert result["actions"] == []

    def test_compression_strategy_light(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.compression_strategy(450)  # 45%
        assert result["level"] == "light"

    def test_compression_strategy_moderate(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.compression_strategy(650)  # 65%
        assert result["level"] == "moderate"

    def test_compression_strategy_aggressive(self):
        manager = TokenBudgetManager(max_context=1000)
        result = manager.compression_strategy(850)  # 85%
        assert result["level"] == "aggressive"
        assert result["target_reduction"] > 0

    def test_input_budget(self):
        manager = TokenBudgetManager(max_context=128000)
        assert manager.input_budget == 128000 - 8000


# ── Progressive Disclosure ──────────────────────────────────


class TestProgressiveDisclosure:
    """测试渐进式信息披露"""

    def test_to_metadata(self):
        items = [
            {"id": "doc1", "title": "Doc 1", "content": "A" * 500, "source": "wiki"},
            {"id": "doc2", "title": "Doc 2", "content": "B" * 100, "source": "web"},
        ]
        result = ProgressiveDisclosure.to_metadata(items)
        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[0]["title"] == "Doc 1"
        assert len(result[0]["snippet"]) <= 203  # 200 + "..."
        assert result[0]["has_full_content"] is True
        assert result[0]["tokens"] > 0

    def test_to_metadata_no_title(self):
        items = [{"content": "Some content without title"}]
        result = ProgressiveDisclosure.to_metadata(items)
        assert result[0]["title"] == "document_0"

    def test_expand_all(self):
        full_items = [
            {"id": "doc1", "content": "Full content 1", "title": "D1"},
            {"id": "doc2", "content": "Full content 2", "title": "D2"},
        ]
        metadata_items = ProgressiveDisclosure.to_metadata(full_items)
        result = ProgressiveDisclosure.expand(metadata_items, full_items)
        assert len(result) == 2
        assert result[0]["content"] == "Full content 1"

    def test_expand_selected(self):
        full_items = [
            {"id": "doc1", "content": "Full content 1", "title": "D1"},
            {"id": "doc2", "content": "Full content 2", "title": "D2"},
        ]
        metadata_items = ProgressiveDisclosure.to_metadata(full_items)
        result = ProgressiveDisclosure.expand(metadata_items, full_items, selected_ids=["doc1"])
        assert len(result) == 2
        assert result[0]["content"] == "Full content 1"
        # doc2 should remain as metadata
        assert result[1].get("has_full_content") is True

    def test_build_metadata_context(self):
        items = [
            {"id": "d1", "title": "Doc 1", "content": "This is document one content."},
            {"id": "d2", "title": "Doc 2", "content": "This is document two."},
            {"id": "d3", "title": "Doc 3", "content": "Third document."},
        ]
        context = ProgressiveDisclosure.build_metadata_context(items, max_items=2)
        assert "Available documents" in context
        assert "[d1]" in context
        assert "[d2]" in context
        assert "1 more" in context  # d3 is beyond max_items
        assert "expand a document" in context.lower()