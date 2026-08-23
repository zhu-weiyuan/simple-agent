# -*- coding: utf-8 -*-
"""
my_agent.core.context_assembler — Context Assembler & Token Budget (JavaGuide Best Practices)

参考 JavaGuide Context Engineering 标准:
- 6 组件分区注入: system / goal / RAG / memory / tools / history
- Token Budget Allocator: 预算估算 + 渐进式压缩策略
- Progressive Disclosure: 先 metadata 后 full content
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence, Iterator

from ..types.message import Message, Role

logger = logging.getLogger(__name__)


def normalize_messages_for_chat_template(messages: List["Message"]) -> List["Message"]:
    """Normalize persisted history for strict local chat templates.

    Local llama.cpp-compatible Jinja templates commonly require one leading
    system message and valid assistant-tool-result groups.  A crashed or old
    session can contain duplicate system messages, empty assistant messages,
    orphan tool results, or an incomplete tool call; those histories often
    surface as an opaque HTTP 500 from the model server.  This function works
    on a copy and never mutates the live session.
    """
    if not messages:
        return []

    def clone(message: "Message", content: Optional[str] = None) -> "Message":
        return Message(
            role=message.role,
            content=message.content if content is None else content,
            tool_calls=list(message.tool_calls),
            tool_call_id=message.tool_call_id,
            metadata=dict(message.metadata or {}),
        )

    source = list(messages)
    systems = [m for m in source if m.role.value == "system"]
    leading: List["Message"] = []
    if systems:
        canonical = clone(systems[0])
        summaries = []
        for message in systems[1:]:
            metadata = getattr(message, "metadata", {}) or {}
            if metadata.get("summary_boundary") or (message.content or "").startswith("[HISTORY SUMMARY]"):
                if message.content and message.content != canonical.content:
                    summaries.append(message.content)
        if summaries:
            canonical.content = (canonical.content or "") + "\n\n" + "\n\n".join(summaries)
        leading = [canonical]

    cleaned: List["Message"] = []
    pending_index: Optional[int] = None
    pending_ids: set[str] = set()

    def drop_pending() -> None:
        nonlocal pending_index, pending_ids
        if pending_index is not None and pending_index < len(cleaned):
            cleaned.pop(pending_index)
        pending_index = None
        pending_ids = set()

    for message in source:
        role = message.role.value
        content = message.content or ""
        if role == "system":
            continue
        if role == "user":
            if pending_ids:
                drop_pending()
            if not content.strip():
                continue
            if cleaned and cleaned[-1].role.value == "user":
                cleaned[-1] = clone(message)
            else:
                cleaned.append(clone(message))
            continue
        if role == "assistant":
            if pending_ids:
                drop_pending()
            if not content.strip() and not message.tool_calls:
                continue
            if cleaned and cleaned[-1].role.value == "assistant":
                cleaned[-1] = clone(message)
            else:
                cleaned.append(clone(message))
            if message.tool_calls:
                pending_index = len(cleaned) - 1
                pending_ids = {tc.id for tc in message.tool_calls if tc.id}
            continue
        if role == "tool":
            if pending_index is None or message.tool_call_id not in pending_ids:
                continue
            cleaned.append(clone(message))
            pending_ids.remove(message.tool_call_id)
            if not pending_ids:
                pending_index = None
            continue
        logger.warning("Dropping unsupported message role from chat history: %s", role)

    if pending_ids:
        drop_pending()
    return leading + cleaned


# ── Token Estimation (统一估算器) ────────────────────────────
#
# 统一规则 (原 token_budget.TokenEstimator 并入此处):
#   tiktoken 可用时优先精确编码 (cl100k_base)
#   否则: CJK 字符 × 0.7 + ASCII 词数 × 1.3 + 其余字符 × 0.3

_TIKTOKEN_ENCODING = None
_TIKTOKEN_TRIED = False


def _get_tiktoken_encoding():
    global _TIKTOKEN_ENCODING, _TIKTOKEN_TRIED
    if not _TIKTOKEN_TRIED:
        _TIKTOKEN_TRIED = True
        try:
            import tiktoken  # type: ignore
            _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENCODING = None
    return _TIKTOKEN_ENCODING


def estimate_tokens(text: str) -> int:
    """统一 token 估算: tiktoken 优先; 回退启发式 CJK×0.7 + ASCII词×1.3 + 其余×0.3。"""
    if not text:
        return 0
    enc = _get_tiktoken_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    cjk = sum(1 for c in text
              if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f"
              or "\u3040" <= c <= "\u30ff" or "\uac00" <= c <= "\ud7af")
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    ascii_word_chars = sum(len(w) for w in ascii_words)
    other = max(0, len(text) - cjk - ascii_word_chars)
    return max(1, round(cjk * 0.7 + len(ascii_words) * 1.3 + other * 0.3))


def estimate_messages_tokens(messages: List[Message]) -> int:
    """估算消息列表的总 token 数"""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.role.value)
        total += estimate_tokens(msg.content or "")
        for tc in msg.tool_calls:
            total += estimate_tokens(tc.name)
            total += estimate_tokens(json.dumps(tc.arguments, ensure_ascii=False))
    return total


# ── ContextPiece ────────────────────────────────────────────


@dataclass
class ContextPiece:
    """上下文组件的元数据 + 内容表示"""

    source_type: str  # system | goal | rag | memory | tools | history
    content: str
    token_estimate: int
    priority: int = 50  # 0-100, higher = more important
    recency: int = 50   # 0-100, higher = more recent
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ContextPiece(source={self.source_type}, tokens={self.token_estimate}, "
            f"priority={self.priority}, recency={self.recency})"
        )


# ── TokenBudgetAllocator ────────────────────────────────────


class TokenBudgetAllocator:
    """渐进式 Token 预算分配器。

    策略:
    1. 保留所有 priority >= 90 (system prompt, current turn)
    2. 按 (priority + recency) 降序填充剩余预算
    3. 丢弃顺序: old memory > history > RAG > tools
    """

    def __init__(self, context_window: int = 128000, reserved_output: int = 8000) -> None:
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        self.context_window = context_window
        self.reserved_output = max(0, reserved_output)

    @property
    def budget(self) -> int:
        """可用输入预算"""
        return max(0, self.context_window - self.reserved_output)

    def allocate_pieces(self, pieces: List[ContextPiece]) -> List[ContextPiece]:
        """渐进式丢弃策略分配预算。

        返回能在预算内容纳的 ContextPiece 列表。
        """
        if not pieces:
            return []

        available = self.budget
        retained: List[ContextPiece] = []
        used = 0

        # Phase 1: 保留所有高优先级 (priority >= 90)
        high_priority = [p for p in pieces if p.priority >= 90]
        low_priority = [p for p in pieces if p.priority < 90]

        for piece in high_priority:
            # system 分区是强制不可丢弃项 (即便超预算也必须保留, 否则模型失去
            # 角色约束与保密指令, 反而更易泄漏/越权)。
            mandatory = piece.source_type == "system"
            if mandatory or used + piece.token_estimate <= available:
                retained.append(piece)
                used += piece.token_estimate
                if mandatory and used > available:
                    logger.warning(
                        "Mandatory system piece (%d tokens) exceeds budget %d — kept anyway",
                        piece.token_estimate, available,
                    )
            else:
                logger.warning(
                    "High-priority piece %s (%d tokens) cannot fit in remaining budget %d",
                    piece.source_type, piece.token_estimate, available - used
                )

        # Phase 2: 按 (priority + recency) 降序填充剩余预算
        available_remaining = available - used
        low_priority.sort(key=lambda p: (p.priority + p.recency), reverse=True)

        for piece in low_priority:
            if available_remaining <= 0:
                break
            if piece.token_estimate <= available_remaining:
                retained.append(piece)
                available_remaining -= piece.token_estimate
            else:
                # 尝试截断内容
                truncated = self._truncate_piece(piece, available_remaining)
                if truncated is not None:
                    retained.append(truncated)
                    available_remaining -= truncated.token_estimate
                else:
                    logger.debug(
                        "Dropping piece %s (%d tokens) — insufficient budget",
                        piece.source_type, piece.token_estimate
                    )

        self._log_budget_usage(retained, used)
        return retained

    def _truncate_piece(self, piece: ContextPiece, max_tokens: int) -> Optional[ContextPiece]:
        """尝试截断 ContextPiece 内容以适应预算"""
        if max_tokens <= 0:
            return None
        # 粗略截断: 按字符比例裁剪
        if piece.token_estimate <= 0:
            return None
        ratio = min(1.0, max_tokens / piece.token_estimate)
        if ratio < 0.3:
            # 内容太少，不值得保留
            return None
        # 简单按字符比例截断
        content = piece.content
        char_limit = int(len(content) * ratio)
        truncated_content = content[:char_limit] + "\n...[truncated]"
        new_tokens = max(1, round(max_tokens * 0.95))
        return ContextPiece(
            source_type=piece.source_type,
            content=truncated_content,
            token_estimate=new_tokens,
            priority=piece.priority,
            recency=piece.recency,
            metadata={**piece.metadata, "truncated": True},
        )

    def _log_budget_usage(self, retained: List[ContextPiece], used: int) -> None:
        total = sum(p.token_estimate for p in retained)
        pct = min(100.0, (total / self.context_window) * 100) if self.context_window > 0 else 0
        if pct > 80:
            logger.error("Token budget usage %.1f%% — Dumb Zone!", pct)
        elif pct > 60:
            logger.warning("Token budget usage %.1f%% — Context Anxiety approaching", pct)
        elif pct > 40:
            logger.info("Token budget usage %.1f%%", pct)


# ── ContextAssembler ────────────────────────────────────────


class ContextAssembler:
    """6 组件分区注入上下文装配器。

    装配流程:
    1. 收集所有组件 → ContextPiece 列表
    2. 按优先级排序
    3. TokenBudgetAllocator 裁剪
    4. 返回 ordered messages + metadata
    """

    def __init__(self, context_window: int = 128000) -> None:
        self.context_window = context_window
        self.allocator = TokenBudgetAllocator(context_window=context_window)

    def assemble(
        self,
        state: dict,
        user_message: str,
        tool_schemas: Optional[List[dict]] = None,
    ) -> Tuple[List[Message], dict]:
        """组装完整上下文。

        Args:
            state: Graph state dict with optional keys:
                - system_prompt: str
                - goal: str
                - rag_chunks: List[str]
                - memory_items: List[str]
                - history: List[Message] or List[dict]
                - tool_schemas: List[dict]
            user_message: 当前用户消息
            tool_schemas: 工具 schema 列表 (可选，也可从 state 获取)

        Returns:
            (ordered_messages, metadata) where metadata includes budget info
        """
        # 1. 收集所有组件
        pieces = self._collect_from_state(state, user_message, tool_schemas)

        # 2. 按优先级排序
        pieces.sort(key=lambda p: (p.priority, p.recency), reverse=True)

        # 3. TokenBudgetAllocator 裁剪
        allocated = self.allocator.allocate_pieces(pieces)

        # 4. 组装为 Message 列表
        messages = self._pieces_to_messages(allocated, user_message)

        # 5. 构建 metadata
        total_tokens = sum(p.token_estimate for p in allocated)
        metadata = {
            "total_tokens": total_tokens,
            "budget_used_pct": (
                min(100.0, (total_tokens / self.context_window) * 100)
                if self.context_window > 0 else 0
            ),
            "pieces_allocated": len(allocated),
            "pieces_total": len(pieces),
            "sources": [p.source_type for p in allocated],
            "dropped_sources": [p.source_type for p in pieces if p not in allocated],
            "truncated_sources": [p.source_type for p in allocated if p.metadata.get("truncated")],
        }

        return messages, metadata

    def _collect_from_state(
        self, state: dict, user_message: str, tool_schemas: Optional[List[dict]] = None
    ) -> List[ContextPiece]:
        """从 state 收集所有 ContextPiece 组件"""
        pieces: List[ContextPiece] = []

        # 1. System Prompt (priority=100)
        system_prompt = state.get("system_prompt", "")
        if system_prompt:
            pieces.append(ContextPiece(
                source_type="system",
                content=system_prompt,
                token_estimate=estimate_tokens(system_prompt),
                priority=100,
                recency=100,
            ))

        # 2. Goal / Task (priority=95)
        goal = state.get("goal", "")
        if goal:
            pieces.append(ContextPiece(
                source_type="goal",
                content=goal,
                token_estimate=estimate_tokens(goal),
                priority=95,
                recency=95,
            ))

        # 3. Tools / Tool Schemas (priority=90)
        schemas = tool_schemas or state.get("tool_schemas", [])
        if schemas:
            tools_text = json.dumps(schemas, ensure_ascii=False, indent=2)
            pieces.append(ContextPiece(
                source_type="tools",
                content=tools_text,
                token_estimate=estimate_tokens(tools_text),
                priority=90,
                recency=50,
            ))

        # 4. RAG Chunks (priority=70)
        rag_chunks = state.get("rag_chunks", [])
        if rag_chunks:
            for i, chunk in enumerate(rag_chunks):
                pieces.append(ContextPiece(
                    source_type="rag",
                    content=chunk,
                    token_estimate=estimate_tokens(chunk),
                    priority=70,
                    recency=50 + i,  # newer chunks slightly higher recency
                    metadata={"chunk_index": i},
                ))

        # 5. Memory Items (priority=60, recency derived from index)
        memory_items = state.get("memory_items", [])
        if memory_items:
            for i, item in enumerate(memory_items):
                recency = 80 - i  # newest first
                pieces.append(ContextPiece(
                    source_type="memory",
                    content=item,
                    token_estimate=estimate_tokens(item),
                    priority=60,
                    recency=max(0, recency),
                    metadata={"memory_index": i},
                ))

        # 6. History (priority=50, recency from position)
        history = state.get("history", [])
        if history:
            for i, msg in enumerate(history):
                if isinstance(msg, Message):
                    content = msg.content or ""
                    role = msg.role.value
                elif isinstance(msg, dict):
                    content = msg.get("content", "") or ""
                    role = msg.get("role", "")
                else:
                    content = str(msg)
                    role = "unknown"
                # 加固: 系统提示词绝不通过 history 通道重新注入 (否则可能被
                # 降级到 user/assistant 角色而泄漏)。system 只从 state["system_prompt"]
                # 走 source_type="system" 分区。
                if role == "system":
                    continue
                piece_content = f"[{role}]: {content}"
                recency = 50 + i  # newer messages higher recency
                pieces.append(ContextPiece(
                    source_type="history",
                    content=piece_content,
                    token_estimate=estimate_tokens(piece_content),
                    priority=50,
                    recency=min(100, recency),
                    metadata={"history_index": i, "role": role},
                ))

        # 7. Current user message (priority=100, always included)
        pieces.append(ContextPiece(
            source_type="history",  # current turn is part of history
            content=f"[user]: {user_message}",
            token_estimate=estimate_tokens(user_message),
            priority=100,
            recency=100,
            metadata={"current_turn": True},
        ))

        return pieces

    def _pieces_to_messages(
        self, pieces: List[ContextPiece], user_message: str
    ) -> List[Message]:
        """将 ContextPiece 列表转换为 Message 列表。

        按 source_type 分组排序:
        system → goal → tools → rag → memory → history → user
        """
        messages: List[Message] = []
        had_system = False

        # 按 source_type 优先级排序
        type_order = {"system": 0, "goal": 1, "tools": 2, "rag": 3, "memory": 4, "history": 5}
        pieces.sort(key=lambda p: type_order.get(p.source_type, 99))

        for piece in pieces:
            if piece.source_type == "system":
                if not had_system:
                    messages.append(Message.system(piece.content))
                    had_system = True
                else:
                    # Additional system content appended
                    messages.append(Message.system(piece.content))

            elif piece.source_type == "goal":
                messages.append(Message.system(
                    f"[TASK]\n{piece.content}"
                ))

            elif piece.source_type == "tools":
                messages.append(Message.system(
                    f"[AVAILABLE TOOLS]\n{piece.content}"
                ))

            elif piece.source_type == "rag":
                prefix = "[RAG CONTEXT (truncated)]\n" if piece.metadata.get("truncated") else "[RAG CONTEXT]\n"
                messages.append(Message.system(
                    f"{prefix}{piece.content}"
                ))

            elif piece.source_type == "memory":
                prefix = "[MEMORY (truncated)]\n" if piece.metadata.get("truncated") else "[MEMORY]\n"
                messages.append(Message.system(
                    f"{prefix}{piece.content}"
                ))

            elif piece.source_type == "history":
                # Check if this is the current turn
                if piece.metadata.get("current_turn"):
                    messages.append(Message.user(user_message))
                elif piece.metadata.get("role") == "user":
                    messages.append(Message.user(piece.content[len("[user]: "):]
                        if piece.content.startswith("[user]: ") else piece.content))
                elif piece.metadata.get("role") == "assistant":
                    messages.append(Message.assistant(piece.content[len("[assistant]: "):]
                        if piece.content.startswith("[assistant]: ") else piece.content))
                elif piece.metadata.get("role") == "system":
                    # 加固: system 角色历史一律作为 system 消息, 绝不降级为 user。
                    messages.append(Message.system(piece.content[len("[system]: "):]
                        if piece.content.startswith("[system]: ") else piece.content))
                elif piece.metadata.get("role") == "tool":
                    # 工具结果保留为 user 可见上下文, 但明确标注, 不伪装成用户话语。
                    messages.append(Message.user(f"[tool result] {piece.content}"))
                else:
                    messages.append(Message.user(piece.content))

        return messages


# ── TokenBudgetManager (Enhanced) ────────────────────────────


class TokenBudgetManager:
    """增强版 Token 预算管理器。

    提供预算检查、分级告警和压缩策略建议。
    """

    def __init__(self, max_context: int = 128000) -> None:
        self.max_context = max_context
        self.reserved_output = 8000

    @property
    def input_budget(self) -> int:
        return max(0, self.max_context - self.reserved_output)

    def check_budget(self, input_tokens: int) -> dict:
        """检查预算并返回建议。

        Returns:
            dict with keys:
            - status: "OK" | "WARNING" | "ERROR"
            - message: human-readable status
            - usage_pct: float (0-100)
            - suggestion: recommended action
            - remaining_tokens: int
        """
        if self.max_context <= 0:
            return {
                "status": "ERROR",
                "message": "Token budget is zero — context window not configured",
                "usage_pct": 100.0,
                "suggestion": "Configure a valid context window size.",
                "remaining_tokens": 0,
            }
        usage_pct = min(100.0, (input_tokens / self.max_context) * 100)
        remaining = max(0, self.input_budget - input_tokens)

        if usage_pct > 80:
            return {
                "status": "ERROR",
                "message": f"Token usage {usage_pct:.1f}% — Dumb Zone!",
                "usage_pct": usage_pct,
                "suggestion": "Must compress or reject. "
                              "Options: (1) summarize/truncate history, "
                              "(2) reduce RAG chunks, "
                              "(3) drop old memory, "
                              "(4) use smaller model",
                "remaining_tokens": remaining,
            }
        elif usage_pct > 60:
            return {
                "status": "WARNING",
                "message": f"Token usage {usage_pct:.1f}% — Context Anxiety approaching",
                "usage_pct": usage_pct,
                "suggestion": "Recommend compression. "
                              "Consider summarizing older history or reducing RAG context.",
                "remaining_tokens": remaining,
            }
        elif usage_pct > 40:
            return {
                "status": "OK",
                "message": f"Token usage {usage_pct:.1f}% — healthy",
                "usage_pct": usage_pct,
                "suggestion": "No action needed.",
                "remaining_tokens": remaining,
            }
        else:
            return {
                "status": "OK",
                "message": f"Token usage {usage_pct:.1f}% — ample headroom",
                "usage_pct": usage_pct,
                "suggestion": "No action needed.",
                "remaining_tokens": remaining,
            }

    def compression_strategy(self, input_tokens: int) -> dict:
        """根据当前使用率返回推荐压缩策略。

        Returns:
            dict with:
            - level: "none" | "light" | "moderate" | "aggressive"
            - actions: List[str] of recommended actions
            - target_reduction: int (target tokens to reduce)
        """
        check = self.check_budget(input_tokens)
        usage_pct = check["usage_pct"]
        target_reduction = 0

        if usage_pct > 80:
            level = "aggressive"
            target_reduction = max(0, input_tokens - int(self.max_context * 0.5))
            actions = [
                "Summarize all history older than 3 turns",
                "Drop all but top-1 RAG chunk",
                "Drop all memory items",
                "Consider switching to shorter system prompt",
            ]
        elif usage_pct > 60:
            level = "moderate"
            target_reduction = max(0, input_tokens - int(self.max_context * 0.5))
            actions = [
                "Summarize history older than 5 turns",
                "Keep only top-3 RAG chunks",
                "Drop oldest memory items",
            ]
        elif usage_pct > 40:
            level = "light"
            target_reduction = max(0, input_tokens - int(self.max_context * 0.4))
            actions = [
                "Truncate oldest history messages",
                "Reduce RAG chunk count to 5",
            ]
        else:
            level = "none"
            actions = []

        return {
            "level": level,
            "actions": actions,
            "target_reduction": target_reduction,
            "current_usage_pct": usage_pct,
        }


# ── Progressive Disclosure ──────────────────────────────────


class ProgressiveDisclosure:
    """渐进式信息披露: 先 metadata 后 full content。

    使用场景:
    - 大量 RAG chunks 时，先展示摘要 (title + snippet)
    - 当模型需要更多上下文时，再展开 full content
    """

    @staticmethod
    def to_metadata(items: List[dict], item_type: str = "document") -> List[dict]:
        """将 items 转换为 metadata-only 表示。

        Args:
            items: List of dicts with at least 'content' key
            item_type: 类型标签 (document, memory, etc.)

        Returns:
            List of metadata dicts with 'title', 'snippet', 'tokens', 'id'
        """
        results = []
        for i, item in enumerate(items):
            content = item.get("content", "")
            title = item.get("title", item.get("source", f"{item_type}_{i}"))
            snippet = content[:200] + "..." if len(content) > 200 else content
            results.append({
                "id": item.get("id", f"{item_type}_{i}"),
                "type": item_type,
                "title": title,
                "snippet": snippet,
                "tokens": estimate_tokens(content),
                "has_full_content": True,
                "metadata": {k: v for k, v in item.items() if k not in ("content", "id", "title", "source")},
            })
        return results

    @staticmethod
    def expand(
        items: List[dict],
        full_items: List[dict],
        selected_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """展开选中的 items 为 full content。

        Args:
            items: metadata-only items (from to_metadata)
            full_items: original full items
            selected_ids: 要展开的 item IDs (None = 全部展开)

        Returns:
            List with full content for selected items
        """
        full_by_id = {}
        for i, item in enumerate(full_items):
            item_id = item.get("id", f"doc_{i}")
            full_by_id[item_id] = item

        if selected_ids is None:
            return full_items

        result = []
        for item in items:
            item_id = item["id"]
            if item_id in selected_ids and item_id in full_by_id:
                result.append(full_by_id[item_id])
            else:
                result.append(item)
        return result

    @staticmethod
    def build_metadata_context(
        items: List[dict],
        item_type: str = "document",
        max_items: int = 10,
    ) -> str:
        """构建 metadata-only 上下文文本。

        Returns:
            Formatted string listing item metadata for model to select from
        """
        metadata_items = ProgressiveDisclosure.to_metadata(items, item_type)
        lines = [f"Available {item_type}s ({len(metadata_items)} total, showing top {min(max_items, len(metadata_items))}):"]
        for item in metadata_items[:max_items]:
            lines.append(f"  [{item['id']}] {item['title']}: {item['snippet']} ({item['tokens']} tokens)")
        if len(metadata_items) > max_items:
            lines.append(f"  ... and {len(metadata_items) - max_items} more")
        lines.append("\nTo expand a document, request its full content by ID.")
        return "\n".join(lines)

# ── Engine-facing chat fitting ──────────────────────────────


def _message_tokens(msg: "Message") -> int:
    total = estimate_tokens(msg.content or "") + 4
    for tc in getattr(msg, "tool_calls", []) or []:
        total += estimate_tokens(tc.name)
        try:
            total += estimate_tokens(json.dumps(tc.arguments, ensure_ascii=False))
        except (TypeError, ValueError):
            pass
    return total


def fit_messages_to_budget(
    messages: List["Message"],
    context_window: int = 131072,
    target_ratio: float = 0.7,
) -> List["Message"]:
    """把会话消息裁剪到 context_window * target_ratio 以内。

    规则 (engine._call_llm/_acall_llm 组装路径):
    - 首条规范 system prompt 始终保留、置于最前；摘要边界等辅助 system
      内容仅在预算允许时保留
    - 历史消息按 旧→新 顺序保留,超预算时从最旧的非 system 消息开始成组丢弃
      (assistant 的 tool_calls 与其后的 tool 结果作为一组,避免孤儿 tool 消息)
    - 最新一条(当前问题)始终保留在最后
    - ``target_ratio`` 是尽力达到的上限。若首条 system prompt 或当前问题
      单独已超限，函数保留它们并记录告警，而不会静默截断指令或用户输入。
    """
    if not messages:
        return []
    budget = max(1, int(context_window * target_ratio))

    system_msgs = [m for m in messages if m.role.value == "system"]
    history = [m for m in messages if m.role.value != "system"]

    # DEBUG: Log system messages
    logger.debug("fit_messages_to_budget: system_msgs count=%d, history count=%d", len(system_msgs), len(history))
    for i, m in enumerate(system_msgs):
        meta = getattr(m, "metadata", {}) or {}
        logger.debug("  system[%d]: summary_boundary=%s, content_len=%d", i, meta.get("summary_boundary"), len(m.content or ""))

    # Keep the canonical prompt, but do not let a large compaction summary take
    # the whole budget before the current turn is considered.  SessionState puts
    # the canonical prompt first and marks summaries explicitly.
    primary_system: List["Message"] = []
    auxiliary_system: List["Message"] = []
    for message in system_msgs:
        if not primary_system and not getattr(message, "metadata", {}).get("summary_boundary"):
            primary_system.append(message)
        else:
            auxiliary_system.append(message)

    if not primary_system and auxiliary_system:
        # 首条 system 就是摘要边界(无规范 system): 提升第一条为规范位
        primary_system = [auxiliary_system.pop(0)]
        logger.debug("Promoted first auxiliary system to primary")

    used = sum(_message_tokens(m) for m in primary_system)
    # Chat template (Qwen3 系) 只允许整个 payload 中 index 0 存在唯一一条
    # system 消息; 摘要边界等辅助 system 内容必须并入规范 system,
    # 否则上游抛 "System message must be at the beginning" (HTTP 500)。
    merged_aux: List[str] = []
    for message in auxiliary_system:
        msg_tokens = _message_tokens(message)
        if used + msg_tokens <= budget:
            merged_aux.append(message.content or "")
            used += msg_tokens
        else:
            logger.info("Dropping auxiliary system context to honor message budget")
    if merged_aux:
        merged = (primary_system[0].content if primary_system else "") or ""
        primary_system[0] = Message.system(merged + "\n\n" + "\n\n".join(merged_aux))
        used = _message_tokens(primary_system[0])
    kept_system = list(primary_system)

    # 把历史按 "组" 切分: assistant(tool_calls) + 其 tool 结果 是一组
    groups: List[List["Message"]] = []
    current: List["Message"] = []
    for m in history:
        if m.role.value == "tool" and current:
            current.append(m)
        else:
            if current:
                groups.append(current)
            current = [m]
    if current:
        groups.append(current)

    # 从最新往回填充 (最后一组 = 当前问题,必留)
    kept_rev: List[List["Message"]] = []
    for gi, group in enumerate(reversed(groups)):
        g_tokens = sum(_message_tokens(m) for m in group)
        if gi == 0 or used + g_tokens <= budget:
            kept_rev.append(group)
            used += g_tokens
        else:
            break  # older groups all dropped once budget is hit
    kept: List["Message"] = []
    for group in reversed(kept_rev):
        kept.extend(group)

    # 若第一条保留的是孤儿 tool 消息,去掉
    while kept and kept[0].role.value == "tool":
        kept.pop(0)

    result = kept_system + kept
    pct = (sum(_message_tokens(m) for m in result) / context_window * 100
           if context_window else 0)
    if pct > 60:
        logger.warning("Context utilization %.1f%% exceeds 60%% target", pct)
    return result
