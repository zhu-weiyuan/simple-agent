"""Priority-aware context assembly for LLM calls."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.
    
    English ~0.4 chars/token, Chinese ~0.6 chars/token (regex [一-鿿]).
    """
    if not text:
        return 0
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    other_chars = len(text) - chinese_chars - len(re.findall(r'[a-zA-Z]+', text))
    tokens = (
        chinese_chars * 1.67 +
        english_words * 2.5 +
        other_chars * 0.4
    )
    return int(tokens)


@dataclass
class ContextPiece:
    source_type: str
    content: str
    token_estimate: int = 0
    priority: int = 50
    recency: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenBudgetManager:
    """Manages token budget with warning/error thresholds."""

    WARNING_THRESHOLD = 0.60
    ERROR_THRESHOLD = 0.80

    def __init__(self, max_context: int = 128_000, reserved_output: int = 8_000):
        self.max_context = max_context
        self.reserved_output = reserved_output
        self.budget = max(0, max_context - reserved_output)

    def check_budget(self, used_tokens: int) -> dict[str, Any]:
        """Check how much budget is being used."""
        if self.budget <= 0:
            return {
                'ok': False,
                'alert': 'ERROR',
                'usage_pct': 1.0,
                'recommendation': 'Context window is zero — cannot process any tokens.'
            }
        
        usage_pct = used_tokens / self.budget
        
        if usage_pct > self.ERROR_THRESHOLD:
            return {
                'ok': False,
                'alert': 'ERROR',
                'usage_pct': min(usage_pct, 1.0),
                'recommendation': f'DUMB ZONE! Usage at {usage_pct:.0%} — context too full. Compaction or splitting required.'
            }
        elif usage_pct > self.WARNING_THRESHOLD:
            return {
                'ok': True,
                'alert': 'WARNING',
                'usage_pct': min(usage_pct, 1.0),
                'recommendation': f'Approaching context limit at {usage_pct:.0%}. Consider early compaction.'
            }
        else:
            return {
                'ok': True,
                'alert': None,
                'usage_pct': min(usage_pct, 1.0),
                'recommendation': 'Within budget, no action needed.'
            }

    def can_fit(self, extra_input: int, extra_output: int) -> bool:
        """Check if both extra input and output fit in remaining budget."""
        return extra_input + extra_output <= self.budget


class ContextAssembler:
    def __init__(self, max_tokens: int = 131072, context_window: int | None = None):
        # Accept both 'max_tokens' and 'context_window' for backward compatibility
        if context_window is not None:
            self.max_tokens = context_window
        else:
            self.max_tokens = max_tokens
        self.context: list[dict] = []

    def add_message(self, role: str, content: str, priority: int = 1) -> None:
        self.context.append({'role': role, 'content': content, 'priority': priority})

    def get_context(self) -> list[dict]:
        if self._estimate_tokens(self.context) <= self.max_tokens:
            return self.context

        compacted = self.summarize_old_history(self.context)
        if self._estimate_tokens(compacted) <= self.max_tokens:
            return compacted

        compacted = self.prune_low_priority(compacted)
        if self._estimate_tokens(compacted) <= self.max_tokens:
            return compacted

        return self.compress_context(compacted)

    def summarize_old_history(self, context: list[dict]) -> list[dict]:
        if len(context) <= 5:
            return context
        summary = "Previous discussion summarized."
        return [{'role': 'system', 'content': summary}] + context[-5:]

    def prune_low_priority(self, context: list[dict]) -> list[dict]:
        return [msg for msg in context if msg.get('priority', 1) >= 2]

    def compress_context(self, context: list[dict]) -> list[dict]:
        return [{'role': msg['role'], 'content': msg['content'][:50] + '...'}
                for msg in context]

    @staticmethod
    def _estimate_tokens(context: list[dict]) -> int:
        return sum(len(msg.get('content', '')) for msg in context) // 4

    def assemble(
        self,
        system_prompt: str | None = None,
        user_message: str | None = None,
        conversation_history: list[tuple[str, str]] | None = None,
        rag_results: list[dict] | None = None,
        goal_context: str | None = None,
        context_window: int | None = None,
    ) -> tuple[list[dict], dict[str, Any]]:
        if context_window is None:
            context_window = self.max_tokens

        total_pieces: list[ContextPiece] = []
        allocated_pieces: list[ContextPiece] = []
        used_budget = 0

        # --- system prompt (highest priority) ---
        if system_prompt:
            piece = ContextPiece(
                source_type='system', content=system_prompt,
                token_estimate=estimate_tokens(system_prompt),
                priority=90, recency=100,
            )
            total_pieces.append(piece)

        # --- goal context (high priority) ---
        if goal_context:
            piece = ContextPiece(
                source_type='goal', content=goal_context,
                token_estimate=estimate_tokens(goal_context),
                priority=85, recency=90,
            )
            total_pieces.append(piece)

        # --- conversation history (variable priority by recency) ---
        if conversation_history:
            for i, (role, content) in enumerate(conversation_history):
                recency = max(1, 100 - i * 20)
                piece = ContextPiece(
                    source_type='history', content=content,
                    token_estimate=estimate_tokens(content),
                    priority=max(5, min(30, recency)),
                    recency=recency, metadata={'role': role},
                )
                total_pieces.append(piece)

        # --- RAG results (medium-high priority, weighted by score) ---
        if rag_results:
            for idx, rag in enumerate(rag_results):
                content_text = rag.get('chunk', '') or rag.get('content', '')
                score = rag.get('score', 0.5)
                piece = ContextPiece(
                    source_type='rag', content=content_text,
                    token_estimate=estimate_tokens(content_text),
                    priority=int(score * 70 + 20), recency=max(1, 80 - idx * 10),
                )
                total_pieces.append(piece)

        # --- user message (latest, high recency) ---
        if user_message:
            piece = ContextPiece(
                source_type='user', content=user_message,
                token_estimate=estimate_tokens(user_message),
                priority=80, recency=100,
            )
            total_pieces.append(piece)

        # --- allocate within budget (sorted by priority desc, then recency desc) ---
        sorted_pieces = sorted(total_pieces, key=lambda p: (-p.priority, -p.recency))
        
        for piece in sorted_pieces:
            tok = piece.token_estimate or estimate_tokens(piece.content)
            if used_budget + tok <= context_window:
                allocated_pieces.append(piece)
                used_budget += tok

        # --- build messages list ---
        messages: list[dict] = []
        source_counts: dict[str, int] = {}

        for piece in allocated_pieces:
            src = piece.source_type
            source_counts[src] = source_counts.get(src, 0) + 1
            
            if src == 'system':
                messages.append({'role': 'system', 'content': piece.content})
            elif src == 'goal':
                messages.append({'role': 'system', 'content': f'Goal: {piece.content}'})
            elif src == 'history':
                role = piece.metadata.get('role', 'user')
                messages.append({'role': role, 'content': piece.content})
            elif src == 'rag':
                messages.append({'role': 'system', 'content': f'[RAG: {piece.content}]'})
            elif src == 'user':
                messages.append({'role': 'user', 'content': piece.content})

        # ensure at least one system message
        if not any(m['role'] == 'system' for m in messages):
            if allocated_pieces:
                highest = max(allocated_pieces, key=lambda p: p.priority)
                messages.insert(0, {'role': 'system', 'content': f'[Context: {highest.content[:80]}]'})
            else:
                messages.append({'role': 'system', 'content': '[Empty context — nothing to assemble.]'})

        total_estimated = sum(p.token_estimate or estimate_tokens(p.content) for p in allocated_pieces)

        meta = {
            'total_pieces': len(total_pieces),
            'allocated_pieces': len(allocated_pieces),
            'dropped_pieces': len(total_pieces) - len(allocated_pieces),
            'by_source_type': source_counts,
            'estimated_tokens': total_estimated,
            'budget_usage_pct': total_estimated / context_window if context_window > 0 else 1.0,
        }

        return messages, meta
