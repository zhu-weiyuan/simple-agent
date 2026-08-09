# -*- coding: utf-8 -*-
"""my_agent.core — 核心引擎层"""
from .engine import QueryEngine
from .hooks import HookPoint, HookContext, HookRegistry, register_hook
from .context_assembler import (
    ContextPiece,
    ContextAssembler,
    TokenBudgetAllocator,
    TokenBudgetManager,
    ProgressiveDisclosure,
    estimate_tokens,
    estimate_messages_tokens,
)

__all__ = [
    "QueryEngine",
    "HookPoint",
    "HookContext",
    "HookRegistry",
    "register_hook",
    "ContextPiece",
    "ContextAssembler",
    "TokenBudgetAllocator",
    "TokenBudgetManager",
    "ProgressiveDisclosure",
    "estimate_tokens",
    "estimate_messages_tokens",
]
