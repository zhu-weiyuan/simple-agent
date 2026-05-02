# -*- coding: utf-8 -*-
"""
my_agent.message — 向后兼容 shim

旧代码 from my_agent.message import Message 仍然可用。
实际实现已迁移到 types/message.py。
"""
from .types.message import Role as MessageRole
from .types.message import Message, ToolCall

# 向后兼容：旧代码使用 MessageRole 而不是 Role
__all__ = ["MessageRole", "Message", "ToolCall"]
