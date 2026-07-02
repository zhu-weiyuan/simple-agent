# -*- coding: utf-8 -*-
"""
Conversation Summarization Module for SimpleAgent

Generates service tickets from conversation history.
Uses LLM for intelligent summarization with structured output.
"""

import json
from typing import Dict, Any, List, Optional

SUMMARY_SYSTEM = """你是一个客服工单生成器。根据对话历史生成结构化工单，返回严格的 JSON（不要其他文字）：
{
  "subject": "简短标题（10字以内）",
  "category": "问题分类",
  "description": "问题描述（50字以内）",
  "resolution": "解决方案或处理结果（50字以内）",
  "priority": "优先级: low/medium/high/critical",
  "status": "状态: resolved/pending/escalated"
}

分类选项：技术咨询、产品问题、账号相关、账单支付、功能建议、投诉建议、其他"""


def _generate_summary_prompt(messages: List[Dict[str, str]]) -> str:
    """Build a prompt from conversation messages."""
    parts = []
    for msg in messages[-20:]:  # Last 20 messages to avoid token overflow
        role = msg.get("role", "user")
        content = msg.get("content", "")[:200]  # Truncate long messages
        if role == "user":
            parts.append(f"用户: {content}")
        else:
            parts.append(f"助手: {content}")
    return "\n".join(parts)


def summarize(messages: List[Dict[str, str]], llm_client=None) -> Dict[str, Any]:
    """Generate a service ticket summary from conversation messages.

    Args:
        messages: Conversation history as list of {"role": ..., "content": ...}
        llm_client: Optional LLM client for generation. If None, uses keyword-based fallback.

    Returns:
        Service ticket dict with subject, category, description, resolution, priority, status
    """
    prompt = _generate_summary_prompt(messages)

    if llm_client is not None:
        try:
            result = llm_client.chat_json(
                [{"role": "user", "content": f"请为以下对话生成工单：\n{prompt}"}],
                SUMMARY_SYSTEM,
                max_tokens=256,
            )
            if result and isinstance(result, dict):
                return {
                    "subject": result.get("subject", _infer_subject(messages)),
                    "category": result.get("category", "其他"),
                    "description": result.get("description", prompt[:100]),
                    "resolution": result.get("resolution", "待处理"),
                    "priority": result.get("priority", "medium"),
                    "status": result.get("status", "pending"),
                }
        except Exception:
            pass  # Fall through to keyword-based

    # Keyword-based fallback
    return _keyword_summary(messages)


def _infer_subject(messages: List[Dict[str, str]]) -> str:
    """Infer a subject from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content[:20] + ("..." if len(content) > 20 else "")
    return "未分类对话"


def _keyword_summary(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Fallback keyword-based summary when LLM is unavailable."""
    all_text = " ".join(m.get("content", "") for m in messages)

    # Category detection
    categories = {
        "技术咨询": ["怎么", "如何", "什么", "为什么", "教程", "方法"],
        "产品问题": ["bug", "错误", "报错", "异常", "闪退", "崩溃", "不能用"],
        "账号相关": ["登录", "注册", "密码", "账号", "权限"],
        "账单支付": ["费用", "付款", "退款", "发票", "价格", "收费"],
        "功能建议": ["建议", "希望", "如果", "能不能", "增加"],
        "投诉建议": ["投诉", "差评", "不满", "太差", "垃圾"],
    }

    scores = {}
    for cat, keywords in categories.items():
        scores[cat] = sum(1 for kw in keywords if kw in all_text)

    category = max(scores, key=scores.get) if any(scores.values()) else "其他"

    # Priority detection
    urgent_words = ["紧急", "急死", "立刻", "马上", "崩溃", "不能用", "投诉"]
    has_urgent = any(w in all_text for w in urgent_words)
    priority = "high" if has_urgent else "medium"

    # Status detection (check if last assistant message resolves)
    status = "pending"
    for msg in reversed(messages):
        if msg.get("role") != "user":
            content = msg.get("content", "").lower()
            if any(w in content for w in ["解决", "搞定", "完成", "好了", "已处理", "已解决", "搞定了"]):
                status = "resolved"
            break

    # Get first user message as description
    description = ""
    for msg in messages:
        if msg.get("role") == "user":
            description = msg.get("content", "")[:100]
            break

    return {
        "subject": _infer_subject(messages),
        "category": category,
        "description": description,
        "resolution": "待处理" if status == "pending" else "已处理",
        "priority": priority,
        "status": status,
    }


def generate_ticket(messages: List[Dict[str, str]], llm_client=None) -> Dict[str, Any]:
    """Alias for summarize - generates a full service ticket."""
    return summarize(messages, llm_client)
