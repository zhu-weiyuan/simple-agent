# -*- coding: utf-8 -*-
"""Tests for conversation summarization module."""

import pytest
from my_agent.summary import summarize, generate_ticket, _keyword_summary


class TestKeywordSummary:
    """Test keyword-based fallback summary (no LLM needed)."""

    def test_basic_summary(self):
        messages = [
            {"role": "user", "content": "我的设备无法开机了，怎么办？"},
            {"role": "assistant", "content": "请检查电源连接，然后长按电源键10秒。"},
        ]
        result = _keyword_summary(messages)
        assert "subject" in result
        assert "category" in result
        assert "description" in result
        assert "resolution" in result
        assert "priority" in result
        assert "status" in result

    def test_category_technical(self):
        messages = [
            {"role": "user", "content": "怎么连接WiFi？为什么连不上？"},
        ]
        result = _keyword_summary(messages)
        assert result["category"] == "技术咨询"

    def test_category_product_issue(self):
        messages = [
            {"role": "user", "content": "设备闪退，有bug，一直报错"},
        ]
        result = _keyword_summary(messages)
        assert result["category"] == "产品问题"

    def test_category_account(self):
        messages = [
            {"role": "user", "content": "我忘记登录密码了，怎么重置账号？"},
        ]
        result = _keyword_summary(messages)
        assert result["category"] == "账号相关"

    def test_category_billing(self):
        messages = [
            {"role": "user", "content": "这个功能要收费吗？价格是多少？"},
        ]
        result = _keyword_summary(messages)
        assert result["category"] == "账单支付"

    def test_category_suggestion(self):
        messages = [
            {"role": "user", "content": "建议增加深色模式，希望能支持导出PDF"},
        ]
        result = _keyword_summary(messages)
        assert result["category"] == "功能建议"

    def test_category_complaint(self):
        messages = [
            {"role": "user", "content": "我要投诉！这服务太差了，差评！"},
        ]
        result = _keyword_summary(messages)
        assert result["category"] == "投诉建议"

    def test_priority_high_on_urgent(self):
        messages = [
            {"role": "user", "content": "紧急！设备崩溃了，立刻帮忙！"},
        ]
        result = _keyword_summary(messages)
        assert result["priority"] == "high"

    def test_priority_medium_default(self):
        messages = [
            {"role": "user", "content": "请问这个功能怎么用？"},
        ]
        result = _keyword_summary(messages)
        assert result["priority"] == "medium"

    def test_status_resolved(self):
        messages = [
            {"role": "user", "content": "设备无法开机"},
            {"role": "assistant", "content": "问题已解决，请重启试试。"},
        ]
        result = _keyword_summary(messages)
        assert result["status"] == "resolved"

    def test_status_pending(self):
        messages = [
            {"role": "user", "content": "设备无法开机"},
        ]
        result = _keyword_summary(messages)
        assert result["status"] == "pending"

    def test_empty_messages(self):
        result = _keyword_summary([])
        assert result["category"] == "其他"
        assert result["subject"] == "未分类对话"


class TestSummarize:
    """Test main summarize function (with None LLM = keyword fallback)."""

    def test_summarize_returns_ticket(self):
        messages = [
            {"role": "user", "content": "我的账号登录不了"},
            {"role": "assistant", "content": "请尝试重置密码。"},
        ]
        result = summarize(messages, llm_client=None)
        assert isinstance(result, dict)
        assert result["category"] == "账号相关"

    def test_generate_ticket_alias(self):
        messages = [
            {"role": "user", "content": "怎么退款？"},
        ]
        result = generate_ticket(messages, llm_client=None)
        assert isinstance(result, dict)
        assert "subject" in result

    def test_description_truncated(self):
        long_msg = "a" * 200
        messages = [{"role": "user", "content": long_msg}]
        result = _keyword_summary(messages)
        assert len(result["description"]) <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
