# -*- coding: utf-8 -*-
"""Tests for sentiment analysis module."""

import pytest
from my_agent.sentiment import (
    analyze, get_tone_adjustment, clear_cache,
    _keyword_sentiment, EmotionTracker,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear sentiment cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestKeywordSentiment:
    """Test keyword-based emotion detection."""

    def test_angry_detection(self):
        result = _keyword_sentiment("这产品太垃圾了，气死我了")
        assert result is not None
        assert result["emotion"] == "angry"
        assert result["intensity"] >= 2

    def test_sad_detection(self):
        result = _keyword_sentiment("我真的好失望，白买了")
        assert result is not None
        assert result["emotion"] == "sad"

    def test_anxious_detection(self):
        result = _keyword_sentiment("怎么办？我的设备坏了，急死了！")
        assert result is not None
        assert result["emotion"] == "anxious"

    def test_happy_detection(self):
        result = _keyword_sentiment("太好了！这个产品真的很棒，很满意！")
        assert result is not None
        assert result["emotion"] == "happy"

    def test_no_emotion(self):
        result = _keyword_sentiment("今天天气不错")
        assert result is None

    def test_strong_word_single_hit(self):
        # "救命" is strong + "开不了机" matches anxious too = 2 hits
        result = _keyword_sentiment("救命！设备开不了机了")
        assert result is not None
        assert result["emotion"] == "anxious"
        assert result["intensity"] >= 3  # at least medium intensity


class TestAnalyze:
    """Test full sentiment analysis with tracking."""

    def test_neutral_default(self):
        result = analyze("你好，请问有什么可以帮忙的？")
        assert result["emotion"] == "neutral"
        assert result["intensity"] == 1

    def test_emotion_tracking_with_session(self):
        # First message - angry (uses keyword match)
        r1 = analyze("这产品太垃圾了！气死我了！", session_id="test1")
        assert r1["emotion"] == "angry"

        # Second message - still angry (worsening)
        r2 = analyze("我真的受够了！投诉！差评！", session_id="test1")
        assert r2["emotion"] == "angry"
        # Trend should be worsening or stable
        assert r2.get("trend") in ("worsening", "stable", "unknown")

    def test_escalation_flag(self):
        result = analyze("这产品太垃圾了，气死我了！", session_id="esc_test")
        assert "should_escalate" in result

    def test_intensity_range(self):
        result = analyze("很满意！")
        assert 1 <= result["intensity"] <= 5


class TestToneAdjustment:
    """Test tone adjustment instructions."""

    def test_neutral_returns_empty(self):
        adj = get_tone_adjustment("neutral", 1)
        assert adj == ""

    def test_angry_low_intensity(self):
        adj = get_tone_adjustment("angry", 2)
        assert "谦逊" in adj or "不满" in adj

    def test_angry_high_intensity(self):
        adj = get_tone_adjustment("angry", 5)
        assert "愤怒" in adj or "道歉" in adj

    def test_anxious_guidance(self):
        adj = get_tone_adjustment("anxious", 3)
        assert "着急" in adj or "解决方案" in adj

    def test_worsening_trend(self):
        adj = get_tone_adjustment("angry", 3, trend="worsening")
        assert "恶化" in adj

    def test_improving_trend(self):
        adj = get_tone_adjustment("angry", 2, trend="improving")
        assert "好转" in adj


class TestEmotionTracker:
    """Test multi-turn emotion tracking."""

    def test_initial_state(self):
        tracker = EmotionTracker()
        trend = tracker.get_trend("new_session")
        assert trend["current"] == "neutral"
        assert trend["trend"] == "unknown"

    def test_trend_worsening(self):
        tracker = EmotionTracker()
        tracker.add_emotion("s1", "neutral", 1)
        tracker.add_emotion("s1", "anxious", 3)
        tracker.add_emotion("s1", "angry", 5)

        trend = tracker.get_trend("s1")
        assert trend["trend"] == "worsening"
        assert trend["current"] == "angry"

    def test_trend_improving(self):
        tracker = EmotionTracker()
        tracker.add_emotion("s2", "angry", 5)
        tracker.add_emotion("s2", "anxious", 3)
        tracker.add_emotion("s2", "neutral", 1)

        trend = tracker.get_trend("s2")
        assert trend["trend"] == "improving"

    def test_escalation_on_worsening(self):
        tracker = EmotionTracker()
        tracker.add_emotion("s3", "angry", 4)
        tracker.add_emotion("s3", "angry", 5)
        assert tracker.should_escalate("s3") is True

    def test_no_escalation_low_intensity(self):
        tracker = EmotionTracker()
        tracker.add_emotion("s4", "neutral", 1)
        assert tracker.should_escalate("s4") is False

    def test_max_history_limit(self):
        tracker = EmotionTracker(max_history=3)
        for i in range(5):
            tracker.add_emotion("s5", "neutral", 1)
        assert len(tracker.history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
