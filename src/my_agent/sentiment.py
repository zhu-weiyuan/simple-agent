# -*- coding: utf-8 -*-
"""
Sentiment Analysis Module for SimpleAgent

Detects user emotion from their messages. Uses keyword-based detection
for speed, with optional LLM fallback for accuracy.

Emotion categories: neutral, angry, sad, anxious, happy
"""

import time
from typing import Dict, Any, Optional
from collections import deque

_KEYWORDS = {
    "angry": [
        "垃圾", "太差", "操你", "傻逼", "废物", "骗子", "黑心", "坑人",
        "愤怒", "气死", "恶心", "无语", "受够了", "滚蛋", "去死",
        "投诉", "举报", "维权", "差评", "退款", "退货",
    ],
    "sad": [
        "失望", "难过", "伤心", "可怜", "无助", "绝望", "心碎",
        "不好用", "没用", "白买了", "浪费钱", "后悔",
    ],
    "anxious": [
        "着急", "急死", "快点", "紧急", "怎么办", "救命", "来不及",
        "坏了", "不能用", "开不了机", "连不上", "闪退",
    ],
    "happy": [
        "太好了", "很棒", "喜欢", "满意", "好用", "赞", "给力",
        "谢谢", "感谢", "不错", "很好", "完美", "开心",
    ],
}

_STRONG_WORDS = {"操你", "傻逼", "去死", "滚蛋", "救命", "紧急", "完美"}


class EmotionTracker:
    """Tracks emotion trends across multiple conversation turns."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)

    def add_emotion(self, session_id: str, emotion: str, intensity: int):
        self.history.append({
            "session_id": session_id,
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": time.time(),
        })

    def get_trend(self, session_id: str) -> Dict[str, Any]:
        session_history = [h for h in self.history if h["session_id"] == session_id]
        if not session_history:
            return {"current": "neutral", "previous": None,
                    "trend": "unknown", "avg_intensity": 0}

        current = session_history[-1]
        previous = session_history[-2] if len(session_history) > 1 else None

        if len(session_history) >= 3:
            recent = [h["intensity"] for h in session_history[-3:]]
            avg_intensity = sum(recent) / len(recent)
            if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
                trend = "worsening"
            elif all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
                trend = "improving"
            else:
                trend = "stable"
        else:
            avg_intensity = current["intensity"]
            trend = "unknown"

        return {
            "current": current["emotion"],
            "previous": previous["emotion"] if previous else None,
            "trend": trend,
            "avg_intensity": round(avg_intensity, 1),
        }

    def should_escalate(self, session_id: str) -> bool:
        session_history = [h for h in self.history if h["session_id"] == session_id]
        if not session_history:
            return False
        current = session_history[-1]
        if current["intensity"] >= 4:
            trend = self.get_trend(session_id)["trend"]
            if trend == "worsening":
                return True
        recent_high = sum(1 for h in session_history[-3:] if h["intensity"] >= 3)
        return recent_high >= 2


# Global tracker
_emotion_tracker = EmotionTracker()
_sentiment_cache: Dict[str, Dict[str, Any]] = {}


def _keyword_sentiment(text: str) -> Optional[Dict[str, Any]]:
    """Fast keyword-based emotion detection."""
    scores = {"angry": 0, "sad": 0, "anxious": 0, "happy": 0}
    for emotion, words in _KEYWORDS.items():
        for word in words:
            if word in text:
                scores[emotion] += 1

    max_emotion = max(scores, key=scores.get)
    max_score = scores[max_emotion]

    if max_score >= 2:
        return {"emotion": max_emotion, "intensity": min(5, max_score + 1)}
    elif max_score == 1:
        for word in _KEYWORDS.get(max_emotion, []):
            if word in text and word in _STRONG_WORDS:
                return {"emotion": max_emotion, "intensity": 4}
    return None


def analyze(text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Analyze the sentiment of a user message.

    Returns:
        {"emotion": str, "intensity": int, "trend": str, "should_escalate": bool}
    """
    ck = f"{session_id}:{text[:50]}" if session_id else text[:50]
    if ck in _sentiment_cache:
        cached = _sentiment_cache[ck].copy()
        if session_id:
            trend_info = _emotion_tracker.get_trend(session_id)
            cached["trend"] = trend_info["trend"]
            cached["should_escalate"] = _emotion_tracker.should_escalate(session_id)
        return cached

    kw_result = _keyword_sentiment(text)
    if kw_result:
        emotion, intensity = kw_result["emotion"], kw_result["intensity"]
    else:
        emotion, intensity = "neutral", 1

    # Context-aware calibration
    if session_id:
        trend_info = _emotion_tracker.get_trend(session_id)
        if trend_info["trend"] == "worsening" and intensity < 5:
            intensity = min(5, intensity + 1)
        elif trend_info["trend"] == "improving" and intensity > 1:
            intensity = max(1, intensity - 1)

    _emotion_tracker.add_emotion(session_id or "anonymous", emotion, intensity)

    result = {"emotion": emotion, "intensity": intensity}
    if session_id:
        trend_info = _emotion_tracker.get_trend(session_id)
        result["trend"] = trend_info["trend"]
        result["should_escalate"] = _emotion_tracker.should_escalate(session_id)

    _sentiment_cache[ck] = result
    return result


def get_tone_adjustment(emotion: str, intensity: int, trend: str = "unknown") -> str:
    """Generate tone adjustment instructions based on detected emotion."""
    adjustments = {
        "angry": {
            (1, 2): "注意：用户有些不满，请语气更加谦逊有礼。",
            (3, 4): "注意：用户比较生气，请先诚恳道歉，表达理解，再耐心解决问题。",
            (5,): "注意：用户非常愤怒！请先真诚道歉，优先给出解决方案。",
        },
        "sad": {
            (1, 2): "注意：用户有些失落，请语气温和，给予鼓励。",
            (3, 4): "注意：用户感到失望，请先表达理解和同情，再积极提供帮助。",
            (5,): "注意：用户非常沮丧，请先温暖安慰，再一步步协助解决。",
        },
        "anxious": {
            (1, 2): "注意：用户有些担心，请给出明确的步骤和预期结果。",
            (3, 4): "注意：用户比较着急，请直接给出解决方案，减少废话。",
            (5,): "注意：用户非常焦虑！请立即给出清晰的解决步骤和时间预期。",
        },
        "happy": {
            (1, 2): "注意：用户心情不错，可以轻松愉快地交流。",
            (3, 4): "注意：用户很开心，继续保持热情友好的态度。",
            (5,): "注意：用户非常高兴！可以活泼一些。",
        },
    }

    if emotion == "neutral":
        return ""

    emotion_adj = adjustments.get(emotion, {})
    instruction = ""
    for intensity_range, adj in emotion_adj.items():
        if intensity in intensity_range:
            instruction = adj
            break

    if trend == "worsening":
        instruction += "\n注意：用户情绪正在恶化，请更加谨慎处理。"
    elif trend == "improving":
        instruction += "\n注意：用户情绪正在好转，继续保持。"

    return f"\n\n{instruction}" if instruction else ""


def clear_cache():
    """Clear sentiment cache and tracker."""
    _sentiment_cache.clear()
    _emotion_tracker.history.clear()
