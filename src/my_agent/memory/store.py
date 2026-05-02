# -*- coding: utf-8 -*-
"""
my_agent.memory.store — 记忆存储

持久化用户事实和经验教训。
- facts.json: 结构化事实（用户偏好、项目状态、稳定事实）
- lessons.md: 经验教训列表
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import json


@dataclass
class MemoryRecallResult:
    facts: Dict[str, Any]
    matched_lessons: List[str]


class MemoryStore:
    """长期记忆存储"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.memory_dir = project_root / "memory"
        self.facts_path = self.memory_dir / "facts.json"
        self.lessons_path = self.memory_dir / "lessons.md"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_seed_files()
        self._injected_memory: List[str] = []

    def _ensure_seed_files(self) -> None:
        if not self.facts_path.exists():
            self.facts_path.write_text(
                json.dumps(
                    {
                        "user_preferences": {},
                        "project_state": {
                            "agent_name": "SimpleAgent",
                            "current_focus": "learn-agent-architecture",
                        },
                        "stable_facts": {},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if not self.lessons_path.exists():
            self.lessons_path.write_text(
                "# lessons.md\n\n"
                "- CLI 参数（如 --help）应优先在本地处理，不要误进模型。\n"
                "- 工具结果应该写回消息历史，再交给模型生成最终回答。\n"
                "- 计算器不要裸 eval，应先做 AST 白名单校验。\n",
                encoding="utf-8",
            )

    # ── facts ────────────────────────────────────────────────

    def load_facts(self) -> Dict[str, Any]:
        try:
            return json.loads(self.facts_path.read_text(encoding="utf-8"))
        except Exception:
            return {"user_preferences": {}, "project_state": {}, "stable_facts": {}}

    def save_facts(self, facts: Dict[str, Any]) -> None:
        self.facts_path.write_text(
            json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def set_fact(self, category: str, key: str, value: Any) -> None:
        facts = self.load_facts()
        bucket = facts.setdefault(category, {})
        if not isinstance(bucket, dict):
            bucket = {}
            facts[category] = bucket
        bucket[key] = value
        self.save_facts(facts)

    # ── lessons ──────────────────────────────────────────────

    def load_lessons(self) -> List[str]:
        text = self.lessons_path.read_text(encoding="utf-8")
        result: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                result.append(stripped[2:].strip())
        return result

    def append_lesson(self, lesson: str) -> bool:
        lesson = lesson.strip()
        if not lesson:
            return False
        lessons = self.load_lessons()
        if lesson in lessons:
            return False
        with self.lessons_path.open("a", encoding="utf-8") as f:
            if not self.lessons_path.read_text(encoding="utf-8").endswith("\n"):
                f.write("\n")
            f.write(f"- {lesson}\n")
        return True

    # ── recall ───────────────────────────────────────────────

    def recall(self, query: str, max_lessons: int = 5) -> MemoryRecallResult:
        query_lower = query.lower()
        facts = self.load_facts()
        lessons = self.load_lessons()

        matched: List[str] = []
        query_terms = [
            term for term in query_lower.replace("_", " ").split() if term
        ]

        for lesson in lessons:
            lesson_lower = lesson.lower()
            if any(term in lesson_lower for term in query_terms):
                matched.append(lesson)

        if not matched:
            matched = lessons[:max_lessons]

        return MemoryRecallResult(facts=facts, matched_lessons=matched[:max_lessons])

    # ── prompt building ──────────────────────────────────────

    def build_memory_prompt(self, query: str) -> str:
        recalled = self.recall(query)
        parts: List[str] = []

        user_prefs = recalled.facts.get("user_preferences", {})
        project_state = recalled.facts.get("project_state", {})
        stable_facts = recalled.facts.get("stable_facts", {})

        if user_prefs:
            parts.append(f"用户偏好：{json.dumps(user_prefs, ensure_ascii=False)}")
        if project_state:
            parts.append(f"项目状态：{json.dumps(project_state, ensure_ascii=False)}")
        if stable_facts:
            parts.append(f"稳定事实：{json.dumps(stable_facts, ensure_ascii=False)}")
        if recalled.matched_lessons:
            filtered = [
                l for l in recalled.matched_lessons
                if not any(ex in l for ex in self._injected_memory)
            ]
            if filtered:
                parts.append("相关经验:\n- " + "\n- ".join(filtered))

        return "\n\n".join(parts).strip()

    # ── memory injection tracking ────────────────────────────

    def mark_memory_injected(self, memory_content: str) -> None:
        if memory_content and memory_content not in self._injected_memory:
            self._injected_memory.append(memory_content)

    def clear_injected_memory(self) -> None:
        self._injected_memory.clear()

    # ── auto-capture ─────────────────────────────────────────

    def remember_from_user_text(self, user_input: str) -> bool:
        text = user_input.strip()
        lower = text.lower()

        triggers = ["记住这个", "记住：", "记住", "remember this", "remember:"]
        if not any(trigger in lower for trigger in triggers):
            return False

        content = text
        for trigger in [
            "记住这个", "记住：", "记住",
            "Remember this", "remember this", "remember:",
        ]:
            content = content.replace(trigger, "")
        content = content.strip(" ：:")
        if not content:
            return False

        facts = self.load_facts()
        key = f"note_{len(facts.get('stable_facts', {})) + 1}"
        self.set_fact("stable_facts", key, content)
        return True

    def auto_capture_lesson(self, user_input: str, assistant_output: str) -> bool:
        text = (user_input + "\n" + assistant_output).lower()

        rules: List[tuple] = [
            (
                ["help", "-help", "--help", "cli"],
                "CLI 参数应优先在本地处理，不要把规则型输入误送进模型。",
            ),
            (
                ["tool_calls", "response.choices", "message history", "工具调用"],
                "分析 Agent 主循环时，优先观察 response、tool_calls 和消息历史三者的关系。",
            ),
            (
                ["memory", "记忆", "long-term"],
                "记忆系统应区分长期事实、经验教训和当前上下文，不要把所有历史都塞进 prompt。",
            ),
        ]

        for keywords, lesson in rules:
            if any(keyword in text for keyword in keywords):
                return self.append_lesson(lesson)
        return False
