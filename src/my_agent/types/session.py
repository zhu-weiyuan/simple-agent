# -*- coding: utf-8 -*-
"""
my_agent.types.session — Session 类型定义

参考 Claude Code Session 设计：
- SessionConfig: 行为配置（压缩阈值、持久化路径）
- SessionState: 会话状态（消息列表、轮数、token 估算）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .message import Message


@dataclass
class SessionConfig:
    """Session 行为配置"""
    max_turns: int = 16
    keep_recent: int = 4
    max_tokens: int = 8000
    summary_max_chars: int = 300
    save_path: Optional[Path] = None


@dataclass
class SessionState:
    """
    Agent session 状态容器。

    职责：
    - 维护消息列表（首条必须是 system）
    - token 估算与压缩判断
    - 序列化 / 反序列化（会话恢复）
    """
    messages: List[Message] = field(default_factory=list)
    turn_count: int = 0
    config: SessionConfig = field(default_factory=SessionConfig)

    @classmethod
    def create(
        cls,
        system_prompt: str,
        config: Optional[SessionConfig] = None,
    ) -> SessionState:
        return cls(
            messages=[Message.system(system_prompt)],
            config=config or SessionConfig(),
        )

    # ── 消息操作 ─────────────────────────────────────────────

    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def update_system(self, new_content: str) -> None:
        if self.messages and self.messages[0].role.value == "system":
            self.messages[0].content = new_content
        else:
            self.messages.insert(0, Message.system(new_content))

    @property
    def system_message(self) -> Optional[Message]:
        return self.messages[0] if self.messages else None

    # ── 压缩 ─────────────────────────────────────────────────

    def should_compact(self) -> bool:
        non_system = [
            m for m in self.messages[1:] if m.role.value != "system"
        ]
        turns = len(non_system) // 2
        return turns > self.config.max_turns or self.estimate_tokens() > self.config.max_tokens

    def estimate_tokens(self) -> int:
        total_chars = sum(len(m.content or "") for m in self.messages)
        has_cjk = any(
            "\u4e00" <= c <= "\uffff"
            for m in self.messages
            for c in (m.content or "")
        )
        return int(total_chars * (0.75 if has_cjk else 0.25))

    def compact(self) -> None:
        if self.estimate_tokens() < self.config.max_tokens * 0.3:
            return

        non_system = [
            m for m in self.messages[1:] if m.role.value != "system"
        ]
        keep_count = self.config.keep_recent * 2
        if len(non_system) <= keep_count:
            return
        keep = non_system[-keep_count:]
        older = non_system[:-keep_count]

        summary = self._summarize_old(older)
        boundary = Message.summary_boundary(summary)
        self.messages = [self.messages[0], boundary] + keep
        self.turn_count += 1

    def _summarize_old(self, older: List[Message]) -> str:
        parts: List[str] = []
        user_msgs = [m for m in older if m.role.value == "user"]
        assistant_msgs = [m for m in older if m.role.value == "assistant"]
        tool_msgs = [m for m in older if m.role.value == "tool"]

        if user_msgs:
            previews = [
                f"  • {(m.content or '').replace(chr(10), ' ').strip()[:60]}"
                for m in user_msgs[:5]
            ]
            parts.append(f"用户输入 ({len(user_msgs)} 条):")
            parts.extend(previews)

        if assistant_msgs:
            previews = [
                f"  • {(m.content or '').replace(chr(10), ' ').strip()[:60]}"
                for m in assistant_msgs[:5]
            ]
            parts.append(f"助手回复 ({len(assistant_msgs)} 条):")
            parts.extend(previews)

        if tool_msgs:
            parts.append(f"工具调用结果 ({len(tool_msgs)} 条): [已压缩]")

        return "\n".join(parts) or f"[{len(older)} messages summarized]"

    # ── 持久化 ───────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        import json
        target = path or self.config.save_path
        if not target:
            return
        data = {
            "turn_count": self.turn_count,
            "messages": [m.to_openai() for m in self.messages],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> SessionState:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = [Message.from_dict(raw) for raw in data.get("messages", [])]
        return cls(messages=messages, turn_count=data.get("turn_count", 0))
