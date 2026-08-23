# -*- coding: utf-8 -*-
"""
my_agent.compaction — DSH-style 会话压缩引擎

核心设计参考 @deepseek-ai/dsh-compaction-basic：
1. 结构化摘要格式 (Markdown sections)
2. LLM 生成摘要，复用 KV cache (replay prefix + compaction instruction as final user msg)
3. Tool-pairing balanced boundaries (不拆分 tool-call/result 对)
4. Prefix preservation (保留最早消息 verbatim)
5. Durable checkpoint format with <compacted-summary> tags
6. Token-meter driven retention policy
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .core.engine import QueryEngine
from .types.message import Message, Role
from .types.session import SessionState, SessionConfig


# ── 常量定义 ────────────────────────────────────────────────

SUMMARY_OPEN_TAG = "<compacted-summary>"
SUMMARY_CLOSE_TAG = "</compacted-summary>"

CHECKPOINT_PREAMBLE = (
    "This is an automatically generated checkpoint condensing an earlier span of the "
    "conversation to free up context. Treat the captured context as established background "
    "and build on it without restating it. Continue the task directly from the messages "
    "that follow, without acknowledging this checkpoint."
)

COMPACTION_INSTRUCTION = """You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.

Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.

## Primary Request and Intent
- [the user's original and evolving goals; quote verbatim where the exact wording matters]

## Key Technical Concepts
- [technologies, frameworks, patterns, and conventions in play]

## Files and Code
- [exact path: why it matters, key changes or snippets]

## Errors and Fixes
- [error: how it was resolved, plus any related user feedback]

## Pending Jobs
- [explicitly requested work not yet completed]

## Current Work
- [precisely what was in progress at this checkpoint]

## Next Step
- [the single next action, directly in line with the most recent request, or "(none)"]

## Critical Context
- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]

Rules:
- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.
- Capture user feedback and explicit instructions faithfully, especially corrections.
- Do NOT mention this summarization request or that the context was compacted.
- Output only the checkpoint text: do not call any tool or take any other action.
- If the conversation already contains a """ + SUMMARY_OPEN_TAG + """ block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated summary under the same structure."""

# ── 数据类 ────────────────────────────────────────────────


@dataclass
class CompactionConfig:
    """压缩配置"""
    # 压力阈值：token 使用率超过此比例触发压缩
    threshold_ratio: float = 0.8
    # 保留尾部比例：保留最近对话的 token 比例
    retain_ratio: float = 0.16
    # 绝对保留 token 数（优先于 retain_ratio）
    retain_tokens: Optional[int] = None
    # 摘要模型配置
    summarization_provider: str = ""
    summarization_model: str = ""
    # 摘要最大 token
    max_tokens: int = 8192
    # 自动压缩开关
    auto: bool = True


@dataclass
class CompactionResult:
    """压缩结果"""
    compaction_id: str
    start_seq: int
    end_seq: int
    summary: str
    shadowed_token_count: int
    checkpoint_token_count: int


@dataclass
class CompactionRange:
    """待压缩范围"""
    start_seq: int
    end_seq: int
    start_idx: int
    end_idx: int
    shadowed_seqs: List[int]
    shadowed_token_count: int


class TokenEstimator:
    """简单的 token 估算器"""

    @staticmethod
    def estimate_message(message: Message) -> int:
        """Conservative estimate aligned with the DSH request guard."""
        content = message.content or ""
        if not content:
            return 1
        cjk = sum(
            1 for ch in content
            if ("\u3400" <= ch <= "\u9fff"
                or "\uf900" <= ch <= "\ufaff"
                or "\u3000" <= ch <= "\u303f"
                or "\u3040" <= ch <= "\u30ff"
                or "\uac00" <= ch <= "\ud7af")
        )
        non_cjk = len(content) - cjk
        # Local tokenizers vary; a CJK character is conservatively one token.
        return max(1, int(cjk + (non_cjk / 3.0) + 0.999))

    @staticmethod
    @staticmethod
    def estimate_messages(messages: List[Message]) -> int:
        return sum(TokenEstimator.estimate_message(m) for m in messages)


class CompactionEngine:
    """
    DSH-style 会话压缩引擎

    工作流程：
    1. TokenMeter 测量会话压力
    2. 超过 threshold_ratio 触发压缩
    3. select_compactable_range 选择 head-anchored 范围，保留 priced tail
    4. 验证边界：toolPairingBalancedBefore/After (不拆分 tool-call/result)
    5. buildSummarizationInput: replay prefix (system + tools + region messages)
    6. LLM 调用：replay prefix + compaction instruction as FINAL user message
    7. frameSummary: wrap in <compacted-summary> tags + preamble
    8. commitCompactionBody: 替换范围为单个 checkpoint user message
    """

    def __init__(
        self,
        engine: QueryEngine,
        config: Optional[CompactionConfig] = None,
    ) -> None:
        self.engine = engine
        self.config = config or CompactionConfig()
        self.token_estimator = TokenEstimator()

    # ── 公共接口 ──────────────────────────────────────────

    def maybe_compact(
        self,
        session: SessionState,
        *,
        threshold_ratio: Optional[float] = None,
    ) -> Optional[CompactionResult]:
        """检查是否需要压缩，需要则执行。

        ``threshold_ratio`` lets the DSH request path compact earlier than the
        hard context limit while preserving the default behaviour of callers
        that use this engine directly.
        """
        if not self.config.auto:
            return None

        pressure = self._measure_pressure(session)
        threshold = self.config.threshold_ratio if threshold_ratio is None else float(threshold_ratio)
        threshold = min(0.95, max(0.05, threshold))
        if pressure.ratio < threshold:
            return None

        # 选择待压缩范围
        range_ = self._select_compactable_range(session, pressure)
        if range_ is None:
            return None

        # 执行压缩
        return self._compact_range(session, range_, pressure)

    def force_compact(self, session: SessionState) -> CompactionResult:
        """强制压缩（手动触发）"""
        pressure = self._measure_pressure(session)
        range_ = self._select_compactable_range(session, pressure)
        if range_ is None:
            raise ValueError("No compactable range available")
        return self._compact_range(session, range_, pressure)

    # ── 内部实现 ──────────────────────────────────────────

    def _measure_pressure(self, session: SessionState) -> "PressureMeasurement":
        """测量会话压力"""
        total_tokens = self.token_estimator.estimate_messages(session.messages)
        # 估算 context window（从 engine 配置获取）
        context_window = getattr(self.engine, 'context_window', 131072)
        return PressureMeasurement(
            total_tokens=total_tokens,
            context_window=context_window,
            ratio=total_tokens / context_window if context_window > 0 else 0.0,
        )

    def _select_compactable_range(
        self,
        session: SessionState,
        measurement: "PressureMeasurement"
    ) -> Optional[CompactionRange]:
        """
        选择待压缩范围：head-anchored，保留 priced tail
        对应 DSH 的 selectCompactableRange
        """
        messages = session.messages
        if len(messages) <= 1:  # 只有 system
            return None

        # 计算保留 token 预算
        retain_tokens = self.config.retain_tokens
        if retain_tokens is None:
            retain_tokens = int(measurement.total_tokens * self.config.retain_ratio)

        # 从尾部向前累积 token，直到达到 retain_tokens
        non_system = [(i, m) for i, m in enumerate(messages[1:], 1)]
        if not non_system:
            return None

        accumulated = 0
        keep_from_idx = len(non_system)
        for idx in range(len(non_system) - 1, -1, -1):
            idx_in_messages, msg = non_system[idx]
            accumulated += self.token_estimator.estimate_message(msg)
            keep_from_idx = idx
            if accumulated >= retain_tokens:
                break

        if keep_from_idx == 0:
            return None  # 全部都要保留

        # 调整到 tool-pairing balanced boundary。
        # ``keep_from_idx`` 指向保留尾部的第一条消息，绝不能从一条
        # tool result 开始；否则既会生成不合法的 chat-template 消息序列，
        # 也会把它和前面的 assistant tool_call 拆开。历史数据有可能
        # 已经包含孤立 tool result（例如旧版本异常中断），这类消息不能
        # 让边界选择陷入循环：直接把孤立 result 收进摘要，继续寻找安全
        # 的起点即可。
        while keep_from_idx < len(non_system):
            _, msg = non_system[keep_from_idx]
            if msg.role != Role.TOOL:
                break

            tool_call_id = str(getattr(msg, "tool_call_id", "") or "")
            owner_idx: Optional[int] = None
            for candidate_idx in range(keep_from_idx - 1, -1, -1):
                _, candidate = non_system[candidate_idx]
                if candidate.role == Role.USER:
                    # tool calls never legally cross a new user turn.
                    break
                if candidate.role != Role.ASSISTANT:
                    continue
                for call in candidate.tool_calls or []:
                    call_id = str(getattr(call, "id", "") or "")
                    if call_id and call_id == tool_call_id:
                        owner_idx = candidate_idx
                        break
                if owner_idx is not None:
                    break

            if owner_idx is not None:
                # Keep the requesting assistant message together with all of
                # its following tool results.
                keep_from_idx = owner_idx
                break

            # Malformed/orphan tool result: compact it instead of retaining an
            # invalid leading tool message.  This must advance the index.
            keep_from_idx += 1

        if keep_from_idx <= 0 or keep_from_idx >= len(non_system):
            return None

        # 转换为 messages 中的索引
        start_idx = 1  # 跳过 system
        end_idx = non_system[keep_from_idx - 1][0]
        shadowed_seqs = list(range(start_idx, end_idx + 1))

        # 计算 shadowed token count
        shadowed_tokens = sum(
            self.token_estimator.estimate_message(messages[i])
            for i in shadowed_seqs
        )

        return CompactionRange(
            start_seq=start_idx,
            end_seq=end_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            shadowed_seqs=shadowed_seqs,
            shadowed_token_count=shadowed_tokens,
        )

    def _build_summarization_input(
        self,
        session: SessionState,
        range_: CompactionRange
    ) -> Tuple[Optional[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        构建摘要输入：system + tools + region messages
        对应 DSH 的 buildSummarizationInput
        """
        # 从 engine 获取 system prompt 和 tools
        system_prompt = None
        tools = None

        if hasattr(self.engine, 'system_prompt_base'):
            system_prompt = self.engine.system_prompt_base

        if hasattr(self.engine, 'tool_registry'):
            tools = self.engine.tool_registry.all_schemas()

        # 提取待压缩区域的消息
        region_messages = []
        for seq in range_.shadowed_seqs:
            if seq < len(session.messages):
                msg = session.messages[seq]
                if msg.role != Role.SYSTEM:
                    region_messages.append(msg.to_openai())

        return system_prompt, tools or [], region_messages

    def _summarize_with_llm(
        self,
        system_prompt: Optional[str],
        tools: List[Dict[str, Any]],
        region_messages: List[Dict[str, Any]],
    ) -> str:
        """
        调用 LLM 生成结构化摘要
        关键：replay prefix + compaction instruction as FINAL user message
        这样可以复用 KV cache
        """
        # 构建 messages: system + tools + region_messages + compaction_instruction
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if tools:
            # tools 以 function calling schema 形式传递给 LLM
            # 这里简化：只在 messages 中体现 tools 存在
            pass

        # 添加待压缩的对话区域
        messages.extend(region_messages)

        # 最后添加 compaction instruction 作为 user message
        # 这是复用 KV cache 的关键！
        messages.append({"role": "user", "content": COMPACTION_INSTRUCTION})

        # 调用 LLM
        try:
            if hasattr(self.engine, '_llm_call_fn') and self.engine._llm_call_fn:
                # 同步调用
                response = self.engine._llm_call_fn(messages, tools)
                # 解析 response
                content = self._extract_content(response)
                return content
        except Exception as e:
            # LLM 调用失败（超时、连接错误等），静默回退
            pass

        # fallback：使用简单摘要
        return self._fallback_summary(region_messages)

    def _fallback_summary(self, region_messages: List[Dict[str, Any]]) -> str:
        """Deterministic, bounded fallback when a summarization LLM is unavailable.

        A previous fallback only wrote ``(auto-summary)`` placeholders.  That
        technically reduced tokens but discarded the investigation state needed
        to resume a long tool-driven task.  Preserve compact excerpts and tool
        identities instead; this is deterministic, local, and cannot create an
        additional oversized LLM request during an already pressured turn.
        """
        def clip(value: Any, limit: int) -> str:
            text = str(value or "").replace("\x00", "").strip()
            if len(text) > limit:
                return text[:limit] + " …[已裁剪]"
            return text or "(empty)"

        requests: List[str] = []
        files_and_tools: List[str] = []
        findings: List[str] = []
        for message in region_messages[-24:]:
            role = str(message.get("role") or "")
            content = clip(message.get("content"), 700 if role == "tool" else 420)
            if role == "user":
                requests.append(f"- {content}")
            elif role == "tool":
                tool_id = clip(message.get("tool_call_id"), 80)
                files_and_tools.append(f"- tool_result {tool_id}: {content}")
            elif role == "assistant":
                calls = message.get("tool_calls") or []
                names = []
                for call in calls:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    name = function.get("name") if isinstance(function, dict) else None
                    if name:
                        names.append(str(name))
                if names:
                    files_and_tools.append(f"- tool_call: {', '.join(names[:8])}")
                elif content != "(empty)":
                    findings.append(f"- {content}")

        return "\n".join([
            "## Primary Request and Intent",
            *(requests[-6:] or [f"- auto-summary of {len(region_messages)} messages"]),
            "## Key Technical Concepts",
            "- Deterministic local fallback; retain recent task and tool facts.",
            "## Files and Code",
            *(files_and_tools[-12:] or ["- (no tool details retained)"]),
            "## Errors and Fixes",
            *(findings[-5:] or ["- (none)"]),
            "## Pending Jobs",
            "- Continue the most recent user request using the retained tool state.",
            "## Current Work",
            "- Historical context was compacted before the next model request.",
            "## Next Step",
            "- Inspect the latest user request and continue from the newest tool result.",
            "## Critical Context",
            "- Older verbose tool output was clipped, not treated as a final answer.",
        ])

    def _checkpoint_budget(self, measurement: "PressureMeasurement") -> int:
        """Bound a checkpoint so it can coexist with the live task tail.

        ``max_tokens`` is an upper ceiling, not a reason to create an 8k
        checkpoint in a 4k test/development context.  Reserve most of the
        input budget for the newest user turn and valid tool-call group.
        """
        window_share = max(128, int(max(1, measurement.context_window) * 0.16))
        return max(128, min(int(self.config.max_tokens), window_share))

    def _fit_summary_to_budget(self, summary: str, token_budget: int) -> str:
        """Trim a deterministic or model summary without breaking the checkpoint."""
        if self.token_estimator.estimate_message(Message.user(summary)) <= token_budget:
            return summary
        suffix = "\n- [Earlier checkpoint details clipped to preserve the active task.]"
        low, high, best = 0, len(summary), "## Primary Request and Intent" + suffix
        while low <= high:
            middle = (low + high) // 2
            candidate = summary[:middle].rstrip()
            if "\n" in candidate:
                candidate = candidate.rsplit("\n", 1)[0].rstrip()
            candidate = (candidate or "## Primary Request and Intent") + suffix
            if self.token_estimator.estimate_message(Message.user(candidate)) <= token_budget:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best


    def _frame_summary(self, summary: str) -> str:
        """包装摘要为 checkpoint 格式"""
        return f"{CHECKPOINT_PREAMBLE}\n\n{SUMMARY_OPEN_TAG}\n{summary}\n{SUMMARY_CLOSE_TAG}"

    def _compact_range(
        self,
        session: SessionState,
        range_: CompactionRange,
        measurement: "PressureMeasurement"
    ) -> CompactionResult:
        """执行压缩：生成摘要 -> 替换消息"""
        # 1. 构建输入
        system_prompt, tools, region_messages = self._build_summarization_input(
            session, range_
        )

        # 2. 生成摘要
        summary = self._summarize_with_llm(system_prompt, tools, region_messages)
        summary = self._fit_summary_to_budget(summary, self._checkpoint_budget(measurement))

        # 3. 包装为 checkpoint 格式
        checkpoint_content = self._frame_summary(summary)

        # 4. 创建 checkpoint user message
        compaction_id = str(uuid.uuid4())
        checkpoint_msg = Message.user(checkpoint_content)

        # 5. 替换消息：移除 range，插入 checkpoint
        # 关键修复：确保 start_idx >= 1，永远保留索引 0 的系统消息
        safe_start_idx = max(1, range_.start_idx)
        new_messages = session.messages[:safe_start_idx] + [checkpoint_msg] + session.messages[range_.end_idx + 1:]
        session.messages = new_messages

        # 6. 计算 checkpoint token 数
        checkpoint_tokens = self.token_estimator.estimate_message(checkpoint_msg)

        return CompactionResult(
            compaction_id=compaction_id,
            start_seq=range_.start_seq,
            end_seq=range_.end_seq,
            summary=summary,
            shadowed_token_count=range_.shadowed_token_count,
            checkpoint_token_count=checkpoint_tokens,
        )


@dataclass
class PressureMeasurement:
    """压力测量结果"""
    total_tokens: int
    context_window: int
    ratio: float

    def __init__(self, total_tokens: int, context_window: int, ratio: float):
        self.total_tokens = total_tokens
        self.context_window = context_window
        self.ratio = ratio


# ── 便捷函数：集成到 SessionState ─────────────────────────


def install_compaction_engine(engine: QueryEngine, config: Optional[CompactionConfig] = None) -> CompactionEngine:
    """安装压缩引擎到 engine"""
    compaction_engine = CompactionEngine(engine, config)
    # 替换 session 的 compact 方法
    original_compact = SessionState.compact

    def enhanced_compact(self: SessionState) -> None:
        result = engine._compaction_engine.maybe_compact(self)
        if result is None:
            # 回退到原有逻辑
            original_compact(self)
        else:
            print(f"[Compaction] Compacted {result.shadowed_token_count} tokens -> {result.checkpoint_token_count} tokens (id: {result.compaction_id[:8]})")

    SessionState.compact = enhanced_compact
    engine._compaction_engine = CompactionEngine(engine, config)
    return engine._compaction_engine
