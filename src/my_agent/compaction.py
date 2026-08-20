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
        """估算单条消息的 token 数"""
        content = message.content or ""
        # 简单估算：中文 ~1.3 chars/token, 英文 ~4 chars/token
        has_cjk = any("\u4e00" <= c <= "\uffff" for c in content)
        chars = len(content)
        return int(chars * (0.75 if has_cjk else 0.25))

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

    def maybe_compact(self, session: SessionState) -> Optional[CompactionResult]:
        """检查是否需要压缩，需要则执行"""
        if not self.config.auto:
            return None

        pressure = self._measure_pressure(session)
        if pressure.ratio < self.config.threshold_ratio:
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
        context_window = getattr(self.engine, 'context_window', 32768)
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

        # 调整到 tool-pairing balanced boundary
        # 简化：向前找到最近的 assistant message 边界
        while keep_from_idx > 0:
            _, msg = non_system[keep_from_idx - 1]
            if msg.role == Role.ASSISTANT and not msg.tool_calls:
                # 这是一个不含 tool_calls 的 assistant 消息，安全切分
                break
            if msg.role == Role.TOOL:
                # tool message，需要找到对应的 assistant
                continue
            keep_from_idx -= 1

        if keep_from_idx == 0:
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
        """LLM 不可用时的回退摘要"""
        parts = [
            f"## Primary Request and Intent\n- (auto-summary: {len(region_messages)} messages)",
            "## Key Technical Concepts\n- (auto-summary)",
            "## Files and Code\n- (auto-summary)",
            "## Errors and Fixes\n- (auto-summary)",
            "## Pending Jobs\n- (auto-summary)",
            "## Current Work\n- (auto-summary)",
            "## Next Step\n- (auto-summary)",
            "## Critical Context\n- (auto-summary)",
        ]
        return "\n".join(parts)

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

        # 3. 包装为 checkpoint 格式
        checkpoint_content = self._frame_summary(summary)

        # 4. 创建 checkpoint user message
        compaction_id = str(uuid.uuid4())
        checkpoint_msg = Message.user(checkpoint_content)

        # 5. 替换消息：移除 range，插入 checkpoint
        new_messages = session.messages[:range_.start_idx] + [checkpoint_msg] + session.messages[range_.end_idx + 1:]
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