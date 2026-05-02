# -*- coding: utf-8 -*-
"""
my_agent.core.engine — QueryEngine 核心循环

参考 Claude Code query.ts → QueryEngine 设计：
- run() 接受用户输入，返回最终回复
- _loop() 是工具调用循环核心
- 流式输出支持
- Hook 集成
- 通过 ToolRegistry 执行工具
"""
from __future__ import annotations

import json
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
)

from ..types.message import Message, ToolCall, Role
from ..types.session import SessionState, SessionConfig
from ..tools.registry import ToolRegistry
from .hooks import HookPoint, HookContext, HookRegistry


# LLM 调用函数签名
LLMCallFn = Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Any]
LLMStreamFn = Callable[[], Generator[str, None, None]]


class QueryEngine:
    """
    Agent 核心引擎。

    职责：
    - 管理 SessionState（消息历史 + 压缩）
    - 驱动工具调用循环
    - 注入 LLM 调用函数
    - 通过 HookRegistry 暴露扩展点
    - 通过 ToolRegistry 执行工具
    """

    def __init__(
        self,
        system_prompt: str,
        tool_registry: Optional[ToolRegistry] = None,
        hooks: Optional[HookRegistry] = None,
        session_config: Optional[SessionConfig] = None,
    ) -> None:
        self.system_prompt_base = system_prompt
        self.tool_registry = tool_registry or ToolRegistry()
        self.hooks = hooks or HookRegistry()
        self.session = SessionState.create(system_prompt, session_config)

        # LLM 调用函数（延迟注入）
        self._llm_call_fn: Optional[LLMCallFn] = None
        self._llm_stream_fn: Optional[LLMStreamFn] = None

    # ── public API ───────────────────────────────────────────

    def set_llm(
        self,
        call_fn: LLMCallFn,
        stream_fn: Optional[LLMStreamFn] = None,
    ) -> None:
        """注入 LLM 调用函数"""
        self._llm_call_fn = call_fn
        self._llm_stream_fn = stream_fn

    def run(self, user_input: str, max_tool_calls: int = 10) -> str:
        """处理一条用户消息，返回完整回复"""
        self.hooks.fire(HookPoint.QUERY_START, data={"user_input": user_input})
        self.session.append(Message.user(user_input))

        result = self._loop(max_tool_calls)

        self.hooks.fire(HookPoint.QUERY_END, data={"result": result})
        return result

    def run_stream(
        self, user_input: str, max_tool_calls: int = 10
    ) -> Generator[str, None, None]:
        """流式版本：先跑完工具轮次，最后流式输出"""
        self.hooks.fire(HookPoint.QUERY_START, data={"user_input": user_input})
        self.session.append(Message.user(user_input))

        self._loop(max_tool_calls, capture_last=True)

        if self._llm_stream_fn:
            yield from self._last_stream()
        else:
            last = self._last_assistant_content()
            if last:
                yield last

    # ── tool loop ────────────────────────────────────────────

    def _loop(self, max_tool_calls: int, capture_last: bool = False) -> str:
        """
        工具调用循环。

        每次迭代：
        1. 调用 LLM（带 tools schema）
        2. 有 tool_calls → 执行 → 追加结果 → 继续
        3. 无 tool_calls → 返回最终回复
        """
        last_response = ""

        for iteration in range(max_tool_calls):
            self.hooks.fire(HookPoint.LLM_START)
            response = self._call_llm()
            self.hooks.fire(HookPoint.LLM_END)

            if response is None:
                return "API 调用失败，请检查网络和配置。"

            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            msg_attr = getattr(choice, "message", None)
            if msg_attr is None:
                return "API 返回空消息"

            assistant_msg = Message.from_openai_choice(msg_attr)

            if capture_last and not assistant_msg.tool_calls:
                last_response = assistant_msg.content or ""
                break

            # content_filter: don't append, return immediately
            if finish_reason == "content_filter":
                return "[内容被安全过滤]"

            if assistant_msg.tool_calls:
                self.session.append(assistant_msg)
                for tc in assistant_msg.tool_calls:
                    result = self._execute_tool(tc)
                    is_error = result.startswith("错误：") or result.startswith("MCP 调用失败")
                    tool_msg = Message.tool_result(tc.id, result, is_error=is_error)
                    self.session.append(tool_msg)
                continue

            # Final response (no tool calls) — append and return
            self.session.append(assistant_msg)
            last_response = assistant_msg.content or ""

            if self.session.should_compact():
                self.hooks.fire(HookPoint.SESSION_COMPACT)
                self.session.compact()

            return last_response

        return "错误：工具调用次数过多，请检查是否陷入循环。"

    # ── LLM ──────────────────────────────────────────────────

    def _call_llm(self) -> Any:
        if self._llm_call_fn is None:
            raise RuntimeError("未设置 LLM 调用函数，请先通过 set_llm() 配置")
        openai_msgs = [m.to_openai() for m in self.session.messages]
        # Safety: remove consecutive assistant messages at the end
        cleaned = self._dedup_assistant(openai_msgs)
        schemas = self.tool_registry.all_schemas()
        return self._llm_call_fn(cleaned, schemas)

    @staticmethod
    def _dedup_assistant(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure no two consecutive assistant messages at the tail."""
        i = len(msgs) - 1
        while i > 0 and msgs[i].get("role") == "assistant" and msgs[i-1].get("role") == "assistant":
            # Remove the earlier duplicate; keep the latest
            msgs.pop(i - 1)
            i -= 1
        return msgs

    def _last_stream(self) -> Generator[str, None, None]:
        if self._llm_stream_fn:
            yield from self._llm_stream_fn()

    def _last_assistant_content(self) -> str:
        for msg in reversed(self.session.messages):
            if msg.role == Role.ASSISTANT and not msg.tool_calls:
                return msg.content or ""
        return ""

    # ── tool execution ───────────────────────────────────────

    def _execute_tool(self, tc: ToolCall) -> str:
        hook_data = {"tool_name": tc.name, "arguments": tc.arguments}
        self.hooks.fire(HookPoint.TOOL_CALL_BEFORE, data=hook_data)

        handler = self.tool_registry.get_handler(tc.name)
        if handler is None:
            error_msg = f"错误：未知工具：{tc.name}"
            self.hooks.fire(HookPoint.TOOL_ERROR, data={**hook_data, "error": error_msg})
            return error_msg

        try:
            params = tc.arguments if isinstance(tc.arguments, dict) else {}
            result = handler(params)
            self.hooks.fire(HookPoint.TOOL_CALL_AFTER, data={**hook_data, "result": result})
            return result
        except Exception as e:
            error_msg = f"工具执行失败 [{tc.name}]：{type(e).__name__}: {e}"
            self.hooks.fire(HookPoint.TOOL_ERROR, data={**hook_data, "error": error_msg})
            return error_msg

    # ── system prompt ────────────────────────────────────────

    def refresh_system_prompt(self, extra_context: str = "") -> None:
        combined = self.system_prompt_base
        if extra_context:
            combined += "\n\n" + extra_context
        self.session.update_system(combined)
