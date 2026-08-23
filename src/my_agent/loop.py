# -*- coding: utf-8 -*-
"""
my_agent.loop — DSH 状态机与 SimpleAgent 组件的桥接层

将 DSH AgentLoop 适配为 SimpleAgent 的核心执行引擎，
替代 QueryEngine 内部的 _loop/_arun 逻辑，同时保持对外 API 兼容。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Generator, List, Optional, TYPE_CHECKING

from .dsh_state_machine import (
    AgentLoop as DSHAgentLoop,
    AgentLoopBuilder,
    PhaseKind,
    Phase,
    Session as DSHSession,
    InboxPosition,
    InboxMessage,
    SessionEventType,
)
from .types.message import Message, Role, ToolCall
from .types.session import SessionState, SessionConfig
from .tools.registry import ToolRegistry

if TYPE_CHECKING:
    from .core.engine import QueryEngine

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    """统一的执行结果"""
    content: str
    stop_reason: str
    stop_detail: str
    usage: Dict[str, Any]
    iterations: int
    model: Optional[str]
    fallback_reason: Optional[str]


class SimpleAgentLoop:
    """
    SimpleAgent 专用的 DSH 状态机包装器
    
    职责：
    1. 适配 SimpleAgent 的工具注册表、LLM 调用、会话管理
    2. 将 DSH 的 SessionEvent 转换为 SimpleAgent 的 Message
    3. 提供与 QueryEngine 兼容的 run/arun/run_stream/arun_stream 接口
    4. 集成 checkpoints、budget guardrails、hooks
    """
    
    def __init__(
        self,
        engine: "QueryEngine",
        system_prompt: str,
        session_state: SessionState,
        max_tool_calls: int = 200,
        checkpoint_path: Optional[str] = None,
    ):
        self._engine = engine
        self._system_prompt = system_prompt
        self._session_state = session_state
        self._max_tool_calls = max_tool_calls
        
        # 创建 DSH AgentLoop
        self._dsh_loop = (
            AgentLoopBuilder()
            .system_prompt(system_prompt)
            .tool_registry(engine.tool_registry)
            # QueryEngine 的内部 _call_llm/_acall_stream 签名与 DSH
            # callback 契约不同；由 set_llm_functions 显式注入适配后的函数。
            .max_tool_calls(max_tool_calls)
            .max_parallel_tool_calls(engine.tool_concurrency)
            .checkpoint_path(checkpoint_path)
            .context_budget(
                context_window=getattr(engine, "context_window", 131072),
                target_ratio=float(getattr(engine, "context_target_ratio", 0.60)),
                reserved_output_tokens=getattr(engine, "reserved_output_tokens", 8192),
                safety_margin=getattr(engine, "context_safety_margin", 1024),
                provider_input_ratio=float(getattr(engine, "provider_input_ratio", 0.50)),
            )
            .build()
        )
        
        # 绑定 SimpleAgent 会话状态到 DSH Session
        self._sync_session_to_dsh()
        
        # 注册事件监听器用于流式输出
        self._stream_queue: asyncio.Queue = asyncio.Queue()
        self._dsh_loop.on_event(self._on_dsh_event)
        
        # 运行时状态
        self._current_session_id: str = ""
        self._llm_idle_timeout: Optional[float] = None
    
    def _sync_session_to_dsh(self) -> None:
        """将 SimpleAgent SessionState 同步到 DSH Session"""
        dsh_session = self._dsh_loop.session
        dsh_session._events.clear()
        dsh_session._seq = 0
        
        # 转换现有消息
        for msg in self._session_state.messages:
            if msg.role == Role.USER:
                dsh_session.append(SessionEventType.USER_MESSAGE.value, {"message": msg.to_openai()})
            elif msg.role == Role.ASSISTANT:
                dsh_session.append(SessionEventType.ASSISTANT_MESSAGE.value, {
                    "message": msg.to_openai(),
                    "usage": msg.metadata.get("usage", {}) if msg.metadata else {}
                })
            elif msg.role == Role.TOOL:
                dsh_session.append(SessionEventType.TOOL_RESULT.value, {
                    "message": msg.to_openai(),
                    "is_error": msg.metadata.get("is_error", False) if msg.metadata else False
                })
            elif msg.role == Role.SYSTEM:
                dsh_session.set_request_header({"system_prompt": msg.content})
    
    def _sync_dsh_to_session(self) -> None:
        """将 DSH Session 的新事件同步回 SimpleAgent SessionState"""
        dsh_session = self._dsh_loop.session
        # 这里可以增量同步，或者在结束时全量同步
        pass
    
    def _on_dsh_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Map DSH lifecycle events to UI-safe stream frames.

        Internal planning prose is never emitted as ``token``.  The browser
        receives explicit thinking/tool progress frames and one final frame.
        """
        try:
            if event_type == "llm/start":
                self._stream_queue.put_nowait({
                    "progress": "thinking",
                    "phase": "llm",
                    "turn": data.get("turn"),
                    "step": data.get("step"),
                })
            elif event_type == "tool/call":
                self._stream_queue.put_nowait({
                    "progress": f"tool:{data.get('name', '')}",
                    "tool_call": {
                        "name": data.get("name"),
                        "arguments": data.get("arguments"),
                    },
                })
            elif event_type == "tool/result":
                self._stream_queue.put_nowait({
                    "progress": "tool:done",
                    "tool_result": {
                        "name": data.get("name"),
                        "ok": not data.get("is_error", False),
                    },
                })
            elif event_type == "context/compacted":
                self._stream_queue.put_nowait({
                    "progress": "context:compacted",
                    "context": {
                        "dropped_messages": data.get("dropped_messages", 0),
                        "estimated_tokens": data.get("estimated_tokens", 0),
                        "budget": data.get("budget", 0),
                    },
                })
            elif event_type == "assistant/message":
                if data.get("visibility") == "final":
                    msg = data.get("message", {}) or {}
                    self._stream_queue.put_nowait({
                        "final": True,
                        "content": msg.get("content", ""),
                        "usage": data.get("usage", {}),
                    })
        except Exception:
            # A disconnected consumer must never break the agent state machine.
            logger.debug("stream event enqueue failed", exc_info=True)

    def set_llm_functions(self, call_fn: Optional[Callable] = None, stream_fn: Optional[Callable] = None) -> None:
        """动态设置 LLM 调用函数（用于运行时切换）。"""
        if call_fn:
            self._dsh_loop._llm_call_fn = call_fn
        if stream_fn:
            self._dsh_loop._llm_stream_fn = stream_fn

    def _dsh_call(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
        """Adapt QueryEngine's async LLM call to DSH's sync callback."""
        raise RuntimeError("DSH sync callback is not configured")
    
    def set_session_id(self, session_id: str) -> None:
        self._current_session_id = session_id
        self._dsh_loop._session.id = session_id
    
    def set_llm_idle_timeout(self, timeout: Optional[float]) -> None:
        self._llm_idle_timeout = timeout
    
    def run(self, user_input: str) -> LoopResult:
        """同步执行（兼容旧接口）"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.arun(user_input))
        finally:
            loop.close()
    
    async def arun(self, user_input: str) -> LoopResult:
        """异步执行，返回本轮唯一的正式回复。"""
        start_seq = self._last_event_seq()
        self._dsh_loop.followup(user_input)
        # The DSH session owns the live user event; mirror it once into the
        # application session so direct /a2a and production chat requests are
        # persisted consistently.
        if user_input:
            self._session_state.append(Message.user(user_input))
        await self._dsh_loop.when_idle()
        result = self._result_from_events(start_seq)
        self._sync_dsh_messages_back(start_seq)
        return result

    def _last_event_seq(self) -> int:
        events = self._dsh_loop.session.events
        return events[-1].seq if events else 0

    def _result_from_events(self, start_seq: int) -> LoopResult:
        """Extract only the terminal result from one DSH run.

        DSH records planning notes and tool-call assistant messages for model
        context.  Those events are deliberately not eligible as the formal
        answer, even when they are the most recent assistant event.
        """
        events = [e for e in self._dsh_loop.session.events if e.seq > start_seq]
        content = ""
        stop_reason: Optional[str] = None
        stop_detail = ""
        usage: Dict[str, Any] = {}
        model = None
        iterations = sum(1 for e in events if e.type == SessionEventType.TURN_START.value)

        for event in events:
            if event.type == SessionEventType.TURN_END.value:
                reason = event.data.get("reason") or {}
                if isinstance(reason, dict):
                    stop_reason = reason.get("kind") or stop_reason or "completed"
                    error = reason.get("error")
                    if error:
                        stop_reason = "error"
                        stop_detail = error.get("message", "") if isinstance(error, dict) else str(error)
            elif event.type == SessionEventType.ASSISTANT_MESSAGE.value:
                msg = event.data.get("message", {}) or {}
                visibility = msg.get("visibility")
                tool_calls = msg.get("tool_calls") or []
                if visibility == "final" or (visibility is None and msg.get("content") and not tool_calls):
                    if msg.get("content"):
                        content = msg.get("content", "")
                        usage = event.data.get("usage", {}) or usage
                        model = event.data.get("model") or model

        if not stop_reason:
            stop_reason = "completed" if content else "empty"
        stop_reason = {
            "max-tool-calls": "max_tool_calls",
            "max-steps": "max_steps",
            "max-tokens": "max_tokens",
        }.get(stop_reason, stop_reason)
        return LoopResult(
            content=content,
            stop_reason=stop_reason,
            stop_detail=stop_detail,
            usage=usage,
            iterations=iterations,
            model=model,
            fallback_reason=None,
        )

    def run_stream(self, user_input: str) -> Generator[Dict[str, Any], None, None]:
        """同步流式（兼容接口）。"""
        loop = asyncio.new_event_loop()
        try:
            async def _collect():
                results = []
                async for chunk in self.arun_stream(user_input):
                    results.append(chunk)
                return results
            chunks = loop.run_until_complete(_collect())
            for chunk in chunks:
                yield chunk
        finally:
            loop.close()

    async def arun_stream(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """真异步流式：思考、工具进度和正式回复分层输出。"""
        # Discard only frames left by a previously completed run.  We never
        # scan the whole session for assistant text, so old replies cannot leak
        # into this stream.
        while not self._stream_queue.empty():
            try:
                self._stream_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        start_seq = self._last_event_seq()
        self._dsh_loop.followup(user_input)
        if user_input:
            self._session_state.append(Message.user(user_input))
        async for chunk in self._arun_stream_internal(start_seq):
            yield chunk

    async def _arun_stream_internal(self, start_seq: int) -> AsyncIterator[Dict[str, Any]]:
        """Consume events generated by exactly one DSH run."""
        loop_task = asyncio.create_task(self._dsh_loop.when_idle())
        result: Optional[LoopResult] = None
        try:
            while not loop_task.done() or not self._stream_queue.empty():
                try:
                    yield await asyncio.wait_for(self._stream_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
            await loop_task
            result = self._result_from_events(start_seq)
            self._sync_dsh_messages_back(start_seq)
        except asyncio.CancelledError:
            self._dsh_loop.cancel()
            raise
        finally:
            if result is None:
                result = self._result_from_events(start_seq)
            yield {
                "done": True,
                "session_id": self._current_session_id,
                "content": result.content,
                "stop_reason": result.stop_reason,
                "stop_detail": result.stop_detail,
                "usage": result.usage,
                "iterations": result.iterations,
                "model": result.model,
            }

    def _sync_dsh_messages_back(self, start_seq: int = 0) -> None:
        """Persist only new model-context messages, not internal progress text."""
        for event in self._dsh_loop.session.events:
            if event.seq <= start_seq:
                continue
            if event.type == SessionEventType.ASSISTANT_MESSAGE.value:
                msg_data = event.data.get("message", {}) or {}
                if msg_data.get("role") != "assistant":
                    continue
                tool_calls = msg_data.get("tool_calls") or []
                visibility = msg_data.get("visibility")
                # Keep tool-call envelopes for context recovery, but suppress
                # their human planning prose from persisted conversation history.
                if visibility != "final" and not tool_calls:
                    continue
                tc_objs = []
                for tc in tool_calls:
                    func = tc.get("function", {}) or {}
                    tc_objs.append(ToolCall(
                        id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", {}) if isinstance(func.get("arguments", {}), dict) else {},
                    ))
                self._session_state.append(Message.assistant(
                    content=msg_data.get("content", "") if visibility == "final" else "",
                    tool_calls=tc_objs or None,
                ))
            elif event.type == SessionEventType.TOOL_RESULT.value:
                msg_data = event.data.get("message", {}) or {}
                if msg_data.get("role") == "tool":
                    self._session_state.append(Message.tool_result(
                        tool_call_id=msg_data.get("tool_call_id", ""),
                        content=msg_data.get("content", ""),
                        is_error=event.data.get("is_error", False),
                    ))

    def save_checkpoint(self) -> Optional[Dict[str, Any]]:
        return self._dsh_loop.save_checkpoint()
    
    def restore_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        ok = self._dsh_loop.restore_checkpoint(checkpoint)
        if ok:
            self._sync_session_to_dsh()
        return ok
    
    def get_session_snapshot(self) -> Dict[str, Any]:
        return self._dsh_loop.session.snapshot()
    
    @property
    def dsh_loop(self) -> DSHAgentLoop:
        return self._dsh_loop
    
    @property
    def phase(self) -> Phase:
        return self._dsh_loop.phase
    
    @property
    def status(self) -> str:
        return self._dsh_loop.status.value


def create_simple_agent_loop(
    engine: "QueryEngine",
    system_prompt: str,
    session_state: SessionState,
    max_tool_calls: int = 200,
    checkpoint_path: Optional[str] = None,
) -> SimpleAgentLoop:
    """工厂函数：创建 SimpleAgentLoop"""
    return SimpleAgentLoop(
        engine=engine,
        system_prompt=system_prompt,
        session_state=session_state,
        max_tool_calls=max_tool_calls,
        checkpoint_path=checkpoint_path,
    )