# -*- coding: utf-8 -*-
"""
my_agent.dsh_state_machine — DSH 风格状态机

参考 DSH (DeepSeek Harness) 的事件驱动状态机设计:
- Phase: 状态阶段 (idle / running / maintenance)
- Turn / Step: 对话轮次与步骤边界
- Inbox: 消息队列 (next-turn / next-step)
- AgentLoop: 主循环驱动器
- Session: 持久化会话状态
- Checkpoint / Recovery: 检查点与恢复
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PhaseKind(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    MAINTENANCE = "maintenance"


@dataclass
class Phase:
    kind: PhaseKind
    turn: int = 0
    step: int = 0
    wake_requested: bool = False
    abort: Optional["AbortController"] = None


class AbortController:
    def __init__(self) -> None:
        self._aborted = False
        self._reason: Optional[BaseException] = None
        self._listeners: List[Callable[[], None]] = []

    @property
    def signal(self) -> "AbortSignal":
        return AbortSignal(self)

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> Optional[BaseException]:
        return self._reason

    def abort(self, reason: Optional[BaseException] = None) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._reason = reason or AbortError("aborted")
        for listener in self._listeners:
            try:
                listener()
            except Exception:
                logger.warning("abort listener raised", exc_info=True)

    def on_abort(self, listener: Callable[[], None]) -> None:
        if self._aborted:
            try:
                listener()
            except Exception:
                pass
        else:
            self._listeners.append(listener)


class AbortSignal:
    def __init__(self, controller: AbortController) -> None:
        self._controller = controller

    @property
    def aborted(self) -> bool:
        return self._controller.aborted

    @property
    def reason(self) -> Optional[BaseException]:
        return self._controller.reason

    def throw_if_aborted(self) -> None:
        if self._controller.aborted:
            raise self._controller.reason or AbortError("aborted")

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._controller.on_abort(listener)


class AbortError(Exception):
    pass


class InboxPosition(str, Enum):
    NEXT_TURN = "next-turn"
    NEXT_STEP = "next-step"


@dataclass
class InboxMessage:
    content: Any
    position: InboxPosition = InboxPosition.NEXT_TURN
    priority: int = 0


class Inbox:
    def __init__(self) -> None:
        self._next_step: List[InboxMessage] = []
        self._next_turn: List[InboxMessage] = []

    @property
    def has_pending(self) -> bool:
        return bool(self._next_step or self._next_turn)

    @property
    def next_step(self) -> List[InboxMessage]:
        return self._next_step

    def splice(self, position: InboxPosition, start: int, count: int,
               messages: List[InboxMessage]) -> None:
        target = self._next_step if position == InboxPosition.NEXT_STEP else self._next_turn
        target[start:start + count] = messages

    def claim(self, position: InboxPosition, turn: int) -> List[InboxMessage]:
        target = self._next_step if position == InboxPosition.NEXT_STEP else self._next_turn
        claimed = list(target)
        target.clear()
        return claimed

    def clear(self) -> None:
        self._next_step.clear()
        self._next_turn.clear()

    def __len__(self) -> int:
        return len(self._next_step) + len(self._next_turn)


class SessionEventType(str, Enum):
    TURN_START = "turn/start"
    TURN_END = "turn/end"
    STEP_START = "step/start"
    STEP_END = "step/end"
    USER_MESSAGE = "user/message"
    ASSISTANT_MESSAGE = "assistant/message"
    ASSISTANT_CHUNK = "assistant/chunk"
    TOOL_CALL = "tool/call"
    TOOL_RESULT = "tool/result"
    REQUEST_HEADER = "request/header"
    REQUEST_CONTEXT = "request/context"
    SESSION_CHECKPOINT = "session/checkpoint"


@dataclass
class SessionEvent:
    seq: int
    type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class Session:
    def __init__(self, session_id: str = "") -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self._events: List[SessionEvent] = []
        self._seq = 0
        self._request_header: Optional[Dict[str, Any]] = None
        self._request_context: Optional[Dict[str, Any]] = None
        self._surface_nodes: set = set()

    @property
    def events(self) -> List[SessionEvent]:
        return self._events

    @property
    def header(self) -> Optional[Dict[str, Any]]:
        return self._request_header

    def append(self, event_type: str, data: Dict[str, Any],
               surface_op: str = "append",
               source_event_seqs: Optional[List[int]] = None) -> SessionEvent:
        self._seq += 1
        event = SessionEvent(seq=self._seq, type=event_type, data=data)
        self._events.append(event)
        if surface_op == "append":
            self._surface_nodes.add(self._seq)
        return event

    def request_header(self) -> Optional[Dict[str, Any]]:
        return self._request_header

    def set_request_header(self, header: Dict[str, Any], reason: str = "change") -> None:
        self._request_header = header
        self.append(SessionEventType.REQUEST_HEADER.value, {"header": header, "reason": reason})

    def request_context(self) -> Optional[Dict[str, Any]]:
        return self._request_context

    def set_request_context(self, ctx: Dict[str, Any]) -> None:
        self._request_context = ctx
        self.append(SessionEventType.REQUEST_CONTEXT.value, ctx)

    def derive_messages(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for event in self._events:
            etype = event.type
            data = event.data
            if etype in (SessionEventType.USER_MESSAGE.value,
                         SessionEventType.ASSISTANT_MESSAGE.value,
                         SessionEventType.TOOL_RESULT.value):
                msg = data.get("message", data)
                messages.append(self._normalize_message(msg))
        return messages

    @staticmethod
    def _normalize_message(msg: Any) -> Dict[str, Any]:
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "to_openai"):
            return msg.to_openai()
        return {"role": "user", "content": str(msg)}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.id,
            "events": [{"seq": e.seq, "type": e.type, "data": e.data} for e in self._events],
            "request_header": self._request_header,
            "request_context": self._request_context,
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        self.id = snapshot.get("session_id", self.id)
        self._events = [
            SessionEvent(seq=e["seq"], type=e["type"], data=e["data"])
            for e in snapshot.get("events", [])
        ]
        self._seq = max((e.seq for e in self._events), default=0)
        self._request_header = snapshot.get("request_header")
        self._request_context = snapshot.get("request_context")
        self._surface_nodes = set(e.seq for e in self._events)


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"


class AgentLoop:
    def __init__(
        self,
        llm_call_fn: Optional[Callable] = None,
        llm_stream_fn: Optional[Callable] = None,
        tool_registry: Optional[Any] = None,
        system_prompt: str = "",
        max_parallel_tool_calls: int = 10,
        max_tool_calls: int = 10,
        checkpoint_path: Optional[str] = None,
        context_window: int = 131072,
        context_target_ratio: float = 0.70,
        reserved_output_tokens: int = 8192,
        context_safety_margin: int = 1024,
        provider_input_ratio: float = 0.50,
    ) -> None:
        self._llm_call_fn = llm_call_fn
        self._llm_stream_fn = llm_stream_fn
        self._tool_registry = tool_registry
        self._system_prompt_base = system_prompt
        self._max_parallel_tool_calls = max_parallel_tool_calls
        self._max_tool_calls = max_tool_calls
        self._checkpoint_path = checkpoint_path
        # DSH does not go through QueryEngine._assemble_messages(), so it has
        # its own last-mile context guard.  Keep a separate output reserve and
        # safety margin: the model server counts template/tool overhead too.
        self._context_window = max(1, int(context_window))
        self._context_target_ratio = min(0.95, max(0.10, float(context_target_ratio)))
        self._reserved_output_tokens = max(0, int(reserved_output_tokens))
        self._context_safety_margin = max(0, int(context_safety_margin))
        # The provider counts chat-template delimiters, tool schemas and output
        # reservation differently from our generic estimator.  Keep a separate,
        # deliberately conservative input cap so one long exploration cannot
        # ever drive a nominal 128K model past its actual server limit.
        self._provider_input_ratio = min(0.85, max(0.15, float(provider_input_ratio)))
        self._last_context_stats: Dict[str, Any] = {}
        self._tool_call_count = 0
        self._step_count = 0
        self._exploration_mode = False
        self._exploration_min_tool_calls = 8
        # A tool call is not a whole exploration step: a long request commonly
        # needs planning, tool execution, and a synthesis turn.  Leave enough
        # room for continuation turns while retaining a hard safety bound.
        self._exploration_max_steps = max(60, max_tool_calls * 4)
        # 显式探索进度追踪（不依赖模型自我评估）
        self._exploration_state: Optional[Dict[str, Any]] = None

        self._session = Session()
        self._inbox = Inbox()
        self._phase: Phase = Phase(kind=PhaseKind.IDLE, turn=0, step=0)
        # Do not bind an idle future to a loop at construction time.  AgentLoop
        # instances are also created by synchronous tests and may be started
        # later inside a different asyncio.run() loop (Python 3.11 no longer
        # creates a current loop implicitly).
        self._activity_done: Optional[asyncio.Future] = None
        self._status_listeners: List[Callable[[AgentStatus], None]] = []
        self._event_listeners: List[Callable[[str, Dict[str, Any]], None]] = []

    @property
    def status(self) -> AgentStatus:
        if self._phase.kind in (PhaseKind.IDLE, PhaseKind.MAINTENANCE):
            return AgentStatus.IDLE
        return AgentStatus.RUNNING

    @property
    def session(self) -> Session:
        return self._session

    @property
    def phase(self) -> Phase:
        return self._phase

    def on_status_change(self, listener: Callable[[AgentStatus], None]) -> None:
        self._status_listeners.append(listener)

    def on_event(self, listener: Callable[[str, Dict[str, Any]], None]) -> None:
        self._event_listeners.append(listener)

    def _emit_status(self, status: AgentStatus) -> None:
        for listener in self._status_listeners:
            try:
                listener(status)
            except Exception:
                logger.warning("status listener raised", exc_info=True)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        for listener in self._event_listeners:
            try:
                listener(event_type, data)
            except Exception:
                logger.warning("event listener raised", exc_info=True)

    def _set_phase(self, next_phase: Phase) -> None:
        previous_status = self.status
        self._phase = next_phase
        status = self.status
        if status != previous_status:
            self._emit_status(status)

    def followup(self, message: Any) -> None:
        self._tool_call_count = 0
        self._step_count = 0
        text = str(message)
        self._exploration_mode = self._is_broad_exploration_request(text)
        if self._exploration_mode:
            self._exploration_state = {
                "required_dirs": set(),
                "listed_dirs": set(),
                "read_files": {},
                "readme_read": set(),
                "core_files_read": {},
                "files_by_dir": {},
                "task_parsed": False,
            }
        else:
            self._exploration_state = None
        self._inbox.splice(InboxPosition.NEXT_TURN, 0, 0, [
            InboxMessage(content=message, position=InboxPosition.NEXT_TURN)
        ])
        self._wake_driver()

    @staticmethod
    def _is_broad_exploration_request(message: str) -> bool:
        """Return True only for a genuinely broad workspace-inspection request.

        A directory/file word alone must not force the agent through the long
        exploration state machine.  For example, ``列出当前目录前 5 个文件``
        should be a one-tool task and can finalize immediately after
        ``list_files``.  The safety floor remains enabled for requests that
        explicitly ask for comprehensive project/workspace exploration.
        """
        text = (message or "").lower()
        broad_phrases = (
            "explore the workspace", "explore workspace", "explore the project",
            "read every file", "read all files", "review the entire project",
            "full project review", "comprehensive project", "entire codebase",
            "完整全面", "全面看", "完整看", "看一遍", "全部源码", "所有源文件",
            "所有文件", "逐个文件", "逐一", "全量", "整体审查", "全面审查",
        )
        if any(phrase in text for phrase in broad_phrases):
            return True

        scope_words = (
            "workspace", "work space", "project", "codebase", "repository",
            "目录", "工作区", "项目", "源码", "代码库", "仓库",
        )
        coverage_words = (
            "explore", "inspect", "review", "read every", "read all",
            "comprehensive", "entire", "all files", "every file",
            "完整", "全面", "探索", "全部", "所有", "每个", "逐个", "整体",
        )
        return any(word in text for word in scope_words) and any(
            word in text for word in coverage_words
        )

    def steer(self, message: Any) -> None:
        self._inbox.splice(InboxPosition.NEXT_STEP, 0, 0, [
            InboxMessage(content=message, position=InboxPosition.NEXT_STEP)
        ])
        self._wake_driver()

    def inject(self, message: Any) -> None:
        self._inbox.splice(InboxPosition.NEXT_STEP, 0, 0, [
            InboxMessage(content=message, position=InboxPosition.NEXT_STEP)
        ])

    def cancel(self, cause: Optional[BaseException] = None) -> None:
        if self._phase.abort:
            self._phase.abort.abort(cause or AbortError("cancelled"))
        self._inbox.clear()

    def _wake_driver(self, wake_after_abort: bool = False) -> None:
        if self._phase.kind != PhaseKind.IDLE:
            if (self._phase.abort and self._phase.abort.aborted
                    and self._phase.abort.reason is not None
                    and not isinstance(self._phase.abort.reason, AbortError)
                    and wake_after_abort):
                self._phase.wake_requested = True
            return

        loop = asyncio.get_event_loop()
        driver = loop.create_future()
        self._activity_done = driver
        self._set_phase(Phase(
            kind=PhaseKind.RUNNING,
            turn=self._phase.turn,
            step=0,
            wake_requested=False,
            abort=AbortController(),
        ))
        loop.create_task(self._kick(driver))

    async def _kick(self, driver: asyncio.Future) -> None:
        try:
            while await self._turn():
                pass
        except Exception as exc:
            logger.exception("DSH driver failed")
            self._session.append(SessionEventType.TURN_END.value, {
                "turn": self._phase.turn,
                "reason": {"kind": "error", "error": {"message": str(exc), "code": "DRIVER_ERROR"}},
            })
        finally:
            if self._phase.kind == PhaseKind.RUNNING:
                turn = self._phase.turn
                self._set_phase(Phase(
                    kind=PhaseKind.IDLE,
                    turn=turn,
                    step=0,
                ))
                if self._phase.wake_requested and self._inbox.has_pending:
                    self._wake_driver()
            if not driver.done():
                driver.set_result(None)

    async def when_idle(self) -> None:
        while True:
            current = self._activity_done
            if current is None:
                return
            await current
            if current is self._activity_done:
                return

    async def _turn(self) -> bool:
        if self._phase.kind != PhaseKind.RUNNING:
            return False

        phase = self._phase
        signal = phase.abort.signal if phase.abort else None

        if signal:
            signal.throw_if_aborted()

        turn = phase.turn + 1
        self._session.append(SessionEventType.TURN_START.value, {"turn": turn})
        phase.turn = turn

        turn_ends: Optional[Dict[str, Any]] = None
        target = InboxPosition.NEXT_TURN

        try:
            while True:
                if signal:
                    signal.throw_if_aborted()

                step = phase.step + 1
                decision = await self._pre_step(target, {"turn": turn, "step": step})

                if decision.get("kind") == "reject":
                    turn_ends = {"kind": "blocked"}
                    return False

                if turn_ends and len(decision.get("messages", [])) == 0:
                    break

                if phase.step == 0 and len(decision.get("messages", [])) == 0:
                    turn_ends = {"kind": "completed"}
                    return False

                if signal:
                    signal.throw_if_aborted()

                self._session.append(SessionEventType.STEP_START.value, {
                    "turn": turn, "step": step
                })
                phase.step = step
                self._step_count += 1
                if self._step_count > self._exploration_max_steps:
                    turn_ends = {"kind": "max-steps"}
                    break

                try:
                    for msg in decision.get("messages", []):
                        self._session.append(SessionEventType.USER_MESSAGE.value, {
                            "message": msg, "turn": turn, "step": step
                        }, surface_op="append")

                    step_end = await self._step(decision.get("assembly"))
                    if turn_ends is None or turn_ends.get("kind") != "max-tokens":
                        turn_ends = step_end
                finally:
                    self._session.append(SessionEventType.STEP_END.value, {
                        "turn": turn, "step": step
                    })

                if signal:
                    signal.throw_if_aborted()

                if turn_ends and len(self._inbox.next_step) == 0:
                    self._emit_event("agent/turn-stopping", {"turn": turn})
                    if signal:
                        signal.throw_if_aborted()

                logger.debug(f"[Turn] turn={turn}, turn_ends={turn_ends}, inbox_next_step_len={len(self._inbox.next_step)}, inbox_pending={self._inbox.has_pending}")
                if turn_ends and len(self._inbox.next_step) == 0:
                    break

                target = InboxPosition.NEXT_STEP

        except Exception as e:
            if signal and signal.aborted:
                turn_ends = {"kind": "aborted", "reason": signal.reason}
                raise
            turn_ends = {
                "kind": "error",
                "error": {"message": str(e), "code": "UNKNOWN"}
            }
            raise
        finally:
            try:
                self._session.append(SessionEventType.TURN_END.value, {
                    "turn": turn, "reason": turn_ends
                })
            except Exception:
                pass

        if not self._inbox.has_pending:
            return False

        phase.abort = AbortController()
        phase.wake_requested = False
        phase.step = 0
        return True

    async def _pre_step(self, target: InboxPosition,
                        position: Dict[str, Any]) -> Dict[str, Any]:
        if self._phase.kind != PhaseKind.RUNNING:
            raise RuntimeError("pre-step outside running phase")

        signal = self._phase.abort.signal if self._phase.abort else None
        if signal:
            signal.throw_if_aborted()

        claimed = self._inbox.claim(target, position.get("turn", 0))
        system = self._system_prompt_base
        messages = []
        # A continuation is logically an internal control instruction, but it
        # must still occupy a user turn in the OpenAI chat transcript.  Earlier
        # code merged ``[系统指令]`` continuations into the system prompt and
        # sent the preceding assistant message again without an intervening
        # user message.  Strict local chat templates then rejected the payload
        # with "Cannot have 2 or more assistant messages at the end of the
        # list."  Keep one immutable system message at index zero and convert
        # a continuation into a hidden/synthetic user turn for the model.
        for m in claimed:
            content = self._normalize_message(m.content)
            text = content.get("content", "") if isinstance(content, dict) else content
            if isinstance(text, str) and text.startswith("[系统指令]"):
                directive = text.removeprefix("[系统指令]").lstrip()
                messages.append({
                    "role": "user",
                    "content": "[继续执行内部指令]\n" + directive,
                })
            else:
                messages.append(content)

        return {
            "kind": "enter",
            "messages": messages,
            "assembly": {"system": system, "tools": []},
        }

    async def _step(self, assembly: Dict[str, Any]) -> Dict[str, Any]:
        if self._phase.kind != PhaseKind.RUNNING:
            raise RuntimeError("step outside running phase")

        phase = self._phase
        signal = phase.abort.signal if phase.abort else None
        if signal:
            signal.throw_if_aborted()

        turn = phase.turn
        step = phase.step
        system = assembly.get("system", self._system_prompt_base)

        while True:
            messages = self._session.derive_messages()
            schemas = []

            if self._tool_registry:
                try:
                    schemas = self._tool_registry.all_schemas()
                except Exception:
                    schemas = []

            request = await self._build_request(turn, step, system, messages, schemas, signal)

            if signal:
                signal.throw_if_aborted()

            try:
                self._emit_event("llm/start", {
                    "turn": turn, "step": step,
                    "message_count": len(self._session.derive_messages()),
                })
                if self._llm_stream_fn:
                    content, tool_calls, usage = await self._call_llm_stream(request, signal)
                elif self._llm_call_fn:
                    content, tool_calls, usage = await self._call_llm(request, signal)
                else:
                    return {"kind": "completed"}
            except Exception as e:
                if signal and signal.aborted:
                    return {"kind": "aborted", "reason": signal.reason}
                return {"kind": "error", "error": {"message": str(e), "code": "LLM_ERROR"}}

            if signal:
                signal.throw_if_aborted()

            self._emit_event("llm/end", {
                "turn": turn, "step": step,
                "content": content,
                "tool_calls": tool_calls,
                "usage": usage,
            })

            # Text accompanying a tool call, or an intermediate exploration
            # update, is internal progress rather than the formal answer.
            # Exploration completion is model-driven with a small safety floor:
            # do not accept a premature summary after only one or two reads, but
            # also do not require every top-level folder to contain a made-up
            # README/core.py pair.
            is_final = bool(content and not tool_calls and (
                not self._exploration_mode
                or self._exploration_can_finalize(content)
            ))

            assistant_msg = {
                "role": "assistant",
                "content": content,
                "visibility": "final" if is_final else "internal",
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": tc.get("arguments", "{}"),
                        }
                    }
                    for i, tc in enumerate(tool_calls)
                ]

            self._session.append(SessionEventType.ASSISTANT_MESSAGE.value, {
                "turn": turn, "step": step,
                "message": assistant_msg,
                "usage": usage,
            }, surface_op="append")
            self._emit_event("assistant/message", {
                "turn": turn, "step": step,
                "message": assistant_msg,
                "usage": usage,
                "visibility": assistant_msg["visibility"],
            })

            if not content and not tool_calls:
                if (self._exploration_mode and self._exploration_state is not None
                        and not self._exploration_can_finalize(content)):
                    self._inbox.splice(InboxPosition.NEXT_STEP, 0, 0, [
                        InboxMessage(content=self._get_continuation_prompt(),
                                     position=InboxPosition.NEXT_STEP)
                    ])
                    return None
                return {"kind": "completed"}

            if tool_calls:
                self._tool_call_count += len(tool_calls)
                if self._tool_call_count > self._max_tool_calls:
                    return {"kind": "max-tool-calls"}
                concluded = await self._execute_tool_calls(turn, step, tool_calls, signal)
                if concluded:
                    return {"kind": "completed"}
                return None
            else:
                # 基于实际探索进度判断完成，不依赖模型文本
                if self._exploration_mode and self._exploration_state is not None:
                    if not self._exploration_can_finalize(content):
                        # 强制要求模型继续使用工具；中间规划文字只保留在
                        # DSH 事件日志中，不会出现在最终助手消息里。
                        prompt = self._get_continuation_prompt()
                        if not tool_calls and "TOOL_CALL:" not in content:
                            prompt = ("[系统指令] 检测到你未调用工具。你的下一个回复必须是工具调用，禁止输出任何文本。\n" + prompt)
                        continue_msg = InboxMessage(
                            content=prompt,
                            position=InboxPosition.NEXT_STEP,
                        )
                        self._inbox.splice(InboxPosition.NEXT_STEP, 0, 0, [continue_msg])
                        return None
                return {"kind": "completed"}

    async def _build_request(self, turn: int, step: int, system: str,
                              messages: List[Dict[str, Any]],
                              schemas: List[Dict[str, Any]],
                              signal: Optional[AbortSignal]) -> Dict[str, Any]:
        config = {
            "provider": "",
            "model": "",
            "system": system,
            "tools": schemas,
            "messages": messages,
            "turn": turn,
            "step": step,
        }
        if signal:
            config["signal"] = signal
        return config

    @staticmethod
    def _request_messages(request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build the chat-template-safe message list for one LLM request.

        The DSH request keeps ``system`` separate for state-machine assembly,
        but the OpenAI-compatible adapters only receive ``messages``. Merge
        the system prompt here, exactly once, and keep it at index zero.
        """
        raw_messages = list(request.get("messages", []) or [])
        messages = [AgentLoop._normalize_message(message) for message in raw_messages]
        system = str(request.get("system", "") or "").strip()
        if not system:
            return messages

        existing_system_parts: List[str] = []
        non_system: List[Dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                if content:
                    existing_system_parts.append(str(content))
            else:
                non_system.append(message)

        combined_system = system
        for part in existing_system_parts:
            if part != system and part not in combined_system:
                combined_system = f"{combined_system}\n\n{part}"
        return [{"role": "system", "content": combined_system}] + non_system

    @staticmethod
    def _estimate_payload_tokens(value: Any) -> int:
        """Conservative token estimate for an OpenAI-style payload fragment."""
        try:
            raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            raw = str(value)
        if not raw:
            return 1
        # The local model tokenizer is not guaranteed to be cl100k_base.
        # Treat CJK characters as roughly one token each: this intentionally
        # overestimates common local-model tokenizers and leaves room for chat
        # template delimiters.  Do not spell the Unicode bounds as ``"\\u..."``:
        # that is a six-character literal and silently classifies all Chinese
        # text as ASCII, which was the direct cause of oversized requests.
        cjk = sum(
            1 for ch in raw
            if ("\u3400" <= ch <= "\u9fff"
                or "\uf900" <= ch <= "\ufaff"
                or "\u3000" <= ch <= "\u303f"
                or "\u3040" <= ch <= "\u30ff"
                or "\uac00" <= ch <= "\ud7af")
        )
        non_cjk_chars = max(0, len(raw) - cjk)
        # Code, JSON, paths and tool schemas tokenize much more densely than
        # ordinary prose on local models.  Two non-CJK characters per token is
        # intentionally conservative and avoids the previous 3-char estimate
        # undercounting large workspace reads by tens of thousands of tokens.
        return max(1, int((non_cjk_chars / 2.0) + cjk + 0.999))

    @classmethod
    def _message_tokens(cls, message: Dict[str, Any]) -> int:
        return cls._estimate_payload_tokens(message) + 4

    @classmethod
    def _messages_payload_tokens(cls, messages: List[Dict[str, Any]]) -> int:
        """Estimate the complete ``messages`` payload rather than each item alone.

        Serialising messages one at a time misses the JSON list delimiters and
        separators that are present in the actual OpenAI-compatible request.
        At a small budget that off-by-a-few error is enough to cross a strict
        provider limit, so the final admission guard must use this whole-payload
        estimate.  An empty list contributes no chat-message payload.
        """
        if not messages:
            return 0
        return cls._estimate_payload_tokens(messages) + (4 * len(messages))

    @classmethod
    def _minimal_message(cls, message: Dict[str, Any]) -> Dict[str, Any]:
        """Return the smallest valid OpenAI-style representation of a message.

        Context fitting must have a real lower bound.  Keeping a 32/64 token
        floor is not safe for tiny test windows and can still make the provider
        reject the whole request.  This helper preserves tool-call envelopes so
        the chat template remains valid while dropping optional prose first.
        """
        role = str(message.get("role") or "user")
        minimal: Dict[str, Any] = {"role": role}
        if role == "tool":
            minimal["tool_call_id"] = str(message.get("tool_call_id") or "call")
            minimal["content"] = ""
        elif role == "assistant" and message.get("tool_calls"):
            calls: List[Dict[str, Any]] = []
            for index, call in enumerate(message.get("tool_calls") or []):
                function = call.get("function") or {}
                calls.append({
                    "id": str(call.get("id") or f"call-{index}"),
                    "type": "function",
                    "function": {
                        "name": str(function.get("name") or "tool"),
                        "arguments": "{}",
                    },
                })
            minimal["tool_calls"] = calls
            minimal["content"] = ""
        else:
            minimal["content"] = ""
        return minimal

    @classmethod
    def _truncate_message(cls, message: Dict[str, Any], token_budget: int) -> Dict[str, Any]:
        """Fit one message to an exact estimated-token ceiling.

        The previous implementation used ``max(64, ...)`` characters, which
        made the advertised hard budget impossible to satisfy for small
        windows.  Binary-searching the content length keeps the structured
        fields intact and guarantees the returned message is no larger than
        ``token_budget`` whenever the minimal envelope itself fits.
        """
        budget = max(1, int(token_budget))
        original = dict(message)
        content = str(original.get("content") or "")
        minimal = cls._minimal_message(original)
        if cls._message_tokens(minimal) > budget:
            return minimal
        if not content or cls._message_tokens(original) <= budget:
            return original

        marker = "\n...[上下文已裁剪]...\n"
        low, high = 0, len(content)
        best = minimal
        while low <= high:
            length = (low + high) // 2
            if length == 0:
                candidate_content = ""
            elif length >= len(content):
                candidate_content = content
            else:
                head = max(1, int(length * 0.67))
                tail = max(0, length - head)
                if tail:
                    candidate_content = content[:head] + marker + content[-tail:]
                else:
                    candidate_content = content[:head]
            candidate = dict(original)
            candidate["content"] = candidate_content
            if cls._message_tokens(candidate) <= budget:
                best = candidate
                low = length + 1
            else:
                high = length - 1
        return best

    @classmethod
    def _truncate_message_for_payload(
        cls,
        messages: List[Dict[str, Any]],
        index: int,
        token_budget: int,
    ) -> Dict[str, Any]:
        """Shrink one message while measuring the complete messages payload.

        Per-message budgets are only an approximation because the provider sees
        one JSON array.  This binary search is used by the final admission guard
        so the newest user message is retained whenever its structured envelope
        can fit alongside the remaining messages.
        """
        original = dict(messages[index])
        content = str(original.get("content") or "")
        minimal = cls._minimal_message(original)

        def fits(candidate: Dict[str, Any]) -> bool:
            probe = list(messages)
            probe[index] = candidate
            return cls._messages_payload_tokens(probe) <= token_budget

        if fits(original):
            return original
        if not content:
            return minimal
        if not fits(minimal):
            return minimal

        low, high = 0, len(content)
        best = minimal
        marker = "\n...[context trimmed]...\n"
        while low <= high:
            length = (low + high) // 2
            if length == 0:
                candidate_content = ""
            elif length >= len(content):
                candidate_content = content
            else:
                head = max(1, int(length * 0.67))
                tail = max(0, length - head)
                candidate_content = content[:head] + (marker + content[-tail:] if tail else "")
            candidate = dict(original)
            candidate["content"] = candidate_content
            if fits(candidate):
                best = candidate
                low = length + 1
            else:
                high = length - 1
        return best

    @classmethod
    def _shrink_group(cls, group: List[Dict[str, Any]], token_budget: int) -> List[Dict[str, Any]]:
        """Fit a tool-call group without breaking its assistant/tool pairing."""
        if token_budget <= 0:
            return []
        shrunk = [dict(message) for message in group]
        if sum(cls._message_tokens(m) for m in shrunk) <= token_budget:
            return shrunk

        # First shrink prose in every member.  Tool-call arguments are reduced
        # by _minimal_message only if the structural envelope is the limiting
        # factor; normal tool calls retain their arguments whenever possible.
        for index, message in enumerate(shrunk):
            remaining = max(1, token_budget - sum(
                cls._message_tokens(m) for j, m in enumerate(shrunk) if j != index
            ))
            shrunk[index] = cls._truncate_message(message, remaining)

        while sum(cls._message_tokens(m) for m in shrunk) > token_budget:
            candidates = [
                (len(str(m.get("content") or "")), i)
                for i, m in enumerate(shrunk)
                if str(m.get("content") or "")
            ]
            if not candidates:
                break
            _, index = max(candidates)
            other_tokens = sum(
                cls._message_tokens(m) for i, m in enumerate(shrunk) if i != index
            )
            next_message = cls._truncate_message(shrunk[index], token_budget - other_tokens)
            if cls._message_tokens(next_message) >= cls._message_tokens(shrunk[index]):
                break
            shrunk[index] = next_message
        return shrunk

    def _prepare_request_messages(
        self,
        request: Dict[str, Any],
        *,
        target_ratio: Optional[float] = None,
        fit_reason: str = "preflight",
    ) -> List[Dict[str, Any]]:
        """Merge system + fit DSH history before *every* LLM call.

        This is intentionally the final boundary before the provider adapter,
        including the streaming path.  It prevents a long DSH exploration from
        bypassing QueryEngine's normal context assembler.
        """
        messages = self._request_messages(request)
        total_budget = max(
            1,
            self._context_window - self._reserved_output_tokens - self._context_safety_margin,
        )
        requested_ratio = self._context_target_ratio if target_ratio is None else target_ratio
        requested_ratio = min(0.95, max(0.10, float(requested_ratio)))
        # A context window is not a safe input window: providers also apply the
        # chat template, tools and max_tokens reservation.  Cap the assembled
        # payload before dispatch, independently of historical compaction.
        provider_cap = max(1, int(self._context_window * self._provider_input_ratio))
        target_budget = min(
            total_budget,
            provider_cap,
            max(1, int(self._context_window * requested_ratio)),
        )
        tool_tokens = self._estimate_payload_tokens(request.get("tools", []))
        message_budget = max(1, target_budget - tool_tokens)

        if not messages:
            self._last_context_stats = {
                "estimated_tokens": tool_tokens,
                "budget": target_budget,
                "dropped_messages": 0,
                "fitted": False,
                "reason": fit_reason,
            }
            return messages

        system = messages[0] if messages[0].get("role") == "system" else None
        history = messages[1:] if system else messages
        system_tokens = self._message_tokens(system) if system else 0
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        pending_tool_ids: set[str] = set()
        for message in history:
            role = message.get("role")
            if role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                if current and current[0].get("role") == "assistant" and call_id in pending_tool_ids:
                    current.append(message)
                    pending_tool_ids.discard(call_id)
                else:
                    # Never send an orphan tool result to strict templates.
                    continue
            else:
                if current:
                    groups.append(current)
                current = [message]
                pending_tool_ids = {
                    str(tc.get("id") or "")
                    for tc in (message.get("tool_calls") or [])
                    if tc.get("id")
                } if role == "assistant" else set()
        if current:
            groups.append(current)
        if pending_tool_ids and groups and groups[-1] and groups[-1][0].get("role") == "assistant":
            # An incomplete tool-call envelope cannot be rendered safely.
            groups.pop()

        # Select the newest complete group first.  This is important for long
        # explorations: the current user request/tool result must survive even
        # when the system prompt is large.  Older groups are only added when
        # their complete assistant/tool envelope fits.
        dropped = 0
        latest_group = groups[-1] if groups else []
        system_min = self._minimal_message(system) if system else None
        system_min_tokens = self._message_tokens(system_min) if system_min else 0
        latest_budget = max(0, message_budget - system_min_tokens)
        latest = self._shrink_group(latest_group, latest_budget) if latest_group else []

        # If the system envelope plus the newest group cannot coexist under a
        # very small budget, prefer the newest user/tool context and omit the
        # system message rather than sending an over-limit request.
        if latest_group and (
            not latest or sum(self._message_tokens(m) for m in latest) > latest_budget
        ):
            latest = self._shrink_group(latest_group, message_budget)
            system = None
            system_min = None
            system_min_tokens = 0
        if latest_group and not latest:
            # A structurally oversized group cannot be made valid within this
            # window.  Keep a final user message when present; otherwise drop
            # the incomplete/oversized envelope instead of violating the cap.
            user_messages = [m for m in latest_group if m.get("role") == "user"]
            if user_messages:
                latest = [self._truncate_message(user_messages[-1], message_budget)]
            else:
                latest = []

        used = sum(self._message_tokens(m) for m in latest)
        if system:
            remaining_for_system = max(0, message_budget - used)
            fitted_system = self._truncate_message(system, remaining_for_system)
            if self._message_tokens(fitted_system) <= remaining_for_system:
                system = fitted_system
                used += self._message_tokens(system)
            else:
                system = None
        result_groups: List[List[Dict[str, Any]]] = []
        if latest:
            result_groups.append(latest)
        # Add older groups in reverse chronological order, then restore order.
        for group in reversed(groups[:-1] if groups else []):
            group_tokens = sum(self._message_tokens(m) for m in group)
            if used + group_tokens <= message_budget:
                result_groups.append(group)
                used += group_tokens
            else:
                dropped += len(group)
                # Once an older group does not fit, still try no further old
                # groups: they cannot be more useful than the newest history.
                break

        kept: List[Dict[str, Any]] = []
        for group in reversed(result_groups):
            kept.extend(group)
        result = ([system] if system else []) + kept

        # Strict chat templates require at least one real ``user`` turn.  The
        # normal newest-group-first policy can otherwise retain only the latest
        # assistant/tool envelope during a long tool run, dropping the original
        # user task before the provider sees the request.  That exact shape is
        # rejected by llama.cpp/Qwen templates as ``No user query found in
        # messages``.  Reinsert the newest available user turn before the
        # retained assistant/tool context; it is a tiny but non-negotiable
        # structural anchor for the request.
        if not any(message.get("role") == "user" for message in result):
            latest_user = next(
                (
                    dict(message)
                    for message in reversed(messages)
                    if message.get("role") == "user"
                    and str(message.get("content") or "").strip()
                ),
                None,
            )
            if latest_user is None:
                # This should only be reachable for a malformed restored
                # checkpoint.  Never send a template-invalid empty transcript;
                # keep the recovery instruction explicit and observable.
                latest_user = {
                    "role": "user",
                    "content": "请继续处理当前任务，并基于已有上下文完成下一步。",
                }
                logger.warning("DSH request had no user event; inserted recovery user turn")
            insert_at = 1 if result and result[0].get("role") == "system" else 0
            result.insert(insert_at, latest_user)
            logger.warning("DSH context fitting restored newest user turn for chat-template validity")

        # Last-mile hard guard.  It is deliberately independent of the normal
        # group selection so a future message-shape change cannot reintroduce
        # the 147k > 131k provider failure.  Shrink content, then remove the
        # oldest non-system group if a structural envelope is still too large.
        while self._messages_payload_tokens(result) > message_budget:
            # Preserve a user anchor whenever possible.  A request containing
            # only system/assistant/tool messages is syntactically invalid for
            # the local chat template, so trim execution history before the
            # latest user instruction.
            candidates = [
                (len(str(m.get("content") or "")), index)
                for index, m in enumerate(result)
                if m.get("role") not in {"system", "user"}
                and str(m.get("content") or "")
            ]
            if not candidates:
                candidates = [
                    (len(str(m.get("content") or "")), index)
                    for index, m in enumerate(result)
                    if m.get("role") != "system" and str(m.get("content") or "")
                ]
            if candidates:
                _, index = max(candidates)
                result[index] = self._truncate_message_for_payload(
                    result, index, message_budget
                )
                if self._messages_payload_tokens(result) <= message_budget:
                    continue
            # Remove one oldest complete non-system group.  Never leave a lone
            # tool result or assistant tool-call envelope in the payload.
            non_system = [
                i for i, m in enumerate(result)
                if m.get("role") not in {"system", "user"}
            ]
            if not non_system:
                # At this point only system + user remain.  The user cannot be
                # removed without recreating the template error; shrink it in
                # place and stop rather than dispatching an invalid transcript.
                user_indices = [i for i, m in enumerate(result) if m.get("role") == "user"]
                if user_indices:
                    result[user_indices[-1]] = self._truncate_message_for_payload(
                        result, user_indices[-1], message_budget
                    )
                if result and result[0].get("role") == "system":
                    result[0] = self._truncate_message(result[0], message_budget)
                break
            remove_index = non_system[0]
            remove_role = result[remove_index].get("role")
            if remove_role == "assistant" and result[remove_index].get("tool_calls"):
                ids = {str(tc.get("id") or "") for tc in result[remove_index].get("tool_calls") or []}
                end_index = remove_index
                while end_index + 1 < len(result) and result[end_index + 1].get("role") == "tool" and str(result[end_index + 1].get("tool_call_id") or "") in ids:
                    end_index += 1
                del result[remove_index:end_index + 1]
                dropped += end_index - remove_index + 1
            else:
                del result[remove_index]
                dropped += 1

        # Defensive final assertion in production code: if even the minimum
        # structured message cannot fit, send an empty message list rather than
        # constructing an invalid over-limit request.
        if self._messages_payload_tokens(result) > message_budget:
            result = []
            system = None
            dropped += len(messages)
        estimated = tool_tokens + self._messages_payload_tokens(result)
        # A newest message may be shortened in place without changing the
        # number of messages.  Treat that as a fit too, otherwise the UI and
        # telemetry falsely claim that automatic compaction never happened.
        fitted = bool(dropped or result != messages)
        self._last_context_stats = {
            "estimated_tokens": estimated,
            "message_tokens": estimated - tool_tokens,
            "budget": target_budget,
            "hard_input_budget": total_budget,
            "provider_input_budget": provider_cap,
            "dropped_messages": dropped,
            "fitted": fitted,
            "reason": fit_reason,
        }
        if fitted:
            logger.warning("DSH context fitted before LLM call: %s", self._last_context_stats)
        return result

    def _emit_context_fit_event(self) -> None:
        """Expose last-mile fitting to the UI without leaking prompt content."""
        stats = dict(self._last_context_stats)
        if not stats.get("fitted"):
            return
        self._emit_event("context/compacted", {
            "dropped_messages": int(stats.get("dropped_messages", 0)),
            "estimated_tokens": int(stats.get("estimated_tokens", 0)),
            "budget": int(stats.get("budget", 0)),
            "reason": str(stats.get("reason", "preflight")),
        })

    def _dynamic_output_tokens(
        self, messages: List[Dict[str, Any]], tools: Any
    ) -> int:
        """Choose an output budget that fits the *actual* fitted request.

        ``max_tokens`` is part of the provider's context reservation even when
        the prompt itself has already been compacted.  Passing a fixed 4096
        therefore leaves the adapter free to over-reserve on a nearly-full
        request.  Keep the configured reserve as an upper bound, but shrink it
        to the remaining capacity after messages, tool schemas and the safety
        margin are accounted for.
        """
        input_tokens = (
            self._messages_payload_tokens(messages)
            + self._estimate_payload_tokens(tools or [])
        )
        available = self._context_window - input_tokens - self._context_safety_margin
        if available <= 0:
            return 1
        configured = self._reserved_output_tokens
        if configured <= 0:
            # A zero reserve means “no preferred output budget”, not an invalid
            # provider value.  Use a small bounded default and never exceed the
            # remaining context capacity.
            configured = min(1024, max(1, self._context_window // 16))
        return max(1, min(configured, available))

    @staticmethod
    def _invoke_callback(
        fn: Callable,
        messages: List[Dict[str, Any]],
        tools: Any,
        max_tokens: int,
    ) -> Any:
        """Call old two-argument and new dynamic-budget callbacks safely."""
        try:
            signature = inspect.signature(fn)
            params = list(signature.parameters.values())
            by_name = signature.parameters.get("max_tokens")
            accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
            positional = [
                p for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            # Unknown/builtin callables are assumed to support the modern form.
            return fn(messages, tools, max_tokens=max_tokens)

        if accepts_kwargs or (by_name is not None and by_name.kind != by_name.POSITIONAL_ONLY):
            return fn(messages, tools, max_tokens=max_tokens)
        if any(p.kind == p.VAR_POSITIONAL for p in params) or len(positional) >= 3:
            return fn(messages, tools, max_tokens)
        if len(positional) >= 2:
            return fn(messages, tools)
        return fn()

    @staticmethod
    def _is_context_overflow_error(exc: BaseException) -> bool:
        """Return true only for deterministic provider context-limit rejections."""
        detail = str(exc).lower()
        markers = (
            "exceeds the available context size",
            "exceeds context window",
            "context length exceeded",
            "maximum context length",
            "prompt is too long",
            "too many tokens",
            "context window exceeded",
        )
        return any(marker in detail for marker in markers)

    def _overflow_retry_ratio(self) -> float:
        """Use one much smaller retry budget after an upstream context rejection."""
        return min(self._context_target_ratio, self._provider_input_ratio, 0.32)

    def _prepare_overflow_retry(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aggressively compact once after a provider rejects an estimated fit."""
        fitted = self._prepare_request_messages(
            request,
            target_ratio=self._overflow_retry_ratio(),
            fit_reason="provider_context_limit_retry",
        )
        self._emit_context_fit_event()
        logger.warning(
            "DSH provider rejected context; retrying once with budget=%s",
            self._last_context_stats.get("budget"),
        )
        return fitted

    async def _call_llm(self, request: Dict[str, Any],
                         signal: Optional[AbortSignal]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        if signal:
            signal.throw_if_aborted()

        fitted_messages = self._prepare_request_messages(request)
        self._emit_context_fit_event()
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._invoke_callback(
                    self._llm_call_fn,
                    fitted_messages,
                    request.get("tools", []),
                    self._dynamic_output_tokens(fitted_messages, request.get("tools", [])),
                )
            )
        except Exception as exc:
            if signal and signal.aborted:
                raise
            if not self._is_context_overflow_error(exc):
                raise
            retry_messages = self._prepare_overflow_retry(request)
            response = await loop.run_in_executor(
                None,
                lambda: self._invoke_callback(
                    self._llm_call_fn,
                    retry_messages,
                    request.get("tools", []),
                    self._dynamic_output_tokens(retry_messages, request.get("tools", [])),
                ),
            )

        content = ""
        tool_calls = []
        usage = {}

        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            msg = getattr(choice, "message", None)
            if msg:
                content = getattr(msg, "content", "") or ""
                raw_calls = getattr(msg, "tool_calls", []) or []
                for tc in raw_calls:
                    func = getattr(tc, "function", None)
                    tool_calls.append({
                        "id": getattr(tc, "id", ""),
                        "name": getattr(func, "name", "") if func else "",
                        "arguments": getattr(func, "arguments", "{}") if func else "{}",
                    })

        if hasattr(response, "usage"):
            u = response.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0),
            }

        return content, tool_calls, usage

    async def _call_llm_stream(self, request: Dict[str, Any],
                                signal: Optional[AbortSignal]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        if signal:
            signal.throw_if_aborted()

        content = ""
        tool_calls = []
        usage = {}

        if self._llm_stream_fn:
            # The stream callback receives the same fitted payload and dynamic
            # output budget as the non-streaming callback.  This keeps browser
            # streaming and direct A2A calls under one context-limit policy.
            stream_fn = self._llm_stream_fn

            async def consume(stream: Any) -> None:
                nonlocal content, tool_calls, usage
                async for chunk in stream:
                    if signal:
                        signal.throw_if_aborted()
                    if isinstance(chunk, dict):
                        if chunk.get("type") == "delta":
                            content += chunk.get("content", "")
                        elif chunk.get("type") == "final":
                            content = chunk.get("content", content)
                            tool_calls = chunk.get("tool_calls", [])
                            usage = chunk.get("usage", {})

            fitted_messages = self._prepare_request_messages(request)
            self._emit_context_fit_event()
            max_tokens = self._dynamic_output_tokens(fitted_messages, request.get("tools", []))
            try:
                await consume(self._invoke_callback(
                    stream_fn, fitted_messages, request.get("tools", []), max_tokens
                ))
            except Exception as exc:
                if signal and signal.aborted:
                    raise
                if not self._is_context_overflow_error(exc):
                    raise
                # The browser only receives final/progress frames from DSH,
                # never raw model deltas, so retrying this rejected request
                # cannot duplicate a visible partial answer.
                content, tool_calls, usage = "", [], {}
                retry_messages = self._prepare_overflow_retry(request)
                await consume(self._invoke_callback(
                    stream_fn, retry_messages, request.get("tools", []),
                    self._dynamic_output_tokens(retry_messages, request.get("tools", [])),
                ))

        return content, tool_calls, usage

    async def _execute_tool_calls(self, turn: int, step: int,
                                   tool_calls: List[Dict[str, Any]],
                                   signal: Optional[AbortSignal]) -> bool:
        if not self._tool_registry:
            return True

        concluded = False
        for tc in tool_calls:
            if signal and signal.aborted:
                self._session.append(SessionEventType.TOOL_CALL.value, {
                    "turn": turn, "step": step,
                    "callId": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", ""),
                })
                self._session.append(SessionEventType.TOOL_RESULT.value, {
                    "turn": turn, "step": step,
                    "message": {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": "Error: tool call aborted before dispatch",
                    },
                    "error": {"message": "tool call aborted before dispatch"},
                })
                continue

            self._session.append(SessionEventType.TOOL_CALL.value, {
                "turn": turn, "step": step,
                "callId": tc.get("id", ""),
                "name": tc.get("name", ""),
                "arguments": tc.get("arguments", ""),
            })
            self._emit_event("tool/call", {
                "turn": turn, "step": step,
                "callId": tc.get("id", ""),
                "name": tc.get("name", ""),
                "arguments": tc.get("arguments", ""),
            })

            try:
                result = await self._execute_tool(tc)
                is_error = isinstance(result, str) and result.startswith("错误:")

                self._session.append(SessionEventType.TOOL_RESULT.value, {
                    "turn": turn, "step": step,
                    "message": {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    },
                    "is_error": is_error,
                })
                self._emit_event("tool/result", {
                    "turn": turn, "step": step,
                    "callId": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "content": result,
                    "is_error": is_error,
                })

                # 追踪探索进度
                if not is_error and self._exploration_state is not None:
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            import json
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    self._update_exploration_state(tc, args, result)

                if is_error:
                    pass
                else:
                    if tc.get("name") in {"run_tests"}:
                        concluded = True

            except Exception as e:
                error_content = f"工具执行失败 [{tc.get('name', '')}]:{type(e).__name__}: {e}"
                self._session.append(SessionEventType.TOOL_RESULT.value, {
                    "turn": turn, "step": step,
                    "message": {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": error_content,
                    },
                    "is_error": True,
                })
                self._emit_event("tool/result", {
                    "turn": turn, "step": step,
                    "callId": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "content": error_content,
                    "is_error": True,
                })

        return concluded

    def _update_exploration_state(self, tc: Dict[str, Any], arguments: Dict[str, Any], result: str) -> None:
        """更新探索进度；只记录工具真实返回的目录/文件，不臆造路径。"""
        if self._exploration_state is None:
            return
        name = tc.get("name", "")
        if name == "list_files":
            import re
            requested = str(arguments.get("path", ".") if isinstance(arguments, dict) else ".")
            requested = requested.rstrip("/") or "."
            dirs = re.findall(r"^\[DIR\]\s+(.+)$", result, flags=re.MULTILINE)
            files = re.findall(r"^\[FILE\]\s+(.+)$", result, flags=re.MULTILINE)
            for d in dirs:
                d = d.strip()
                full = d if requested in ("", ".") else f"{requested}/{d}"
                self._exploration_state["listed_dirs"].add(full)
            self._exploration_state["files_by_dir"][requested] = set(f.strip() for f in files if f.strip())
            if dirs or files:
                self._exploration_state["task_parsed"] = True
                # The root listing is enough to establish the exploration
                # scope.  We do not demand a README in every directory.
                if requested in ("", ".") and dirs:
                    self._exploration_state["required_dirs"] = {
                        d.strip() for d in dirs
                        if d.strip() and not d.strip().startswith((".", "__"))
                    }
        elif name == "read_file":
            path = str(arguments.get("path", "") if isinstance(arguments, dict) else "").replace("\\", "/")
            if not path:
                return
            parts = [p for p in path.split("/") if p]
            parent = "/".join(parts[:-1]) or "."
            filename = parts[-1]
            self._exploration_state["read_files"].setdefault(parent, set()).add(filename)
            if filename.lower() == "readme.md":
                self._exploration_state["readme_read"].add(parent)
            if filename.lower().endswith((".py", ".js", ".ts", ".tsx", ".jsx")) and not filename.lower().startswith("test"):
                self._exploration_state["core_files_read"][parent] = self._exploration_state["core_files_read"].get(parent, 0) + 1

    @staticmethod
    def _has_explicit_final_marker(content: str) -> bool:
        lowered = (content or "").lower()
        return any(marker in lowered for marker in (
            "任务完成", "探索完成", "task complete", "exploration complete",
            "final architecture summary", "final answer", "总结如下", "综上",
        ))

    @staticmethod
    def _looks_like_continuation(content: str) -> bool:
        lowered = (content or "").lower()
        return any(marker in lowered for marker in (
            "let me continue", "i'll continue", "继续探索", "再看一下", "让我继续",
            "接下来", "now let me", "let me read", "i will read", "我再看",
        ))

    def _exploration_can_finalize(self, content: str) -> bool:
        """Return whether prose is a real answer rather than a progress note.

        DSH does not infer completion from a fixed directory checklist.  A
        checklist breaks on real workspaces (many folders have no README and
        generated/hidden folders should not be opened).  We instead require a
        small number of actual tool calls, reject obvious continuation prose,
        and honor an explicit final marker immediately.
        """
        if self._exploration_state is None:
            return True
        if not (content or "").strip():
            return False
        if self._has_explicit_final_marker(content):
            return self._tool_call_count > 0
        if self._looks_like_continuation(content):
            return False
        return self._tool_call_count >= self._exploration_min_tool_calls

    def _is_exploration_complete(self) -> bool:
        """Compatibility helper used by callers/tests."""
        if self._exploration_state is None:
            return True
        return self._tool_call_count >= self._exploration_min_tool_calls

    def _get_continuation_prompt(self) -> str:
        """Generate a safe, model-directed continuation instruction."""
        if self._exploration_state is None:
            return "继续探索。"
        state = self._exploration_state
        # Prefer a real, unvisited file from a listing instead of guessing
        # paths such as ``<dir>/engine.py`` that may not exist.
        for directory, files in sorted(state.get("files_by_dir", {}).items()):
            read = state.get("read_files", {}).get(directory, set())
            for filename in sorted(files):
                if filename not in read and not filename.lower().startswith(("test", ".")):
                    path = filename if directory in ("", ".") else f"{directory}/{filename}"
                    return ("[系统指令] 探索未完成。下一个动作必须是工具调用，禁止输出文本。\n"
                            f"TOOL_CALL: read_file(path='{path}')")
        return ("[系统指令] 探索未完成。请继续系统地浏览尚未检查的目录，"
                "优先调用 list_files，再读取关键 README 和源文件；禁止输出阶段性总结。")

    async def _execute_tool(self, tc: Dict[str, Any]) -> str:
        if not self._tool_registry:
            return "错误: 工具注册表未初始化"

        name = tc.get("name", "")
        arguments = tc.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}

        handler = None
        try:
            handler = self._tool_registry.get_handler(name)
        except Exception:
            pass

        if handler is None:
            return f"错误: 未知工具:{name}"

        try:
            result = handler(arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result) if result is not None else ""
        except Exception as e:
            return f"工具执行失败 [{name}]:{type(e).__name__}: {e}"

    def save_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not self._checkpoint_path:
            return None
        try:
            os.makedirs(self._checkpoint_path, exist_ok=True)
            checkpoint = {
                "session": self._session.snapshot(),
                "phase": {
                    "kind": self._phase.kind.value,
                    "turn": self._phase.turn,
                    "step": self._phase.step,
                },
                "timestamp": time.time(),
            }
            checkpoint_file = os.path.join(
                self._checkpoint_path,
                f"agent_{self._session.id}_{int(time.time())}.json"
            )
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            return checkpoint
        except Exception as e:
            logger.warning("checkpoint save failed: %s", e)
            return None

    def restore_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        try:
            self._session.restore(checkpoint["session"])
            phase_data = checkpoint.get("phase", {})
            self._phase = Phase(
                kind=PhaseKind(phase_data.get("kind", PhaseKind.IDLE.value)),
                turn=phase_data.get("turn", 0),
                step=phase_data.get("step", 0),
            )
            return True
        except Exception as e:
            logger.warning("checkpoint restore failed: %s", e)
            return False

    async def run_maintenance(self, job: Callable[[AbortSignal], Awaitable[Any]]) -> Any:
        if self._phase.kind != PhaseKind.IDLE:
            raise RuntimeError("agent already has active work")

        maintenance = Phase(
            kind=PhaseKind.MAINTENANCE,
            abort=AbortController(),
            turn=self._phase.turn,
            step=0,
        )
        self._set_phase(maintenance)
        signal = maintenance.abort.signal

        try:
            result = await job(signal)
            return result
        finally:
            self._set_phase(Phase(
                kind=PhaseKind.IDLE,
                turn=maintenance.turn,
                step=0,
            ))
            if maintenance.wake_requested and self._inbox.has_pending:
                self._wake_driver()

    def run(self, user_input: str) -> str:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.arun(user_input))
        finally:
            loop.close()

    async def arun(self, user_input: str) -> str:
        self.followup(user_input)
        await self.when_idle()

        for event in reversed(self._session.events):
            if event.type == SessionEventType.ASSISTANT_MESSAGE.value:
                msg = event.data.get("message", {})
                content = msg.get("content", "")
                if content and not msg.get("tool_calls"):
                    return content

        return "任务已完成，但未生成回复。"

    def run_stream(self, user_input: str) -> Generator[str, None, None]:
        loop = asyncio.new_event_loop()
        try:
            async def _collect():
                self.followup(user_input)
                results = []
                async for chunk in self.arun_stream():
                    results.append(chunk)
                return results

            chunks = loop.run_until_complete(_collect())
            for chunk in chunks:
                yield chunk
        finally:
            loop.close()

    async def arun_stream(self, user_input: str) -> AsyncIterator[str]:
        self.followup(user_input)
        async for chunk in self._stream_events():
            yield chunk

    async def _stream_events(self) -> AsyncIterator[str]:
        await self.when_idle()
        for event in self._session.events:
            if event.type == SessionEventType.ASSISTANT_MESSAGE.value:
                msg = event.data.get("message", {})
                content = msg.get("content", "")
                if content:
                    yield content

    @staticmethod
    def _normalize_message(msg: Any) -> Dict[str, Any]:
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "to_openai"):
            return msg.to_openai()
        return {"role": "user", "content": str(msg)}


class AgentLoopBuilder:
    def __init__(self) -> None:
        self._llm_call_fn: Optional[Callable] = None
        self._llm_stream_fn: Optional[Callable] = None
        self._tool_registry: Optional[Any] = None
        self._system_prompt: str = ""
        self._max_parallel_tool_calls: int = 10
        self._max_tool_calls: int = 10
        self._checkpoint_path: Optional[str] = None
        self._context_window: int = 131072
        self._context_target_ratio: float = 0.70
        self._reserved_output_tokens: int = 8192
        self._context_safety_margin: int = 1024
        self._provider_input_ratio: float = 0.50

    def context_budget(self, context_window: int, target_ratio: float = 0.70,
                       reserved_output_tokens: int = 8192, safety_margin: int = 1024,
                       provider_input_ratio: float = 0.50
                       ) -> "AgentLoopBuilder":
        self._context_window = context_window
        self._context_target_ratio = target_ratio
        self._reserved_output_tokens = reserved_output_tokens
        self._context_safety_margin = safety_margin
        self._provider_input_ratio = provider_input_ratio
        return self

    def llm_call(self, fn: Callable) -> "AgentLoopBuilder":
        self._llm_call_fn = fn
        return self

    def llm_stream(self, fn: Callable) -> "AgentLoopBuilder":
        self._llm_stream_fn = fn
        return self

    def tool_registry(self, registry: Any) -> "AgentLoopBuilder":
        self._tool_registry = registry
        return self

    def system_prompt(self, prompt: str) -> "AgentLoopBuilder":
        self._system_prompt = prompt
        return self

    def max_parallel_tool_calls(self, n: int) -> "AgentLoopBuilder":
        self._max_parallel_tool_calls = n
        return self

    def max_tool_calls(self, n: int) -> "AgentLoopBuilder":
        self._max_tool_calls = n
        return self

    def checkpoint_path(self, path: str) -> "AgentLoopBuilder":
        self._checkpoint_path = path
        return self

    def build(self) -> AgentLoop:
        return AgentLoop(
            llm_call_fn=self._llm_call_fn,
            llm_stream_fn=self._llm_stream_fn,
            tool_registry=self._tool_registry,
            system_prompt=self._system_prompt,
            max_parallel_tool_calls=self._max_parallel_tool_calls,
            max_tool_calls=self._max_tool_calls,
            checkpoint_path=self._checkpoint_path,
            context_window=self._context_window,
            context_target_ratio=self._context_target_ratio,
            reserved_output_tokens=self._reserved_output_tokens,
            context_safety_margin=self._context_safety_margin,
            provider_input_ratio=self._provider_input_ratio,
        )


def create_agent_loop(
    llm_call_fn: Optional[Callable] = None,
    llm_stream_fn: Optional[Callable] = None,
    tool_registry: Optional[Any] = None,
    system_prompt: str = "",
    max_parallel_tool_calls: int = 10,
    max_tool_calls: int = 10,
    checkpoint_path: Optional[str] = None,
    context_window: int = 131072,
    context_target_ratio: float = 0.70,
    reserved_output_tokens: int = 8192,
    context_safety_margin: int = 1024,
    provider_input_ratio: float = 0.50,
) -> AgentLoop:
    return AgentLoop(
        llm_call_fn=llm_call_fn,
        llm_stream_fn=llm_stream_fn,
        tool_registry=tool_registry,
        system_prompt=system_prompt,
        max_parallel_tool_calls=max_parallel_tool_calls,
        max_tool_calls=max_tool_calls,
        checkpoint_path=checkpoint_path,
        context_window=context_window,
        context_target_ratio=context_target_ratio,
        reserved_output_tokens=reserved_output_tokens,
        context_safety_margin=context_safety_margin,
        provider_input_ratio=provider_input_ratio,
    )


__all__ = [
    "PhaseKind", "Phase", "AbortController", "AbortSignal", "AbortError",
    "InboxPosition", "InboxMessage", "Inbox",
    "SessionEventType", "SessionEvent", "Session",
    "AgentStatus", "AgentLoop", "AgentLoopBuilder", "create_agent_loop",
]
