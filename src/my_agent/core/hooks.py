"""
my_agent.core.hooks — 钩子系统

参考 Claude Code Hooks.ts + OpenClaw extension hooks。
Agent 核心循环的关键节点暴露钩子，允许外部注入行为。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional
from enum import Enum


class HookPoint(Enum):
    """钩子触发点"""
    QUERY_START = "on_query_start"
    QUERY_END = "on_query_end"
    TOOL_CALL_BEFORE = "on_tool_call_before"
    TOOL_CALL_AFTER = "on_tool_call_after"
    TOOL_ERROR = "on_tool_error"
    LLM_START = "on_llm_start"
    LLM_END = "on_llm_end"
    SESSION_COMPACT = "on_session_compact"
    STREAM_CHUNK = "on_stream_chunk"


@dataclass
class HookContext:
    """钩子执行上下文"""
    point: HookPoint
    agent: Any = None
    data: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class HookRegistry:
    """钩子注册中心"""

    def __init__(self) -> None:
        self._handlers: Dict[HookPoint, List[tuple]] = {}

    def register(
        self,
        point: HookPoint,
        handler: Callable[..., Any],
        priority: int = 0,
        async_: bool = False,
    ) -> None:
        if point not in self._handlers:
            self._handlers[point] = []
        self._handlers[point].append((priority, async_, handler))
        self._handlers[point].sort(key=lambda x: -x[0])

    def fire(
        self, point: HookPoint, ctx: Optional[HookContext] = None, **kwargs
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        handlers = self._handlers.get(point, [])
        context = ctx or HookContext(point=point, data=kwargs)

        for priority, is_async, handler in handlers:
            try:
                result = handler(context, **kwargs)
                key = f"{handler.__name__}_{priority}"
                results[key] = result
            except Exception as e:
                results[f"error_{handler.__name__}"] = str(e)
        return results

    async def fire_async(
        self, point: HookPoint, ctx: Optional[HookContext] = None, **kwargs
    ) -> Dict[str, Any]:
        import asyncio
        results: Dict[str, Any] = {}
        handlers = self._handlers.get(point, [])
        context = ctx or HookContext(point=point, data=kwargs)

        coros = []
        for priority, is_async, handler in handlers:
            try:
                result = handler(context, **kwargs)
                if asyncio.iscoroutine(result) or asyncio.iscoroutinefunction(handler):
                    coros.append((handler.__name__, result))
                else:
                    results[handler.__name__] = result
            except Exception as e:
                results[f"error_{handler.__name__}"] = str(e)

        for name, coro in coros:
            try:
                results[name] = await coro
            except Exception as e:
                results[f"error_{name}"] = str(e)
        return results

    def list_handlers(self, point: HookPoint) -> List[str]:
        handlers = self._handlers.get(point, [])
        return [h[2].__name__ for _, _, h in handlers]


# ── 全局 hooks ──────────────────────────────────────────────

_global_hooks = HookRegistry()


def register_hook(
    point: HookPoint,
    handler: Callable[..., Any],
    priority: int = 0,
) -> None:
    """模块级钩子注册"""
    _global_hooks.register(point, handler, priority)
