# -*- coding: utf-8 -*-
"""
流式输出系统
基于论文：AutoGen 2.0: Next-Generation Multi-Agent Framework with Native Tool Use and Streaming
arXiv: 2604.09871v2
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, AsyncGenerator, Optional
from enum import Enum


class StreamEventType(Enum):
    CHUNK = "chunk"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class StreamEvent:
    event_type: StreamEventType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: "2026-04-27")


class StreamingOutput:
    """流式输出系统"""
    
    def __init__(self):
        self.events: List[StreamEvent] = []
        self.is_streaming = False
    
    async def stream_response(self, response: str, chunk_size: int = 10) -> AsyncGenerator[StreamEvent, None]:
        """流式输出响应"""
        self.is_streaming = True
        self.events.append(StreamEvent(
            event_type=StreamEventType.CHUNK,
            content="Starting stream...",
        ))
        
        # 按 chunk 大小分割响应
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            event = StreamEvent(
                event_type=StreamEventType.CHUNK,
                content=chunk,
                metadata={"chunk_index": i // chunk_size},
            )
            self.events.append(event)
            yield event
            # 模拟延迟
            await asyncio.sleep(0.01)
        
        self.events.append(StreamEvent(
            event_type=StreamEventType.COMPLETE,
            content="Stream complete.",
        ))
        self.is_streaming = False
    
    async def stream_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        """流式输出工具调用"""
        self.is_streaming = True
        event = StreamEvent(
            event_type=StreamEventType.TOOL_CALL,
            content=f"Calling tool: {tool_name}",
            metadata={"tool_name": tool_name, "arguments": arguments},
        )
        self.events.append(event)
        yield event
        
        # 模拟工具执行
        await asyncio.sleep(0.1)
        
        self.events.append(StreamEvent(
            event_type=StreamEventType.COMPLETE,
            content=f"Tool {tool_name} executed.",
        ))
        self.is_streaming = False
    
    async def stream_error(self, error_message: str) -> AsyncGenerator[StreamEvent, None]:
        """流式输出错误"""
        event = StreamEvent(
            event_type=StreamEventType.ERROR,
            content=error_message,
            metadata={"error_type": type(error_message).__name__},
        )
        self.events.append(event)
        yield event
    
    def get_events(self) -> List[StreamEvent]:
        """获取所有事件"""
        return self.events
    
    def get_event_summary(self) -> str:
        """获取事件摘要"""
        summary_parts = []
        for event in self.events:
            summary_parts.append(f"[{event.event_type.value}] {event.content}")
        return "\n".join(summary_parts)


# 测试代码
if __name__ == "__main__":
    streaming_output = StreamingOutput()
    
    async def test_streaming():
        print("=" * 80)
        print("Streaming Output Test")
        print("=" * 80)
        
        # 测试响应流式输出
        response = "This is a test response. It contains multiple chunks of text."
        async for event in streaming_output.stream_response(response, chunk_size=10):
            print(f"Event: {event.event_type.value} | Content: {event.content}")
        
        # 测试工具调用流式输出
        async for event in streaming_output.stream_tool_call("get_time", {}):
            print(f"Event: {event.event_type.value} | Content: {event.content}")
        
        # 测试错误流式输出
        async for event in streaming_output.stream_error("Test error"):
            print(f"Event: {event.event_type.value} | Content: {event.content}")
        
        # 打印事件摘要
        print("\nEvent Summary:")
        print(streaming_output.get_event_summary())
    
    asyncio.run(test_streaming())
