# SimpleAgent 面试 Q&A

## Q1：为什么自己实现 Agent 框架，而不是直接用 LangChain / LangGraph？
为了理解底层机制。现成框架能快速做应用，但会隐藏消息模型、工具调用循环、上下文压缩、hook 插入点等关键细节。自己实现一遍后，我对 Agent 的工程边界更清楚。

## Q2：QueryEngine 的职责是什么？
它负责驱动 Agent 主循环，而不是单纯产出回答。核心职责是调模型、解析工具调用、执行工具、回填观察结果，再决定继续还是停止。

## Q3：为什么需要 ToolRegistry？
因为工具 schema、handler、权限和异常处理不能散落在业务代码里。ToolRegistry 是让工具管理变成框架能力。

## Q4：为什么要限制工具调用轮数？
防止 Agent 陷入调用循环，避免 token 和时间失控。

## Q5：Session 压缩解决什么问题？
长对话成本高、延迟高，而且会让关键指令被旧历史稀释。压缩是 Context Engineering 的一部分。

## Q6：Hook 的价值是什么？
Hook 让框架可插拔。可以在不改主逻辑前提下插入日志、缓存、审计、实验和权限逻辑。

## Q7：MCP 和 Tool Calling 是什么关系？
Tool Calling 是模型发出调用意图的机制；MCP 是工具/资源标准化接入协议层。

## Q8：A2A 在这个项目里代表什么？
代表 Agent 间互操作能力，比如暴露 Agent Card、任务状态和调用协作接口。
