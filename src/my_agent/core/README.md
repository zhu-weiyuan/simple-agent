# core

Agent 核心执行层。engine.py 负责消息、LLM、工具调用、guardrail 和流式输出；hooks.py 是 pre/post hook；context_assembler.py 负责上下文片段和裁剪；token_budget.py 负责 token 估算和预算。这里是最值得完整通读的目录。
