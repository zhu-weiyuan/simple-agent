# lessons.md

- CLI 参数（如 --help）应优先在本地处理，不要误进模型。
- 工具结果应该写回消息历史，再交给模型生成最终回答。
- 计算器不要裸 eval，应先做 AST 白名单校验。
- 当问题涉及工具、消息流或 response 结构时，应优先展示 message history 和 tool_calls。
- 学 Claude Code 时，应该始终把大系统映射回 SimpleAgent 的最小主链。
- 分析 Agent 主循环时，优先观察 response、tool_calls 和消息历史三者的关系。
- CLI 参数应优先在本地处理，不要把规则型输入误送进模型。
- 记忆系统应区分长期事实、经验教训和当前上下文，不要把所有历史都塞进 prompt。
