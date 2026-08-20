#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: SimpleAgent 自动分发子 Agent（Agent-as-Tool 模式）

运行:
    python demo_multiagent.py
"""
from __future__ import annotations

from my_agent import SimpleAgent
from my_agent.multiagent import SupervisorAgent, AgentRole


def create_specialized_agents():
    """创建一组专用子 Agent"""
    
    # 1. 代码审查员
    reviewer = SimpleAgent(
        name="code-reviewer",
        description="资深代码审查专家。输入：文件路径或代码片段。输出：结构化 issue 列表（严重度、行号、建议修复）",
        system_prompt=(
            "你是资深工程师，专门做代码审查。\n"
            "收到代码时：\n"
            "1. 逐行分析潜在 bug、性能、安全、可读性问题\n"
            "2. 按严重度分级：🔴 Critical / 🟡 Major / 🟢 Minor / 🔵 Suggestion\n"
            "3. 输出格式：\n"
            "   ## 审查报告: <文件名>\n"
            "   ### 🔴 Critical\n"
            "   - L<行号>: <问题> → <建议修复>\n"
            "   ### 🟡 Major\n"
            "   ...\n"
            "只输出审查报告，不闲聊。"
        ),
    )
    
    # 2. 代码生成器
    coder = SimpleAgent(
        name="code-writer",
        description="全栈代码生成器。输入：需求描述、技术栈、约束条件。输出：可运行的完整代码文件",
        system_prompt=(
            "你是全栈工程师，专门写生产级代码。\n"
            "收到需求时：\n"
            "1. 确认技术栈、接口契约、边界条件\n"
            "2. 写完整可运行代码，包含：类型注解、文档字符串、错误处理、日志\n"
            "3. 遵循项目现有风格（如有上下文）\n"
            "输出：单个文件完整代码，用 ```python 包裹。"
        ),
    )
    
    # 3. 技术调研员
    researcher = SimpleAgent(
        name="tech-researcher",
        description="技术调研专家。输入：技术问题、对比选型、最佳实践查询。输出：结构化调研报告（结论、依据、参考链接）",
        system_prompt=(
            "你是技术调研员。\n"
            "收到问题时：\n"
            "1. 拆解关键决策点\n"
            "2. 给出对比表格（维度：性能、生态、学习曲线、维护成本）\n"
            "3. 给出明确推荐结论 + 理由\n"
            "4. 列出参考来源（官方文档、知名博客、GitHub issue）\n"
            "输出：Markdown 结构化报告。"
        ),
    )
    
    # 4. 测试工程师
    tester = SimpleAgent(
        name="test-engineer",
        description="测试工程师。输入：代码文件路径或功能描述。输出：pytest 测试用例（覆盖正常、边界、异常）",
        system_prompt=(
            "你是测试工程师，专写 pytest。\n"
            "收到代码时：\n"
            "1. 分析公共接口、依赖、边界条件\n"
            "2. 生成测试类：正常流、边界值、异常抛出、Mock 外部依赖\n"
            "3. 用 pytest 风格：fixture、parametrize、assert 语义化\n"
            "输出：完整 test_*.py 文件。"
        ),
    )
    
    return {
        "reviewer": reviewer,
        "coder": coder,
        "researcher": researcher,
        "tester": tester,
    }


def build_main_agent(sub_agents: dict) -> SimpleAgent:
    """构建主 Agent，把子 Agent 注册为工具"""
    
    main = SimpleAgent(
        name="main-coordinator",
        description="主协调者：理解用户意图，自动分发任务给专用子 Agent，汇总结果",
        system_prompt=(
            "你是主协调者，拥有以下专用子工具（**有需要必须调用**，别自己硬抗）：\n\n"
            "🔧 **可用子 Agent 工具**：\n"
            "1. `code_reviewer` —— 代码审查（输入文件路径或代码，输出 issue 列表）\n"
            "2. `code_writer` —— 代码生成（输入需求描述，输出完整可运行代码）\n"
            "3. `tech_researcher` —— 技术调研（输入技术问题，输出对比报告+推荐结论）\n"
            "4. `test_engineer` —— 测试生成（输入代码路径，输出 pytest 用例）\n\n"
            "**决策规则**：\n"
            "- 用户要**写代码** → 先调 `code_writer`\n"
            "- 用户要**审查/检查代码** → 直接调 `code_reviewer`\n"
            "- 用户问**技术选型/对比/最佳实践** → 调 `tech_researcher`\n"
            "- 用户要**写测试** → 调 `test_engineer`\n"
            "- 复杂任务可**链式调用**：如 先 writer → 再 reviewer → 再 tester\n\n"
            "**输出风格**：简洁汇总，注明调用了哪个子 Agent，关键结论前置。"
        ),
    )
    
    # 关键：把子 Agent 包装为工具注册进去
    main.add_tool(sub_agents["reviewer"].as_tool(
        name="code_reviewer",
        description="代码审查专家。参数: {\"code_path\": \"文件路径\"} 或 {\"code_content\": \"代码字符串\"}"
    ))
    main.add_tool(sub_agents["coder"].as_tool(
        name="code_writer",
        description="代码生成器。参数: {\"requirement\": \"需求描述\", \"tech_stack\": \"可选技术栈\"}"
    ))
    main.add_tool(sub_agents["researcher"].as_tool(
        name="tech_researcher",
        description="技术调研员。参数: {\"question\": \"技术问题\", \"context\": \"可选背景\"}"
    ))
    main.add_tool(sub_agents["tester"].as_tool(
        name="test_engineer",
        description="测试工程师。参数: {\"code_path\": \"文件路径\"} 或 {\"code_content\": \"代码字符串\"}"
    ))
    
    return main


def demo_conversation(main: SimpleAgent):
    """演示多轮对话，展示自动分发"""
    
    test_cases = [
        # 1. 直接触发代码生成
        "帮我写一个 Python 类，实现 LRU 缓存，要求线程安全、支持 TTL 过期",
        
        # 2. 触发代码审查（复用刚生成的代码，或指定现有文件）
        # "帮我审查一下 src/my_agent/core/engine.py 的 run 方法",
        
        # 3. 触发技术调研
        # "Python 异步 HTTP 客户端选型：httpx vs aiohttp vs requests-threads，哪个适合高并发爬虫？",
        
        # 4. 触发测试生成
        # "给刚才的 LRU 缓存类写完整 pytest 测试",
        
        # 5. 复合任务：写 → 审 → 测（看主 Agent 是否自动链式调用）
        # "实现一个简单的配置管理类（支持 YAML/环境变量/默认值优先级），并帮我审查和写测试",
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🗣️  用户 [{i}]: {user_input}")
        print(f"{'='*60}\n")
        
        result = main.run(user_input)
        print(f"🤖 主 Agent 回复:\n{result}\n")


def demo_supervisor_mode(sub_agents: dict):
    """备选：Supervisor 模式（显式编排，非工具调用）"""
    print("\n" + "="*60)
    print("🎯 Supervisor 模式演示")
    print("="*60)
    
    supervisor = SupervisorAgent(
        name="supervisor",
        roles=[
            AgentRole("researcher", "技术调研", sub_agents["researcher"], priority=10),
            AgentRole("coder", "代码生成", sub_agents["coder"], priority=8),
            AgentRole("reviewer", "代码审查", sub_agents["reviewer"], priority=6),
            AgentRole("tester", "测试生成", sub_agents["tester"], priority=4),
        ],
        max_rounds=3,
    )
    
    # Supervisor 会根据关键词自动路由
    tasks = [
        "调研一下 Python 里实现分布式锁的几种方案对比",
        "写一个基于 Redis 的分布式锁工具类",
        "帮我审查一下这个分布式锁的实现",
    ]
    
    for task in tasks:
        print(f"\n🗣️  任务: {task}")
        result = supervisor.run(task)
        print(f"📋 Supervisor 选择: {result.supervisor_choice}")
        print(f"📝 结果:\n{result.final_response[:500]}...")


if __name__ == "__main__":
    print("🚀 启动 SimpleAgent 多 Agent 自动分发演示")
    print("="*60)
    
    # 1. 创建专用子 Agent
    sub_agents = create_specialized_agents()
    print(f"✅ 创建了 {len(sub_agents)} 个专用子 Agent: {list(sub_agents.keys())}")
    
    # 2. 构建主 Agent（自动注册工具）
    main = build_main_agent(sub_agents)
    print(f"✅ 主 Agent 已注册 {len(main.tool_registry.all_names())} 个工具: {main.tool_registry.all_names()}")
    
    # 3. 运行演示对话
    demo_conversation(main)
    
    # 4. 可选：Supervisor 模式对比
    # demo_supervisor_mode(sub_agents)
    
    # 5. 清理
    main.close()
    for a in sub_agents.values():
        a.close()
    
    print("\n✅ 演示结束")