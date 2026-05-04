# AI Code Review Assistant

基于 SimpleAgent 框架实现的 AI 代码审查助手。

## 功能

- **Git Diff 解析**：解析 git diff 输出，提取变更摘要
- **代码质量检查**：调用 flake8/pylint 进行静态分析
- **安全扫描**：调用 bandit 检测 Python 安全问题
- **智能审查报告**：LLM 综合生成结构化审查报告

## 架构

```
┌─────────────┐     ┌──────────┐     ┌──────────────────┐
│  Git Diff   │────▶│  Linter  │────▶│                  │
└─────────────┘     └──────────┘     │  LLM Reviewer    │
                                      │  (SimpleAgent)   │
┌─────────────┐     ┌──────────┐────▶│                  │
│  Git Diff   │────▶│ Security │────▶│                  │
└─────────────┘     └──────────┘     └──────────────────┘
                                              │
                                      ┌───────▼────────┐
                                      │  Review Report  │
                                      │  (JSON/Markdown)│
                                      └─────────────────┘
```

## 使用

```bash
# 审查本地 diff 文件
python examples/code_review/main.py --diff-file changes.diff

# 审查指定分支
python examples/code_review/main.py --repo /path/to/repo --branch feature-branch

# 审查指定提交
python examples/code_review/main.py --repo /path/to/repo --commit abc1234
```

## 输出示例

```json
{
  "summary": "本次变更涉及 3 个文件，主要改进了用户认证模块...",
  "score": 85,
  "issues": [
    {
      "file": "auth.py",
      "line": 42,
      "severity": "warning",
      "message": "硬编码的密码盐值",
      "suggestion": "使用 secrets.token_hex() 生成随机盐值"
    }
  ],
  "recommendations": [
    "建议添加单元测试覆盖新增逻辑",
    "考虑使用 typing 注解提升代码可读性"
  ]
}
```

## 体现的 SimpleAgent 能力

- **图状态编排**：git_diff → [linter + security](并行) → LLM review → report
- **工具注册机制**：每个检查步骤注册为独立工具
- **多 Agent 协作**：代码分析 Agent + 安全扫描 Agent + 审查 Agent
