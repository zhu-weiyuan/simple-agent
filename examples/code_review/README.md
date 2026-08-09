# AI Code Review Assistant

基于 SimpleAgent 的代码审查示例，展示如何把专用工具、静态检查和 LLM 审查组合成一个应用。

## 文件

- main.py：示例入口和审查流程。
- tools/git_diff.py：读取并整理 Git diff。
- tools/：示例专用工具。
- prompts/：提示模板目录。

## 学习重点

先看 src/my_agent/agent.py、src/my_agent/tools/ 和 src/my_agent/graph/，再回来看本示例如何注册工具、组织步骤和生成报告。

## 运行

python examples/code_review/main.py --diff-file changes.diff

示例不参与生产服务启动，也不要把真实仓库的敏感代码提交到测试数据中。
