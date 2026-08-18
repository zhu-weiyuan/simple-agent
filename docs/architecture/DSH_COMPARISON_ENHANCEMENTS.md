# DSH 对照增强说明

本轮以 DeepSeek Harness 已安装包的公开设计文档为参照，对 SimpleAgent 做了一个可立即复用的可靠性增强：**有界工具输出保留（bounded output retention）**。

## 参照的 DSH 设计

- `@deepseek-ai/dsh-output-retention`：把「保留了什么、因预算省略了什么」从工具业务逻辑中独立出来，且不把输出截断误报为执行不完整。
- `@deepseek-ai/dsh-fs`：文件能力强调明确的错误边界、有限上下文输出和恢复路径。
- `@deepseek-ai/dsh-timeout`：超时仅负责时序/分类，真正终止动作仍由具体能力负责。

## 已落地实现

### 1. 通用工具输出保留层
新增 `src/my_agent/tools/retention.py`：

- `ItemRetainer`：保留前 N 个逻辑项，精确统计省略项数。
- `retain_text`：保留文本前缀，并返回精确省略字符数。
- 保留通知带恢复指引；例如建议缩小 `path`、`pattern` 或改用按行读取。

### 2. 已接入的工具

- `read_file` 与 `read_json`：截断后说明省略量及下一步读取建议。
- `list_files`、`search_files`、`search_text`：不再在达到上限时提前终止扫描；会扫描完整候选集合，稳定地保留前 N 项并报告实际省略数。
- `git_status`、`git_diff`、`run_tests`、`execute_powershell`：统一为有界输出 + 明确截断事实，避免静默丢失上下文。

### 3. 超时语义基础设施
新增 `src/my_agent/timeout.py`，提供 `Deadline` 与 `TimeoutReason`：

- 区分本地 deadline 到期与外部取消。
- 不假装能强制停止底层工作；执行器仍应负责杀掉子进程、关闭连接等动作。
- 作为后续统一改造 LLM 流、MCP 与后台任务超时的基础。

### 4. 文件观察与原子版本防护

新增 `src/my_agent/file_observation.py`，并引入 `write_file`、`edit_file`：

- 读取后的文件会按 session 记录 SHA-256 版本观察值。
- 覆盖与文本编辑必须基于当前会话的观察值；外部修改会触发陈旧版本拒绝。
- 新建文件必须显式传 `create_if_absent=true`。
- 写入与编辑使用临时文件 + `os.replace` 原子提交；字面量编辑拒绝空匹配和歧义匹配。

### 5. 后台任务输出保留上限

参考 DSH 的 TextRetainer / output-retention 思路，后台子进程现在会持续排空 stdout 管道，但只保留有界日志（默认 5 MB）。超过上限时：

- 不再继续写磁盘，避免测试/子进程刷屏导致磁盘耗尽；
- 追加明确的“后续内容未保存”标记；
- SQLite 记录 output_bytes 与 output_truncated；
- job_output 分页接口返回截断事实和 finish_reason；
- 即使日志被截断，管道仍会继续读取，避免子进程因 pipe 写满而卡死。

### 6. 统一权限策略接线

`PermissionPolicy` 已接入 `QueryEngine` 的 preflight：

- 工具声明的 `deny` 始终拒绝。
- `overrides` 和 `allow_patterns` 可在运行时统一允许或拒绝工具。
- `ask` 保留原有 Hook 审批流；无明确批准的非交互调用不会被策略自动提升。

## 后续建议（尚未直接改造）

1. **文件观察与版本保护**：为读过的文件记录版本/hash；写入或编辑必须带预期版本，防止并发覆盖（DSH `fs` 的 observation + stale-version 模式）。
2. **权限策略真正接线**：当前 `PermissionPolicy` 已定义，但工具注册与执行路径仍可继续强化为策略事件/审批决策，而不是只依赖工具自身约定。
3. **取消传播**：将 `Deadline` / request cancellation 贯穿到 LLM HTTP、MCP 进程与 JobManager，让超时可以实际终止对应资源。
4. **子 Agent 生命周期**：为多 agent 编排增加最大委派深度、子任务独立会话、可取消句柄和只回传最终结果的边界。
5. **大结果 spill/pagination**：把完整 grep/glob/测试日志落到 artifact，再只把有界预览送入模型，并提供后续分页读取。
