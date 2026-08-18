# SimpleAgent 评测集与评测入口

SimpleAgent 不应只评测 RAG 的 Recall/Precision 和生成质量。真正上线后，最容易出问题的是“选错工具、参数错误、失败后死循环、越权读取、输出格式不合规、成本和延迟失控”。因此这里增加一套**不依赖 RAG** 的基础评测集。

## 数据集

- `datasets/tool_calling.jsonl`：工具选择、参数、调用顺序和不必要调用。
- `datasets/agent_e2e.jsonl`：多轮任务完成、轮次/工具次数、失败恢复。
- `datasets/skill_routing.jsonl`：应触发/不应触发、缺参和相似意图。
- `datasets/structured_output.jsonl`：JSON 合法性、Schema、枚举和自动修复。
- `datasets/safety.jsonl`：越权、敏感文件、删除/外传和正常请求误拒答。
- `datasets/reliability.jsonl`：LLM/工具超时、坏 JSON、重复提交、重启恢复。
- `datasets/performance.jsonl`：TTFT、E2E 延迟、token、重试率和预算护栏。

harness.py 提供可复用的本地评测数据结构、确定性评分器和 LLM Judge 封装；它与 pytest 回归测试分离，避免把评测数据混入 	ests/。

## 运行

先只检查数据集格式（不启动模型）：

```powershell
python evals/run_eval.py --validate
```

从 JSONL 运行时迹（每行至少包含 `dataset`, `case_id`, `actual`）生成报告：

```powershell
python evals/run_eval.py --trace runtime/eval_trace.jsonl --output runtime/eval_report.json
```

评测结果必须记录：`dataset_version`、`model_id`、`prompt_version`、运行时间和失败样例。LLM Judge 只作为补充，不能替代工具调用、状态机和安全规则的确定性指标。

## 第一阶段门槛建议

- 工具选择准确率 ≥ 95%
- 参数准确率 ≥ 95%
- 任务完成率 ≥ 90%
- 错误恢复率 ≥ 80%
- 敏感信息泄漏率 = 0
- 拒答率和误拒答率分开看，不能只追求“拒答越多越好”
- JSON 合法率 ≥ 99%，Schema 通过率 ≥ 95%
- 超时/重复提交必须最终收敛，不能无限调用 LLM

RAG 数据集仍单独维护 Recall@k、Hit Rate@k、MRR、Context Precision/Recall、Faithfulness 等指标；不要把检索分数和 Agent E2E 成功率混成一个总分。

## Production reliability 50

`datasets/production_reliability_50.jsonl` contains exactly 50 deliberately
unambiguous online samples: 43 tool-call cases and 7 JSON-contract cases. It
covers simple requests, precise argument extraction, explicitly ordered
multi-step tasks, safety/adversarial boundaries, tool-choice disambiguation,
and nested/enum/array/union JSON schemas.

Run only this suite against an isolated service with:

```powershell
.\.venv\Scripts\python.exe evals\run_live_eval.py --suite reliability50 --base-url http://127.0.0.1:8005 --timeout 90 --output runtime\live_eval_reliability50_report.json --trace runtime\live_eval_reliability50_trace.jsonl
```

A `200` structured response is revalidated independently by the evaluator. A
`422 structured_output_invalid` means the server correctly rejected invalid
model output; it is counted as a generation failure, not as a successful JSON
answer. `structured_first_pass_valid` distinguishes first-attempt quality from
recovery after the bounded corrective retry.

## 评测结果的解释

`evals/reports/reliability50/controlled-online-run/README.md` 记录了本次 50 条受控在线运行的范围与限制。该结果不能替代与规则不重合的盲测改写集、重复稳定性测试和故障注入测试。
