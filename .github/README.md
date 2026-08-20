# .github

GitHub 自动化配置目录。

## 目录用途

| 文件/目录 | 作用 | 维护者关注点 |
|-----------|------|--------------|
| `workflows/ci.yml` | CI 流水线：lint → type-check → test → build | 每次 PR 自动跑，红了不让合 |
| `workflows/release.yml` | 语义化发布：tag → changelog → PyPI | 打 tag 自动发版 |
| `dependabot.yml` | 依赖自动升级 PR | 周一早起看有没有安全漏洞 |
| `CODEOWNERS` | 目录级审批人 | 改 core/engine 必须 @core-maintainer 审 |
| `PULL_REQUEST_TEMPLATE.md` | PR 描述模板 | 强制填：动机、测试、回滚方案 |
| `ISSUE_TEMPLATE/` | Bug/Feature 模板 | 复现步骤、预期行为、环境信息 |

## 本项目 CI 关键节点

```yaml
# .github/workflows/ci.yml 简版
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ruff check --output-format=github .
      - run: ruff format --check .
  typecheck:
    runs-on: ubuntu-latest
    steps:
      - run: mypy src/
  test:
    runs-on: ubuntu-latest
    services:
      # 需要 LLM 服务的集成测试用 mock 或 testcontainers
    steps:
      - run: pytest tests/ -x -q
  build:
    needs: [lint, typecheck, test]
    runs-on: ubuntu-latest
    steps:
      - run: pip build --wheel
```

## 贡献者快速上手

1. Fork → Clone → `pip install -e ".[dev]"`
2. `pre-commit install`（提交前自动跑 ruff/mypy）
3. 改代码 → `pytest tests/ -x` 跑通
4. 提 PR，模板填满，等 CI 绿灯 + Codeowner 批准

> **源码学习通常不需要先读这里** —— 直接看 `src/my_agent/` 和 `docs/LEARNING_GUIDE.md` 更高效。