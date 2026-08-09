# PM_LOG.md - SimpleAgent 工程化改造

## 目标
将 simple-agent 提升到智能客服 (langgraph-customer-service-agent) 的工程化水平。

## Phase 规划

### Phase 1: 安全层 🔒 ✅ DONE
- [x] PII 脱敏器 (phone/ID/email/bank card)
- [x] Prompt 注入检测器
- [x] app.py 集成（请求前扫描）
- [x] `/api/security/scan` 端点
- [x] 18 tests (test_security.py)

### Phase 2: 情感分析 + 对话摘要 💬 ✅ DONE
- [x] 情感分析模块 (5种情绪检测 + 趋势追踪 + 升级判断)
- [x] 对话摘要/工单生成 (7分类 + 优先级 + 状态)
- [x] app.py 集成 (`/api/sentiment`, `/api/summary`)
- [x] 30 tests (test_sentiment.py + test_summary.py)

### Phase 3: 可观测性增强 📊 ✅ DONE
- [x] MetricsCollector（Counter/Histogram/Gauge，Prometheus格式）
- [x] AlertService（延迟告警 + 冷却机制 + 回调）
- [x] `/api/observability` JSON 端点
- [x] `/api/metrics` Prometheus 格式（合并旧+新）
- [x] 12 tests (test_observability.py)

### Phase 4: Web UI 增强 🎨 ✅ DONE
- [x] 中英文切换 (CN/EN) — i18n 系统, localStorage 持久化
- [x] 语音 I/O (Web Speech API) — 语音输入 + TTS 朗读
- [x] 情感强度指示器 — 自动分析用户情绪, 5秒显示
- [x] 快捷回复按钮 — 已有, 支持 CN/EN 切换
- [x] SSE 流式打字动画 — cursor blink 效果
- [x] 会话导出 (JSON) — 完整消息历史下载
- [x] 🔊 朗读按钮 — 每条助手消息可语音播放

### Phase 5: 测试补全 🧪 ✅ Phase 1-3 DONE
- [x] security tests (18)
- [x] sentiment tests (20)
- [x] summary tests (16)
- [x] observability tests (12)
- [ ] integration tests (API level)

### Phase 6: SQLite Checkpointing 💾 ⬜ TODO
- [ ] SQLite 持久化（替代 facts.json）
- [ ] 用户画像自动检测
- [ ] 会话管理改进

---

## 当前状态: Phase 1-4 完成 (2026-07-02)

### 后端工程化 ✅
- **安全层**: PII 脱敏 + Prompt 注入检测，集成到 /api/chat
- **情感分析**: 5种情绪 + 趋势追踪 + 升级判断 + 语气自适应
- **对话摘要**: 7分类工单生成 (LLM + keyword fallback)
- **可观测性**: Prometheus 格式 metrics + AlertService (延迟告警)
- **新增 API**: /api/security/scan, /api/sentiment, /api/summary, /api/observability

### Web UI 工程化 ✅
- **CN/EN 切换**: 完整 i18n，欢迎页/快捷按钮/placeholder 全部双语
- **语音输入**: Web Speech API，点击 🎤 说话即发送
- **TTS 朗读**: 每条助手消息有 🔊 按钮，支持中英文朗读
- **情感指示器**: 自动分析用户情绪，显示 emoji + 强度条 (5秒)
- **会话导出**: 📥 按钮下载完整 JSON 会话历史
- **打字动画**: cursor blink 效果

### 测试覆盖 ✅
- **总计**: 66 tests, all pass
- test_security.py (18) — PII + Prompt Guard
- test_sentiment.py (20) — Emotion + Tracker + Tone
- test_summary.py (16) — Categories + Priority + Status
- test_observability.py (12) — Metrics + Alerts

---

## Phase 5: ���ɲ��� + Bug�޸� (2026-07-03)

### �������ɲ��� (test_integration.py)
- **12��API�˵����**: �������/��֤/��ȫɨ��/��з���/ժҪ/�����б�
- ��֤��Ӧͷ��Content-Type��Prometheus��ʽ
- ���� auth_required װ������Ϊ

### Bug�޸�
- **app.py**: chat�˵����� equest: Request ������authװ������Ҫ��
- **auth.py**: uth_required װ����֧�� kwargs ע�루FastAPI TestClient���ݣ�
- **registry.py**: ToolRegistry ���� 	ools �������ԣ����˽�� _tools��

### ����״̬
- **�ܼ�**: 78 tests pass (security/sentiment/summary/observability/integration)

---

## Phase 6: SQLite�־û� + API�˵� (2026-07-03)

### SQLite�Ի��洢ģ��
- src/my_agent/memory/sqlite_store.py: ����SQLite�Ի��洢
  - sessions/messages/user_profiles��
  - ȫ������/��ҳ/����/ͳ��
  - :memory:ģʽ(�����Ѻ�) + WALģʽ(����)
  
### API�˵�
- GET /api/sqlite/sessions: �Ự�б�
- GET /api/sqlite/session/{id}: �����Ự
- GET /api/sqlite/search?query=xxx: ȫ������
- GET /api/sqlite/stats: ͳ����Ϣ

### ����
- test_sqlite_store.py: 10����Ԫ����
- **�ܼ�: 88 tests pass**
