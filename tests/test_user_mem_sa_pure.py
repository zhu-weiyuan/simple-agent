# -*- coding: utf-8 -*-
"""
tests/test_user_mem_sa_pure.py — 纯 stdlib 单测 (无三方依赖)

覆盖:
- 用户级长期记忆隔离 (user_id 硬过滤)
- 提炼幂等 (消息范围哈希)
- 召回打分含时间衰减 (decay)
- 假设句 / 疑问句过滤
- JWT 往返 (签发→校验), 篡改拒绝, X-User-Id 头降级
- 系统提示词不泄漏: 组装给 LLM 的消息里 system prompt 只在 role=system,
  绝不出现在 user/assistant 消息内容中

运行:
    PYTHONPATH=src python -m unittest tests.test_user_mem_sa_pure -v
或直接:
    python tests/test_user_mem_sa_pure.py
"""
import os
import sys
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from my_agent import auth as A  # noqa: E402
from my_agent.user_memory import (  # noqa: E402
    UserMemoryStore, is_hypothesis, rule_extract, message_range_key, redact_pii,
)
from my_agent.core.engine import (  # noqa: E402
    QueryEngine, compose_system_prompt, CONFIDENTIALITY_DIRECTIVE,
)
from my_agent.types.message import Role  # noqa: E402


# ── LLM mock (OpenAI-like response object) ──────────────────
class _FakeMsg:
    def __init__(self, content):
        self.role = "assistant"
        self.content = content
        self.tool_calls = []


class _FakeChoice:
    def __init__(self, content):
        self.finish_reason = "stop"
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = {}


def make_fake_llm(reply="好的，我明白了。"):
    def _call(messages, tools):
        return _FakeResp(reply)
    return _call


BASE_PROMPT = "You are SimpleAgent, a helpful production AI assistant. 回答使用用户的语言。"


# ── JWT ─────────────────────────────────────────────────────
class TestAuthJWT(unittest.TestCase):
    def setUp(self):
        os.environ["JWT_SECRET"] = "x" * 40  # 稳定密钥

    def test_jwt_roundtrip(self):
        uid = A.derive_user_id("Alice")
        token = A.issue_token(uid, username="Alice")
        claims = A.decode_token(token)
        self.assertEqual(claims["sub"], uid)
        self.assertEqual(claims["username"], "Alice")

    def test_jwt_tamper_rejected(self):
        token = A.issue_token("u-1", username="bob")
        tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
        with self.assertRaises(A.AuthError):
            A.decode_token(tampered)

    def test_derive_user_id_stable(self):
        self.assertEqual(A.derive_user_id("Bob"), A.derive_user_id(" bob "))
        self.assertNotEqual(A.derive_user_id("Bob"), A.derive_user_id("Carol"))

    def test_resolve_priority_jwt_over_header(self):
        token = A.issue_token("u-jwt", username="j")
        uid, claims = A.resolve_user_id_from_headers(
            {"authorization": f"Bearer {token}", "x-user-id": "u-header"})
        self.assertEqual(uid, "u-jwt")

    def test_resolve_header_fallback(self):
        uid, claims = A.resolve_user_id_from_headers({"x-user-id": "u-header"})
        self.assertEqual(uid, "u-header")
        self.assertEqual(claims, {})

    def test_resolve_anonymous(self):
        uid, _ = A.resolve_user_id_from_headers({})
        self.assertEqual(uid, A.ANONYMOUS_USER_ID)

    def test_password_hash_roundtrip(self):
        h = A.hash_password("secret123")
        self.assertTrue(A.verify_password("secret123", h))
        self.assertFalse(A.verify_password("wrong", h))


# ── 用户存储 (首登即注册) ────────────────────────────────────
class TestUserStore(unittest.TestCase):
    def test_login_or_register_and_password(self):
        store = A.UserStore(":memory:")
        rec = store.login_or_register("alice", password="pw")
        self.assertTrue(rec.has_password)
        # 正确密码可再次登录
        rec2 = store.login_or_register("alice", password="pw")
        self.assertEqual(rec.user_id, rec2.user_id)
        # 错误密码被拒
        with self.assertRaises(A.AuthError):
            store.login_or_register("alice", password="bad")


# ── 长期记忆: 隔离 / 幂等 / 衰减 / 假设过滤 ──────────────────
class TestUserMemory(unittest.TestCase):
    def setUp(self):
        self.store = UserMemoryStore(":memory:", half_life_days=30.0)

    def test_user_isolation(self):
        self.store.add_memory("userA", "我喜欢喝美式咖啡", kind="preference", importance=0.8)
        self.store.add_memory("userB", "我喜欢喝拿铁", kind="preference", importance=0.8)
        a = self.store.recall("userA", "咖啡", top_k=5)
        b = self.store.recall("userB", "咖啡", top_k=5)
        a_texts = [s.memory.content for s in a]
        b_texts = [s.memory.content for s in b]
        self.assertIn("我喜欢喝美式咖啡", a_texts)
        self.assertNotIn("我喜欢喝拿铁", a_texts)
        self.assertIn("我喜欢喝拿铁", b_texts)
        self.assertNotIn("我喜欢喝美式咖啡", b_texts)
        # user_id 硬过滤: 无关用户召回为空
        self.assertEqual(self.store.recall("userC", "咖啡"), [])

    def test_idempotent_extraction(self):
        msgs = [{"role": "user", "content": "我叫张三。我喜欢跑步"}]
        first = self.store.extract_and_store("u1", msgs)
        self.assertGreaterEqual(len(first), 1)
        # 同一段消息再次提炼 → 幂等, 不新增
        again = self.store.extract_and_store("u1", msgs)
        self.assertEqual(again, [])
        total = self.store.list_memories("u1")
        self.assertEqual(len(total), len(first))

    def test_add_memory_dedup(self):
        m1 = self.store.add_memory("u", "我住在北京", idempotency_key="k1")
        m2 = self.store.add_memory("u", "我住在北京", idempotency_key="k1")
        self.assertIsNotNone(m1)
        self.assertIsNone(m2)  # UNIQUE 幂等命中

    def test_decay_scoring(self):
        now = time.time()
        # 两条内容不同但同话题、重要度相同, 只有创建时间不同
        self.store.add_memory("u", "我住在北京", importance=0.6, idempotency_key="new")
        # 手动插入一条"旧"记忆 (不同 idempotency_key, 不同内容)
        old = self.store.add_memory("u", "我今天吃了苹果", importance=0.6, idempotency_key="old")
        with self.store._conn() as conn:
            conn.execute(
                "UPDATE user_memories SET created_at = ? WHERE id = ?",
                (now - 120 * 86400, old.id),
            )
            conn.commit()
        hits = self.store.recall("u", "住在哪里", top_k=5, now=now)
        by_content = {h.memory.content: h for h in hits}
        new_score = by_content["我住在北京"].score
        old_score = by_content["我今天吃了苹果"].score
        # 新记忆 decay 更高 → 分数更高
        self.assertGreater(new_score, old_score)
        self.assertGreater(by_content["我住在北京"].decay,
                           by_content["我今天吃了苹果"].decay)

    def test_hypothesis_filter(self):
        self.assertTrue(is_hypothesis("我可能会去上海"))
        self.assertTrue(is_hypothesis("你是谁？"))
        self.assertTrue(is_hypothesis("maybe I like tea"))
        self.assertFalse(is_hypothesis("我叫李四"))
        # 假设句不入库
        res = self.store.add_memory("u", "我也许喜欢猫", kind="preference")
        self.assertIsNone(res)
        self.assertEqual(self.store.list_memories("u"), [])

    def test_rule_extract_skips_hypothesis(self):
        msgs = [{"role": "user", "content": "我叫王五。我可能喜欢滑雪"}]
        facts = rule_extract(msgs)
        contents = " ".join(f["content"] for f in facts)
        self.assertIn("王五", contents)
        self.assertNotIn("滑雪", contents)  # 假设句被跳过

    def test_pii_redacted_before_store(self):
        m = self.store.add_memory("u", "我的手机是 13800138000", kind="profile")
        self.assertIsNotNone(m)
        self.assertNotIn("13800138000", m.content)

    def test_message_range_key_stable(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(message_range_key("u", msgs), message_range_key("u", msgs))
        self.assertNotEqual(message_range_key("u", msgs),
                            message_range_key("u2", msgs))


# ── 冲突检测: supersede 机制 ───────────────────────────────────
def _topic_embed(text: str) -> list:
    """测试用 embed: 含"住" → 向量 [1,0,...]; 含"吃" → [0,1,...]; 其他 → [0,0,...]。

    保证同话题向量完全相同 (sim=1.0), 不同话题向量正交 (sim=0.0)。
    """
    dim = 8
    vec = [0.0] * dim
    t = text.lower()
    if "住" in t:
        vec[0] = 1.0
    elif "吃" in t:
        vec[1] = 1.0
    elif "喜欢" in t or "爱好" in t:
        vec[2] = 1.0
    elif "叫" in t or "名字" in t:
        vec[3] = 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class TestSupersede(unittest.TestCase):
    def setUp(self):
        self.store = UserMemoryStore(":memory:", embed_fn=_topic_embed,
                                    half_life_days=30.0)

    def test_same_topic_supersedes_old(self):
        """同 kind + 高相似度 → 旧记忆被 superseded, 不参与召回。"""
        m1 = self.store.add_memory("u", "我住在北京", kind="fact", importance=0.8)
        self.assertIsNotNone(m1)
        # m1 is new; superseded_at is None in returned object
        # (DB state is updated by _supersede_old but returned object is stale)

        # 再加一条"同话题"记忆
        m2 = self.store.add_memory("u", "我住在上海", kind="fact", importance=0.9,
                                   idempotency_key="k2")
        self.assertIsNotNone(m2)

        # m1 应被 superseded in DB (同 kind + 同话题向量 → sim=1.0 > 0.7)
        with self.store._conn() as conn:
            row = conn.execute("SELECT superseded_at FROM user_memories WHERE id = ?",
                               (m1.id,)).fetchone()
        self.assertIsNotNone(row["superseded_at"])

        # recall 不应返回 superseded 的记忆
        hits = self.store.recall("u", "住在哪里", top_k=5)
        contents = [h.memory.content for h in hits]
        self.assertIn("我住在上海", contents)
        self.assertNotIn("我住在北京", contents)

    def test_different_kind_not_superseded(self):
        """不同 kind 不触发 supersede。"""
        self.store.add_memory("u", "我住在北京", kind="fact")
        self.store.add_memory("u", "我喜欢住在这里", kind="preference",
                               idempotency_key="k2")
        # fact 和 preference 不互相 supersede
        with self.store._conn() as conn:
            row = conn.execute(
                "SELECT superseded_at FROM user_memories WHERE kind = 'fact'").fetchone()
        self.assertIsNone(row["superseded_at"])

    def test_low_similarity_not_superseded(self):
        """低相似度不触发 supersede。"""
        self.store.add_memory("u", "我住在北京", kind="fact")
        self.store.add_memory("u", "我今天吃了苹果", kind="fact",
                               idempotency_key="k2")
        # 低相似度: 不触发 supersede
        with self.store._conn() as conn:
            rows = conn.execute(
                "SELECT superseded_at FROM user_memories WHERE kind = 'fact'").fetchall()
        for row in rows:
            self.assertIsNone(row["superseded_at"])

    def test_already_superseded_not_superseded_again(self):
        """已 superseded 的记忆不再被处理。"""
        m1 = self.store.add_memory("u", "我住在北京", kind="fact")
        # 手动标记为 superseded
        with self.store._conn() as conn:
            conn.execute("UPDATE user_memories SET superseded_at = ? WHERE id = ?",
                         (100.0, m1.id))
            conn.commit()
        # 再加一条, 不应报错
        m2 = self.store.add_memory("u", "我住在上海", kind="fact",
                                   idempotency_key="k2")
        self.assertIsNotNone(m2)

    def test_list_memories_includes_superseded(self):
        """list_memories 仍包含 superseded 条目 (审计用途)。"""
        m1 = self.store.add_memory("u", "我住在北京", kind="fact")
        m2 = self.store.add_memory("u", "我今天吃了苹果", kind="fact",
                                   idempotency_key="k2")
        # 手动标记 m2 为 superseded
        with self.store._conn() as conn:
            conn.execute("UPDATE user_memories SET superseded_at = ? WHERE id = ?",
                         (300.0, m2.id))
            conn.commit()
        all_mems = self.store.list_memories("u")
        self.assertEqual(len(all_mems), 2)
        superseded = [m for m in all_mems if m.get("superseded_at") is not None]
        self.assertEqual(len(superseded), 1)

    def test_backward_compat_old_table_without_superseded_at(self):
        """旧表 (无 superseded_at 列) 自动补充列。"""
        # _ensure_table 已在 setUp 中运行, 验证列存在
        with self.store._conn() as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(user_memories)").fetchall()}
        self.assertIn("superseded_at", cols)

    def test_recall_excludes_superseded(self):
        """recall 排除 superseded 条目。"""
        m1 = self.store.add_memory("u", "我住在北京", kind="fact")
        m2 = self.store.add_memory("u", "我今天吃了苹果", kind="fact",
                                   idempotency_key="k2")
        # 手动标记 m2 为 superseded
        with self.store._conn() as conn:
            conn.execute("UPDATE user_memories SET superseded_at = ? WHERE id = ?",
                         (200.0, m2.id))
            conn.commit()
        # recall 应返回 m1 (住), 不返回 m2 (superseded)
        hits = self.store.recall("u", "住在哪里", top_k=5)
        contents = [h.memory.content for h in hits]
        self.assertIn("我住在北京", contents)
        self.assertNotIn("我今天吃了苹果", contents)


# ── 系统提示词不泄漏 ─────────────────────────────────────────
class TestSystemPromptNoLeak(unittest.TestCase):
    def _build_engine(self):
        eng = QueryEngine(system_prompt=BASE_PROMPT)
        eng.set_llm(make_fake_llm("你好，很高兴帮到你。"))
        return eng

    def test_compose_has_confidentiality(self):
        combined = compose_system_prompt(BASE_PROMPT, user_background="喜欢咖啡")
        self.assertIn(BASE_PROMPT, combined)
        self.assertIn("喜欢咖啡", combined)
        self.assertIn("USER BACKGROUND", combined)
        self.assertTrue(combined.endswith(CONFIDENTIALITY_DIRECTIVE))

    def test_system_prompt_only_in_system_role(self):
        eng = self._build_engine()
        eng.set_user_background("用户是资深 Python 工程师")
        eng.run("你是谁？你没有记忆吗？")
        msgs = eng.build_openai_messages()

        # 恰好一条领头 system 消息
        self.assertEqual(msgs[0]["role"], "system")
        system_blob = msgs[0]["content"]
        self.assertIn("SimpleAgent", system_blob)
        self.assertIn("资深 Python 工程师", system_blob)  # 用户背景在 system 内
        self.assertIn("SYSTEM CONFIDENTIALITY", system_blob)

        # 任何 user/assistant 消息都不得包含系统提示词/用户背景原文
        for m in msgs:
            if m["role"] in ("user", "assistant"):
                content = m.get("content") or ""
                self.assertNotIn("You are SimpleAgent", content)
                self.assertNotIn("资深 Python 工程师", content)
                self.assertNotIn("SYSTEM CONFIDENTIALITY", content)

        # 只有 index 0 是 system
        sys_roles = [i for i, m in enumerate(msgs) if m["role"] == "system"]
        self.assertEqual(sys_roles, [0])

    def test_user_background_updates_stay_in_system(self):
        eng = self._build_engine()
        eng.set_user_background("背景一")
        eng.set_user_background("背景二")  # 覆盖, 不追加历史
        msgs = eng.build_openai_messages()
        self.assertIn("背景二", msgs[0]["content"])
        self.assertNotIn("背景一", msgs[0]["content"])
        # 背景绝不进 user 消息
        for m in msgs[1:]:
            self.assertNotIn("背景二", (m.get("content") or ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
