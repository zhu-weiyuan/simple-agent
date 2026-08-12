# -*- coding: utf-8 -*-
"""
my_agent.user_memory — 跨会话用户长期记忆 (向量召回)

与会话级 SqliteConversationStore 互补: 本模块存放**按 user_id 隔离**的结构化
长期事实, 会话结束时异步提炼, 并在下一轮组装 prompt 时以"用户背景"分区召回。

核心能力
--------
1. 提炼 (extract_and_store): 注入式 ``llm_fn`` 提炼结构化事实, 缺失时规则降级。
   - 幂等键 = 消息范围内容哈希 (同一段对话重复提炼不重复入库)。
   - 假设句 / 疑问句不入库 (只保留稳定陈述性事实)。
   - 入库前 PII 脱敏。
2. 召回 (recall): 打分 = relevance(cosine) × importance × decay(时间衰减),
   ``user_id`` 硬过滤, 返回 top-k。
3. embedding: 注入式 ``embed_fn`` (可接 AsyncLLMClient / MY_AGENT_* 配置),
   缺失时用确定性 stdlib 词袋 hash 向量降级 (保证离线/单测可跑)。
4. 存储: SQLite (stdlib) + 内存余弦; pgvector 可选 (未安装则透明降级)。

纯 stdlib: 仅依赖 sqlite3 / hashlib / math / json / time / re。
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

EMBED_DIM = 256  # 降级词袋向量维度

# 假设/疑问/不稳定语气标记 — 命中则该句不入库
_HYPOTHESIS_MARKERS = (
    "可能", "也许", "大概", "或许", "假设", "如果", "要是", "说不定", "似乎", "好像",
    "maybe", "perhaps", "probably", "might", "could be", "i think", "i guess",
    "suppose", "what if", "not sure", "unsure",
)

# 简单 PII 正则 (与 security.pii_redactor 对齐, 但本模块自持以免耦合)
_PII_PATTERNS = {
    "id_card": (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), lambda s: s[:6] + "*" * 9 + s[-4:]),
    "phone": (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), lambda s: s[:3] + "****" + s[-4:]),
    "bank_card": (re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
                  lambda s: "****" + re.sub(r"[\s-]", "", s)[-4:]),
    "email": (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
              lambda s: s.split("@")[0][:2] + "***@" + s.split("@")[1]),
}


def redact_pii(text: str) -> str:
    """入库前脱敏 (优先复用 security.pii_redactor, 缺失则用内置)。"""
    try:  # 复用主实现 (保持一致的脱敏格式)
        from .security.pii_redactor import redact as _redact  # type: ignore
        return _redact(text).redacted_text
    except Exception:  # noqa: BLE001 - 降级到内置
        out = text
        for _kind, (pat, repl) in _PII_PATTERNS.items():
            out = pat.sub(lambda m: repl(m.group()), out)
        return out


def is_hypothesis(sentence: str) -> bool:
    """判断句子是否为假设/疑问 (不入库)。"""
    s = sentence.strip()
    if not s:
        return True
    if s.endswith(("?", "？")):
        return True
    low = s.lower()
    return any(marker in low for marker in _HYPOTHESIS_MARKERS)


# ── embedding ────────────────────────────────────────────────
def _hash_embed(text: str, dim: int = EMBED_DIM) -> List[float]:
    """确定性词袋 hash 向量 (离线降级)。同义词无法泛化, 但保证可复现、可比对。"""
    vec = [0.0] * dim
    tokens = re.findall(r"[\w一-鿿]+", text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度 (已归一化时即点积); 维度不等取较短。"""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── 数据模型 ─────────────────────────────────────────────────
@dataclass
class UserMemory:
    user_id: str
    content: str
    kind: str = "fact"  # fact | preference | profile | lesson
    importance: float = 0.5  # 0..1
    embedding: List[float] = field(default_factory=list)
    created_at: float = 0.0
    expires_at: Optional[float] = None
    idempotency_key: str = ""
    id: Optional[int] = None
    superseded_at: Optional[float] = None  # 被更新时设置; 非 None 则不参与召回

    def to_public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "importance": round(self.importance, 3),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "superseded_at": self.superseded_at,
        }


@dataclass
class ScoredMemory:
    memory: UserMemory
    relevance: float
    decay: float
    score: float
    rerank_score: Optional[float] = None


# ── 幂等键 ───────────────────────────────────────────────────
def message_range_key(user_id: str, messages: Sequence[Dict[str, Any]]) -> str:
    """消息范围内容哈希 → 幂等键 (同段对话重复提炼幂等)。"""
    h = hashlib.sha256()
    h.update(user_id.encode("utf-8"))
    for m in messages:
        h.update(b"\x00")
        h.update(str(m.get("role", "")).encode("utf-8"))
        h.update(b"\x01")
        h.update(str(m.get("content", "")).encode("utf-8"))
    return h.hexdigest()


# ── 规则降级提炼 ─────────────────────────────────────────────
# "我叫X" / "我是X" / "我的名字是X" / "我喜欢X" / "我在X工作" 等稳定陈述
_RULE_PATTERNS = [
    (re.compile(r"(?:我叫|我的名字(?:是|叫)|我是)\s*([^\s,。,.!！?？的]{1,20})"), "profile", 0.8),
    (re.compile(r"我(?:的)?(?:邮箱|邮件|email)(?:是|为)?\s*([^\s,。,.!！?？]{3,60})", re.I), "profile", 0.7),
    (re.compile(r"我(?:喜欢|爱好|偏好|喜爱)\s*([^,。,.!！?？]{1,40})"), "preference", 0.6),
    (re.compile(r"我(?:不喜欢|讨厌|不想)\s*([^,。,.!！?？]{1,40})"), "preference", 0.6),
    (re.compile(r"我(?:在|于)\s*([^,。,.!！?？]{1,30})\s*(?:工作|上班|上学|学习)"), "profile", 0.7),
    (re.compile(r"我(?:住在|来自|在)\s*([^,。,.!！?？]{1,30})", ), "profile", 0.5),
]


def rule_extract(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """规则降级: 从 user 消息里抽稳定陈述性事实。返回 [{content,kind,importance}]。"""
    facts: List[Dict[str, Any]] = []
    seen = set()
    for m in messages:
        if m.get("role") != "user":
            continue
        text = str(m.get("content", "") or "")
        # 逐句切分, 假设句丢弃
        for sentence in re.split(r"[。.!！?？\n]", text):
            s = sentence.strip()
            if not s or is_hypothesis(s):
                continue
            for pat, kind, imp in _RULE_PATTERNS:
                mt = pat.search(s)
                if mt:
                    val = mt.group(0).strip()
                    if val and val not in seen:
                        seen.add(val)
                        facts.append({"content": val, "kind": kind, "importance": imp})
                    break
    return facts


# ── 主存储 ───────────────────────────────────────────────────
class UserMemoryStore:
    """按 user_id 隔离的长期记忆存储 (SQLite + 内存余弦)。"""

    # 默认相似度阈值: 余弦归一后超过此值视为"同一话题"的更新
    SUPERSEDE_SIMILARITY_THRESHOLD: float = 0.7

    def __init__(
        self,
        db_path: str = "conversations.db",
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
        half_life_days: float = 30.0,
        rerank_fn: Optional[Callable[[str, Sequence[str]], Sequence[float]]] = None,
        rerank_min_score: float = 0.0,
        rerank_candidate_limit: int = 20,
    ) -> None:
        self.db_path = db_path
        self._embed_fn = embed_fn or _hash_embed
        self._rerank_fn = rerank_fn
        self.rerank_min_score = float(rerank_min_score)
        self.rerank_candidate_limit = max(1, int(rerank_candidate_limit))
        self.half_life_seconds = max(1.0, half_life_days * 86400.0)
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._persistent_conn = self._new_conn()
        self._ensure_table()

    # ── 连接 ─────────────────────────────────────────────────
    def _new_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _conn(self) -> sqlite3.Connection:
        return self._persistent_conn or self._new_conn()

    def _ensure_table(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'fact',
                    importance REAL NOT NULL DEFAULT 0.5,
                    embedding TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    idempotency_key TEXT NOT NULL DEFAULT '',
                    superseded_at REAL,
                    UNIQUE(user_id, idempotency_key, content)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_mem_uid ON user_memories(user_id)"
            )
            # 兼容旧表: 补充 superseded_at 列 (幂等)
            try:
                conn.execute("ALTER TABLE user_memories ADD COLUMN superseded_at REAL")
            except sqlite3.OperationalError:
                pass  # 列已存在
            conn.commit()

    # ── embed ────────────────────────────────────────────────
    def embed(self, text: str) -> List[float]:
        try:
            vec = self._embed_fn(text)
            return [float(x) for x in vec]
        except Exception as e:  # noqa: BLE001 - embedding 失败降级
            logger.warning("embed_fn failed (%s); falling back to hash embed", e)
            return _hash_embed(text)

    # ── 写入 ─────────────────────────────────────────────────
    def add_memory(
        self,
        user_id: str,
        content: str,
        kind: str = "fact",
        importance: float = 0.5,
        idempotency_key: str = "",
        ttl_seconds: Optional[float] = None,
        skip_hypothesis: bool = True,
    ) -> Optional[UserMemory]:
        """写一条记忆 (PII 脱敏 + 假设句过滤 + 幂等)。返回入库对象或 None。"""
        content = (content or "").strip()
        if not content:
            return None
        if skip_hypothesis and is_hypothesis(content):
            return None
        content = redact_pii(content)
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds else None
        emb = self.embed(content)
        mem = UserMemory(
            user_id=user_id, content=content, kind=kind,
            importance=max(0.0, min(1.0, importance)), embedding=emb,
            created_at=now, expires_at=expires_at, idempotency_key=idempotency_key,
        )
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO user_memories
                    (user_id, content, kind, importance, embedding, created_at,
                     expires_at, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, content, kind, mem.importance, json.dumps(emb),
                 now, expires_at, idempotency_key),
            )
            conn.commit()
            # rowcount==1 表示确实插入; INSERT OR IGNORE 命中冲突时 rowcount==0
            # (lastrowid 会保留上一次成功的 rowid, 不可靠, 故用 rowcount 判断)。
            if cur.rowcount == 1:
                mem.id = cur.lastrowid
                # 冲突检测: 同 kind + 高相似度 → 旧记忆标记为 superseded
                self._supersede_old(user_id, kind, emb, now, exclude_id=mem.id)
                return mem
        return None  # 幂等命中, 未新增

    def _supersede_old(
        self,
        user_id: str,
        kind: str,
        new_embedding: List[float],
        now: float,
        exclude_id: Optional[int] = None,
    ) -> int:
        """标记同 kind、高相似度的旧记忆为 superseded。返回标记数量。

        仅处理 fact / preference / profile 类型 (lesson 等不限)。
        阈值: cosine 归一后 > SUPERSEDE_SIMILARITY_THRESHOLD。
        exclude_id: 刚插入的新记忆 ID, 不参与对比。
        """
        if kind not in ("fact", "preference", "profile"):
            return 0
        if not new_embedding:
            return 0
        superseded = 0
        with self._conn() as conn:
            if exclude_id is not None:
                rows = conn.execute(
                    """SELECT id, embedding FROM user_memories
                       WHERE user_id = ? AND kind = ?
                         AND superseded_at IS NULL
                         AND id != ?
                         AND (expires_at IS NULL OR expires_at > ?)""",
                    (user_id, kind, exclude_id, now),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, embedding FROM user_memories
                       WHERE user_id = ? AND kind = ?
                         AND superseded_at IS NULL
                         AND (expires_at IS NULL OR expires_at > ?)""",
                    (user_id, kind, now),
                ).fetchall()
            for row in rows:
                try:
                    old_emb = json.loads(row["embedding"])
                except (ValueError, json.JSONDecodeError):
                    continue
                if not old_emb:
                    continue
                sim = (cosine(new_embedding, old_emb) + 1.0) / 2.0  # 归一到 0..1
                if sim > self.SUPERSEDE_SIMILARITY_THRESHOLD:
                    conn.execute(
                        "UPDATE user_memories SET superseded_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
                    superseded += 1
            conn.commit()
        if superseded > 0:
            logger.info(
                "superseded %d old %s memories for user %s (sim > %.2f)",
                superseded, kind, user_id, self.SUPERSEDE_SIMILARITY_THRESHOLD,
            )
        return superseded

    def extract_and_store(
        self,
        user_id: str,
        messages: Sequence[Dict[str, Any]],
        llm_fn: Optional[Callable[[Sequence[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    ) -> List[UserMemory]:
        """会话结束提炼: llm_fn 优先, 失败/缺失则规则降级。幂等键=消息范围哈希。

        llm_fn 约定: 接收 messages, 返回 [{content, kind?, importance?}]。
        """
        if not messages:
            return []
        idem = message_range_key(user_id, messages)
        # 幂等: 该消息范围已提炼过则跳过
        if self._range_already_extracted(user_id, idem):
            return []

        candidates: List[Dict[str, Any]] = []
        if llm_fn is not None:
            try:
                raw = llm_fn(messages) or []
                for item in raw:
                    if isinstance(item, dict) and item.get("content"):
                        candidates.append(item)
            except Exception as e:  # noqa: BLE001 - LLM 提炼失败 → 规则降级
                logger.warning("llm_fn extraction failed (%s); rule fallback", e)
        if not candidates:
            candidates = rule_extract(messages)

        stored: List[UserMemory] = []
        for item in candidates:
            content = str(item.get("content", ""))
            if is_hypothesis(content):
                continue
            mem = self.add_memory(
                user_id=user_id,
                content=content,
                kind=str(item.get("kind", "fact")),
                importance=float(item.get("importance", 0.5)),
                idempotency_key=idem,
            )
            if mem is not None:
                stored.append(mem)
        # 即便没有产出事实, 也写一条哨兵幂等标记, 避免重复空提炼
        if not stored:
            self._mark_range(user_id, idem)
        return stored

    def _range_already_extracted(self, user_id: str, idem: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM user_memories WHERE user_id = ? AND idempotency_key = ? LIMIT 1",
                (user_id, idem),
            ).fetchone()
        return row is not None

    def _mark_range(self, user_id: str, idem: str) -> None:
        """空提炼哨兵 (importance=0, 立即过期, 不参与召回)。"""
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO user_memories
                   (user_id, content, kind, importance, embedding, created_at,
                    expires_at, idempotency_key)
                   VALUES (?, ?, 'sentinel', 0.0, '[]', ?, ?, ?)""",
                (user_id, f"[sentinel:{idem[:8]}]", now, now - 1, idem),
            )
            conn.commit()

    # ── 召回 ─────────────────────────────────────────────────
    def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        now: Optional[float] = None,
    ) -> List[ScoredMemory]:
        """score = relevance × importance × decay; user_id 硬过滤; 排除过期/哨兵/已 supersede。"""
        now = now if now is not None else time.time()
        q_emb = self.embed(query) if query else []
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM user_memories
                   WHERE user_id = ? AND kind != 'sentinel'
                     AND superseded_at IS NULL
                     AND (expires_at IS NULL OR expires_at > ?)""",
                (user_id, now),
            ).fetchall()
        scored: List[ScoredMemory] = []
        for row in rows:
            mem = self._row_to_mem(row)
            relevance = cosine(q_emb, mem.embedding) if q_emb and mem.embedding else 0.0
            # 归一到 0..1 (cosine ∈ [-1,1])
            relevance = (relevance + 1.0) / 2.0
            age = max(0.0, now - mem.created_at)
            decay = math.exp(-age / self.half_life_seconds)
            score = relevance * mem.importance * decay
            if score >= min_score:
                scored.append(ScoredMemory(mem, relevance, decay, score))
        scored.sort(key=lambda s: s.score, reverse=True)
        if not scored or self._rerank_fn is None:
            return scored[:top_k]

        # The vector/decay score first narrows candidates. A cross-encoder then
        # refines semantic relevance. Provider errors intentionally preserve the
        # local ranking, so a remote reranker cannot block the primary chat path.
        candidates = scored[:self.rerank_candidate_limit]
        try:
            rerank_scores = list(self._rerank_fn(query, [item.memory.content for item in candidates]))
            if len(rerank_scores) != len(candidates):
                raise ValueError("rerank score count does not match candidate count")
            reranked: List[ScoredMemory] = []
            for item, raw_score in zip(candidates, rerank_scores):
                score = float(raw_score)
                if not math.isfinite(score):
                    raise ValueError("rerank score is not finite")
                item.rerank_score = score
                if score >= self.rerank_min_score:
                    reranked.append(item)
            reranked.sort(key=lambda item: (item.rerank_score, item.score), reverse=True)
            return reranked[:top_k]
        except Exception as e:  # noqa: BLE001 - remote ranker is strictly best-effort
            logger.warning("rerank_fn failed (%s); retaining local memory ranking", e)
            return scored[:top_k]

    def build_background_block(
        self, user_id: str, query: str, top_k: int = 5
    ) -> str:
        """构造注入 engine 的"用户背景"文本 (无记忆返回空串)。"""
        hits = self.recall(user_id, query, top_k=top_k)
        hits = [h for h in hits if h.score > 0]
        if not hits:
            return ""
        lines = [f"- {h.memory.content}" for h in hits]
        return "\n".join(lines)

    def list_memories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM user_memories
                   WHERE user_id = ? AND kind != 'sentinel'
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [self._row_to_mem(r).to_public() for r in rows]

    def _row_to_mem(self, row: sqlite3.Row) -> UserMemory:
        try:
            emb = json.loads(row["embedding"]) if row["embedding"] else []
        except (ValueError, json.JSONDecodeError):
            emb = []
        return UserMemory(
            id=row["id"], user_id=row["user_id"], content=row["content"],
            kind=row["kind"], importance=row["importance"], embedding=emb,
            created_at=row["created_at"], expires_at=row["expires_at"],
            idempotency_key=row["idempotency_key"],
            superseded_at=row["superseded_at"] if "superseded_at" in row.keys() else None,
        )

    def close(self) -> None:
        if self._persistent_conn is not None:
            try:
                self._persistent_conn.close()
            finally:
                self._persistent_conn = None
