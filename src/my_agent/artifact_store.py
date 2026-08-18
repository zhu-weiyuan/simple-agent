"""Private session-scoped storage for oversized tool results."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    session_id: str
    tool_name: str
    path: str
    size_bytes: int
    created_at: float


class ArtifactStore:
    """Stores full text privately and returns bounded, model-safe previews."""

    def __init__(self, root: str | Path = "runtime/artifacts", max_inline_bytes: int = 12_000,
                 max_artifact_bytes: int = 5_000_000) -> None:
        self.root = Path(root).resolve()
        self.max_inline_bytes = max(512, int(max_inline_bytes))
        self.max_artifact_bytes = max(self.max_inline_bytes, int(max_artifact_bytes))
        self._index_dir = self.root / "index"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(value: str, fallback: str) -> str:
        cleaned = _SAFE_SEGMENT.sub("-", str(value or "")).strip(".-")
        return cleaned[:80] or fallback

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)

    def save_text(self, text: str, session_id: str, tool_name: str) -> ArtifactRecord:
        raw = str(text)
        encoded = raw.encode("utf-8", errors="replace")
        if len(encoded) > self.max_artifact_bytes:
            raw = encoded[:self.max_artifact_bytes].decode("utf-8", errors="ignore")
            raw += "\n\n[完整工具结果超过归档上限，后续内容未保存。]"
            encoded = raw.encode("utf-8")
        aid = f"art-{uuid.uuid4().hex}"
        safe_session = self._safe(session_id, "default")
        safe_tool = self._safe(tool_name, "tool")
        data_path = self.root / safe_session / f"{aid}-{safe_tool}.txt"
        record = ArtifactRecord(aid, safe_session, safe_tool, str(data_path), len(encoded), time.time())
        with self._lock:
            self._atomic_write(data_path, raw)
            self._atomic_write(self._index_dir / f"{aid}.json", json.dumps(asdict(record), ensure_ascii=False))
        return record

    def retain(self, text: str, session_id: str, tool_name: str) -> tuple[str, Optional[ArtifactRecord]]:
        raw = str(text)
        encoded = raw.encode("utf-8", errors="replace")
        if len(encoded) <= self.max_inline_bytes:
            return raw, None
        record = self.save_text(raw, session_id, tool_name)
        notice_template = (
            "\n\n（工具输出过大，已省略 {omitted} 字节；完整结果归档为 {artifact_id}。"
            "可调用 read_artifact 分页读取，或调用 search_artifact 搜索。）"
        )
        # Allocate the notice first. A byte budget matters more than character count.
        omitted = max(0, len(encoded) - self.max_inline_bytes)
        notice = notice_template.format(omitted=omitted, artifact_id=record.artifact_id)
        budget = max(0, self.max_inline_bytes - len(notice.encode("utf-8")))
        if budget <= 0:
            return notice.encode("utf-8")[:self.max_inline_bytes].decode("utf-8", errors="ignore"), record
        head_budget = budget // 2
        tail_budget = budget - head_budget
        head = encoded[:head_budget].decode("utf-8", errors="ignore")
        tail = encoded[-tail_budget:].decode("utf-8", errors="ignore")
        preview = f"{head}\n…（中间内容已归档）…\n{tail}{notice}"
        while len(preview.encode("utf-8")) > self.max_inline_bytes and tail:
            tail = tail[:-1]
            preview = f"{head}\n…（中间内容已归档）…\n{tail}{notice}"
        return preview, record

    def _record(self, artifact_id: str) -> ArtifactRecord:
        if not re.fullmatch(r"art-[0-9a-f]{32}", str(artifact_id or "")):
            raise KeyError("artifact_id 格式无效")
        index = self._index_dir / f"{artifact_id}.json"
        if not index.exists():
            raise KeyError("artifact 不存在或已清理")
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            record = ArtifactRecord(**data)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise KeyError("artifact 元数据不可读取") from exc
        target = Path(record.path).resolve()
        if self.root not in target.parents:
            raise KeyError("artifact 路径非法")
        return record

    def describe(self, artifact_id: str, session_id: str = "") -> Dict[str, Any]:
        record = self._record(artifact_id)
        if session_id and self._safe(session_id, "default") != record.session_id:
            raise PermissionError("artifact 不属于当前会话")
        return asdict(record)

    def read(self, artifact_id: str, offset: int = 0, limit: int = 12_000,
             session_id: str = "") -> Dict[str, Any]:
        record = self._record(artifact_id)
        if session_id and self._safe(session_id, "default") != record.session_id:
            raise PermissionError("artifact 不属于当前会话")
        text = Path(record.path).read_text(encoding="utf-8", errors="replace")
        offset = max(0, int(offset))
        limit = min(max(1, int(limit)), 100_000)
        chunk = text[offset:offset + limit]
        return {"artifact_id": artifact_id, "offset": offset, "limit": limit,
                "total_chars": len(text), "has_more": offset + len(chunk) < len(text),
                "content": chunk}

    def search(self, artifact_id: str, query: str, session_id: str = "", limit: int = 50) -> Dict[str, Any]:
        record = self._record(artifact_id)
        if session_id and self._safe(session_id, "default") != record.session_id:
            raise PermissionError("artifact 不属于当前会话")
        needle = str(query or "").strip().lower()
        if not needle:
            raise ValueError("query 不能为空")
        matches: List[Dict[str, Any]] = []
        for number, line in enumerate(Path(record.path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if needle in line.lower():
                matches.append({"line": number, "content": line[:1000]})
                if len(matches) >= min(max(1, int(limit)), 200):
                    break
        return {"artifact_id": artifact_id, "query": query, "matches": matches, "count": len(matches)}

    def stats(self) -> Dict[str, Any]:
        """Return bounded-storage facts without exposing artifact content."""
        total_bytes = 0
        count = 0
        malformed = 0
        oldest_created_at: Optional[float] = None
        newest_created_at: Optional[float] = None
        with self._lock:
            for index in self._index_dir.glob("art-*.json"):
                try:
                    record = ArtifactRecord(**json.loads(index.read_text(encoding="utf-8")))
                    target = Path(record.path).resolve()
                    if self.root not in target.parents:
                        raise ValueError("artifact path escapes store")
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    malformed += 1
                    continue
                count += 1
                total_bytes += max(0, int(record.size_bytes))
                oldest_created_at = record.created_at if oldest_created_at is None else min(oldest_created_at, record.created_at)
                newest_created_at = record.created_at if newest_created_at is None else max(newest_created_at, record.created_at)
        return {
            "count": count,
            "total_bytes": total_bytes,
            "malformed_index_count": malformed,
            "oldest_created_at": oldest_created_at,
            "newest_created_at": newest_created_at,
        }

    def cleanup(self, max_age_seconds: float = 0, max_total_bytes: int = 0) -> Dict[str, Any]:
        """Remove expired artifacts, then oldest artifacts until the byte cap fits.

        A non-positive limit disables the corresponding policy. Only index records
        whose data path resolves inside this store are eligible for deletion.
        """
        max_age_seconds = max(0.0, float(max_age_seconds))
        max_total_bytes = max(0, int(max_total_bytes))
        now = time.time()
        records: list[ArtifactRecord] = []
        malformed = 0
        with self._lock:
            for index in self._index_dir.glob("art-*.json"):
                try:
                    record = ArtifactRecord(**json.loads(index.read_text(encoding="utf-8")))
                    target = Path(record.path).resolve()
                    if self.root not in target.parents:
                        raise ValueError("artifact path escapes store")
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    malformed += 1
                    continue
                records.append(record)

            records.sort(key=lambda row: (row.created_at, row.artifact_id))
            selected: dict[str, str] = {}
            if max_age_seconds:
                cutoff = now - max_age_seconds
                for record in records:
                    if record.created_at < cutoff:
                        selected[record.artifact_id] = "expired"

            remaining_bytes = sum(max(0, int(row.size_bytes)) for row in records
                                  if row.artifact_id not in selected)
            if max_total_bytes and remaining_bytes > max_total_bytes:
                for record in records:
                    if record.artifact_id in selected:
                        continue
                    selected[record.artifact_id] = "capacity"
                    remaining_bytes -= max(0, int(record.size_bytes))
                    if remaining_bytes <= max_total_bytes:
                        break

            removed = 0
            removed_bytes = 0
            reasons = {"expired": 0, "capacity": 0}
            by_id = {record.artifact_id: record for record in records}
            for artifact_id, reason in selected.items():
                record = by_id[artifact_id]
                target = Path(record.path).resolve()
                index = self._index_dir / f"{artifact_id}.json"
                try:
                    if target.exists():
                        target.unlink()
                    if index.exists():
                        index.unlink()
                except OSError:
                    continue
                removed += 1
                removed_bytes += max(0, int(record.size_bytes))
                reasons[reason] += 1

        stats = self.stats()
        return {
            "removed": removed,
            "removed_bytes": removed_bytes,
            "expired_removed": reasons["expired"],
            "capacity_removed": reasons["capacity"],
            "malformed_index_count": malformed,
            "remaining": stats,
        }


__all__ = ["ArtifactRecord", "ArtifactStore"]
