"""Append-only, JSONL session events for recovery, diagnostics and evaluation."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]+")


class SessionEventLog:
    def __init__(self, root: str | Path = "runtime/session-events") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._seq: Dict[str, int] = {}

    @staticmethod
    def _safe_session(session_id: str) -> str:
        value = _SAFE_SEGMENT.sub("-", str(session_id or "default")).strip(".-")
        return value[:80] or "default"

    def _path(self, session_id: str) -> Path:
        return self.root / f"{self._safe_session(session_id)}.jsonl"

    def append(self, session_id: str, event_type: str, payload: Dict[str, Any] | None = None,
               trace_id: str = "") -> Dict[str, Any]:
        safe_id = self._safe_session(session_id)
        with self._lock:
            seq = self._seq.get(safe_id)
            if seq is None:
                path = self._path(safe_id)
                try:
                    seq = sum(1 for _ in path.open("r", encoding="utf-8")) if path.exists() else 0
                except OSError:
                    seq = 0
            seq += 1
            self._seq[safe_id] = seq
            event = {"event_id": f"evt-{uuid.uuid4().hex}", "seq": seq,
                     "timestamp": time.time(), "session_id": safe_id,
                     "trace_id": trace_id or "", "type": event_type,
                     "payload": payload or {}}
            path = self._path(safe_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return event

    def _read_rows(self, path: Path) -> tuple[List[Dict[str, Any]], int]:
        rows: List[Dict[str, Any]] = []
        malformed = 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
                    else:
                        malformed += 1
                except json.JSONDecodeError:
                    malformed += 1
        except OSError:
            return [], 0
        return rows, malformed

    def list(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        rows, _ = self._read_rows(path)
        return rows[-min(max(1, int(limit)), 1000):]

    def list_page(self, session_id: str, limit: int = 100, before_seq: int | None = None) -> Dict[str, Any]:
        """Read a stable, newest-first event page and report damaged JSONL rows."""
        path = self._path(session_id)
        if not path.exists():
            return {"events": [], "next_before_seq": None, "malformed_lines": 0}
        rows, malformed = self._read_rows(path)
        if before_seq is not None:
            try:
                cursor = max(0, int(before_seq))
                rows = [row for row in rows if int(row.get("seq", 0) or 0) < cursor]
            except (TypeError, ValueError):
                raise ValueError("before_seq 必须是正整数")
        rows.sort(key=lambda row: int(row.get("seq", 0) or 0), reverse=True)
        page = rows[:min(max(1, int(limit)), 1000)]
        next_before_seq = int(page[-1].get("seq", 0) or 0) if len(rows) > len(page) and page else None
        return {"events": page, "next_before_seq": next_before_seq, "malformed_lines": malformed}

    def stats(self) -> Dict[str, Any]:
        session_files = list(self.root.glob("*.jsonl"))
        total_bytes = 0
        oldest_mtime: float | None = None
        newest_mtime: float | None = None
        for path in session_files:
            try:
                info = path.stat()
            except OSError:
                continue
            total_bytes += max(0, int(info.st_size))
            oldest_mtime = info.st_mtime if oldest_mtime is None else min(oldest_mtime, info.st_mtime)
            newest_mtime = info.st_mtime if newest_mtime is None else max(newest_mtime, info.st_mtime)
        return {
            "session_files": len(session_files),
            "total_bytes": total_bytes,
            "oldest_mtime": oldest_mtime,
            "newest_mtime": newest_mtime,
        }

    def cleanup(self, max_age_seconds: float = 0, max_total_bytes: int = 0) -> Dict[str, Any]:
        """Delete whole, inactive session logs by age and then by oldest-first capacity."""
        max_age_seconds = max(0.0, float(max_age_seconds))
        max_total_bytes = max(0, int(max_total_bytes))
        now = time.time()
        with self._lock:
            entries = []
            for path in self.root.glob("*.jsonl"):
                try:
                    info = path.stat()
                except OSError:
                    continue
                entries.append((path, float(info.st_mtime), max(0, int(info.st_size))))
            entries.sort(key=lambda row: (row[1], row[0].name))
            selected: dict[Path, str] = {}
            if max_age_seconds:
                cutoff = now - max_age_seconds
                for path, mtime, _ in entries:
                    if mtime < cutoff:
                        selected[path] = "expired"
            remaining_bytes = sum(size for path, _, size in entries if path not in selected)
            if max_total_bytes and remaining_bytes > max_total_bytes:
                for path, _, size in entries:
                    if path in selected:
                        continue
                    selected[path] = "capacity"
                    remaining_bytes -= size
                    if remaining_bytes <= max_total_bytes:
                        break
            removed = removed_bytes = 0
            reasons = {"expired": 0, "capacity": 0}
            for path, reason in selected.items():
                try:
                    size = path.stat().st_size if path.exists() else 0
                    path.unlink(missing_ok=True)
                except OSError:
                    continue
                self._seq.pop(path.stem, None)
                removed += 1
                removed_bytes += max(0, int(size))
                reasons[reason] += 1
        return {
            "removed": removed,
            "removed_bytes": removed_bytes,
            "expired_removed": reasons["expired"],
            "capacity_removed": reasons["capacity"],
            "remaining": self.stats(),
        }


__all__ = ["SessionEventLog"]
