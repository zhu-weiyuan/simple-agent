# -*- coding: utf-8 -*-
"""File observation and optimistic-concurrency guards for mutable tools.

An edit/write must be based on an observation made during the same session.
The observed file fingerprint becomes a version precondition, preventing a
model from silently overwriting an intervening external change.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Optional


@dataclass(frozen=True)
class FileObservation:
    path: str
    version: str
    exists: bool


class FileObservationStore:
    def __init__(self) -> None:
        self._observations: dict[tuple[str, str], FileObservation] = {}
        self._lock = RLock()

    @staticmethod
    def version_of(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(64 * 1024), b""):
                    digest.update(block)
        except OSError:
            return None
        return digest.hexdigest()

    def observe(self, session_id: str, path: Path) -> FileObservation:
        observation = FileObservation(
            path=str(path.resolve()),
            version=self.version_of(path) or "",
            exists=path.exists(),
        )
        with self._lock:
            self._observations[(session_id or "default", observation.path)] = observation
        return observation

    def require_current(self, session_id: str, path: Path, *, allow_create: bool = False) -> Optional[str]:
        key = (session_id or "default", str(path.resolve()))
        with self._lock:
            observation = self._observations.get(key)
        if observation is None:
            if allow_create and not path.exists():
                return None
            return "安全限制:修改前必须在当前会话中先读取目标文件"
        if not observation.exists:
            if allow_create and not path.exists():
                return None
            return "安全限制:文件观察状态已失效，请重新读取目标路径"
        current = self.version_of(path)
        if current != observation.version:
            return "安全限制:文件自读取后已变化，请重新读取后再修改"
        return None

    def forget(self, session_id: str, path: Path) -> None:
        with self._lock:
            self._observations.pop((session_id or "default", str(path.resolve())), None)


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file in its existing directory."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


__all__ = ["FileObservation", "FileObservationStore", "atomic_write_text"]
