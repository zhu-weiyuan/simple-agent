# -*- coding: utf-8 -*-
"""Small timeout/deadline helpers inspired by dsh-timeout.

The helper classifies timeout versus cancellation. The caller still owns the
actual cancellation mechanism (subprocess/process termination, HTTP close, or
thread cancellation).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimeoutReason:
    code: str
    timeout_seconds: float


class Deadline:
    def __init__(self, timeout_seconds: float, code: str = "TOOL_TIMEOUT") -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.code = code
        self.timeout_seconds = float(timeout_seconds)
        self._timed_out = False
        self._cancelled = False
        self._timer = threading.Timer(self.timeout_seconds, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def _on_timeout(self) -> None:
        self._timed_out = True

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    @property
    def cancelled(self) -> bool:
        return self._cancelled and not self._timed_out

    @property
    def reason(self) -> Optional[TimeoutReason]:
        if not self._timed_out:
            return None
        return TimeoutReason(self.code, self.timeout_seconds)

    def cancel(self) -> None:
        self._cancelled = True
        self._timer.cancel()

    def close(self) -> None:
        self._timer.cancel()

    def __enter__(self) -> "Deadline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


__all__ = ["Deadline", "TimeoutReason"]
