# -*- coding: utf-8 -*-
"""
Observability: Trace + Metrics + Alerts for SimpleAgent

Production-grade observability layer with:
1. MetricsCollector - Prometheus-style counters, histograms, gauges
2. AlertService - Threshold-based alerting with cooldown
"""

import json
import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional
from datetime import datetime


class MetricsCollector:
    """
    Production-grade metrics collector (Prometheus style).

    Supports: Counter, Histogram (latency buckets), Gauge (instantaneous)
    All metrics support labels for dimensional aggregation.
    """

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, dict] = {}
        self._gauges: Dict[str, float] = {}
        # P6: bounded event buffer to prevent unbounded memory growth
        self._events: "deque[Dict[str, object]]" = deque(maxlen=5000)
        self._lock = threading.Lock()
        # (metric_name -> deque[(ts, value)]) raw observations for sliding windows
        self._observations: Dict[str, deque] = {}
        self._default_buckets = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    def increment_counter(self, name: str, labels: Dict[str, str] = None) -> None:
        key = self._build_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1

    def observe_histogram(self, name: str, value_ms: float,
                          labels: Dict[str, str] = None,
                          buckets: List[float] = None) -> None:
        key = self._build_key(name, labels)
        bucket_list = buckets or self._default_buckets
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = {b: 0 for b in bucket_list}
                self._histograms[key]['total'] = 0
                self._histograms[key]['sum'] = 0.0

            hist = self._histograms[key]
            hist['total'] += 1
            hist['sum'] += value_ms
            obs = self._observations.setdefault(key, deque(maxlen=10000))
            obs.append((time.time(), value_ms))
            for bucket in bucket_list:
                if value_ms <= bucket:
                    hist[bucket] += 1

    def record_event(self, event_name: str, request_id: str = "", duration_ms: float = 0, **extra) -> None:
        with self._lock:
            self._events.append({"event_name": event_name, "request_id": request_id, "duration_ms": duration_ms, **extra})

    def get_events(self) -> List[Dict[str, object]]:
        with self._lock:
            return list(self._events)

    def window_average(self, metric_name: str, window_seconds: int) -> Optional[float]:
        """P6: 真滑动窗口 — 返回 window 内该指标观测值的平均 (无观测返回 None)。"""
        cutoff = time.time() - window_seconds
        with self._lock:
            values = []
            for key, obs in self._observations.items():
                if metric_name in key:
                    values.extend(v for ts, v in obs if ts >= cutoff)
        if not values:
            return None
        return sum(values) / len(values)

    def set_gauge(self, name: str, value: float,
                  labels: Dict[str, str] = None) -> None:
        key = self._build_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            metrics = {
                'counters': dict(self._counters),
                'histograms': {},
                'gauges': dict(self._gauges),
            }
            for key, hist in self._histograms.items():
                total = hist.get('total', 0)
                sum_val = hist.get('sum', 0.0)
                avg = sum_val / total if total > 0 else 0
                metrics['histograms'][key] = {
                    'count': total,
                    'sum_ms': round(sum_val, 2),
                    'avg_ms': round(avg, 2),
                    'buckets': {str(b): c for b, c in hist.items()
                                if b not in ('total', 'sum')},
                }
            return metrics

    def get_prometheus_text(self) -> str:
        """合法 Prometheus 文本: # TYPE 行只含基础指标名 (不带 label),每个基础名只输出一次。"""
        lines = []
        typed = set()

        def base_name(key: str) -> str:
            return key.split("{", 1)[0]

        for key, value in sorted(self._counters.items()):
            b = base_name(key)
            if b not in typed:
                lines.append(f"# TYPE {b} counter")
                typed.add(b)
            lines.append(f"{key} {value}")

        for key, hist in sorted(self._histograms.items()):
            b = base_name(key)
            if b not in typed:
                lines.append(f"# TYPE {b} histogram")
                typed.add(b)
            total = hist.get('total', 0)
            sum_val = hist.get('sum', 0.0)
            # 带 label 的 key: 合法格式要求 base_bucket{labels,le=...}
            if "{" in key:
                inner = key[key.index("{") + 1:key.rindex("}")]
                label_prefix = inner + ","
            else:
                label_prefix = ""
            for bucket in sorted([k for k in hist.keys()
                                  if k not in ('total', 'sum')]):
                lines.append(f'{b}_bucket{{{label_prefix}le="{bucket}"}} {hist[bucket]}')
            lines.append(f'{b}_bucket{{{label_prefix}le="+Inf"}} {total}')
            lines.append(f'{b}_count{{{inner}}} {total}' if "{" in key else f'{b}_count {total}')
            lines.append(f'{b}_sum{{{inner}}} {sum_val:.2f}' if "{" in key else f'{b}_sum {sum_val:.2f}')

        for key, value in sorted(self._gauges.items()):
            b = base_name(key)
            if b not in typed:
                lines.append(f"# TYPE {b} gauge")
                typed.add(b)
            lines.append(f"{key} {value}")

        return "\n".join(lines)

    def _build_key(self, name: str, labels: Dict[str, str] = None) -> str:
        if not labels:
            return name
        label_str = ",".join(
            f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


class AlertService:
    """Threshold-based alerting with cooldown to prevent alert fatigue."""

    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self._alerts_triggered: Dict[str, float] = {}
        self._rules: List[Dict[str, Any]] = []
        self._callbacks: Dict[str, callable] = {}

    def add_rule(self, name: str, metric_name: str, threshold: float,
                 window_seconds: int = 300,
                 cooldown_seconds: int = 600) -> None:
        self._rules.append({
            'name': name,
            'metric_name': metric_name,
            'threshold': threshold,
            'window': window_seconds,
            'cooldown': cooldown_seconds,
        })

    def on_alert(self, name: str, callback: callable) -> None:
        """Register a callback for an alert."""
        self._callbacks[name] = callback

    def check_and_alert(self) -> List[str]:
        now = time.time()
        triggered = []

        for rule in self._rules:
            name = rule['name']
            if name in self._alerts_triggered:
                elapsed = now - self._alerts_triggered[name]
                if elapsed < rule['cooldown']:
                    continue

            # P6: 真滑动窗口 — 只统计 rule['window'] 秒内的观测,
            # 而非自进程启动以来的全量平均。
            avg_latency = self.metrics.window_average(
                rule['metric_name'], rule['window'])

            if avg_latency is not None and avg_latency > rule['threshold']:
                self._alerts_triggered[name] = now
                triggered.append(name)
                print(f"[Alert] {name}: avg_latency={avg_latency:.2f}ms "
                      f"> threshold={rule['threshold']}ms")
                cb = self._callbacks.get(name)
                if cb:
                    try:
                        cb(name, avg_latency, rule['threshold'])
                    except Exception as e:
                        print(f"[Alert] Callback error for {name}: {e}")

        return triggered


# Global singleton
_metrics = MetricsCollector()
_alerts = AlertService(_metrics)


def get_metrics() -> MetricsCollector:
    return _metrics


def get_alerts() -> AlertService:
    return _alerts
