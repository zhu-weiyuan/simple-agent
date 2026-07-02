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
        self._lock = threading.Lock()
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
            for bucket in bucket_list:
                if value_ms <= bucket:
                    hist[bucket] += 1

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
        lines = []
        for key, value in self._counters.items():
            lines.append(f"# TYPE {key} counter")
            lines.append(f"{key} {value}")

        for key, hist in self._histograms.items():
            lines.append(f"# TYPE {key} histogram")
            total = hist.get('total', 0)
            sum_val = hist.get('sum', 0.0)
            for bucket in sorted([b for b in hist.keys()
                                  if b not in ('total', 'sum')]):
                lines.append(f'{key}_bucket{{le="{bucket}"}} {hist[bucket]}')
            lines.append(f'{key}_bucket{{le="+Inf"}} {total}')
            lines.append(f'{key}_count {total}')
            lines.append(f'{key}_sum {sum_val:.2f}')

        for key, value in self._gauges.items():
            lines.append(f"# TYPE {key} gauge")
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

            metrics = self.metrics.get_metrics()
            avg_latency = None
            for key, hist in metrics['histograms'].items():
                if rule['metric_name'] in key:
                    avg_latency = hist.get('avg_ms', 0)
                    break

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
