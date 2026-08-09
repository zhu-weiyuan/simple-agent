# -*- coding: utf-8 -*-
"""
my_agent.cost_tracker — Usage tracking and cost attribution

This module provides:
- UsageRecord: Individual usage record
- CostTracker: Main tracker for aggregating usage and costs
- CostSummary: Aggregated cost summary
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

# ── Price table ─────────────────────────────────────────────
# 单位: USD / 1M tokens。以各官网最新公布为准 (price_version 追踪)。
PRICE_VERSION = "2026-07"
DEFAULT_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o":            {"input_rate": 2.50,  "output_rate": 10.00},
    "gpt-4o-mini":       {"input_rate": 0.15,  "output_rate": 0.60},
    "gpt-4.1":           {"input_rate": 2.00,  "output_rate": 8.00},
    "gpt-4.1-mini":      {"input_rate": 0.40,  "output_rate": 1.60},
    "o3-mini":           {"input_rate": 1.10,  "output_rate": 4.40},
    "claude-3-5-haiku":  {"input_rate": 0.80,  "output_rate": 4.00},
    "claude-sonnet-4":   {"input_rate": 3.00,  "output_rate": 15.00},
    "claude-opus-4":     {"input_rate": 15.00, "output_rate": 75.00},
    "deepseek-chat":     {"input_rate": 0.27,  "output_rate": 1.10},
    "deepseek-reasoner": {"input_rate": 0.55,  "output_rate": 2.19},
    "qwen-plus":         {"input_rate": 0.40,  "output_rate": 1.20},
    "qwen-turbo":        {"input_rate": 0.05,  "output_rate": 0.20},
}


@dataclass
class UsageRecord:
    """Individual usage record for a single request"""
    id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    
    # Context metadata
    budget_id: Optional[str] = None
    tenant_id: Optional[str] = None
    scene: Optional[str] = None
    user_id: Optional[str] = None
    request_type: Optional[str] = None
    
    # Timing
    created_at: float = field(default_factory=time.time)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("UsageRecord requires an id")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("Token counts must be non-negative")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")


@dataclass
class CostSummary:
    """Aggregated cost summary for a period or dimension"""
    period_start: float
    period_end: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    currency: str = "USD"
    request_count: int = 0
    
    # Breakdown by model
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Breakdown by tenant
    by_tenant: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Breakdown by scene
    by_scene: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def average_cost_per_request(self) -> float:
        return self.total_cost / self.request_count if self.request_count > 0 else 0.0
    
    @property
    def average_tokens_per_request(self) -> float:
        return self.total_tokens / self.request_count if self.request_count > 0 else 0.0


class CostTracker:
    """Tracker for LLM usage and costs with multi-dimensional aggregation.

    通过 :meth:`make_llm_end_hook` 挂到 engine 的 ``HookPoint.LLM_END`` 记账;
    可选 ``db_path`` 时把记录落到 SQLite ``cost_records`` 表。
    """

    price_version = PRICE_VERSION

    def __init__(self, db_path: Optional[str] = None, flush_every: int = 200,
                 load_default_prices: bool = True):
        self.records: List[UsageRecord] = []
        self.model_costs: Dict[str, Dict[str, float]] = (
            {k: dict(v) for k, v in DEFAULT_PRICE_TABLE.items()}
            if load_default_prices else {})
        self.tenant_budgets: Dict[str, float] = {}  # tenant_id -> monthly_budget
        self.alert_thresholds: Dict[str, float] = {}  # tenant_id -> alert_threshold percentage
        # RLock allows record_usage to trigger flush_to_sqlite safely.
        self._lock = threading.RLock()
        self.db_path = db_path
        self.flush_every = max(1, flush_every)
        self._unflushed = 0
        if db_path:
            try:
                self._ensure_cost_table()
            except Exception as e:  # noqa: BLE001 — 记账失败不能拖垮启动
                logger.warning("cost_records table init failed: %s", e)

    # ── SQLite persistence ───────────────────────────────────

    def _ensure_cost_table(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_records (
                    id TEXT PRIMARY KEY,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_cost REAL,
                    currency TEXT,
                    budget_id TEXT,
                    tenant_id TEXT,
                    scene TEXT,
                    user_id TEXT,
                    price_version TEXT,
                    created_at REAL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def flush_to_sqlite(self) -> int:
        """把内存记录写入 SQLite (INSERT OR IGNORE 按 id 去重)。"""
        if not self.db_path:
            return 0
        with self._lock:
            snapshot = list(self.records)
            self._unflushed = 0
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO cost_records "
                "(id, model, input_tokens, output_tokens, total_cost, currency, "
                " budget_id, tenant_id, scene, user_id, price_version, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(r.id, r.model, r.input_tokens, r.output_tokens, r.total_cost,
                  r.currency, r.budget_id, r.tenant_id, r.scene, r.user_id,
                  self.price_version, r.created_at) for r in snapshot],
            )
            conn.commit()
            return len(snapshot)
        finally:
            conn.close()

    # ── engine wiring ────────────────────────────────────────

    def make_llm_end_hook(self, default_model: str = "unknown"):
        """返回可注册到 ``HookPoint.LLM_END`` 的 handler: 从 usage 记账。

        hook data 里的 ``context`` (QueryContext) 提供 tenant/scene/user 归因维度,
        ``model`` 为本次**实际使用**的模型 (可能是 fallback/降级后的那个)。
        """
        def _on_llm_end(hook_ctx, **kwargs):
            data = kwargs or getattr(hook_ctx, "data", {}) or {}
            data = data.get("data", data)
            usage = data.get("usage") or {}
            if not usage:
                return None
            ctx = data.get("context")
            record = self.record_usage(
                model=data.get("model") or default_model,
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                budget_id=getattr(ctx, "tenant_id", None),
                tenant_id=getattr(ctx, "tenant_id", None),
                scene=getattr(ctx, "scene", None),
                user_id=getattr(ctx, "user_id", None),
                request_type="chat",
                metadata={"fallback_reason": data.get("fallback_reason")}
                if data.get("fallback_reason") else None,
            )
            with self._lock:
                self._unflushed += 1
                due = self.db_path and self._unflushed >= self.flush_every
            if due:
                try:
                    self.flush_to_sqlite()
                except Exception as e:  # noqa: BLE001
                    logger.warning("cost flush failed: %s", e)
            return record
        return _on_llm_end


    def set_model_cost(self, model_name: str, input_per_1m: float, output_per_1m: float):
        """Set pricing for a model (per 1M tokens)"""
        self.model_costs[model_name] = {
            "input_rate": input_per_1m,
            "output_rate": output_per_1m
        }
    
    def get_model_cost(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get pricing for a model"""
        return self.model_costs.get(model_name)
    
    def set_tenant_budget(self, tenant_id: str, budget: float, alert_threshold: float = 0.8):
        """Set monthly budget and alert threshold for a tenant"""
        self.tenant_budgets[tenant_id] = budget
        self.alert_thresholds[tenant_id] = alert_threshold
    
    def record_usage(self, 
                    model: str,
                    input_tokens: int,
                    output_tokens: int,
                    budget_id: Optional[str] = None,
                    tenant_id: Optional[str] = None,
                    scene: Optional[str] = None,
                    user_id: Optional[str] = None,
                    request_type: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> UsageRecord:
        """Record a single usage event
        
        Returns:
            The created UsageRecord
        """
        # Calculate costs
        model_cost = self.get_model_cost(model) or {"input_rate": 0.0, "output_rate": 0.0}
        
        input_cost = (input_tokens / 1_000_000) * model_cost["input_rate"]
        output_cost = (output_tokens / 1_000_000) * model_cost["output_rate"]
        total_cost = input_cost + output_cost
        
        # Create record
        record_id = f"usec_{int(time.time() * 1000000)}"
        record = UsageRecord(
            id=record_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            budget_id=budget_id,
            tenant_id=tenant_id,
            scene=scene,
            user_id=user_id,
            request_type=request_type,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.records.append(record)
            self._unflushed += 1
            # Auto-flush when threshold is reached (avoids silent data loss).
            if self.db_path and self._unflushed >= self.flush_every:
                try:
                    self.flush_to_sqlite()
                except Exception as e:  # noqa: BLE001
                    logger.warning("cost flush failed: %s", e)
        return record
    
    def record_batch(self, records_data: List[Dict[str, Any]]) -> List[UsageRecord]:
        """Record multiple usage events at once
        
        Args:
            records_data: List of dicts with keys matching UsageRecord fields
            
        Returns:
            List of created UsageRecords
        """
        return [self.record_usage(**data) for data in records_data]
    
    def get_records(self, 
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   model: Optional[str] = None,
                   tenant_id: Optional[str] = None,
                   budget_id: Optional[str] = None,
                   user_id: Optional[str] = None) -> List[UsageRecord]:
        """Get filtered usage records"""
        filtered = self.records
        
        if start_time:
            filtered = [r for r in filtered if r.created_at >= start_time]
        if end_time:
            filtered = [r for r in filtered if r.created_at <= end_time]
        if model:
            filtered = [r for r in filtered if r.model == model]
        if tenant_id:
            filtered = [r for r in filtered if r.tenant_id == tenant_id]
        if budget_id:
            filtered = [r for r in filtered if r.budget_id == budget_id]
        if user_id:
            filtered = [r for r in filtered if r.user_id == user_id]
            
        return filtered
    
    def get_summary(self,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   group_by: Optional[str] = None) -> CostSummary:
        """Get aggregated cost summary
        
        Args:
            start_time: Start of period (timestamp)
            end_time: End of period (timestamp)
            group_by: Optional dimension to group by ("model", "tenant", "scene")
            
        Returns:
            CostSummary with totals and optional breakdowns
        """
        filtered_records = self.get_records(start_time, end_time)
        
        summary = CostSummary(
            period_start=start_time or 0,
            period_end=end_time or time.time(),
            currency="USD"
        )
        
        for record in filtered_records:
            summary.total_input_tokens += record.input_tokens
            summary.total_output_tokens += record.output_tokens
            summary.total_tokens += record.total_tokens
            summary.total_cost += record.total_cost
            summary.request_count += 1
            
            # Group by dimension
            if group_by == "model":
                if record.model not in summary.by_model:
                    summary.by_model[record.model] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost": 0.0,
                        "requests": 0
                    }
                by_dim = summary.by_model[record.model]
            elif group_by == "tenant":
                key = record.tenant_id or "unassigned"
                if key not in summary.by_tenant:
                    summary.by_tenant[key] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost": 0.0,
                        "requests": 0
                    }
                by_dim = summary.by_tenant[key]
            elif group_by == "scene":
                key = record.scene or "unknown"
                if key not in summary.by_scene:
                    summary.by_scene[key] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost": 0.0,
                        "requests": 0
                    }
                by_dim = summary.by_scene[key]
            else:
                continue
            
            by_dim["input_tokens"] += record.input_tokens
            by_dim["output_tokens"] += record.output_tokens
            by_dim["total_tokens"] += record.total_tokens
            by_dim["cost"] += record.total_cost
            by_dim["requests"] += 1
        
        return summary
    
    def get_tenant_costs(self, tenant_id: str, 
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None) -> Dict[str, Any]:
        """Get cost details for a specific tenant
        
        Returns:
            Dictionary with cost breakdown and budget status
        """
        summary = self.get_summary(start_time, end_time, group_by="tenant")
        tenant_data = summary.by_tenant.get(tenant_id, {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "requests": 0
        })
        
        result = {
            "tenant_id": tenant_id,
            **tenant_data,
            "currency": "USD"
        }
        
        # Add budget info if available
        if tenant_id in self.tenant_budgets:
            budget = self.tenant_budgets[tenant_id]
            result["budget"] = {
                "monthly_limit": budget,
                "used": tenant_data["cost"],
                "remaining": max(0, budget - tenant_data["cost"]),
                "usage_percent": min(1.0, tenant_data["cost"] / budget) if budget > 0 else 0.0
            }
            
            # Check alert threshold
            if tenant_id in self.alert_thresholds:
                threshold = self.alert_thresholds[tenant_id]
                if result["budget"]["usage_percent"] >= threshold:
                    result["alert"] = {
                        "threshold": threshold,
                        "message": f"Budget usage at {result['budget']['usage_percent']*100:.1f}%"
                    }
        
        return result
    
    def get_model_breakdown(self,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """Get usage breakdown by model"""
        summary = self.get_summary(start_time, end_time, group_by="model")
        
        result = {}
        for model, data in summary.by_model.items():
            result[model] = {
                "input_tokens": data["input_tokens"],
                "output_tokens": data["output_tokens"],
                "total_tokens": data["total_tokens"],
                "cost": data["cost"],
                "requests": data["requests"],
                "avg_tokens_per_request": data["total_tokens"] / data["requests"] if data["requests"] > 0 else 0,
                "avg_cost_per_request": data["cost"] / data["requests"] if data["requests"] > 0 else 0
            }
        
        return result
    
    def check_budget_alerts(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Check all tenants against their budgets and return alerts
        
        Args:
            tenant_id: Optional specific tenant to check
            
        Returns:
            List of alert dictionaries for tenants over threshold
        """
        alerts = []
        tenants_to_check = [tenant_id] if tenant_id else list(self.tenant_budgets.keys())
        
        current_time = time.time()
        # Approximate month start (30 days ago)
        month_ago = current_time - (30 * 24 * 60 * 60)
        
        for tid in tenants_to_check:
            tenant_info = self.get_tenant_costs(tid, month_ago, current_time)
            
            if "budget" in tenant_info:
                budget = tenant_info["budget"]
                if "alert" in tenant_info:
                    alerts.append({
                        "type": "budget_warning",
                        "tenant_id": tid,
                        "usage_percent": budget["usage_percent"],
                        "threshold": tenant_info["alert"]["threshold"],
                        "message": tenant_info["alert"]["message"],
                        "cost_used": budget["used"],
                        "budget_limit": budget["monthly_limit"]
                    })
        
        return alerts
    
    def export_report(self,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     format: str = "summary") -> Dict[str, Any]:
        """Export a cost report
        
        Args:
            start_time: Report start time
            end_time: Report end time  
            format: "summary", "detailed", or "by_tenant"
            
        Returns:
            Report dictionary
        """
        base_summary = self.get_summary(start_time, end_time)
        
        if format == "summary":
            return {
                "report_type": "summary",
                "period": {
                    "start": start_time,
                    "end": end_time or time.time()
                },
                "totals": {
                    "input_tokens": base_summary.total_input_tokens,
                    "output_tokens": base_summary.total_output_tokens,
                    "total_tokens": base_summary.total_tokens,
                    "total_cost": base_summary.total_cost,
                    "request_count": base_summary.request_count,
                    "average_cost_per_request": base_summary.average_cost_per_request,
                    "average_tokens_per_request": base_summary.average_tokens_per_request
                },
                "currency": "USD"
            }
        
        elif format == "detailed":
            return {
                "report_type": "detailed",
                "period": {
                    "start": start_time,
                    "end": end_time or time.time()
                },
                "totals": {
                    "input_tokens": base_summary.total_input_tokens,
                    "output_tokens": base_summary.total_output_tokens,
                    "total_tokens": base_summary.total_tokens,
                    "total_cost": base_summary.total_cost,
                    "request_count": base_summary.request_count
                },
                "by_model": self.get_model_breakdown(start_time, end_time),
                "currency": "USD"
            }
        
        elif format == "by_tenant":
            tenants = {}
            current_time = end_time or time.time()
            month_ago = current_time - (30 * 24 * 60 * 60)
            
            for tenant_id in self.tenant_budgets.keys():
                tenants[tenant_id] = self.get_tenant_costs(tenant_id, month_ago, current_time)
            
            return {
                "report_type": "by_tenant",
                "period": {
                    "start": month_ago,
                    "end": current_time
                },
                "tenants": tenants,
                "currency": "USD"
            }
        
        raise ValueError(f"Unknown format: {format}")
    
    def clear_old_records(self, older_than_days: int = 90):
        """Remove records older than specified days (garbage collection)"""
        cutoff = time.time() - (older_than_days * 24 * 60 * 60)
        self.records = [r for r in self.records if r.created_at >= cutoff]
