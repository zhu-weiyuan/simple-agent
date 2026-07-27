# -*- coding: utf-8 -*-
"""
my_agent.gateway — Model Gateway with multi-model routing, token budget, and cost attribution

This module provides:
- ModelRoute: Configuration for routing to specific models
- TokenBudget: Per-session/per-user token budget management
- ModelGateway: Main gateway class that handles model selection, budget checking, and fallback chains
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from enum import Enum
import time


class BudgetStatus(Enum):
    """Status of budget check"""
    OK = "ok"
    EXCEEDED = "exceeded"
    WARNING = "warning"  # Near limit


class BudgetPolicy(str, Enum):
    """预算超限时的处理策略 (默认 DEGRADE — 温和降级, 不直接拒绝服务)。

    - REJECT : 直接拒绝, 抛 BudgetExceededError
    - DEGRADE: 降级到 gateway 中更便宜的 tier 模型继续服务; 降不了再 reject
    - WARN   : 只告警不拦截
    """
    REJECT = "reject"
    DEGRADE = "degrade"
    WARN = "warn"

    @classmethod
    def coerce(cls, value: Any, default: "BudgetPolicy" = None) -> "BudgetPolicy":
        """把字符串/枚举/None 归一化为 BudgetPolicy (未知值回落到 default)。"""
        default = default or cls.DEGRADE
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                return default
        return default


class BudgetExceededError(RuntimeError):
    """租户 token 预算超限且无法降级时抛出。"""

    def __init__(self, budget_id: str, needed_tokens: int = 0,
                 remaining: int = 0, message: str = ""):
        self.budget_id = budget_id
        self.needed_tokens = needed_tokens
        self.remaining = remaining
        super().__init__(message or (
            f"token budget exceeded for '{budget_id}': "
            f"needed {needed_tokens}, remaining {remaining}"))


@dataclass
class ModelRoute:
    """Configuration for a single model route"""
    name: str
    endpoint: str
    provider: str  # e.g., "ollama", "openai", "anthropic", "local"
    context_window: int  # Maximum context size in tokens
    max_tokens: int  # Maximum response tokens
    priority: int = 0  # Lower number = higher priority (used for fallback ordering)
    cost_per_1m_input: float = 0.0  # Cost per 1M input tokens
    cost_per_1m_output: float = 0.0  # Cost per 1M output tokens
    health_check_endpoint: Optional[str] = None
    is_healthy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass 
class TokenBudget:
    """Token budget for a session/user/project"""
    id: str
    max_tokens: int
    used_tokens: int = 0
    warning_threshold: float = 0.8  # 80% warning
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)
    
    @property
    def usage_percent(self) -> float:
        if self.max_tokens <= 0:
            return 1.0
        return min(1.0, self.used_tokens / self.max_tokens)
    
    @property
    def status(self) -> BudgetStatus:
        if self.usage_percent >= 1.0:
            return BudgetStatus.EXCEEDED
        elif self.usage_percent >= self.warning_threshold:
            return BudgetStatus.WARNING
        else:
            return BudgetStatus.OK
    
    def check_available(self, needed: int) -> tuple[BudgetStatus, int]:
        """Check if budget can accommodate needed tokens
        
        Returns:
            Tuple of (status, remaining_after_if_ok)
        """
        if self.used_tokens + needed > self.max_tokens:
            return BudgetStatus.EXCEEDED, 0
        return self.status, self.remaining - needed
    
    def consume(self, tokens: int) -> bool:
        """Consume tokens from budget
        
        Returns:
            True if successful, False if would exceed budget
        """
        if self.used_tokens + tokens > self.max_tokens:
            return False
        self.used_tokens += tokens
        self.updated_at = time.time()
        return True
    
    def consume_force(self, tokens: int) -> int:
        """无条件扣减 (用于真实 usage 回写 reconcile)。

        与 ``consume`` 不同: 即使会超出上限也照实累计, 使 status 如实变成
        EXCEEDED (预算是"事后如实记账 + 事前拦截", 不能因为超限就丢掉真实用量)。

        Returns:
            扣减后的 used_tokens
        """
        if tokens <= 0:
            return self.used_tokens
        self.used_tokens += int(tokens)
        self.updated_at = time.time()
        return self.used_tokens

    def reset(self):
        """Reset budget usage"""
        self.used_tokens = 0
        self.updated_at = time.time()


class ModelGateway:
    """Main gateway for model routing with budget enforcement and fallback chain"""
    
    def __init__(self, routes: Optional[List[ModelRoute]] = None):
        self.routes: Dict[str, ModelRoute] = {}
        self.budgets: Dict[str, TokenBudget] = {}
        self._route_order: List[str] = []
        
        if routes:
            for route in routes:
                self.add_route(route)
    
    def add_route(self, route: ModelRoute):
        """Add or update a model route"""
        self.routes[route.name] = route
        # Maintain sorted order by priority
        self._route_order = sorted(
            self.routes.keys(),
            key=lambda r: self.routes[r].priority
        )
    
    def remove_route(self, name: str):
        """Remove a model route"""
        if name in self.routes:
            del self.routes[name]
            self._route_order = sorted(
                self.routes.keys(),
                key=lambda r: self.routes[r].priority
            )
    
    def get_route(self, name: str) -> Optional[ModelRoute]:
        """Get a specific route by name"""
        return self.routes.get(name)
    
    def get_available_routes(self) -> List[ModelRoute]:
        """Get all healthy routes sorted by priority"""
        return [
            self.routes[name] 
            for name in self._route_order 
            if self.routes[name].is_healthy
        ]
    
    def create_budget(self, budget_id: str, max_tokens: int, 
                      warning_threshold: float = 0.8) -> TokenBudget:
        """Create a new token budget"""
        budget = TokenBudget(
            id=budget_id,
            max_tokens=max_tokens,
            warning_threshold=warning_threshold
        )
        self.budgets[budget_id] = budget
        return budget
    
    def get_budget(self, budget_id: str) -> Optional[TokenBudget]:
        """Get a budget by ID"""
        return self.budgets.get(budget_id)
    
    def check_budget(self, budget_id: str, needed_tokens: int) -> tuple[BudgetStatus, Optional[str]]:
        """Check if a budget can accommodate the needed tokens
        
        Returns:
            Tuple of (status, rejected_model_name if applicable)
        """
        if budget_id not in self.budgets:
            # No budget = unlimited
            return BudgetStatus.OK, None
        
        budget = self.budgets[budget_id]
        status, _ = budget.check_available(needed_tokens)
        
        if status == BudgetStatus.EXCEEDED:
            return status, budget_id
        
        return status, None
    
    def select_model(self, budget_id: Optional[str] = None, 
                     needed_tokens: int = 0,
                     preferred_models: Optional[List[str]] = None) -> Optional[str]:
        """Select the best available model considering budget and preferences
        
        Args:
            budget_id: Optional budget to check against
            needed_tokens: Number of tokens expected for this request
            preferred_models: Optional ordered list of preferred model names
            
        Returns:
            Name of selected model, or None if no suitable model found
        """
        # Get candidates based on preference
        if preferred_models:
            candidates = [m for m in preferred_models if m in self.routes]
            unavailable = set(preferred_models) - set(self.routes.keys())
            if unavailable:
                pass  # Log warning about unavailable models
        else:
            candidates = self._route_order
        
        # Filter to healthy routes only
        candidates = [
            name for name in candidates 
            if self.routes[name].is_healthy
        ]
        
        # Check budget constraints
        for model_name in candidates:
            if budget_id:
                status, _ = self.check_budget(budget_id, needed_tokens)
                if status == BudgetStatus.EXCEEDED:
                    continue  # Skip this model, try next
            
            # Check model capacity: available input = context_window - max_tokens (reserved for output)
            route = self.routes[model_name]
            if route.context_window - route.max_tokens >= needed_tokens and route.is_healthy:
                return model_name
        
        return None
    
    def record_usage(self, budget_id: str, model_name: str, 
                     input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """Record usage for a request
        
        Returns:
            Dictionary with cost breakdown and budget status
        """
        route = self.get_route(model_name)
        if not route:
            return {"error": f"Unknown model: {model_name}"}
        
        total_tokens = input_tokens + output_tokens
        consumed = False
        
        if budget_id and budget_id in self.budgets:
            consumed = self.budgets[budget_id].consume(total_tokens)
            if not consumed:
                # Refund if it would exceed budget
                return {
                    "status": "rejected",
                    "reason": "budget_exceeded",
                    "budget_id": budget_id,
                    "needed": total_tokens,
                    "available": self.budgets[budget_id].remaining
                }
        elif budget_id:
            # Create auto-budget if doesn't exist but was specified
            return {
                "status": "error",
                "reason": "budget_not_found",
                "budget_id": budget_id
            }
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * route.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * route.cost_per_1m_output
        total_cost = input_cost + output_cost
        
        result = {
            "status": "recorded",
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }
        
        if budget_id:
            budget = self.budgets[budget_id]
            result["budget"] = {
                "id": budget_id,
                "used": budget.used_tokens,
                "remaining": budget.remaining,
                "usage_percent": budget.usage_percent,
                "status": budget.status.value
            }
        
        return result
    
    def get_fallback_chain(self, primary_model: str) -> List[str]:
        """Get ordered list of fallback models excluding the primary.

        Only routes with strictly lower priority (higher number) than the primary
        are included — a higher-priority route must never appear as a fallback.
        """
        primary_priority = self.routes[primary_model].priority if primary_model in self.routes else float('inf')

        fallbacks = [
            name for name in self._route_order
            if name != primary_model
            and self.routes[name].is_healthy
            and self.routes[name].priority >= primary_priority
        ]

        return fallbacks
    
    # ── cost tiering / budget reconcile (engine wiring) ──────────────────

    @staticmethod
    def route_cost(route: ModelRoute) -> float:
        """粗粒度"贵不贵"排序键: 输入+输出单价之和 (USD / 1M tokens)。"""
        return float(route.cost_per_1m_input) + float(route.cost_per_1m_output)

    def cheaper_routes(self, model_name: Optional[str] = None,
                       needed_tokens: int = 0) -> List[str]:
        """返回比 ``model_name`` 更便宜的健康路由 (由便宜到贵)。

        ``model_name`` 为 None 时返回全部健康路由 (由便宜到贵), 供"没选出主模型
        但需要降级"的场景使用。``needed_tokens`` 用于过滤放不下的上下文窗口。
        """
        current = self.get_route(model_name) if model_name else None
        ceiling = self.route_cost(current) if current else float("inf")
        candidates = [
            r for r in self.get_available_routes()
            if r.name != model_name
            and self.route_cost(r) < ceiling
            and r.context_window > needed_tokens
        ]
        candidates.sort(key=lambda r: (self.route_cost(r), r.priority, r.name))
        return [r.name for r in candidates]

    def reconcile_usage(self, budget_id: str, total_tokens: int) -> Optional[TokenBudget]:
        """用真实 usage 回写扣减预算 (estimate→reserve→reconcile 的 reconcile 步)。

        预算不存在时静默跳过 (无预算 = 不限量)。
        """
        budget = self.budgets.get(budget_id)
        if budget is None or total_tokens <= 0:
            return budget
        budget.consume_force(total_tokens)
        return budget

    def budget_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """只读快照: 各预算的 used/remaining/percent/status (供 /api/budgets)。"""
        return {
            bid: {
                "max_tokens": b.max_tokens,
                "used_tokens": b.used_tokens,
                "remaining_tokens": b.remaining,
                "usage_percent": round(b.usage_percent, 4),
                "status": b.status.value,
            }
            for bid, b in self.budgets.items()
        }

    def mark_unhealthy(self, model_name: str):
        """Mark a model as unhealthy"""
        if model_name in self.routes:
            self.routes[model_name].is_healthy = False
    
    def mark_healthy(self, model_name: str):
        """Mark a model as healthy"""
        if model_name in self.routes:
            self.routes[model_name].is_healthy = True
    
    def health_check(self) -> Dict[str, bool]:
        """Run health checks on all routes"""
        results = {}
        for name, route in self.routes.items():
            if route.health_check_endpoint:
                # In real implementation, would make HTTP request
                # For now, just report current state
                results[name] = route.is_healthy
            else:
                results[name] = route.is_healthy
        return results
