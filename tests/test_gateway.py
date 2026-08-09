# -*- coding: utf-8 -*-
"""
tests.test_gateway — Tests for ModelGateway, ModelRoute, and TokenBudget
"""

import pytest
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from my_agent.gateway import ModelGateway, ModelRoute, TokenBudget, BudgetStatus


class TestModelRoute:
    """Tests for ModelRoute dataclass"""
    
    def test_basic_route_creation(self):
        route = ModelRoute(
            name="test-model",
            endpoint="https://api.example.com/v1/chat",
            provider="openai",
            context_window=4096,
            max_tokens=2048,
            priority=1,
            cost_per_1m_input=10.0,
            cost_per_1m_output=30.0
        )
        
        assert route.name == "test-model"
        assert route.context_window == 4096
        assert route.max_tokens == 2048
        assert route.priority == 1
        assert route.cost_per_1m_input == 10.0
        assert route.cost_per_1m_output == 30.0
        assert route.is_healthy is True
    
    def test_default_values(self):
        route = ModelRoute(
            name="default-route",
            endpoint="http://localhost:8000",
            provider="local",
            context_window=2048,
            max_tokens=1024
        )
        
        assert route.priority == 0
        assert route.cost_per_1m_input == 0.0
        assert route.cost_per_1m_output == 0.0
        assert route.health_check_endpoint is None
        assert route.metadata == {}
    
    def test_negative_priority_rejected(self):
        with pytest.raises(ValueError, match="Priority must be non-negative"):
            ModelRoute(
                name="bad",
                endpoint="http://x",
                provider="test",
                context_window=100,
                max_tokens=50,
                priority=-1
            )
    
    def test_invalid_context_window_rejected(self):
        with pytest.raises(ValueError, match="context_window must be positive"):
            ModelRoute(
                name="bad",
                endpoint="http://x",
                provider="test",
                context_window=0,
                max_tokens=50
            )
    
    def test_invalid_max_tokens_rejected(self):
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ModelRoute(
                name="bad",
                endpoint="http://x",
                provider="test",
                context_window=100,
                max_tokens=-5
            )


class TestTokenBudget:
    """Tests for TokenBudget dataclass"""
    
    def test_budget_creation(self):
        budget = TokenBudget(
            id="test-budget",
            max_tokens=10000,
            warning_threshold=0.7
        )
        
        assert budget.id == "test-budget"
        assert budget.max_tokens == 10000
        assert budget.used_tokens == 0
        assert budget.remaining == 10000
        assert budget.usage_percent == 0.0
        assert budget.status == BudgetStatus.OK
    
    def test_remaining_calculation(self):
        budget = TokenBudget(id="b1", max_tokens=1000)
        budget.used_tokens = 300
        
        assert budget.remaining == 700
    
    def test_usage_percent_clamped(self):
        budget = TokenBudget(id="b1", max_tokens=1000)
        budget.used_tokens = 1500  # Over limit
        
        assert budget.usage_percent == 1.0
        assert budget.status == BudgetStatus.EXCEEDED
    
    def test_status_transitions(self):
        budget = TokenBudget(id="b1", max_tokens=1000, warning_threshold=0.8)
        
        # OK status
        assert budget.status == BudgetStatus.OK
        
        # WARNING status (80% usage)
        budget.used_tokens = 800
        assert budget.status == BudgetStatus.WARNING
        
        # EXCEEDED status
        budget.used_tokens = 1000
        assert budget.status == BudgetStatus.EXCEEDED
    
    def test_check_available_succeeds(self):
        budget = TokenBudget(id="b1", max_tokens=1000)
        budget.used_tokens = 300
        
        status, remaining = budget.check_available(200)
        
        assert status == BudgetStatus.OK
        assert remaining == 500  # 700 - 200
    
    def test_check_available_exceeds(self):
        budget = TokenBudget(id="b1", max_tokens=1000)
        budget.used_tokens = 900
        
        status, remaining = budget.check_available(200)
        
        assert status == BudgetStatus.EXCEEDED
        assert remaining == 0
    
    def test_consume_success(self):
        budget = TokenBudget(id="b1", max_tokens=1000)
        old_updated = budget.updated_at
        
        # Small delay to ensure timestamp changes
        time.sleep(0.01)
        
        result = budget.consume(500)
        
        assert result is True
        assert budget.used_tokens == 500
        assert budget.updated_at > old_updated
    
    def test_consume_failure_exceeds(self):
        budget = TokenBudget(id="b1", max_tokens=1000)
        budget.used_tokens = 900
        
        result = budget.consume(200)  # Would exceed
        
        assert result is False
        assert budget.used_tokens == 900  # Not modified
    
    def test_reset(self):
        budget = TokenBudget(id="b1", max_tokens=1000)
        budget.used_tokens = 750
        
        budget.reset()
        
        assert budget.used_tokens == 0
        assert budget.remaining == 1000


class TestModelGateway:
    """Tests for ModelGateway class"""
    
    @pytest.fixture
    def gateway(self):
        """Create a gateway with test routes"""
        gw = ModelGateway()
        
        # Add primary route
        gw.add_route(ModelRoute(
            name="primary",
            endpoint="https://api.primary.com/v1",
            provider="openai",
            context_window=8192,
            max_tokens=4096,
            priority=1,
            cost_per_1m_input=10.0,
            cost_per_1m_output=30.0
        ))
        
        # Add fallback route
        gw.add_route(ModelRoute(
            name="fallback",
            endpoint="https://api.fallback.com/v1",
            provider="anthropic",
            context_window=4096,
            max_tokens=2048,
            priority=2,
            cost_per_1m_input=5.0,
            cost_per_1m_output=15.0
        ))
        
        # Add unhealthy route
        gw.add_route(ModelRoute(
            name="unhealthy",
            endpoint="https://broken.com/v1",
            provider="local",
            context_window=2048,
            max_tokens=1024,
            priority=0,  # Highest priority but unhealthy
            is_healthy=False
        ))
        
        return gw
    
    def test_add_route(self, gateway):
        assert "primary" in gateway.routes
        assert gateway.routes["primary"].name == "primary"
        assert len(gateway._route_order) == 3
    
    def test_remove_route(self, gateway):
        gateway.remove_route("fallback")
        
        assert "fallback" not in gateway.routes
        assert len(gateway._route_order) == 2
    
    def test_get_route(self, gateway):
        route = gateway.get_route("primary")
        
        assert route is not None
        assert route.name == "primary"
    
    def test_get_unknown_route(self, gateway):
        route = gateway.get_route("nonexistent")
        assert route is None
    
    def test_get_available_routes_only_healthy(self, gateway):
        available = gateway.get_available_routes()
        
        assert len(available) == 2
        names = {r.name for r in available}
        assert "primary" in names
        assert "fallback" in names
        assert "unhealthy" not in names
    
    def test_routes_sorted_by_priority(self, gateway):
        order = gateway.get_available_routes()
        names = [r.name for r in order]
        
        assert names.index("primary") < names.index("fallback")
    
    def test_create_budget(self, gateway):
        budget = gateway.create_budget("session-123", max_tokens=50000)
        
        assert budget.id == "session-123"
        assert budget.max_tokens == 50000
        assert gateway.get_budget("session-123") is budget
    
    def test_get_unknown_budget(self, gateway):
        assert gateway.get_budget("nonexistent") is None
    
    def test_check_budget_ok(self, gateway):
        gateway.create_budget("budget-1", max_tokens=10000)
        
        status, rejected = gateway.check_budget("budget-1", 5000)
        
        assert status == BudgetStatus.OK
        assert rejected is None
    
    def test_check_budget_exceeded(self, gateway):
        budget = gateway.create_budget("budget-2", max_tokens=1000)
        budget.used_tokens = 900
        
        status, rejected = gateway.check_budget("budget-2", 200)
        
        assert status == BudgetStatus.EXCEEDED
        assert rejected == "budget-2"
    
    def test_no_budget_means_unlimited(self, gateway):
        status, rejected = gateway.check_budget("unknown-budget", 1000000)
        
        assert status == BudgetStatus.OK
        assert rejected is None
    
    def test_select_model_prefers_higher_priority(self, gateway):
        selected = gateway.select_model()
        
        assert selected == "primary"
    
    def test_select_model_respects_health(self, gateway):
        # Even though "unhealthy" has priority 0, it should be skipped
        selected = gateway.select_model()
        
        assert selected != "unhealthy"
        assert selected == "primary"
    
    def test_select_model_with_budget_constraint(self, gateway):
        # Create exhausted budget
        gateway.create_budget("exhausted", max_tokens=100)
        gateway.budgets["exhausted"].used_tokens = 100
        
        selected = gateway.select_model(budget_id="exhausted", needed_tokens=500)
        
        # Should skip models if budget exceeded
        assert selected is None
    
    def test_select_model_with_preferences(self, gateway):
        # Prefer fallback over primary
        selected = gateway.select_model(preferred_models=["fallback", "primary"])
        
        assert selected == "fallback"
    
    def test_select_model_unknown_preference(self, gateway):
        # Request unknown model as preference
        selected = gateway.select_model(preferred_models=["unknown", "fallback"])
        
        # Should fall through to known healthy models
        assert selected == "fallback"
    
    def test_record_usage_success(self, gateway):
        gateway.create_budget("sess-1", max_tokens=10000)
        
        result = gateway.record_usage(
            budget_id="sess-1",
            model_name="primary",
            input_tokens=1000,
            output_tokens=500
        )
        
        assert result["status"] == "recorded"
        assert result["model"] == "primary"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["total_tokens"] == 1500
        
        # Check costs: (1M/1M * 10) + (0.5M/1M * 30) = 10 + 15 = 25 cents
        assert abs(result["input_cost"] - 0.01) < 0.001
        assert abs(result["output_cost"] - 0.015) < 0.001
        assert abs(result["total_cost"] - 0.025) < 0.001
        
        # Check budget was consumed
        assert gateway.budgets["sess-1"].used_tokens == 1500
    
    def test_record_usage_budget_exceeded(self, gateway):
        gateway.create_budget("small-budget", max_tokens=100)
        
        result = gateway.record_usage(
            budget_id="small-budget",
            model_name="primary",
            input_tokens=200,
            output_tokens=200
        )
        
        assert result["status"] == "rejected"
        assert result["reason"] == "budget_exceeded"
        assert gateway.budgets["small-budget"].used_tokens == 0  # Not consumed
    
    def test_record_usage_unknown_model(self, gateway):
        result = gateway.record_usage(
            budget_id=None,
            model_name="unknown-model",
            input_tokens=100,
            output_tokens=100
        )
        
        assert "error" in result
        assert "Unknown model" in result["error"]
    
    def test_get_fallback_chain(self, gateway):
        gateway.mark_unhealthy("primary")
        
        fallbacks = gateway.get_fallback_chain("primary")
        
        assert "fallback" in fallbacks
        assert "primary" not in fallbacks
        assert "unhealthy" not in fallbacks
    
    def test_mark_unhealthy(self, gateway):
        gateway.mark_unhealthy("primary")
        
        assert gateway.routes["primary"].is_healthy is False
    
    def test_mark_healthy(self, gateway):
        gateway.mark_unhealthy("fallback")
        gateway.mark_healthy("fallback")
        
        assert gateway.routes["fallback"].is_healthy is True
    
    def test_health_check(self, gateway):
        results = gateway.health_check()
        
        assert "primary" in results
        assert "fallback" in results
        assert "unhealthy" in results
        assert results["unhealthy"] is False


class TestIntegration:
    """Integration tests combining gateway features"""
    
    def test_full_request_flow(self):
        """Test complete request flow: select -> record -> check budget"""
        gateway = ModelGateway()
        
        # Setup routes
        gateway.add_route(ModelRoute(
            name="gpt-4",
            endpoint="https://api.openai.com/v1",
            provider="openai",
            context_window=8192,
            max_tokens=4096,
            priority=1,
            cost_per_1m_input=30.0,
            cost_per_1m_output=60.0
        ))
        
        # Create session budget
        gateway.create_budget("user-session-42", max_tokens=50000)
        
        # Select model for request
        selected = gateway.select_model(
            budget_id="user-session-42",
            needed_tokens=2000,
            preferred_models=["gpt-4", "gpt-3.5"]
        )
        
        assert selected == "gpt-4"
        
        # Record actual usage
        result = gateway.record_usage(
            budget_id="user-session-42",
            model_name="gpt-4",
            input_tokens=1500,
            output_tokens=800
        )
        
        assert result["status"] == "recorded"
        assert result["total_tokens"] == 2300
        
        # Check remaining budget
        budget = gateway.get_budget("user-session-42")
        assert budget.used_tokens == 2300
        assert budget.remaining == 47700
        assert budget.status == BudgetStatus.OK
    
    def test_budget_warning_scenario(self):
        """Test scenario where budget approaches warning threshold"""
        gateway = ModelGateway()
        
        gateway.add_route(ModelRoute(
            name="standard",
            endpoint="http://localhost:8000",
            provider="local",
            context_window=4096,
            max_tokens=2048
        ))
        
        # Create budget with low warning threshold
        gateway.create_budget(
            "tight-budget",
            max_tokens=10000,
            warning_threshold=0.5
        )
        
        # Consume up to warning threshold
        gateway.budgets["tight-budget"].used_tokens = 5000
        
        status, _ = gateway.check_budget("tight-budget", 1000)
        
        assert status == BudgetStatus.WARNING
        
        # One more request exceeds
        status, _ = gateway.check_budget("tight-budget", 6000)
        
        assert status == BudgetStatus.EXCEEDED
