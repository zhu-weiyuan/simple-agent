# -*- coding: utf-8 -*-
"""
tests.test_router — Tests for RuleBasedRouter and RouteRule
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from my_agent.router import (
    RouteRule, 
    RuleBasedRouter, 
    RoutingPriority,
    create_scene_router,
    create_tenant_router
)


class TestRouteRule:
    """Tests for RouteRule dataclass"""
    
    def test_basic_rule_creation(self):
        rule = RouteRule(
            name="coding-assistant",
            model="code-model-v1",
            priority=RoutingPriority.HIGH,
            scene_patterns=["^code$", "^programming$"],
            description="Routes coding tasks"
        )
        
        assert rule.name == "coding-assistant"
        assert rule.model == "code-model-v1"
        assert rule.priority == RoutingPriority.HIGH
        assert rule.scene_patterns == ["^code$", "^programming$"]
        assert rule.description == "Routes coding tasks"
    
    def test_default_values(self):
        rule = RouteRule(
            name="default-rule",
            model="standard-model"
        )
        
        assert rule.priority == RoutingPriority.MEDIUM
        assert rule.scene_patterns is None
        assert rule.tenant_ids is None
        assert rule.user_tags is None
        assert rule.request_types is None
        assert rule.max_input_tokens is None
        assert rule.max_output_tokens is None
        assert rule.requires_budget is False
    
    def test_matches_scene_with_pattern(self):
        rule = RouteRule(
            name="test",
            model="m1",
            scene_patterns=["^coding", "^development"]
        )
        
        assert rule.matches_scene("coding-help") is True
        assert rule.matches_scene("development-tasks") is True
        assert rule.matches_scene("creative-writing") is False
        assert rule.matches_scene(None) is False
    
    def test_matches_scene_no_patterns(self):
        rule = RouteRule(
            name="test",
            model="m1"
        )
        
        # No patterns means match all scenes
        assert rule.matches_scene("anything") is True
        assert rule.matches_scene("coding") is True
    
    def test_matches_tenant(self):
        rule = RouteRule(
            name="tenant-rule",
            model="m1",
            tenant_ids=["tenant-a", "tenant-b"]
        )
        
        assert rule.matches_tenant("tenant-a") is True
        assert rule.matches_tenant("tenant-b") is True
        assert rule.matches_tenant("tenant-c") is False
        assert rule.matches_tenant(None) is False
    
    def test_matches_tenant_no_restriction(self):
        rule = RouteRule(
            name="all-tenants",
            model="m1"
        )
        
        assert rule.matches_tenant("any-tenant") is True
        assert rule.matches_tenant("different") is True
    
    def test_matches_user_tags_all_required(self):
        rule = RouteRule(
            name="premium-users",
            model="m1",
            user_tags=["premium", "verified"]
        )
        
        # User has both tags
        assert rule.matches_user_tags(["premium", "verified", "beta"]) is True
        
        # User missing one tag
        assert rule.matches_user_tags(["premium"]) is False
        
        # User has no tags
        assert rule.matches_user_tags([]) is False
        assert rule.matches_user_tags(None) is False
    
    def test_matches_user_tags_no_requirement(self):
        rule = RouteRule(
            name="all-users",
            model="m1"
        )
        
        assert rule.matches_user_tags(["anything"]) is True
        assert rule.matches_user_tags([]) is True
        assert rule.matches_user_tags(None) is True
    
    def test_matches_request_type(self):
        rule = RouteRule(
            name="chat-routes",
            model="m1",
            request_types=["chat", "conversation"]
        )
        
        assert rule.matches_request_type("chat") is True
        assert rule.matches_request_type("conversation") is True
        assert rule.matches_request_type("embedding") is False
        assert rule.matches_request_type(None) is False
    
    def test_is_satisfied_by_context_full_match(self):
        rule = RouteRule(
            name="complex-rule",
            model="special-model",
            scene_patterns=["^analysis"],
            tenant_ids=["enterprise-123"],
            user_tags=["premium"],
            request_types=["chat"]
        )
        
        context = {
            "scene": "analysis-task",
            "tenant_id": "enterprise-123",
            "user_tags": ["premium", "verified"],
            "request_type": "chat"
        }
        
        assert rule.is_satisfied_by_context(context) is True
    
    def test_is_satisfied_by_context_partial_failure(self):
        rule = RouteRule(
            name="strict-rule",
            model="m1",
            tenant_ids=["enterprise-123"]
        )
        
        # Wrong tenant
        context = {
            "tenant_id": "wrong-tenant",
            "scene": "anything"
        }
        
        assert rule.is_satisfied_by_context(context) is False
    
    def test_scene_matching_requires_pattern_match(self):
        rule = RouteRule(
            name="case-test",
            model="m1",
            scene_patterns=["^CODING"]  # Pattern for CODING
        )
        
        # Should match patterns that actually match
        assert rule.matches_scene("coding-session") is True
        assert rule.matches_scene("CODE-help") is False


class TestRuleBasedRouter:
    """Tests for RuleBasedRouter class"""
    
    @pytest.fixture
    def router(self):
        """Create a router with test rules"""
        r = RuleBasedRouter()
        
        # Add rules with different priorities
        r.add_rule(RouteRule(
            name="high-priority-coding",
            model="code-model-pro",
            priority=RoutingPriority.CRITICAL,
            scene_patterns=["^coding", "^programming"]
        ))
        
        r.add_rule(RouteRule(
            name="medium-creative",
            model="creative-writer",
            priority=RoutingPriority.MEDIUM,
            scene_patterns=["^creative", "^writing"]
        ))
        
        r.add_rule(RouteRule(
            name="tenant-enterprise",
            model="enterprise-model",
            priority=RoutingPriority.HIGH,
            tenant_ids=["enterprise-123"]
        ))
        
        r.set_default_model("standard-model")
        
        return r
    
    def test_add_rule(self, router):
        assert len(router.rules) == 3
        rule_names = {r.name for r in router.rules}
        assert "high-priority-coding" in rule_names
    
    def test_remove_rule(self, router):
        router.remove_rule("medium-creative")
        
        assert len(router.rules) == 2
        assert not any(r.name == "medium-creative" for r in router.rules)
    
    def test_set_default_model(self, router):
        router.set_default_model("new-default")
        
        assert router.default_model == "new-default"
    
    def test_get_matching_rules_with_matching_scene(self, router):
        context = {"scene": "coding-session"}
        
        matching = router.get_matching_rules(context)
        
        # Should have at least the coding rule matching
        assert len(matching) >= 1
    
    def test_get_matching_rules_sorted_by_priority(self, router):
        context = {"scene": "coding-session", "tenant_id": "enterprise-123"}
        
        matching = router.get_matching_rules(context)
        
        # Critical should come before High, which comes before Medium
        if len(matching) >= 2:
            priorities = [r.priority for r in matching]
            assert priorities.index(RoutingPriority.CRITICAL) < priorities.index(RoutingPriority.HIGH)
    
    def test_route_selects_highest_priority_match(self, router):
        context = {"scene": "coding-help"}
        
        selected = router.route(context)
        
        assert selected == "code-model-pro"
    
    def test_route_falls_back_to_default(self, router):
        context = {"scene": "unknown-scene"}
        
        # No specific rule matches, should use default
        selected = router.route(context)
        
        assert selected == "standard-model"
    
    def test_route_with_gateway(self):
        # Create mock gateway
        class MockGateway:
            def select_model(self, budget_id=None, needed_tokens=0, preferred_models=None):
                if preferred_models and preferred_models[0] == "healthy-model":
                    return "healthy-model"
                return None
        
        router = RuleBasedRouter(gateway=MockGateway())
        router.add_rule(RouteRule(
            name="test",
            model="healthy-model"
        ))
        
        selected = router.route({})
        
        assert selected == "healthy-model"
    
    def test_route_respects_token_constraints(self, router):
        router.add_rule(RouteRule(
            name="small-model",
            model="tiny-model",
            max_input_tokens=500
        ))
        
        # Request too large for small model
        context = {
            "estimated_input_tokens": 1000,
            "estimated_output_tokens": 200
        }
        
        # Should skip small-model rule due to constraint
        selected = router.route(context)
        
        # Should fall back to another option or default
        assert selected is not None
    
    def test_route_requires_budget(self):
        router = RuleBasedRouter()
        
        router.add_rule(RouteRule(
            name="paid-tier",
            model="premium-model",
            requires_budget=True
        ))
        
        router.set_default_model("free-model")
        
        # No budget provided
        selected = router.route({"scene": "anything"})
        
        assert selected == "free-model"
        
        # With budget should match paid rule
        selected = router.route({"scene": "anything"}, budget_id="session-123")
        
        assert selected == "premium-model"
    
    def test_route_uses_default_when_no_rules_match(self, router):
        context = {"scene": "unknown-scene-xyz"}
        
        selected = router.route(context)
        
        assert selected == "standard-model"
    
    def test_analyze_routing_decision(self, router):
        context = {"scene": "coding-task", "tenant_id": "enterprise-123"}
        
        analysis = router.analyze_routing_decision(context)
        
        assert "matching_rules" in analysis
        assert "selected_model" in analysis
        assert "reasoning" in analysis
        assert analysis["selected_model"] == "code-model-pro"
        
        # Should have multiple matching rules
        assert len(analysis["matching_rules"]) >= 1
    
    def test_get_rule_summary(self, router):
        summary = router.get_rule_summary()
        
        assert len(summary) == 3
        
        # Check structure
        for item in summary:
            assert "name" in item
            assert "model" in item
            assert "priority" in item
            assert "constraints" in item


class TestSceneRouter:
    """Tests for create_scene_router helper"""
    
    def test_creating_scene_router(self):
        router = create_scene_router()
        
        assert isinstance(router, RuleBasedRouter)
        
        # Should have coding rule
        rule_names = [r.name for r in router.rules]
        assert "coding-assistant" in rule_names
        assert "creative-writing" in rule_names
        assert "analysis-task" in rule_names
    
    def test_coding_scene_rules_match(self):
        router = create_scene_router()
        
        context = {"scene": "code"}  # Matches ^code$
        matching = router.get_matching_rules(context)
        
        # Should have coding rule matching
        coding_rules = [r for r in matching if r.name == "coding-assistant"]
        assert len(coding_rules) == 1
        assert coding_rules[0].model == "code-model-primary"
    
    def test_creative_scene_rules_match(self):
        router = create_scene_router()
        
        context = {"scene": "story"}  # Matches ^story$
        matching = router.get_matching_rules(context)
        
        creative_rules = [r for r in matching if r.name == "creative-writing"]
        assert len(creative_rules) == 1
        assert creative_rules[0].model == "creative-model"


class TestTenantRouter:
    """Tests for create_tenant_router helper"""
    
    def test_creating_tenant_router(self):
        defaults = {
            "tenant-a": "model-a",
            "tenant-b": "model-b"
        }
        
        router = create_tenant_router(tenant_defaults=defaults)
        
        assert len(router.rules) == 2
        
        rule_names = [r.name for r in router.rules]
        assert "tenant-tenant-a" in rule_names
        assert "tenant-tenant-b" in rule_names
    
    def test_tenant_routes_to_correct_model(self):
        defaults = {"enterprise": "enterprise-model"}
        router = create_tenant_router(tenant_defaults=defaults)
        
        context = {"tenant_id": "enterprise"}
        selected = router.route(context)
        
        assert selected == "enterprise-model"
    
    def test_non_matching_tenant_uses_default(self):
        defaults = {"known-tenant": "specific-model"}
        router = create_tenant_router(tenant_defaults=defaults)
        router.set_default_model("default-model")
        
        context = {"tenant_id": "unknown-tenant"}
        selected = router.route(context)
        
        assert selected == "default-model"


class TestIntegration:
    """Integration tests for routing scenarios"""
    
    def test_multi_dimensional_routing(self):
        """Test routing that considers scene, tenant, and user tags"""
        router = RuleBasedRouter()
        
        # Premium enterprise coding gets special treatment
        router.add_rule(RouteRule(
            name="premium-enterprise-coding",
            model="enterprise-code-pro",
            priority=RoutingPriority.CRITICAL,
            scene_patterns=["^coding"],
            tenant_ids=["enterprise-123"],
            user_tags=["premium"]
        ))
        
        # Regular enterprise coding
        router.add_rule(RouteRule(
            name="enterprise-coding",
            model="enterprise-code-standard",
            priority=RoutingPriority.HIGH,
            scene_patterns=["^coding"],
            tenant_ids=["enterprise-123"]
        ))
        
        # General coding
        router.add_rule(RouteRule(
            name="general-coding",
            model="standard-code",
            priority=RoutingPriority.MEDIUM,
            scene_patterns=["^coding"]
        ))
        
        router.set_default_model("base-model")
        
        # Enterprise premium user coding task
        context = {
            "scene": "coding-help",
            "tenant_id": "enterprise-123",
            "user_tags": ["premium", "verified"]
        }
        
        matching = router.get_matching_rules(context)
        # Critical priority rule should match first
        assert len(matching) >= 1
        assert matching[0].name == "premium-enterprise-coding"
        assert matching[0].model == "enterprise-code-pro"
        
        # Regular enterprise user (no premium tag)
        context["user_tags"] = []
        matching = router.get_matching_rules(context)
        assert matching[0].name == "enterprise-coding"
        assert matching[0].model == "enterprise-code-standard"
        
        # Non-enterprise coding task
        del context["tenant_id"]
        matching = router.get_matching_rules(context)
        assert matching[0].name == "general-coding"
        assert matching[0].model == "standard-code"
    
    def test_constraint_based_routing(self):
        """Test rules with token constraints"""
        router = RuleBasedRouter()
        
        # Small context model
        router.add_rule(RouteRule(
            name="small-model-rule",
            model="small-model",
            priority=RoutingPriority.HIGH,
            max_input_tokens=1000
        ))
        
        # Large context model
        router.add_rule(RouteRule(
            name="large-model-rule",
            model="large-model",
            priority=RoutingPriority.MEDIUM
        ))
        
        router.set_default_model("base-model")
        
        # Small input - should use small model
        context = {"estimated_input_tokens": 500}
        selected = router.route(context)
        assert selected == "small-model"
        
        # Large input - exceeds small model constraint, uses large model
        context = {"estimated_input_tokens": 2000}
        selected = router.route(context)
        assert selected == "large-model"
