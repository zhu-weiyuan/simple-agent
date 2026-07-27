# -*- coding: utf-8 -*-
"""
my_agent.router — Rule-based router with scene/tenant routing capabilities

This module provides:
- RouteRule: Definition of a routing rule
- RuleBasedRouter: Router that applies rules to select models based on context
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import re


class RoutingPriority(Enum):
    """Priority levels for route rules"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RouteRule:
    """A single routing rule"""
    name: str
    model: str
    priority: RoutingPriority = RoutingPriority.MEDIUM
    
    # Match conditions
    scene_patterns: Optional[List[str]] = None  # Regex patterns for scene matching
    tenant_ids: Optional[List[str]] = None  # Specific tenant IDs
    user_tags: Optional[List[str]] = None  # Required user tags
    request_types: Optional[List[str]] = None  # e.g., "chat", "completion", "embedding"
    
    # Constraints
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    requires_budget: bool = False
    
    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches_scene(self, scene: Optional[str]) -> bool:
        """Check if this rule matches the given scene"""
        if not self.scene_patterns:
            return True  # No pattern means match all
        
        if not scene:
            return False
            
        for pattern in self.scene_patterns:
            if re.search(pattern, scene, re.IGNORECASE):
                return True
        return False
    
    def matches_tenant(self, tenant_id: Optional[str]) -> bool:
        """Check if this rule matches the given tenant"""
        if not self.tenant_ids:
            return True  # No restriction means match all
        
        if not tenant_id:
            return False
            
        return tenant_id in self.tenant_ids
    
    def matches_user_tags(self, user_tags: Optional[List[str]]) -> bool:
        """Check if user has all required tags"""
        if not self.user_tags:
            return True  # No requirement means match all
        
        if not user_tags:
            return False
            
        return all(tag in user_tags for tag in self.user_tags)
    
    def matches_request_type(self, request_type: Optional[str]) -> bool:
        """Check if this rule matches the request type"""
        if not self.request_types:
            return True  # No restriction
        
        if not request_type:
            return False
            
        return request_type in self.request_types
    
    def is_satisfied_by_context(self, context: Dict[str, Any]) -> bool:
        """Check if all rule conditions are satisfied by the context"""
        scene = context.get("scene")
        tenant_id = context.get("tenant_id") or context.get("tenant")
        user_tags = context.get("user_tags")
        request_type = context.get("request_type")
        
        return (
            self.matches_scene(scene) and
            self.matches_tenant(tenant_id) and
            self.matches_user_tags(user_tags) and
            self.matches_request_type(request_type)
        )


class RuleBasedRouter:
    """Router that selects models based on configurable rules"""
    
    def __init__(self, gateway=None):
        self.rules: List[RouteRule] = []
        self.default_model: Optional[str] = None
        self.gateway = gateway
        self._rules_by_priority: Dict[RoutingPriority, List[RouteRule]] = {
            p: [] for p in RoutingPriority
        }
    
    def add_rule(self, rule: RouteRule):
        """Add a routing rule"""
        self.rules.append(rule)
        self._rules_by_priority[rule.priority].append(rule)
        # Re-sort rules within priority by insertion order (stable)
        self._rules_by_priority[rule.priority].sort(key=lambda r: self.rules.index(r))
    
    def remove_rule(self, name: str):
        """Remove a rule by name"""
        self.rules = [r for r in self.rules if r.name != name]
        for priority_rules in self._rules_by_priority.values():
            priority_rules[:] = [r for r in priority_rules if r.name != name]
    
    def set_default_model(self, model_name: str):
        """Set the default fallback model"""
        self.default_model = model_name
    
    def get_matching_rules(self, context: Dict[str, Any]) -> List[RouteRule]:
        """Get all rules that match the current context, sorted by priority"""
        matching = [
            rule for rule in self.rules
            if rule.is_satisfied_by_context(context)
        ]
        
        # Sort by priority (higher first), then by position in rules list
        matching.sort(
            key=lambda r: (-r.priority.value, self.rules.index(r))
        )
        
        return matching
    
    def route(self, context: Dict[str, Any], 
              budget_id: Optional[str] = None) -> Optional[str]:
        """Route a request to a model based on context
        
        Args:
            context: Request context containing scene, tenant, tags, etc.
            budget_id: Optional budget ID to check against
            
        Returns:
            Model name to use, or None if no suitable model found
        """
        matching_rules = self.get_matching_rules(context)
        
        # Try each matching rule in priority order
        for rule in matching_rules:
            # Check constraints
            input_tokens = context.get("estimated_input_tokens", 0)
            output_tokens = context.get("estimated_output_tokens", 0)
            
            if rule.max_input_tokens and input_tokens > rule.max_input_tokens:
                continue
            if rule.max_output_tokens and output_tokens > rule.max_output_tokens:
                continue
            if rule.requires_budget and not budget_id:
                continue
            
            # Check if model exists and is healthy
            if self.gateway:
                selected = self.gateway.select_model(
                    budget_id=budget_id,
                    needed_tokens=input_tokens + output_tokens,
                    preferred_models=[rule.model]
                )
                if selected:
                    return selected
            else:
                # No gateway, just return the rule's model
                return rule.model
        
        # Fall back to default model
        if self.default_model:
            if self.gateway:
                return self.gateway.select_model(
                    budget_id=budget_id,
                    preferred_models=[self.default_model]
                )
            return self.default_model
        
        return None
    
    def route_with_fallback(self, context: Dict[str, Any],
                           budget_id: Optional[str] = None,
                           health_check_callback: Optional[Callable[[str], bool]] = None) -> Optional[str]:
        """Route with automatic fallback to healthy models
        
        Args:
            context: Request context
            budget_id: Optional budget ID
            health_check_callback: Optional callback to verify model health
            
        Returns:
            Selected model name or None
        """
        primary = self.route(context, budget_id)
        
        if not primary:
            return None
        
        # Verify primary is actually usable
        if health_check_callback and not health_check_callback(primary):
            # Try next best option
            matching_rules = self.get_matching_rules(context)
            for rule in matching_rules:
                if rule.model != primary:
                    candidate = rule.model
                    if not health_check_callback(candidate) or \
                       (self.gateway and not self.gateway.get_route(candidate)):
                        continue
                    
                    if self.gateway:
                        selected = self.gateway.select_model(
                            budget_id=budget_id,
                            preferred_models=[candidate]
                        )
                        if selected:
                            return selected
                    else:
                        return candidate
        
        return primary
    
    def analyze_routing_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which rules matched and why a decision was made
        
        Returns:
            Detailed breakdown of routing decision process
        """
        matching_rules = self.get_matching_rules(context)
        
        result = {
            "context": context,
            "matching_rules": [],
            "selected_model": None,
            "fallback_used": False,
            "reasoning": []
        }
        
        for rule in matching_rules:
            rule_info = {
                "name": rule.name,
                "model": rule.model,
                "priority": rule.priority.name,
                "matched_conditions": []
            }
            
            if rule.scene_patterns and context.get("scene"):
                rule_info["matched_conditions"].append(f"scene={context.get('scene')}")
            if rule.tenant_ids and context.get("tenant_id"):
                rule_info["matched_conditions"].append(f"tenant={context.get('tenant_id')}")
            if rule.user_tags and context.get("user_tags"):
                rule_info["matched_conditions"].append(f"tags={context.get('user_tags')}")
            if rule.request_types and context.get("request_type"):
                rule_info["matched_conditions"].append(f"type={context.get('request_type')}")
            
            result["matching_rules"].append(rule_info)
        
        # Determine what would be selected
        selected = self.route(context)
        result["selected_model"] = selected
        
        if selected:
            result["reasoning"].append(f"Selected model: {selected}")
        elif self.default_model:
            result["reasoning"].append(f"No matching rule, using default: {self.default_model}")
        else:
            result["reasoning"].append("No suitable model found")
        
        return result
    
    def get_rule_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all configured rules"""
        return [
            {
                "name": rule.name,
                "model": rule.model,
                "priority": rule.priority.name,
                "description": rule.description,
                "scene_patterns": rule.scene_patterns,
                "tenant_ids": rule.tenant_ids,
                "constraints": {
                    "max_input_tokens": rule.max_input_tokens,
                    "max_output_tokens": rule.max_output_tokens,
                    "requires_budget": rule.requires_budget
                }
            }
            for rule in self.rules
        ]


# Convenience functions for common routing scenarios

def create_scene_router(gateway=None) -> RuleBasedRouter:
    """Create a router pre-configured for scene-based routing"""
    router = RuleBasedRouter(gateway=gateway)
    
    # Common scene patterns
    router.add_rule(RouteRule(
        name="coding-assistant",
        model="code-model-primary",
        priority=RoutingPriority.HIGH,
        scene_patterns=["^code$", "^programming$", "^development$"],
        description="Route coding tasks to specialized code model"
    ))
    
    router.add_rule(
        RouteRule(
            name="creative-writing",
            model="creative-model",
            priority=RoutingPriority.MEDIUM,
            scene_patterns=["^creative$", "^writing$", "^story$"],
            description="Route creative writing to specialized model"
        )
    )
    
    router.add_rule(
        RouteRule(
            name="analysis-task",
            model="analysis-model",
            priority=RoutingPriority.MEDIUM,
            scene_patterns=["^analysis$", "^research$", "^data$"],
            description="Route analysis tasks to reasoning-optimized model"
        )
    )
    
    return router


def create_tenant_router(gateway=None, tenant_defaults: Optional[Dict[str, str]] = None) -> RuleBasedRouter:
    """Create a router pre-configured for tenant-based routing"""
    router = RuleBasedRouter(gateway=gateway)
    tenant_defaults = tenant_defaults or {}
    
    for tenant_id, model in tenant_defaults.items():
        router.add_rule(
            RouteRule(
                name=f"tenant-{tenant_id}",
                model=model,
                priority=RoutingPriority.HIGH,
                tenant_ids=[tenant_id],
                description=f"Default model for tenant {tenant_id}"
            )
        )
    
    return router
