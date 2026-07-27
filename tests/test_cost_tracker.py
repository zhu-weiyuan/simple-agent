# -*- coding: utf-8 -*-
"""
tests.test_cost_tracker — Tests for UsageRecord and CostTracker
"""

import pytest
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from my_agent.cost_tracker import (
    UsageRecord,
    CostSummary,
    CostTracker
)


class TestUsageRecord:
    """Tests for UsageRecord dataclass"""
    
    def test_basic_record_creation(self):
        record = UsageRecord(
            id="rec-123",
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            input_cost=0.03,
            output_cost=0.03,
            total_cost=0.06
        )
        
        assert record.id == "rec-123"
        assert record.model == "gpt-4"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.total_tokens == 1500
    
    def test_metadata_fields(self):
        record = UsageRecord(
            id="rec-456",
            model="claude-3",
            input_tokens=2000,
            output_tokens=1000,
            total_tokens=3000,
            input_cost=0.015,
            output_cost=0.075,
            total_cost=0.09,
            budget_id="budget-789",
            tenant_id="tenant-alpha",
            scene="coding",
            user_id="user-42",
            request_type="chat",
            metadata={"session_id": "sess-xyz"}
        )
        
        assert record.budget_id == "budget-789"
        assert record.tenant_id == "tenant-alpha"
        assert record.scene == "coding"
        assert record.user_id == "user-42"
        assert record.request_type == "chat"
        assert record.metadata["session_id"] == "sess-xyz"
    
    def test_timestamp_default(self):
        before = time.time()
        record = UsageRecord(
            id="rec-time",
            model="test-model",
            input_tokens=100,
            output_tokens=100,
            total_tokens=200,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0
        )
        after = time.time()
        
        assert before <= record.created_at <= after
    
    def test_missing_id_rejected(self):
        with pytest.raises(ValueError, match="requires an id"):
            UsageRecord(
                id="",
                model="m1",
                input_tokens=100,
                output_tokens=100,
                total_tokens=200,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0
            )
    
    def test_negative_tokens_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            UsageRecord(
                id="bad",
                model="m1",
                input_tokens=-100,
                output_tokens=100,
                total_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0
            )
    
    def test_total_tokens_must_match_sum(self):
        with pytest.raises(ValueError, match="total_tokens must equal"):
            UsageRecord(
                id="bad",
                model="m1",
                input_tokens=100,
                output_tokens=100,
                total_tokens=300,  # Wrong sum!
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0
            )


class TestCostSummary:
    """Tests for CostSummary dataclass"""
    
    def test_basic_summary_creation(self):
        summary = CostSummary(
            period_start=1000.0,
            period_end=2000.0,
            total_input_tokens=5000,
            total_output_tokens=2000,
            total_tokens=7000,
            total_cost=0.21,
            currency="USD",
            request_count=10
        )
        
        assert summary.period_start == 1000.0
        assert summary.period_end == 2000.0
        assert summary.request_count == 10
    
    def test_average_cost_per_request(self):
        summary = CostSummary(
            period_start=0,
            period_end=0,
            total_cost=1.50,
            request_count=3
        )
        
        assert summary.average_cost_per_request == 0.50
    
    def test_average_cost_no_requests(self):
        summary = CostSummary(
            period_start=0,
            period_end=0,
            total_cost=0.0,
            request_count=0
        )
        
        assert summary.average_cost_per_request == 0.0
    
    def test_average_tokens_per_request(self):
        summary = CostSummary(
            period_start=0,
            period_end=0,
            total_tokens=10000,
            request_count=5
        )
        
        assert summary.average_tokens_per_request == 2000.0
    
    def test_by_model_breakdown(self):
        summary = CostSummary(
            period_start=0,
            period_end=0,
            by_model={
                "gpt-4": {"cost": 0.30, "requests": 2},
                "gpt-3.5": {"cost": 0.10, "requests": 3}
            }
        )
        
        assert "gpt-4" in summary.by_model
        assert summary.by_model["gpt-4"]["cost"] == 0.30


class TestCostTracker:
    """Tests for CostTracker class"""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker with some test data"""
        t = CostTracker()
        
        # Set model costs
        t.set_model_cost("gpt-4", input_per_1m=30.0, output_per_1m=60.0)
        t.set_model_cost("gpt-3.5-turbo", input_per_1m=0.5, output_per_1m=1.5)
        t.set_model_cost("claude-3", input_per_1m=15.0, output_per_1m=75.0)
        
        return t
    
    def test_set_model_cost(self, tracker):
        tracker.set_model_cost("new-model", input_per_1m=10.0, output_per_1m=20.0)
        
        cost = tracker.get_model_cost("new-model")
        assert cost["input_rate"] == 10.0
        assert cost["output_rate"] == 20.0
    
    def test_get_unknown_model_cost(self, tracker):
        cost = tracker.get_model_cost("unknown-model")
        assert cost is None
    
    def test_set_tenant_budget(self, tracker):
        tracker.set_tenant_budget("tenant-123", budget=100.0, alert_threshold=0.7)
        
        assert tracker.tenant_budgets["tenant-123"] == 100.0
        assert tracker.alert_thresholds["tenant-123"] == 0.7
    
    def test_record_usage_basic(self, tracker):
        record = tracker.record_usage(
            model="gpt-4",
            input_tokens=1000000,  # 1M tokens
            output_tokens=500000   # 0.5M tokens
        )
        
        # gpt-4: $30/1M input, $60/1M output
        # Cost: (1 * 30) + (0.5 * 60) = 30 + 30 = $60
        assert record.model == "gpt-4"
        assert record.input_tokens == 1000000
        assert record.output_tokens == 500000
        assert abs(record.input_cost - 30.0) < 0.001
        assert abs(record.output_cost - 30.0) < 0.001
        assert abs(record.total_cost - 60.0) < 0.001
    
    def test_record_usage_zero_cost_model(self, tracker):
        # Model without configured cost should use zeros
        record = tracker.record_usage(
            model="local-model",
            input_tokens=1000,
            output_tokens=500
        )
        
        assert record.input_cost == 0.0
        assert record.output_cost == 0.0
        assert record.total_cost == 0.0
    
    def test_record_batch(self, tracker):
        records_data = [
            {
                "model": "gpt-4",
                "input_tokens": 1000,
                "output_tokens": 500,
                "tenant_id": "tenant-a"
            },
            {
                "model": "gpt-3.5-turbo",
                "input_tokens": 2000,
                "output_tokens": 1000,
                "tenant_id": "tenant-b"
            }
        ]
        
        records = tracker.record_batch(records_data)
        
        assert len(records) == 2
        assert all(isinstance(r, UsageRecord) for r in records)
        assert records[0].tenant_id == "tenant-a"
        assert records[1].tenant_id == "tenant-b"
    
    def test_get_records_all(self, tracker):
        tracker.record_usage(model="m1", input_tokens=100, output_tokens=100)
        tracker.record_usage(model="m2", input_tokens=200, output_tokens=200)
        
        records = tracker.get_records()
        
        assert len(records) == 2
    
    def test_get_records_filtered_by_time(self, tracker):
        before = time.time()
        tracker.record_usage(model="m1", input_tokens=100, output_tokens=100)
        time.sleep(0.01)
        after = time.time()
        tracker.record_usage(model="m2", input_tokens=200, output_tokens=200)
        
        # Filter to only first record
        records = tracker.get_records(start_time=before, end_time=after)
        
        # Should get at least one record
        assert len(records) >= 1
    
    def test_get_records_filtered_by_model(self, tracker):
        tracker.record_usage(model="gpt-4", input_tokens=100, output_tokens=100)
        tracker.record_usage(model="gpt-3.5-turbo", input_tokens=200, output_tokens=200)
        tracker.record_usage(model="gpt-4", input_tokens=300, output_tokens=300)
        
        records = tracker.get_records(model="gpt-4")
        
        assert len(records) == 2
        assert all(r.model == "gpt-4" for r in records)
    
    def test_get_records_filtered_by_tenant(self, tracker):
        tracker.record_usage(model="m1", input_tokens=100, output_tokens=100, tenant_id="tenant-a")
        tracker.record_usage(model="m1", input_tokens=200, output_tokens=200, tenant_id="tenant-b")
        tracker.record_usage(model="m1", input_tokens=300, output_tokens=300, tenant_id="tenant-a")
        
        records = tracker.get_records(tenant_id="tenant-a")
        
        assert len(records) == 2
        assert all(r.tenant_id == "tenant-a" for r in records)
    
    def test_get_summary_basic(self, tracker):
        tracker.record_usage(model="gpt-4", input_tokens=1000000, output_tokens=1000000)
        tracker.record_usage(model="gpt-4", input_tokens=1000000, output_tokens=1000000)
        
        summary = tracker.get_summary()
        
        assert summary.total_input_tokens == 2000000
        assert summary.total_output_tokens == 2000000
        assert summary.total_tokens == 4000000
        assert summary.request_count == 2
        # Cost: 2 * ((1M * $30/1M) + (1M * $60/1M)) = 2 * $90 = $180
        assert abs(summary.total_cost - 180.0) < 0.01
    
    def test_get_summary_grouped_by_model(self, tracker):
        tracker.record_usage(model="gpt-4", input_tokens=1000, output_tokens=1000)
        tracker.record_usage(model="gpt-3.5-turbo", input_tokens=2000, output_tokens=2000)
        tracker.record_usage(model="gpt-4", input_tokens=3000, output_tokens=3000)
        
        summary = tracker.get_summary(group_by="model")
        
        assert "gpt-4" in summary.by_model
        assert "gpt-3.5-turbo" in summary.by_model
        
        assert summary.by_model["gpt-4"]["requests"] == 2
        assert summary.by_model["gpt-3.5-turbo"]["requests"] == 1
    
    def test_get_summary_grouped_by_tenant(self, tracker):
        tracker.record_usage(model="m1", input_tokens=100, output_tokens=100, tenant_id="tenant-a")
        tracker.record_usage(model="m1", input_tokens=200, output_tokens=200, tenant_id="tenant-b")
        
        summary = tracker.get_summary(group_by="tenant")
        
        assert "tenant-a" in summary.by_tenant
        assert "tenant-b" in summary.by_tenant
    
    def test_get_tenant_costs(self, tracker):
        tracker.set_tenant_budget("enterprise", budget=500.0, alert_threshold=0.8)
        
        tracker.record_usage(
            model="gpt-4",
            input_tokens=5000000,  # 5M tokens
            output_tokens=5000000,
            tenant_id="enterprise"
        )
        
        # Cost: (5 * $30) + (5 * $60) = $150 + $300 = $450
        result = tracker.get_tenant_costs("enterprise")
        
        assert result["tenant_id"] == "enterprise"
        assert abs(result["cost"] - 450.0) < 0.01
        assert result["budget"]["monthly_limit"] == 500.0
        assert abs(result["budget"]["used"] - 450.0) < 0.01
        assert abs(result["budget"]["remaining"] - 50.0) < 0.01
        assert result["budget"]["usage_percent"] > 0.8  # Above threshold
        assert "alert" in result
    
    def test_get_tenant_costs_no_budget_configured(self, tracker):
        tracker.record_usage(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=1000,
            tenant_id="no-budget-tenant"
        )
        
        result = tracker.get_tenant_costs("no-budget-tenant")
        
        assert "budget" not in result
    
    def test_get_model_breakdown(self, tracker):
        tracker.record_usage(model="gpt-4", input_tokens=1000, output_tokens=1000)
        tracker.record_usage(model="gpt-4", input_tokens=2000, output_tokens=2000)
        tracker.record_usage(model="gpt-3.5-turbo", input_tokens=3000, output_tokens=3000)
        
        breakdown = tracker.get_model_breakdown()
        
        assert "gpt-4" in breakdown
        assert breakdown["gpt-4"]["total_tokens"] == 6000
        assert breakdown["gpt-4"]["requests"] == 2
        assert breakdown["gpt-4"]["avg_tokens_per_request"] == 3000.0
    
    def test_check_budget_alerts(self, tracker):
        tracker.set_tenant_budget("warning-tenant", budget=100.0, alert_threshold=0.8)
        tracker.set_tenant_budget("ok-tenant", budget=1000.0, alert_threshold=0.8)
        
        # Create high usage for warning-tenant
        tracker.record_usage(
            model="gpt-4",
            input_tokens=2000000,
            output_tokens=2000000,
            tenant_id="warning-tenant"
        )
        # Cost: (2 * 30) + (2 * 60) = $60 + $120 = $180 (> $100 budget)
        
        alerts = tracker.check_budget_alerts()
        
        # Should have alert for warning-tenant
        alert_tenants = [a["tenant_id"] for a in alerts]
        assert "warning-tenant" in alert_tenants
        assert "ok-tenant" not in alert_tenants
    
    def test_check_budget_alerts_for_specific_tenant(self, tracker):
        tracker.set_tenant_budget("tenant-x", budget=50.0, alert_threshold=0.5)
        
        tracker.record_usage(
            model="gpt-4",
            input_tokens=1000000,
            output_tokens=1000000,
            tenant_id="tenant-x"
        )
        # Cost: $90 > $50 budget
        
        alerts = tracker.check_budget_alerts(tenant_id="tenant-x")
        
        assert len(alerts) == 1
        assert alerts[0]["tenant_id"] == "tenant-x"
    
    def test_export_report_summary(self, tracker):
        tracker.record_usage(model="gpt-4", input_tokens=1000, output_tokens=1000)
        tracker.record_usage(model="gpt-3.5-turbo", input_tokens=2000, output_tokens=2000)
        
        report = tracker.export_report(format="summary")
        
        assert report["report_type"] == "summary"
        assert report["totals"]["request_count"] == 2
        assert "currency" in report
    
    def test_export_report_detailed(self, tracker):
        tracker.record_usage(model="gpt-4", input_tokens=1000, output_tokens=1000)
        tracker.record_usage(model="gpt-3.5-turbo", input_tokens=2000, output_tokens=2000)
        
        report = tracker.export_report(format="detailed")
        
        assert report["report_type"] == "detailed"
        assert "by_model" in report
        assert "gpt-4" in report["by_model"]
    
    def test_export_report_invalid_format(self, tracker):
        with pytest.raises(ValueError, match="Unknown format"):
            tracker.export_report(format="invalid")
    
    def test_clear_old_records(self, tracker):
        # Record some old data (simulate by manually setting timestamp)
        import time as time_module
        
        current_time = time_module.time()
        old_time = current_time - (100 * 24 * 60 * 60)  # 100 days ago
        
        tracker.record_usage(model="m1", input_tokens=100, output_tokens=100)
        # Add an artificially old record
        old_record = UsageRecord(
            id="old-rec",
            model="m1",
            input_tokens=1000,
            output_tokens=1000,
            total_tokens=2000,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0
        )
        old_record.created_at = old_time
        tracker.records.append(old_record)
        
        assert len(tracker.records) == 2
        
        # Clear records older than 90 days
        tracker.clear_old_records(older_than_days=90)
        
        assert len(tracker.records) == 1
        assert tracker.records[0].id != "old-rec"


class TestIntegration:
    """Integration tests for cost tracking scenarios"""
    
    def test_full_cost_tracking_workflow(self):
        """Test complete workflow: setup -> record -> analyze -> report"""
        tracker = CostTracker()
        
        # Configure models and tenants
        tracker.set_model_cost("gpt-4-turbo", input_per_1m=10.0, output_per_1m=30.0)
        tracker.set_model_cost("llama-3", input_per_1m=0.0, output_per_1m=0.0)  # Free local model
        
        tracker.set_tenant_budget("startup-123", budget=50.0, alert_threshold=0.7)
        
        # Record various usage patterns
        tracker.record_usage(
            model="gpt-4-turbo",
            input_tokens=500000,
            output_tokens=250000,
            tenant_id="startup-123",
            scene="code-generation"
        )
        
        tracker.record_usage(
            model="llama-3",
            input_tokens=1000000,
            output_tokens=500000,
            tenant_id="startup-123",
            scene="general-chat"
        )
        
        tracker.record_usage(
            model="gpt-4-turbo",
            input_tokens=300000,
            output_tokens=150000,
            tenant_id="startup-123",
            scene="code-review"
        )
        
        # Check costs
        tenant_costs = tracker.get_tenant_costs("startup-123")
        
        # gpt-4-turbo first call: (0.5M * $10/1M) + (0.25M * $30/1M) = $5 + $7.5 = $12.5
        # llama-3: $0
        # gpt-4-turbo second call: (0.3M * $10/1M) + (0.15M * $30/1M) = $3 + $4.5 = $7.5
        # Total: $12.5 + $0 + $7.5 = $20.0
        assert abs(tenant_costs["cost"] - 20.0) < 0.01
        assert tenant_costs["budget"]["usage_percent"] < 0.7  # Below alert threshold (20/50 = 0.4)
        
        # Get model breakdown
        breakdown = tracker.get_model_breakdown()
        assert breakdown["gpt-4-turbo"]["total_tokens"] == 1200000
        assert breakdown["llama-3"]["total_tokens"] == 1500000
        assert abs(breakdown["gpt-4-turbo"]["cost"] - 20.0) < 0.01
        assert breakdown["llama-3"]["cost"] == 0.0
        
        # Export report
        report = tracker.export_report(format="detailed")
        assert report["totals"]["total_tokens"] == 2700000
        assert report["totals"]["request_count"] == 3
    
    def test_multi_tenant_cost_isolation(self):
        """Verify costs are properly isolated across tenants"""
        tracker = CostTracker()
        
        tracker.set_model_cost("standard-model", input_per_1m=10.0, output_per_1m=20.0)
        
        tracker.set_tenant_budget("tenant-a", budget=100.0)
        tracker.set_tenant_budget("tenant-b", budget=200.0)
        
        # Tenant A usage
        tracker.record_usage(
            model="standard-model",
            input_tokens=1000000,
            output_tokens=1000000,
            tenant_id="tenant-a"
        )
        # Cost: (1 * 10) + (1 * 20) = $30
        
        # Tenant B usage
        tracker.record_usage(
            model="standard-model",
            input_tokens=5000000,
            output_tokens=5000000,
            tenant_id="tenant-b"
        )
        # Cost: (5 * 10) + (5 * 20) = $150
        
        # Verify isolation
        tenant_a_costs = tracker.get_tenant_costs("tenant-a")
        tenant_b_costs = tracker.get_tenant_costs("tenant-b")
        
        assert abs(tenant_a_costs["cost"] - 30.0) < 0.01
        assert abs(tenant_b_costs["cost"] - 150.0) < 0.01
        
        # Summary by tenant
        summary = tracker.get_summary(group_by="tenant")
        assert abs(summary.by_tenant["tenant-a"]["cost"] - 30.0) < 0.01
        assert abs(summary.by_tenant["tenant-b"]["cost"] - 150.0) < 0.01
    
    def test_scene_based_cost_analysis(self):
        """Analyze costs broken down by scene"""
        tracker = CostTracker()
        
        tracker.set_model_cost("pro-model", input_per_1m=50.0, output_per_1m=100.0)
        
        # Record various scenes
        tracker.record_usage(
            model="pro-model",
            input_tokens=100000,
            output_tokens=50000,
            scene="coding"
        )
        tracker.record_usage(
            model="pro-model",
            input_tokens=200000,
            output_tokens=100000,
            scene="writing"
        )
        tracker.record_usage(
            model="pro-model",
            input_tokens=150000,
            output_tokens=75000,
            scene="coding"
        )
        
        summary = tracker.get_summary(group_by="scene")
        
        assert "coding" in summary.by_scene
        assert "writing" in summary.by_scene
        
        assert summary.by_scene["coding"]["requests"] == 2
        assert summary.by_scene["writing"]["requests"] == 1
        # coding: (100k+50k) + (150k+75k) = 150k + 225k = 375k
        assert summary.by_scene["coding"]["total_tokens"] == 375000
        # writing: 200k + 100k = 300k
        assert summary.by_scene["writing"]["total_tokens"] == 300000
