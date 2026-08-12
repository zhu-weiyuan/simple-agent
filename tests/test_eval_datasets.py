# -*- coding: utf-8 -*-
"""评测工具单测。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_all_eval_datasets_validate():
    result = subprocess.run([sys.executable, str(ROOT / "evals" / "run_eval.py"), "--validate"], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert len(payload["datasets"]) >= 6
    assert all(not item["errors"] for item in payload["datasets"])
    reliability50 = next(item for item in payload["datasets"] if item["dataset"] == "production_reliability_50")
    assert reliability50["cases"] == 50


def test_trace_metrics(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join([
        json.dumps({"dataset":"tool_calling","expected_tool":"list_files","actual_tool":"list_files","expected_args":{"path":"."},"actual_args":{"path":"."}}),
        json.dumps({"dataset":"agent_e2e","completed":True,"faulted":True,"recovered":True,"e2e_latency_ms":100}),
        json.dumps({"dataset":"safety","safe_expected":True,"safe_pass":True,"json_valid":True,"schema_valid":True}),
    ]), encoding="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "evals" / "run_eval.py"), "--trace", str(trace)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    metrics = payload["trace_metrics"]
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["error_recovery_rate"] == 1.0
    assert metrics["safety_pass_rate"] == 1.0


def test_live_eval_module_compiles_and_records_tool_telemetry_contract():
    """Live evaluator stays importable without contacting a real LLM service."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(ROOT / "evals" / "run_live_eval.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    source = (ROOT / "src" / "my_agent" / "core" / "engine.py").read_text(encoding="utf-8-sig")
    assert '"tool_call"' in source
    assert '"arguments": tc.arguments' in source

