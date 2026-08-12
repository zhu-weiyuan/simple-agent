# -*- coding: utf-8 -*-
"""Deterministic evaluation dataset validation and trace metrics."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
DEFAULT_VERSION = "1.0"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_no}: each row must be a JSON object")
        rows.append(value)
    return rows


def validate_dataset(path: Path) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    errors: List[str] = []
    for index, row in enumerate(rows, 1):
        if not row.get("id"):
            errors.append(f"row {index}: missing id")
        if not row.get("input"):
            errors.append(f"row {index}: missing input")
        if path.name == "tool_calling.jsonl":
            if "allow_no_tool" not in row:
                errors.append(f"row {index}: missing allow_no_tool")
            if not row.get("expected_tool") and not bool(row.get("allow_no_tool")):
                errors.append(f"row {index}: missing expected_tool without no-tool allowance")
        if path.name == "tool_discovery.jsonl":
            candidates = row.get("candidate_tools")
            if not isinstance(candidates, list) or not candidates:
                errors.append(f"row {index}: candidate_tools must be a non-empty list")
            expected = row.get("expected_tool")
            if expected is not None and expected not in candidates:
                errors.append(f"row {index}: expected_tool must be a candidate")
        if path.name == "structured_output.jsonl" and not isinstance(row.get("schema"), dict):
            errors.append(f"row {index}: schema must be an object")
        if path.name == "performance.jsonl" and not isinstance(row.get("budget"), dict):
            errors.append(f"row {index}: budget must be an object")
        if path.name == "production_reliability_50.jsonl":
            if row.get("kind") not in {"tool", "structured"}:
                errors.append(f"row {index}: kind must be tool or structured")
            if not isinstance(row.get("category"), str) or not row["category"].strip():
                errors.append(f"row {index}: category must be a non-empty string")
            if not isinstance(row.get("difficulty"), str) or not row["difficulty"].strip():
                errors.append(f"row {index}: difficulty must be a non-empty string")
            if row.get("kind") == "tool":
                if "allow_no_tool" not in row:
                    errors.append(f"row {index}: tool case missing allow_no_tool")
                if not row.get("expected_tool") and not bool(row.get("allow_no_tool")):
                    errors.append(f"row {index}: tool case missing expected_tool without no-tool allowance")
                if row.get("expected_args") is not None and not isinstance(row.get("expected_args"), dict):
                    errors.append(f"row {index}: expected_args must be an object when provided")
                if row.get("expected_calls") is not None:
                    expected_calls = row["expected_calls"]
                    if not isinstance(expected_calls, list) or not expected_calls:
                        errors.append(f"row {index}: expected_calls must be a non-empty list when provided")
                    elif any(not isinstance(call, dict) or not call.get("name") or not isinstance(call.get("arguments", {}), dict) for call in expected_calls):
                        errors.append(f"row {index}: each expected_call requires name and object arguments")
            if row.get("kind") == "structured" and not isinstance(row.get("schema"), dict):
                errors.append(f"row {index}: structured case schema must be an object")
    if path.name == "production_reliability_50.jsonl":
        if len(rows) != 50:
            errors.append(f"dataset must contain exactly 50 cases, got {len(rows)}")
        ids = [str(row.get("id", "")) for row in rows]
        duplicates = sorted(item for item, count in Counter(ids).items() if item and count > 1)
        if duplicates:
            errors.append("duplicate ids: " + ", ".join(duplicates))
    return {"dataset": path.stem, "version": DEFAULT_VERSION, "cases": len(rows), "errors": errors}


def load_all() -> List[Dict[str, Any]]:
    return [validate_dataset(path) for path in sorted(DATASETS.glob("*.jsonl"))]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: List[float], fraction: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return round(values[index], 2)


def score_trace(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate deterministic metrics from trace rows; missing fields are not successes."""
    rows = list(rows)
    counts = Counter()
    latency: List[float] = []
    for row in rows:
        expected_tool = row.get("expected_tool")
        actual_tool = row.get("actual_tool")
        if expected_tool is None:
            counts["tool_no_call_correct"] += int(actual_tool in (None, ""))
        else:
            counts["tool_selection_correct"] += int(actual_tool == expected_tool)
        if "expected_args" in row:
            counts["tool_args_correct"] += int(row.get("actual_args") == row.get("expected_args"))
        if row.get("unnecessary_call") is not None:
            counts["unnecessary_total"] += 1
            counts["unnecessary_calls"] += int(bool(row["unnecessary_call"]))
        if row.get("completed") is not None:
            counts["completed"] += int(bool(row["completed"]))
        if row.get("faulted"):
            counts["fault_total"] += 1
            counts["recovered"] += int(bool(row.get("recovered")))
        if row.get("safe_expected") is not None:
            counts["safety_total"] += 1
            counts["safe_pass"] += int(bool(row.get("safe_pass")))
        if row.get("json_valid") is not None:
            counts["json_total"] += 1
            counts["json_valid"] += int(bool(row["json_valid"]))
        if row.get("schema_valid") is not None:
            counts["schema_total"] += 1
            counts["schema_valid"] += int(bool(row["schema_valid"]))
        value = _as_float(row.get("e2e_latency_ms"))
        if value is not None:
            latency.append(value)

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "samples": len(rows),
        "tool_selection_accuracy": rate(counts["tool_selection_correct"] + counts["tool_no_call_correct"], len(rows)),
        "tool_argument_accuracy": rate(counts["tool_args_correct"], sum(1 for row in rows if "expected_args" in row)),
        "unnecessary_call_rate": rate(counts["unnecessary_calls"], counts["unnecessary_total"]),
        "task_completion_rate": rate(counts["completed"], sum(1 for row in rows if row.get("completed") is not None)),
        "error_recovery_rate": rate(counts["recovered"], counts["fault_total"]),
        "safety_pass_rate": rate(counts["safe_pass"], counts["safety_total"]),
        "json_valid_rate": rate(counts["json_valid"], counts["json_total"]),
        "schema_pass_rate": rate(counts["schema_valid"], counts["schema_total"]),
        "latency_ms": {"count": len(latency), "p50": _percentile(latency, 0.50), "p95": _percentile(latency, 0.95)} if latency else {},
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SimpleAgent non-RAG evaluation tool")
    parser.add_argument("--validate", action="store_true", help="validate all JSONL datasets")
    parser.add_argument("--trace", type=Path, help="runtime trace JSONL")
    parser.add_argument("--output", type=Path, help="output report JSON")
    args = parser.parse_args(argv)
    if not args.validate and not args.trace:
        parser.error("specify --validate or --trace")
    report: Dict[str, Any] = {"generated_at": time.time(), "dataset_version": DEFAULT_VERSION}
    if args.validate:
        report["datasets"] = load_all()
    if args.trace:
        report["trace_metrics"] = score_trace(_read_jsonl(args.trace))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if any(item.get("errors") for item in report.get("datasets", [])) else 0


if __name__ == "__main__":
    raise SystemExit(main())
