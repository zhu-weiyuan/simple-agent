"""Reusable local evaluation harness for deterministic agent regression checks."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass
class EvalRecord:
    eval_id: str
    task_id: str
    trial_id: str
    prompt_version: str
    model_id: str
    dataset_version: str
    input_hash: str
    raw_input: str
    reference_output: Optional[Any]
    actual_output: Any
    outcome_status: str
    scores: Dict[str, float]
    judge_reasoning: str = ""
    error_category: Optional[str] = None
    confidence: float = 1.0
    evaluated_at: float = field(default_factory=time.time)


@dataclass
class Trial:
    trial_id: str
    input: str
    reference_output: Optional[Any] = None


@dataclass
class Task:
    task_id: str
    trials: List[Trial] = field(default_factory=list)


class GoldenSet:
    def __init__(self, version: str = "1.0", tasks: Optional[Dict[str, Task]] = None) -> None:
        self.version, self.tasks = version, tasks or {}

    def add_trial(self, task_id: str, input: str, reference_output: Optional[Any] = None) -> Trial:
        task = self.tasks.setdefault(task_id, Task(task_id))
        trial = Trial(uuid.uuid4().hex, input, reference_output)
        task.trials.append(trial)
        return trial

    def get_task(self, task_id: str) -> Task:
        return self.tasks[task_id]

    def export_to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"version": self.version, "tasks": [asdict(task) for task in self.tasks.values()]}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_from_json(cls, path: str | Path) -> "GoldenSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tasks = {item["task_id"]: Task(item["task_id"], [Trial(**trial) for trial in item.get("trials", [])]) for item in data.get("tasks", [])}
        return cls(data.get("version", "1.0"), tasks)

    def get_pass_rate(self, records: Optional[List[EvalRecord]] = None) -> float:
        return sum(record.outcome_status == "SUCCESS" for record in records) / len(records) if records else 0.0


class Scorer:
    @staticmethod
    def format_scorer(actual_output: str, expected_format: str = "json") -> float:
        try:
            if expected_format.lower() == "json": json.loads(actual_output)
            elif expected_format.lower() == "xml":
                from xml.etree import ElementTree
                ElementTree.fromstring(actual_output)
            else: raise ValueError("unsupported format")
            return 1.0
        except Exception:
            return 0.0

    @staticmethod
    def schema_scorer(actual_output: str | Dict[str, Any], schema_dict: Dict[str, Any]) -> float:
        try: data = json.loads(actual_output) if isinstance(actual_output, str) else actual_output
        except json.JSONDecodeError: return 0.0
        type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
        for key in schema_dict.get("required", []):
            if key not in data: return 0.0
        for key, rule in schema_dict.get("properties", {}).items():
            if key in data and rule.get("type") and (not isinstance(data[key], type_map[rule["type"]]) or (rule["type"] == "integer" and isinstance(data[key], bool))): return 0.0
            if key in data and "enum" in rule and data[key] not in rule["enum"]: return 0.0
        return 1.0

    @staticmethod
    def field_accuracy_scorer(actual_output_dict: Dict[str, Any], reference_dict: Dict[str, Any], key_fields: List[str]) -> float:
        return sum(actual_output_dict.get(key) == reference_dict.get(key) for key in key_fields) / len(key_fields) if key_fields else 1.0

    @staticmethod
    def faithfulness_scorer(query: str, context: str, answer: str) -> float:
        if not context: return 1.0
        context_terms, answer_terms = ({word.lower() for word in text.split() if len(word) > 3} for text in (context, answer))
        unsupported = answer_terms - context_terms - {word.lower() for word in query.split()}
        return 1.0 if len(unsupported) <= max(3, len(answer_terms) // 3) else 0.0


class LLMJudgeScorer:
    """Score an evaluation answer with an OpenAI-compatible local LLM."""
    def __init__(self, endpoint: str = "http://localhost:8080/v1/chat/completions", timeout: float = 15.0) -> None:
        self.endpoint, self.timeout = endpoint, timeout

    def judge(self, prompt: str, context: str, candidate_output: Any, reference_output: Any = None) -> Dict[str, Any]:
        rubric = """You are a strict evaluation judge. Compare the candidate to the task, context, and reference. Score correctness, relevance, faithfulness, and format from 0-10. Return ONLY JSON: {\"score\": number, \"reasoning\": string, \"dimensions\": [{\"name\": string, \"score\": number}]}."""
        message = f"{rubric}\nTask: {prompt}\nContext: {context}\nReference: {reference_output}\nCandidate: {candidate_output}"
        try:
            response = requests.post(self.endpoint, json={"messages": [{"role": "user", "content": message}], "temperature": 0}, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, str):
                content = content.removeprefix("```json").removesuffix("```").strip()
            result = json.loads(content) if isinstance(content, str) else content
            score = max(0.0, min(10.0, float(result["score"])))
            dimensions = result.get("dimensions", [])
            return {"score": score, "reasoning": str(result.get("reasoning", ""))[:1000], "dimensions": dimensions if isinstance(dimensions, list) else []}
        except (requests.RequestException, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {"score": 0.0, "reasoning": f"Judge unavailable: {type(exc).__name__}", "dimensions": []}


class EvalRunner:
    def __init__(self, agent: Any, golden_set: GoldenSet, prompt_version: str = "default", model_id: str = "unknown", *, llm_judge_enabled: bool = True, judge_scorer: Optional[LLMJudgeScorer] = None) -> None:
        self.agent, self.golden_set, self.prompt_version, self.model_id = agent, golden_set, prompt_version, model_id
        self.llm_judge_enabled, self.judge_scorer = llm_judge_enabled, judge_scorer or LLMJudgeScorer()

    def run(self) -> List[EvalRecord]:
        records = []
        for task in self.golden_set.tasks.values():
            for trial in task.trials:
                try:
                    actual = self.agent.run(trial.input, prompt_version=self.prompt_version)
                    score = 1.0 if trial.reference_output is None else float(str(actual).strip() == str(trial.reference_output).strip())
                    status, error = ("SUCCESS" if score == 1.0 else "PARTIAL"), None
                except Exception as exc:
                    actual, score, status, error = "", 0.0, "FAILED", type(exc).__name__
                scores, reasoning = {"exact_match": score}, "exact reference comparison"
                if self.llm_judge_enabled:
                    judge = self.judge_scorer.judge(trial.input, "", actual, trial.reference_output)
                    scores["llm_judge"] = float(judge["score"])
                    reasoning = judge["reasoning"]
                records.append(EvalRecord(uuid.uuid4().hex, task.task_id, trial.trial_id, self.prompt_version, self.model_id, self.golden_set.version, hashlib.sha256(trial.input.encode()).hexdigest(), trial.input, trial.reference_output, actual, status, scores, reasoning, error))
        return records

    def export_summary_report(self, path: str | Path, records: List[EvalRecord]) -> None:
        summary = {"pass_rate": self.golden_set.get_pass_rate(records), "records": [asdict(record) for record in records], "failed_cases": [asdict(record) for record in records if record.outcome_status != "SUCCESS"]}
        Path(path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
