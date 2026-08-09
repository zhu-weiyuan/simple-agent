import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from my_agent.async_task import AsyncTaskQueue
from my_agent.core.token_budget import TokenBudgetManager, TokenEstimator
from my_agent.prompt_registry import PromptRegistry


def test_token_estimator_and_budget_alerts():
    assert TokenEstimator().estimate("hello world") > 0
    manager = TokenBudgetManager(100, reserved_for_output=20)
    assert manager.allocate(10, 10, 10, 70) == ["system", "memory", "rag"]
    assert "Dumb Zone" in manager.alerts_for_usage(81)[0]


def test_prompt_registry_validates_and_renders():
    registry = PromptRegistry()
    registry.register("greeting", "2", "Hello {{name}}")
    assert registry.render("greeting", name="Ada") == "Hello Ada"
    try:
        registry.render("greeting")
        assert False
    except ValueError as exc:
        assert "name" in str(exc)


def test_async_task_queue_result():
    queue = AsyncTaskQueue(1)
    task_id = queue.submit("add", lambda a, b: a + b, args=(2, 3))
    for _ in range(30):
        if queue.get_status(task_id) == "COMPLETED":
            break
        time.sleep(.01)
    assert queue.get_status(task_id) == "COMPLETED"
    assert queue.get_result(task_id) == 5
    queue.shutdown()


def test_sample_golden_set_is_valid():
    data = json.loads((Path(__file__).parent / "tests" / "data" / "sample_golden_set.json").read_text(encoding="utf-8"))
    assert 5 <= sum(len(task["trials"]) for task in data["tasks"]) <= 10
