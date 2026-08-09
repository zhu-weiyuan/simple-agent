"""Integration tests for the FastAPI A2A routes."""
import io, os, sys, time
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))

import pytest
from fastapi.testclient import TestClient

import app_prod


@pytest.fixture(scope="module")
def client():
    with TestClient(app_prod.app) as c:
        yield c


def test_a2a_card_endpoint(client):
    r = client.get("/a2a/.well-known/agent.json")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "SimpleAgent"
    assert body["version"] == "2.1.0"


def test_a2a_submit_and_poll_local_task(client):
    r = client.post("/a2a/messages", json={"message": "你好", "task_id": "it-001"})
    assert r.status_code in (200, 202)
    body = r.json()
    assert body["taskId"] == "it-001"
    assert body["state"] in {"submitted", "working", "completed"}

    deadline = time.time() + 10
    terminal = {"completed", "failed", "timed_out", "cancelled"}
    status = None
    while time.time() < deadline:
        s = client.get(f"/a2a/tasks/{body['taskId']}")
        assert s.status_code == 200
        status = s.json()
        if status["state"] in terminal:
            break
        time.sleep(0.2)
    assert status is not None
    assert status["state"] in terminal


def test_a2a_idempotent_task_id(client):
    first = client.post("/a2a/messages", json={"message": "A2A 幂等测试", "task_id": "it-idem"})
    second = client.post("/a2a/messages", json={"message": "A2A 幂等测试", "task_id": "it-idem"})
    assert second.status_code in (200, 202)
    assert second.json()["taskId"] == first.json()["taskId"]
    conflict = client.post("/a2a/messages", json={"message": "不同内容", "task_id": "it-idem"})
    assert conflict.status_code == 409


def test_a2a_list_and_stats(client):
    r = client.get("/a2a/tasks?limit=10")
    assert r.status_code == 200
    assert isinstance(r.json()["tasks"], list)
    s = client.get("/a2a/stats")
    assert s.status_code == 200
    assert "counters" in s.json()
    h = client.get("/api/health")
    assert h.status_code == 200
    assert "a2a" in h.json()


def test_a2a_unknown_agent_and_404(client):
    r = client.post("/a2a/messages?agent=nope", json={"message": "x"})
    assert r.status_code in (409, 502)
    r404 = client.get("/a2a/tasks/does-not-exist")
    assert r404.status_code == 404

def test_a2a_demo_page_served(client):
    r = client.get("/a2a")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "A2A" in r.text or "a2a" in r.text


def test_a2a_metrics_exposed(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert "a2a_tasks_total" in r.text
    assert "a2a_tasks_persisted" in r.text
