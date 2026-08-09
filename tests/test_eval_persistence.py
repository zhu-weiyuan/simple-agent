import pytest
from my_agent.eval_persistence import EvalPersistence
import json

def test_logging(tmp_path):
    db_path = tmp_path / "evals.db"
    eval_store = EvalPersistence(str(db_path))
    
    eval_store.log_result({"accuracy": 0.85, "time": 2.3})
    trends = eval_store.get_trends("accuracy")
    assert trends == [0.85]
    
    eval_store.export_json(tmp_path / "export.json")
    with open(tmp_path / "export.json") as f:
        data = json.load(f)
    assert len(data) == 1