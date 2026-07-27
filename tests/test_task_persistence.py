import pytest
from my_agent.task_persistence import TaskPersistence
import json

def test_checkpoint(tmp_path):
    db_path = tmp_path / "tasks.db"
    persistence = TaskPersistence(str(db_path))
    
    state = {"step": 3}
    persistence.save_checkpoint("task1", state, 5, checkpoint_interval=5)
    loaded = persistence.load_checkpoint("task1")
    assert loaded == state
    
    # Should not save at turn 6
    persistence.save_checkpoint("task1", {"step": 6}, 6, 5)
    loaded = persistence.load_checkpoint("task1")
    assert loaded["step"] == 3  # Still old state