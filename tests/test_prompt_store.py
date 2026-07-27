import pytest
from my_agent.prompt_store import PromptStore
import sqlite3

def test_versioning(tmp_path):
    db_path = tmp_path / "test.db"
    store = PromptStore(str(db_path))
    store.add_version("v1", 50.0)
    store.add_version("v2", 50.0)
    
    # Check versions
    assert store.get_latest_version() == 2
    
    # Test gray release
    results = [store.get_current_prompt() for _ in range(1000)]
    assert results.count("v1") > 400
    assert results.count("v2") > 400

def test_rollback(tmp_path):
    db_path = tmp_path / "test.db"
    store = PromptStore(str(db_path))
    store.add_version("v1", 100.0)
    store.add_version("v2", 100.0)
    
    store.rollback_to(1)
    assert store.get_current_prompt() == "v1"