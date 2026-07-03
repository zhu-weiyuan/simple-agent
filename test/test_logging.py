"""Tests for structured logging configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_agent.logging_config import setup_logging, JsonFormatter


def test_json_formatter():
    """JSON formatter should produce valid JSON with required fields."""
    import json
    from logging import LogRecord
    
    formatter = JsonFormatter()
    # Create a proper LogRecord instance
    record = LogRecord(
        name='test.logger',
        level=20,  # INFO
        pathname='test_module.py',
        lineno=42,
        msg='Test message',
        args=None,
        exc_info=None,
    )
    
    output = formatter.format(record)
    data = json.loads(output)
    
    assert "timestamp" in data
    assert data["level"] == "INFO"
    assert data["message"] == "Test message"


def test_setup_logging():
    """setup_logging should configure root logger."""
    logger = setup_logging("DEBUG")
    
    # Should have at least one handler
    assert len(logger.handlers) >= 1
