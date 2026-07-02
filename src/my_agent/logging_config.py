"""
Structured logging configuration for SimpleAgent.

Outputs JSON-formatted logs with request/response tracking.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add context fields
        for field in ["session_id", "user_message", "request_id"]:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(level: str = "INFO"):
    """Configure structured logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(console_handler)
    
    return root_logger


logger = setup_logging("INFO")
