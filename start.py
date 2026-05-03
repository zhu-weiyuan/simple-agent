#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleAgent v2.0 — Unified Entry Point

Usage:
    python start.py              # Interactive CLI (same as run.py)
    python start.py web          # Start web server on :8000 (FastAPI, same as app.py)
    python start.py "your query" # Single-shot query
"""
import os
import sys
from pathlib import Path

# Windows UTF-8 support
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors="replace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors="replace")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Default env
os.environ.setdefault("OPENAI_API_KEY", "your_key_here")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8080")
os.environ.setdefault("OPENAI_MODEL", "Qwen3.6-27B-IQ4_NL.gguf")


def run_cli(query=None):
    """Interactive or single-shot CLI."""
    from my_agent import SimpleAgent

    agent = SimpleAgent()

    if query:
        # Single-shot
        print(agent.run(query))
        agent.close()
        return

    # Interactive loop
    print("=" * 50)
    print("SimpleAgent v2.0 (type 'quit' to exit)")
    print("=" * 50)
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            response = agent.run(user_input)
            print(f"\nAgent: {response}")
    except KeyboardInterrupt:
        print("\n\nBye!")
    finally:
        agent.close()


def run_web():
    """Start FastAPI web server."""
    from app import app
    import uvicorn

    print("Starting SimpleAgent web server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "web":
            run_web()
        else:
            # Treat as query string
            run_cli(" ".join(sys.argv[1:]))
    else:
        run_cli()


if __name__ == "__main__":
    main()
