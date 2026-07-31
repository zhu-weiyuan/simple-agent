# Phase D — Persistence, Durable Queue, and LLM-as-Judge

## Completed
- Added `PromptStore` to `src/my_agent/memory/sqlite_store.py`: SQLite prompt CRUD with automatic table creation, JSON variable schemas, default-version tracking, and an in-memory fallback if SQLite is unavailable.
- Updated `PromptRegistry` for lazy `PROMPT_DB_PATH` initialization (`runtime/prompts.db` by default). Registered prompts persist; stored prompts are lazily read while legacy in-memory/file-style usage continues to work.
- Added durable metadata persistence to `AsyncTaskQueue`, using `TASK_DB_PATH` or `runtime/tasks.db`. Tasks are persisted at submission and terminal state; `get_all_tasks()` returns newest metadata first. Interrupted pending/running tasks are recovered as failed because Python callables cannot be serialized safely for rerun.
- Added `LLMJudgeScorer` and optional `llm_judge_enabled` support to the evaluation runner. It uses an OpenAI-compatible local endpoint and records `llm_judge` scores and judge reasoning alongside exact-match scoring. Endpoint/network/JSON failures degrade to a zero judge score instead of failing the evaluation run.
- Added targeted persistence and mocked-judge tests in `tests/test_persistence.py`.

## Verification
- `python -m pytest -q`: **161 passed** (9.99s).

## Phase E gaps
- Durable task result payloads and callable/job serialization are not implemented; current persistence deliberately stores metadata only, per the requested schema.
- The LLM judge has a mocked integration test; a live endpoint contract test can be added when a stable local model fixture is available.
