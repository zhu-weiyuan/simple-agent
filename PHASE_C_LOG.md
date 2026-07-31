# Phase C Implementation Log

## Findings
- Existing Phase A/B working-tree changes were preserved; no reset, clean, migration, commit, push, or external write was performed.
- `QueryEngine` already had basic context usage logging; Phase C adds local estimator/budget manager integration without API calls.
- The web application currently exposes authenticated APIs, so the queue endpoint uses the existing `@auth_required` guard and deliberately permits only `agent_run` (prevents arbitrary callable dispatch via HTTP).

## Changes
- Added `src/my_agent/core/token_budget.py`: optional-tiktoken/heuristic estimator, allocation with history-first compaction/RAG trimming semantics, threshold alerts.
- Added `src/my_agent/prompt_registry.py`: versioned registry, `{{variable}}` validation, rendering.
- Added `src/my_agent/async_task.py`: tracked `ThreadPoolExecutor` queue.
- Updated engine to estimate context during `run()` and bind prompt version to LLM hook trace data.
- Updated `SimpleAgent.run(..., prompt_version=...)` to select registered prompt variants and propagate version trace context.
- Added authenticated `POST /api/task` (`task_name=agent_run`, `params.message`) returning immediately with task id/status.
- Added evaluation harness, sample golden data (7 cases), and Phase C targeted tests.

## Commands and outcomes
- `python -m pytest -q test_phase_c.py` initially found one prompt-registry argument-name issue; fixed.
- `python -m pytest -q test_phase_c.py test_phase_b.py`: **8 passed**.
- `python -m pytest -q --maxfail=1`: **155 passed in 53.53s**.

## Remaining gaps / Phase D candidates
- EvalRunner uses deterministic exact-match by default; add model-judge/scorer configuration and richer trace-source integration if desired.
- Queue is process-local/in-memory and lacks result/status HTTP endpoint or durable retries; use a managed queue for multi-process production.
- Prompt versions are in-memory; add file/database persistence and deployment governance if prompt lifecycle expands.

## Production Improvement Run: Plain SSE Gateway Reliability (2026-07-25)

### Finding and scope
- Selected one low-risk P1 model-gateway reliability gap after inspecting `app.py`, `SimpleAgent` LLM injection, the circuit breaker, and Phase B tests: the plain-text `POST /api/chat` SSE path called `LLMClient.chat_stream()` directly and therefore bypassed the circuit breaker used by non-streaming and tool calls.
- This could repeatedly send traffic to a known-unavailable upstream, expose an inconsistent streaming failure shape, and leave a half-open circuit probe reserved when a client disconnected.
- Existing working-tree changes in both target repositories were preserved. No PostgreSQL/pgvector work, reset, clean, commit, push, external write, or migration was performed.

### Changes
- Updated `app.py:_stream_plain_reply()` to reserve the existing `CircuitBreaker` before starting an upstream stream, record normal completion as success, and record upstream exceptions as failures.
- Open circuits and upstream streaming failures now emit the existing safe `LLM_UNAVAILABLE` degraded response as an SSE data event, including the request ID, without provider exception text.
- Added `CircuitBreaker.release_probe()` in `src/my_agent/resilience.py`; a generator close releases a cancelled half-open probe without changing circuit state.
- Passed the established request ID into the plain SSE helper while preserving the existing token event format, completion markers, session behavior after received tokens, and tool-stream route.
- Added focused tests for open-circuit rejection, failure accounting/safe error content, and cancellation probe release in `test_phase_b.py`.

### Evidence
- `python -m pytest -q test_phase_b.py`: **7 passed**.
- `python -m py_compile app.py src\\my_agent\\resilience.py`: passed.
- `python -m pytest -q`: **158 passed in 14.19s**.
- `python -m ruff check app.py test_phase_b.py src\\my_agent\\resilience.py`: **All checks passed**.
- `git diff --check`: passed. A repository-wide Ruff run remains blocked by 14 pre-existing E701/E702 violations in `src/my_agent/core/token_budget.py` and `tests/eval_harness.py`; this run did not modify them.

### Limitations and next risk
- The synchronous upstream stream still cannot cancel the underlying `requests` call after a browser disconnect; this run releases circuit state but does not claim transport cancellation. Move this route to the existing async streaming client only after an end-to-end cancellation design and integration test are agreed.
- Test coverage uses local fakes; validate a real provider disconnect/failure scenario in a non-production environment before making production-readiness claims.
