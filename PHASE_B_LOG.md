# PHASE_B_LOG

## Scope
Production hardening Phase B completed without resetting or cleaning the pre-existing working tree. No PostgreSQL/pgvector work, commits, pushes, or external writes were performed.

## Findings and implementation
- Added `my_agent.resilience.CircuitBreaker`: thread-safe CLOSED/OPEN/HALF_OPEN state machine, default 5 consecutive failures and 30-second recovery timeout (environment configurable with `LLM_CIRCUIT_FAILURE_THRESHOLD` / `LLM_CIRCUIT_RECOVERY_TIMEOUT`). HALF_OPEN permits one probe only.
- Wired breaker into `SimpleAgent`'s LLM call adapter. Open circuits reject before any upstream request; failures increment the breaker and successful calls reset it.
- Added API degradation helper `get_degraded_response()`. `/api/chat` maps open-circuit, connection, and requests-layer upstream failures to a UTF-8 structured 503 response with `status=degraded`, `code=LLM_UNAVAILABLE`, safe Chinese fallback text, and request_id.
- Threaded optional `context` through `SimpleAgent.run_stream`, `QueryEngine.run_stream`, and `QueryEngine.run_tool_stream`; hooks and `_loop` now receive the request context. Also made normal `run()` set engine `_context`, restoring request_id visibility for tool hooks.
- Added `_track_context_usage()` to QueryEngine. It uses the existing session estimator and logs at >60% (INFO) and >80% Dumb Zone (WARNING), before each sync/stream request.
- Added `test_phase_b.py` targeted tests covering circuit transitions/probe, stream context hooks, context alerts, and degraded response contract.

## Commands and outcomes
1. `py -m py_compile app.py src/my_agent/core/engine.py src/my_agent/resilience.py` — PASS.
2. `py -W ignore -m pytest` — initial run: 150 passed, 1 failed (test fixture used 12 ASCII chars, estimated at 3 tokens, so did not cross the intended threshold). Updated only the new fixture to 40 chars.
3. `py -W ignore -m pytest` — PASS: 151 passed in 12.39s.

## Phase C gaps / concerns
- Streaming plain replies (`_stream_plain_reply`) bypass LLM adapter, so it does not participate in the circuit breaker; it should be migrated to a breaker-aware streaming adapter in Phase C.
- `run_stream_async` is not part of the requested B3 API pair and still lacks context propagation; extend it together with a truly breaker-aware async upstream path.
- Circuit breaker state is process-local; multi-worker deployments need shared state/metrics if cross-worker protection is required.
- Token estimate is intentionally heuristic from existing SessionState; provider tokenizer integration would make thresholds more precise.
