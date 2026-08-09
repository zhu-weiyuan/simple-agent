import sqlite3
import time

from my_agent.idempotency import IdempotencyStore


def test_claim_finalize_and_replay(tmp_path):
    store = IdempotencyStore(str(tmp_path / "idempotency.db"))
    first = store.claim("tenant:user", "req-1", "hash-1")
    assert first.status == "claimed"
    assert first.token

    assert store.finalize(
        "tenant:user", "req-1", first.token,
        {"reply": "??"}, 201, "application/json",
    )
    replay = store.claim("tenant:user", "req-1", "hash-1")
    assert replay.status == "replay"
    assert replay.status_code == 201
    assert replay.content_type == "application/json"
    assert replay.response_body == '{"reply":"??"}'


def test_conflict_and_in_progress(tmp_path):
    store = IdempotencyStore(str(tmp_path / "idempotency.db"))
    first = store.claim("tenant:user", "req-1", "hash-1")
    assert first.status == "claimed"
    assert store.claim("tenant:user", "req-1", "hash-2").status == "conflict"
    assert store.claim("tenant:user", "req-1", "hash-1").status == "in_progress"


def test_release_allows_retry_and_token_is_bound(tmp_path):
    store = IdempotencyStore(str(tmp_path / "idempotency.db"))
    first = store.claim("scope", "key", "hash")
    assert first.token
    assert not store.finalize("scope", "key", "wrong-token", {"bad": True})
    assert not store.release("scope", "key", "wrong-token")
    assert store.release("scope", "key", first.token)

    retry = store.claim("scope", "key", "hash")
    assert retry.status == "claimed"
    assert retry.token != first.token
    assert not store.finalize("scope", "key", first.token, {"stale": True})
    assert store.finalize("scope", "key", retry.token, {"ok": True})


def test_scope_isolation_and_shared_database(tmp_path):
    db = str(tmp_path / "idempotency.db")
    store_a = IdempotencyStore(db)
    store_b = IdempotencyStore(db)
    first = store_a.claim("tenant-a:user", "same-key", "same-hash")
    assert first.status == "claimed"
    assert store_b.claim("tenant-b:user", "same-key", "same-hash").status == "claimed"
    assert store_b.claim("tenant-a:user", "same-key", "same-hash").status == "in_progress"


def test_stale_pending_can_be_reclaimed(tmp_path):
    db = str(tmp_path / "idempotency.db")
    store = IdempotencyStore(db, pending_ttl_seconds=1)
    first = store.claim("scope", "key", "hash")
    assert first.token
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE idempotency_records SET updated_at=? WHERE scope=? AND key=?",
            (time.time() - 10, "scope", "key"),
        )
        conn.commit()
    reclaimed = store.claim("scope", "key", "hash")
    assert reclaimed.status == "claimed"
    assert reclaimed.token != first.token


def test_only_digest_and_response_are_persisted(tmp_path):
    db = str(tmp_path / "idempotency.db")
    store = IdempotencyStore(db)
    raw_message = "?????????"
    digest = store.fingerprint({"message": raw_message})
    claimed = store.claim("scope", "key", digest)
    assert store.finalize("scope", "key", claimed.token, {"reply": "ok"})

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT request_hash, response_body FROM idempotency_records"
        ).fetchall()
    assert len(rows) == 1
    assert raw_message not in rows[0][0]
    assert raw_message not in rows[0][1]


def test_in_memory_store_is_supported():
    store = IdempotencyStore(":memory:")
    claimed = store.claim("scope", "key", "hash")
    assert claimed.status == "claimed"
    assert store.finalize("scope", "key", claimed.token, "sse-data")
    replay = store.claim("scope", "key", "hash")
    assert replay.status == "replay"
    assert replay.response_body == "sse-data"
