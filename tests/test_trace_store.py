from __future__ import annotations

from bridgetrend_vision.trace_store import InMemoryTraceStore


def test_in_memory_store_preserves_order_and_returns_copies() -> None:
    store = InMemoryTraceStore()
    store.put({"session_id": "one", "created_at": "2026-09-23T01:00:00Z"})
    store.put({"session_id": "two", "created_at": "2026-09-23T02:00:00Z"})

    record = store.get("one")
    assert record is not None
    record["mutated"] = True
    assert "mutated" not in store.get("one")
    assert [item["session_id"] for item in store.list_recent()] == ["two", "one"]


def test_in_memory_store_rejects_missing_session_id() -> None:
    store = InMemoryTraceStore()

    try:
        store.put({"created_at": "2026-09-23T01:00:00Z"})
    except ValueError as exc:
        assert "session_id" in str(exc)
    else:
        raise AssertionError("missing session_id should fail")
