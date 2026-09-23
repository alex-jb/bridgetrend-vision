"""Persistence adapters for competition decision traces and human reviews.

The judge console defaults to an in-memory store for local development.  AWS
deployments can select DynamoDB through environment variables without changing
the evidence-agent policy or its deterministic hash chain.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from typing import Any, Protocol


class TraceStore(Protocol):
    """Minimal storage contract used by the judge console."""

    @property
    def backend_name(self) -> str:
        """Return a public, non-secret backend label."""

    def put(self, record: Mapping[str, Any]) -> None:
        """Persist one complete session record."""

    def get(self, session_id: str) -> dict[str, Any] | None:
        """Fetch one session by its stable ID."""

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent sessions, newest first."""


class InMemoryTraceStore:
    """Thread-safe process-local trace storage for tests and local demos."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    @property
    def backend_name(self) -> str:
        return "memory"

    def put(self, record: Mapping[str, Any]) -> None:
        session_id = _require_session_id(record)
        payload = deepcopy(dict(record))
        with self._lock:
            if session_id not in self._records:
                self._order.append(session_id)
            self._records[session_id] = payload

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(session_id)
            return deepcopy(record) if record is not None else None

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            return [
                deepcopy(self._records[session_id])
                for session_id in reversed(self._order[-limit:])
            ]


class DynamoDBTraceStore:
    """DynamoDB-backed store used by the AWS App Runner deployment."""

    def __init__(
        self,
        *,
        table_name: str,
        region_name: str | None = None,
        resource: Any | None = None,
    ) -> None:
        if not table_name.strip():
            raise ValueError("table_name cannot be empty")
        if resource is None:
            import boto3

            resource = boto3.resource("dynamodb", region_name=region_name)
        self._table = resource.Table(table_name)
        self._table_name = table_name

    @property
    def backend_name(self) -> str:
        return "dynamodb"

    def put(self, record: Mapping[str, Any]) -> None:
        _require_session_id(record)
        payload = _to_dynamodb(dict(record))
        self._table.put_item(Item=payload)

    def get(self, session_id: str) -> dict[str, Any] | None:
        response = self._table.get_item(
            Key={"session_id": session_id},
            ConsistentRead=True,
        )
        item = response.get("Item")
        return _from_dynamodb(item) if isinstance(item, dict) else None

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        response = self._table.scan(Limit=min(limit, 100))
        items = [_from_dynamodb(item) for item in response.get("Items", [])]
        return sorted(
            items,
            key=lambda item: str(item.get("created_at", "")),
            reverse=True,
        )[:limit]


def _require_session_id(record: Mapping[str, Any]) -> str:
    session_id = record.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("record requires a non-empty session_id")
    return session_id


def _to_dynamodb(value: Any) -> Any:
    """Convert floats recursively because DynamoDB rejects binary floats."""

    return json.loads(json.dumps(value), parse_float=Decimal)


def _from_dynamodb(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    if isinstance(value, dict):
        return {key: _from_dynamodb(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_dynamodb(item) for item in value]
    return value
