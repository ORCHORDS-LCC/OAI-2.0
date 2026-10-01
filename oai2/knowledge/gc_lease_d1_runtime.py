"""Async Cloudflare D1 binding helpers for the GC deletion-lease schema.

This module is intentionally binding-facing but deployment-neutral. It consumes
the same prepare/bind/run/batch surface exposed by Cloudflare Python Workers'
D1Database binding without importing the workers package, so the core package
can still be imported and tested outside the Workers runtime.

It does not create credentials, resource identifiers, or a Worker entrypoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol, Self

from .gc_lease_d1 import (
    GC_LEASE_REFERENCE_COUNT_SQL,
    GC_LEASE_UPSERT_SQL,
    GC_LEASE_VALIDATE_SQL,
    GC_LEASE_WRITER_BLOCK_SQL,
    gc_lease_schema_statements,
)


class D1PreparedStatementBinding(Protocol):
    def bind(self, *values: object) -> Self: ...

    async def run(self) -> object: ...

    async def first(self, column_name: str | None = None) -> object | None: ...


class D1DatabaseBinding(Protocol):
    def prepare(self, query: str) -> D1PreparedStatementBinding: ...

    async def batch(
        self, statements: Sequence[D1PreparedStatementBinding]
    ) -> Sequence[object]: ...


class D1GcLeaseStore:
    """Persist/query GC lease state through a bound Cloudflare D1 database."""

    def __init__(self, database: D1DatabaseBinding) -> None:
        self._db = database

    async def ensure_schema(self) -> None:
        """Apply the versioned public-safe schema in one D1 batch transaction."""
        statements = [self._db.prepare(sql) for sql in gc_lease_schema_statements()]
        results = await self._db.batch(statements)
        if len(results) != len(statements):
            raise RuntimeError("D1 schema batch returned an unexpected result count")
        for result in results:
            if not _result_success(result):
                raise RuntimeError("D1 schema batch reported an unsuccessful statement")

    async def retained_reference_count(self, object_key: str) -> int:
        """Read authoritative retained references from knowledge_index."""
        key = _normalized(object_key, "object_key")
        value = await (
            self._db.prepare(GC_LEASE_REFERENCE_COUNT_SQL)
            .bind(key)
            .first("retained_reference_count")
        )
        return _non_negative_int(value, "retained_reference_count")

    async def writer_blocked(self, object_key: str, *, now: float) -> bool:
        """Return whether an active, unexpired lease blocks a new reference."""
        key = _normalized(object_key, "object_key")
        timestamp = _finite_non_negative(now, "now")
        value = await (
            self._db.prepare(GC_LEASE_WRITER_BLOCK_SQL)
            .bind(key, timestamp)
            .first("writer_blocked")
        )
        return _strict_boolean_int(value, "writer_blocked")

    async def lease_valid(self, object_key: str, token: str, *, now: float) -> bool:
        """Revalidate token, expiry and no-reference state immediately pre-delete."""
        key = _normalized(object_key, "object_key")
        lease_token = _normalized(token, "token")
        timestamp = _finite_non_negative(now, "now")
        value = await (
            self._db.prepare(GC_LEASE_VALIDATE_SQL)
            .bind(key, lease_token, timestamp)
            .first("lease_valid")
        )
        return _strict_boolean_int(value, "lease_valid")

    async def upsert_lease(
        self,
        *,
        object_key: str,
        token: str,
        owner: str,
        acquired_at: float,
        expires_at: float,
        state: str,
        failure_count: int,
        finalized_decision: str | None,
        authority_revision: int,
        updated_at: float,
    ) -> None:
        """Persist a lease row using prepared parameter binding."""
        key = _normalized(object_key, "object_key")
        lease_token = _normalized(token, "token")
        lease_owner = _normalized(owner, "owner")
        acquired = _finite_non_negative(acquired_at, "acquired_at")
        expires = _finite_non_negative(expires_at, "expires_at")
        if expires <= acquired:
            raise ValueError("expires_at must be greater than acquired_at")
        failures = _non_negative_int(failure_count, "failure_count")
        revision = _non_negative_int(authority_revision, "authority_revision")
        updated = _finite_non_negative(updated_at, "updated_at")
        if finalized_decision is not None:
            _normalized(finalized_decision, "finalized_decision")

        result = await (
            self._db.prepare(GC_LEASE_UPSERT_SQL)
            .bind(
                key,
                lease_token,
                lease_owner,
                acquired,
                expires,
                _normalized(state, "state"),
                failures,
                finalized_decision,
                revision,
                updated,
            )
            .run()
        )
        if not _result_success(result):
            raise RuntimeError("D1 lease upsert reported an unsuccessful statement")


def _result_success(result: object) -> bool:
    if isinstance(result, Mapping):
        value = result.get("success")
    else:
        value = getattr(result, "success", None)
    if not isinstance(value, bool):
        raise RuntimeError("D1 result does not expose a boolean success field")
    return value


def _normalized(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


def _finite_non_negative(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _strict_boolean_int(value: object, name: str) -> bool:
    flag = _non_negative_int(value, name)
    if flag not in (0, 1):
        raise RuntimeError(f"D1 {name} query returned a non-boolean integer")
    return bool(flag)


__all__ = [
    "D1PreparedStatementBinding",
    "D1DatabaseBinding",
    "D1GcLeaseStore",
]
