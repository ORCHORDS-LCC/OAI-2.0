from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from oai2.knowledge.gc_lease_d1 import (
    GC_LEASE_ACQUIRE_SQL,
    GC_LEASE_FINALIZE_SQL,
    GC_LEASE_RECORD_FAILURE_SQL,
    GC_LEASE_REFERENCE_COUNT_SQL,
    GC_LEASE_RELEASE_SQL,
    GC_LEASE_UPSERT_SQL,
    GC_LEASE_VALIDATE_SQL,
    GC_LEASE_WRITER_BLOCK_SQL,
    gc_lease_schema_statements,
)
from oai2.knowledge.gc_lease_d1_runtime import D1GcLeaseStore


@dataclass
class FakeMeta:
    changes: int = 0


@dataclass
class FakeResult:
    success: bool = True
    meta: FakeMeta = field(default_factory=FakeMeta)


@dataclass
class FakeStatement:
    query: str
    first_value: object | None = None
    run_result: object = field(default_factory=FakeResult)
    bound: tuple[object, ...] = ()

    def bind(self, *values: object) -> FakeStatement:
        self.bound = values
        return self

    async def run(self) -> object:
        return self.run_result

    async def first(self, column_name: str | None = None) -> object | None:
        return self.first_value


class FakeDatabase:
    def __init__(self) -> None:
        self.prepared: list[FakeStatement] = []
        self.first_values: dict[str, object] = {}
        self.run_results: dict[str, object] = {}
        self.batch_results: list[object] | None = None
        self.batched: list[FakeStatement] = []

    def prepare(self, query: str) -> FakeStatement:
        stmt = FakeStatement(
            query=query,
            first_value=self.first_values.get(query),
            run_result=self.run_results.get(query, FakeResult()),
        )
        self.prepared.append(stmt)
        return stmt

    async def batch(self, statements: list[FakeStatement]) -> list[object]:
        self.batched = list(statements)
        if self.batch_results is not None:
            return self.batch_results
        return [FakeResult() for _ in statements]


@pytest.mark.asyncio
async def test_ensure_schema_uses_one_d1_batch_with_all_schema_statements() -> None:
    db = FakeDatabase()
    store = D1GcLeaseStore(db)

    await store.ensure_schema()

    expected = list(gc_lease_schema_statements())
    assert [stmt.query for stmt in db.batched] == expected


@pytest.mark.asyncio
async def test_ensure_schema_fails_closed_on_unsuccessful_result() -> None:
    db = FakeDatabase()
    statements = list(gc_lease_schema_statements())
    db.batch_results = [FakeResult(True), FakeResult(False)]
    assert len(statements) == 2
    store = D1GcLeaseStore(db)

    with pytest.raises(RuntimeError, match="unsuccessful"):
        await store.ensure_schema()


@pytest.mark.asyncio
async def test_reference_count_uses_prepared_bound_scalar_query() -> None:
    db = FakeDatabase()
    db.first_values[GC_LEASE_REFERENCE_COUNT_SQL] = 2
    store = D1GcLeaseStore(db)

    assert await store.retained_reference_count("oai2-blobs/shared") == 2
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_REFERENCE_COUNT_SQL
    assert stmt.bound == ("oai2-blobs/shared",)


@pytest.mark.asyncio
async def test_writer_blocked_uses_object_key_and_timestamp() -> None:
    db = FakeDatabase()
    db.first_values[GC_LEASE_WRITER_BLOCK_SQL] = 1
    store = D1GcLeaseStore(db)

    assert await store.writer_blocked("oai2-blobs/a", now=12.5) is True
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_WRITER_BLOCK_SQL
    assert stmt.bound == ("oai2-blobs/a", 12.5)


@pytest.mark.asyncio
async def test_writer_blocked_rejects_non_boolean_integer() -> None:
    db = FakeDatabase()
    db.first_values[GC_LEASE_WRITER_BLOCK_SQL] = 2
    store = D1GcLeaseStore(db)

    with pytest.raises(RuntimeError, match="non-boolean"):
        await store.writer_blocked("oai2-blobs/a", now=12.5)


@pytest.mark.asyncio
async def test_upsert_lease_binds_all_values_and_checks_success() -> None:
    db = FakeDatabase()
    store = D1GcLeaseStore(db)

    await store.upsert_lease(
        object_key="oai2-blobs/a",
        token="lease-1",
        owner="gc-sweep",
        acquired_at=10.0,
        expires_at=20.0,
        state="active",
        failure_count=0,
        finalized_decision=None,
        authority_revision=7,
        updated_at=10.0,
    )

    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_UPSERT_SQL
    assert stmt.bound == (
        "oai2-blobs/a",
        "lease-1",
        "gc-sweep",
        10.0,
        20.0,
        "active",
        0,
        None,
        7,
        10.0,
    )


@pytest.mark.asyncio
async def test_upsert_lease_propagates_d1_unsuccessful_result() -> None:
    db = FakeDatabase()
    db.run_results[GC_LEASE_UPSERT_SQL] = {"success": False}
    store = D1GcLeaseStore(db)

    with pytest.raises(RuntimeError, match="unsuccessful"):
        await store.upsert_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            owner="gc-sweep",
            acquired_at=10.0,
            expires_at=20.0,
            state="active",
            failure_count=0,
            finalized_decision=None,
            authority_revision=7,
            updated_at=10.0,
        )



@pytest.mark.asyncio
async def test_lease_valid_rechecks_token_expiry_and_authoritative_references() -> None:
    db = FakeDatabase()
    db.first_values[GC_LEASE_VALIDATE_SQL] = 1
    store = D1GcLeaseStore(db)

    assert await store.lease_valid(
        "oai2-blobs/a",
        "lease-1",
        now=19.0,
    ) is True
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_VALIDATE_SQL
    assert stmt.bound == ("oai2-blobs/a", "lease-1", 19.0)


@pytest.mark.asyncio
async def test_lease_valid_fails_closed_on_non_boolean_scalar() -> None:
    db = FakeDatabase()
    db.first_values[GC_LEASE_VALIDATE_SQL] = 3
    store = D1GcLeaseStore(db)

    with pytest.raises(RuntimeError, match="lease_valid"):
        await store.lease_valid("oai2-blobs/a", "lease-1", now=19.0)



@pytest.mark.asyncio
async def test_acquire_lease_uses_single_conditional_mutation_and_change_count() -> None:
    db = FakeDatabase()
    db.run_results[GC_LEASE_ACQUIRE_SQL] = FakeResult(
        success=True,
        meta=FakeMeta(changes=1),
    )
    store = D1GcLeaseStore(db)

    acquired = await store.acquire_lease(
        object_key="oai2-blobs/a",
        token="lease-7",
        owner="gc-sweep",
        now=10.0,
        ttl_seconds=5.0,
        authority_revision=12,
    )

    assert acquired is True
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_ACQUIRE_SQL
    assert stmt.bound == (
        "oai2-blobs/a",
        "lease-7",
        "gc-sweep",
        10.0,
        15.0,
        12,
    )


@pytest.mark.asyncio
async def test_acquire_lease_returns_false_when_conditional_sql_changes_zero_rows() -> None:
    db = FakeDatabase()
    db.run_results[GC_LEASE_ACQUIRE_SQL] = {
        "success": True,
        "meta": {"changes": 0},
    }
    store = D1GcLeaseStore(db)

    assert await store.acquire_lease(
        object_key="oai2-blobs/a",
        token="lease-7",
        owner="gc-sweep",
        now=10.0,
        ttl_seconds=5.0,
        authority_revision=12,
    ) is False


@pytest.mark.asyncio
async def test_acquire_lease_fails_closed_on_bad_change_count_or_ttl() -> None:
    db = FakeDatabase()
    db.run_results[GC_LEASE_ACQUIRE_SQL] = {
        "success": True,
        "meta": {"changes": 2},
    }
    store = D1GcLeaseStore(db)

    with pytest.raises(RuntimeError, match="unexpected number of rows"):
        await store.acquire_lease(
            object_key="oai2-blobs/a",
            token="lease-7",
            owner="gc-sweep",
            now=10.0,
            ttl_seconds=5.0,
            authority_revision=12,
        )

    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        await store.acquire_lease(
            object_key="oai2-blobs/a",
            token="lease-7",
            owner="gc-sweep",
            now=10.0,
            ttl_seconds=0.0,
            authority_revision=12,
        )



@pytest.mark.asyncio
async def test_delete_failure_transition_binds_token_time_and_revision() -> None:
    db = FakeDatabase()
    db.run_results[GC_LEASE_RECORD_FAILURE_SQL] = FakeResult(
        success=True,
        meta=FakeMeta(changes=1),
    )
    store = D1GcLeaseStore(db)

    assert await store.record_delete_failure(
        object_key="oai2-blobs/a",
        token="lease-1",
        now=12.0,
        authority_revision=8,
    ) is True
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_RECORD_FAILURE_SQL
    assert stmt.bound == ("oai2-blobs/a", "lease-1", 12.0, 8)


@pytest.mark.asyncio
async def test_finalize_delete_binds_terminal_decision_and_revision() -> None:
    db = FakeDatabase()
    db.run_results[GC_LEASE_FINALIZE_SQL] = {
        "success": True,
        "meta": {"changes": 1},
    }
    store = D1GcLeaseStore(db)

    assert await store.finalize_delete(
        object_key="oai2-blobs/a",
        token="lease-1",
        now=12.0,
        already_absent=True,
        authority_revision=9,
    ) is True
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_FINALIZE_SQL
    assert stmt.bound == (
        "oai2-blobs/a",
        "lease-1",
        12.0,
        "already_absent",
        9,
    )


@pytest.mark.asyncio
async def test_release_lease_returns_false_when_guard_matches_no_row() -> None:
    db = FakeDatabase()
    db.run_results[GC_LEASE_RELEASE_SQL] = {
        "success": True,
        "meta": {"changes": 0},
    }
    store = D1GcLeaseStore(db)

    assert await store.release_lease(
        object_key="oai2-blobs/a",
        token="wrong-token",
        now=12.0,
        authority_revision=10,
    ) is False


@pytest.mark.asyncio
async def test_outcome_transition_fails_closed_on_multirow_change_count() -> None:
    db = FakeDatabase()
    db.run_results[GC_LEASE_RELEASE_SQL] = {
        "success": True,
        "meta": {"changes": 2},
    }
    store = D1GcLeaseStore(db)

    with pytest.raises(RuntimeError, match="unexpected number of rows"):
        await store.release_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            now=12.0,
            authority_revision=10,
        )
