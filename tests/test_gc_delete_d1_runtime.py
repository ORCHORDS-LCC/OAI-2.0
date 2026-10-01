from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from oai2.knowledge.gc_delete_d1_runtime import (
    D1DeleteOutcome,
    delete_candidate_with_d1_lease,
)


@dataclass
class FakeLeaseStore:
    acquire_result: bool = True
    valid_result: bool = True
    finalize_result: bool = True
    failure_result: bool = True
    calls: list[tuple[str, object]] = field(default_factory=list)

    async def acquire_lease(self, **kwargs: object) -> bool:
        self.calls.append(("acquire", kwargs))
        return self.acquire_result

    async def lease_valid(self, key: str, token: str, *, now: float) -> bool:
        self.calls.append(("validate", (key, token, now)))
        return self.valid_result

    async def record_delete_failure(self, **kwargs: object) -> bool:
        self.calls.append(("failure", kwargs))
        return self.failure_result

    async def finalize_delete(self, **kwargs: object) -> bool:
        self.calls.append(("finalize", kwargs))
        return self.finalize_result


@pytest.mark.asyncio
async def test_delete_boundary_stops_when_lease_acquisition_is_denied() -> None:
    store = FakeLeaseStore(acquire_result=False)
    touched: list[str] = []

    async def exists(key: str) -> bool:
        touched.append(f"exists:{key}")
        return True

    async def delete(key: str) -> None:
        touched.append(f"delete:{key}")

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )

    assert result.outcome is D1DeleteOutcome.LEASE_DENIED
    assert touched == []
    assert [name for name, _ in store.calls] == ["acquire"]


@pytest.mark.asyncio
async def test_delete_boundary_finalizes_already_absent_without_delete_call() -> None:
    store = FakeLeaseStore()
    touched: list[str] = []

    async def exists(key: str) -> bool:
        touched.append(f"exists:{key}")
        return False

    async def delete(key: str) -> None:
        touched.append(f"delete:{key}")

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )

    assert result.outcome is D1DeleteOutcome.ALREADY_ABSENT
    assert touched == ["exists:oai2-blobs/a"]
    assert [name for name, _ in store.calls] == ["acquire", "validate", "finalize"]
    finalize = store.calls[-1][1]
    assert isinstance(finalize, dict)
    assert finalize["already_absent"] is True


@pytest.mark.asyncio
async def test_delete_boundary_revalidates_before_delete_and_finalizes() -> None:
    store = FakeLeaseStore()
    blobs = {"oai2-blobs/a"}
    events: list[str] = []

    async def exists(key: str) -> bool:
        events.append("exists")
        return key in blobs

    async def delete(key: str) -> None:
        events.append("delete")
        assert [name for name, _ in store.calls] == ["acquire", "validate"]
        blobs.remove(key)

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )

    assert result.outcome is D1DeleteOutcome.DELETED
    assert events == ["exists", "delete", "exists"]
    assert [name for name, _ in store.calls] == ["acquire", "validate", "finalize"]


@pytest.mark.asyncio
async def test_delete_boundary_records_failure_when_r2_delete_raises() -> None:
    store = FakeLeaseStore()

    async def exists(_key: str) -> bool:
        return True

    async def delete(_key: str) -> None:
        raise RuntimeError("simulated R2 outage")

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )

    assert result.outcome is D1DeleteOutcome.FAILED
    assert [name for name, _ in store.calls] == ["acquire", "validate", "failure"]


@pytest.mark.asyncio
async def test_delete_boundary_stops_if_lease_revalidation_fails() -> None:
    store = FakeLeaseStore(valid_result=False)
    deleted: list[str] = []

    async def exists(_key: str) -> bool:
        return True

    async def delete(key: str) -> None:
        deleted.append(key)

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )

    assert result.outcome is D1DeleteOutcome.LEASE_DENIED
    assert deleted == []
    assert [name for name, _ in store.calls] == ["acquire", "validate"]


@pytest.mark.asyncio
async def test_delete_boundary_fails_closed_when_finalize_does_not_apply() -> None:
    store = FakeLeaseStore(finalize_result=False)
    blobs = {"oai2-blobs/a"}

    async def exists(key: str) -> bool:
        return key in blobs

    async def delete(key: str) -> None:
        blobs.remove(key)

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )

    assert result.outcome is D1DeleteOutcome.FAILED
    assert blobs == set()
    assert [name for name, _ in store.calls] == ["acquire", "validate", "finalize"]
