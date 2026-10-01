"""Tests for the async conservative sweep planner over the live D1/R2 boundary.

These tests pin the end-to-end wiring described in #233 (WI-GC-003): the
conservative sweep planner delegates the destructive candidate to
``delete_candidate_with_d1_lease`` so the planner cannot accidentally bypass
the acquire → revalidate → delete → finalize/failure flow. Each test exercises
one AC-GC-03x acceptance criterion or one conservative-sweep failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from oai2.knowledge.gc import (
    GcDryRunReport,
    GcObjectDisposition,
    GcObjectRecord,
)
from oai2.knowledge.gc_delete_d1_runtime import D1DeleteOutcome
from oai2.knowledge.sweep import GcSweepDisposition, GcSweepState


@dataclass
class FakeD1LeaseStore:
    acquire_result: bool = True
    valid_result: bool = True
    finalize_result: bool = True
    failure_result: bool = True
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def acquire_lease(self, **kwargs: Any) -> bool:
        self.calls.append(("acquire", kwargs))
        return self.acquire_result

    async def lease_valid(self, key: str, token: str, *, now: float) -> bool:
        self.calls.append(("validate", {"key": key, "token": token, "now": now}))
        return self.valid_result

    async def record_delete_failure(self, **kwargs: Any) -> bool:
        self.calls.append(("failure", kwargs))
        return self.failure_result

    async def finalize_delete(self, **kwargs: Any) -> bool:
        self.calls.append(("finalize", kwargs))
        return self.finalize_result


def _report(*keys: str, observed_at: float = 10.0) -> GcDryRunReport:
    records = tuple(
        GcObjectRecord(
            key=key,
            disposition=GcObjectDisposition.UNREFERENCED_CANDIDATE,
            size_bytes=5,
            uploaded_at=1.0,
            age_seconds=observed_at - 1.0,
        )
        for key in keys
    )
    return GcDryRunReport(
        observed_at=observed_at,
        pages_processed=1,
        records=records,
        referenced_present_count=0,
        referenced_missing_count=0,
        unreferenced_candidate_count=len(records),
        inventory_object_count=len(records),
        inventory_bytes_known=5 * len(records),
        unreferenced_candidate_bytes_known=5 * len(records),
        unknown_size_object_count=0,
        oldest_unreferenced_candidate_age_seconds=observed_at - 1.0,
    )


# --- Gate preservation (mirrors tests/test_knowledge_sweep.py semantics) ----


@pytest.mark.asyncio
async def test_grace_period_defers_without_touching_lease_store() -> None:
    store = FakeD1LeaseStore()
    state = GcSweepState.from_report(_report("oai2-blobs/a", observed_at=10.0))

    result = await state.aprocess_batch(
        now=15.0,
        grace_seconds=10.0,
        reference_lookup=_async_lookup(_forbidden),
        blob_exists=_async_exists(_forbidden),
        delete_blob=_async_delete(_forbidden),
        lease_store=store,  # type: ignore[arg-type]
    )

    assert len(result.records) == 1
    assert result.records[0].disposition is GcSweepDisposition.DEFERRED_GRACE
    assert store.calls == []


@pytest.mark.asyncio
async def test_new_authoritative_reference_blocks_before_lease_attempt() -> None:
    store = FakeD1LeaseStore()
    state = GcSweepState.from_report(_report("oai2-blobs/shared"))

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=_async_lookup(["ko-new"]),
        blob_exists=_async_exists(_forbidden),
        delete_blob=_async_delete(_forbidden),
        lease_store=store,
    )

    assert result.records[0].disposition is GcSweepDisposition.RE_REFERENCED
    assert result.records[0].knowledge_ids == ("ko-new",)
    assert store.calls == [], "lease store must not be touched when references exist"


@pytest.mark.asyncio
async def test_destructive_without_authorization_emits_unapproved() -> None:
    store = FakeD1LeaseStore()
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=False,
        recovery_ready=True,
        reference_lookup=_async_lookup([]),
        blob_exists=_async_exists(_forbidden),
        delete_blob=_async_delete(_forbidden),
        lease_store=store,
    )

    assert result.records[0].disposition is GcSweepDisposition.UNAPPROVED
    assert store.calls == []


@pytest.mark.asyncio
async def test_destructive_without_recovery_readiness_emits_recovery_blocked() -> None:
    store = FakeD1LeaseStore()
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=False,
        reference_lookup=_async_lookup([]),
        blob_exists=_async_exists(_forbidden),
        delete_blob=_async_delete(_forbidden),
        lease_store=store,
    )

    assert result.records[0].disposition is GcSweepDisposition.RECOVERY_BLOCKED
    assert store.calls == []


@pytest.mark.asyncio
async def test_destructive_without_lease_store_fails_closed() -> None:
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    with pytest.raises(ValueError, match="destructive async sweeps require"):
        await state.aprocess_batch(
            now=100.0,
            grace_seconds=0.0,
            destructive=True,
            authorized=True,
            recovery_ready=True,
            reference_lookup=_async_lookup([]),
            blob_exists=_async_exists(False),
            delete_blob=_async_delete(None),
            lease_store=None,
        )


# --- AC-GC-031/032/033: writer-before / writer-after / expired lease -----


@pytest.mark.asyncio
async def test_lease_acquisition_denied_emits_lease_denied_without_delete() -> None:
    """AC-GC-031: a reference created before claim acquisition prevents the lease."""
    store = FakeD1LeaseStore(acquire_result=False)
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    touched: list[str] = []

    async def exists(key: str) -> bool:
        touched.append(f"exists:{key}")
        return True

    async def delete(key: str) -> None:
        touched.append(f"delete:{key}")

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=_async_lookup([]),
        blob_exists=exists,
        delete_blob=delete,
        lease_store=store,
    )

    assert result.records[0].disposition is GcSweepDisposition.LEASE_DENIED
    assert touched == []
    assert [name for name, _ in store.calls] == ["acquire"]


@pytest.mark.asyncio
async def test_lease_invalid_before_delete_blocks_race_without_r2_call() -> None:
    """AC-GC-032: a writer racing after claim acquisition invalidates the lease."""
    store = FakeD1LeaseStore(valid_result=False)
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    touched: list[str] = []

    async def exists(key: str) -> bool:
        touched.append(f"exists:{key}")
        return True

    async def delete(key: str) -> None:
        touched.append(f"delete:{key}")

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=_async_lookup([]),
        blob_exists=exists,
        delete_blob=delete,
        lease_store=store,
    )

    assert result.records[0].disposition is GcSweepDisposition.LEASE_DENIED
    # blob_exists was checked (it's part of the boundary), but delete was never
    # called and no finalize/failure was attempted for a writer-race invalidation.
    assert "exists:oai2-blobs/a" in touched
    assert not any(line.startswith("delete:") for line in touched)
    assert [name for name, _ in store.calls] == ["acquire", "validate"]


@pytest.mark.asyncio
async def test_stale_lease_after_expiry_is_rejected() -> None:
    """AC-GC-033: expired/crashed claims cannot proceed to delete.

    The boundary revalidates the lease token immediately before R2, so a stale
    lease surfaces as LEASE_DENIED and the planner never reaches the R2 call.
    """
    store = FakeD1LeaseStore(valid_result=False)
    state = GcSweepState.from_report(_report("oai2-blobs/expired"))

    touched: list[str] = []

    async def exists(key: str) -> bool:
        touched.append(f"exists:{key}")
        return True

    async def delete(key: str) -> None:
        touched.append(f"delete:{key}")

    result = await state.aprocess_batch(
        now=200.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=_async_lookup([]),
        blob_exists=exists,
        delete_blob=delete,
        lease_store=store,
    )

    assert result.records[0].disposition is GcSweepDisposition.LEASE_DENIED
    assert not any(line.startswith("delete:") for line in touched)
    assert store.calls[-1][0] == "validate"


# --- AC-GC-034/035: R2 failure vs. confirmed deletion (idempotency) ------


@pytest.mark.asyncio
async def test_r2_failure_records_failure_and_does_not_advance_cursor() -> None:
    """AC-GC-034: R2 failure leaves the claim recoverable."""
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))

    async def exists(key: str) -> bool:
        return True

    async def delete(key: str) -> None:
        raise RuntimeError("simulated R2 failure")

    store = FakeD1LeaseStore()

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=_async_lookup([]),
        blob_exists=exists,
        delete_blob=delete,
        lease_store=store,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert result.records[0].detail == "R2 deletion failed"
    assert result.complete is False, "cursor must not advance on R2 failure"
    assert [name for name, _ in store.calls] == [
        "acquire",
        "validate",
        "failure",
    ]


@pytest.mark.asyncio
async def test_r2_remained_present_records_failure_and_does_not_advance() -> None:
    """R2 returns success but the object remains: blur after delete is unsafe."""
    state = GcSweepState.from_report(_report("oai2-blobs/stubborn"))

    async def exists(key: str) -> bool:
        return True

    async def delete(key: str) -> None:
        return None

    store = FakeD1LeaseStore()

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=_async_lookup([]),
        blob_exists=exists,
        delete_blob=delete,
        lease_store=store,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert result.records[0].detail == "R2 object remained present after deletion"
    assert result.complete is False
    assert [name for name, _ in store.calls] == [
        "acquire",
        "validate",
        "failure",
    ]


@pytest.mark.asyncio
async def test_already_absent_finalizes_lease_idempotently() -> None:
    """AC-GC-035: already-absent is a recoverable terminal outcome."""
    state = GcSweepState.from_report(_report("oai2-blobs/ghost"))

    async def exists(key: str) -> bool:
        return False

    async def delete(key: str) -> None:
        raise AssertionError("delete must not be called for already-absent")

    store = FakeD1LeaseStore()

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=_async_lookup([]),
        blob_exists=exists,
        delete_blob=delete,
        lease_store=store,
    )

    assert result.records[0].disposition is GcSweepDisposition.ALREADY_ABSENT
    assert result.complete is True
    assert [name for name, _ in store.calls] == [
        "acquire",
        "validate",
        "finalize",
    ]
    finalize_kwargs = store.calls[-1][1]
    assert finalize_kwargs["already_absent"] is True
    assert finalize_kwargs["object_key"] == "oai2-blobs/ghost"


@pytest.mark.asyncio
async def test_confirmed_delete_advances_cursor_and_finalizes_lease() -> None:
    """AC-GC-035 + AC-GC-036: confirmed deletion produces one terminal record."""
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))
    deleted: list[str] = []
    # First exists() call per key is the pre-delete check; second is the
    # post-delete confirmation that authorizes the finalize step.
    seen: set[str] = set()

    async def confirmable_exists(key: str) -> bool:
        if key in seen:
            return False
        seen.add(key)
        return True

    async def delete(key: str) -> None:
        deleted.append(key)

    store = FakeD1LeaseStore()

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=_async_lookup([]),
        blob_exists=confirmable_exists,
        delete_blob=delete,
        lease_store=store,
        max_items=10,
    )

    assert len(result.records) == 2
    assert all(r.disposition is GcSweepDisposition.DELETED for r in result.records)
    assert deleted == ["oai2-blobs/a", "oai2-blobs/b"]
    assert result.complete is True
    finalize_keys = [
        kwargs["object_key"]
        for name, kwargs in store.calls
        if name == "finalize"
    ]
    assert finalize_keys == ["oai2-blobs/a", "oai2-blobs/b"]


# --- AC-GC-036: shared reference cannot be deleted ------------------------


@pytest.mark.asyncio
async def test_concurrent_reference_race_cannot_silently_delete() -> None:
    """AC-GC-032/036 end-to-end: an after-claim reference invalidates the lease
    and prevents any R2 call, even if the boundary's exists check still says
    true.

    The key invariant is: when the lease is invalid pre-delete, no R2
    deletion occurs and the candidate is recorded as LEASE_DENIED.
    """
    store = FakeD1LeaseStore(valid_result=False)
    state = GcSweepState.from_report(_report("oai2-blobs/race"))

    touched: list[str] = []

    async def exists(key: str) -> bool:
        touched.append("exists")
        return True

    async def delete(key: str) -> None:
        touched.append("delete")

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=_async_lookup([]),
        blob_exists=exists,
        delete_blob=delete,
        lease_store=store,
    )

    assert "delete" not in touched
    assert touched == ["exists"]
    assert result.records[0].disposition is GcSweepDisposition.LEASE_DENIED


# --- Async test helpers ----------------------------------------------------


def _forbidden() -> Any:
    raise AssertionError("callback must not be invoked")


def _async_lookup(values: Any) -> Any:
    async def _lookup(_key: str) -> Any:
        if values is _forbidden:
            _forbidden()
        return values

    return _lookup


def _async_exists(value: Any) -> Any:
    async def _exists(_key: str) -> Any:
        if value is _forbidden:
            _forbidden()
        return value

    return _exists


def _async_delete(value: Any) -> Any:
    async def _delete(_key: str) -> Any:
        if value is _forbidden:
            _forbidden()
        return value

    return _delete