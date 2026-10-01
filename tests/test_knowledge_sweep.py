"""Tests for conservative, resumable R2 orphan sweep behavior."""

from __future__ import annotations

import pytest

from oai2.knowledge.gc import (
    GcDryRunReport,
    GcObjectDisposition,
    GcObjectRecord,
)
from oai2.knowledge.gc_lease import (
    GcDeleteLeaseAuthority,
    GcReferenceDecision,
)
from oai2.knowledge.sweep import (
    GcSweepDisposition,
    GcSweepState,
)


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


def test_grace_period_defers_without_authoritative_or_delete_calls() -> None:
    state = GcSweepState.from_report(_report("oai2-blobs/a", observed_at=10.0))

    result = state.process_batch(
        now=15.0,
        grace_seconds=10.0,
        reference_lookup=lambda _key: (_ for _ in ()).throw(AssertionError()),
        blob_exists=lambda _key: (_ for _ in ()).throw(AssertionError()),
        delete_blob=lambda _key: (_ for _ in ()).throw(AssertionError()),
    )

    assert result.records[0].disposition is GcSweepDisposition.DEFERRED_GRACE
    assert result.complete is True


def test_new_authoritative_reference_prevents_delete() -> None:
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/shared"))

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: ("ko-new",),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )

    assert result.records[0].disposition is GcSweepDisposition.RE_REFERENCED
    assert result.records[0].knowledge_ids == ("ko-new",)
    assert deleted == []


def test_dry_run_and_missing_authorization_never_delete() -> None:
    deleted: list[str] = []
    report = _report("oai2-blobs/a")

    dry = GcSweepState.from_report(report)
    dry_result = dry.process_batch(
        now=100.0,
        grace_seconds=0.0,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )
    assert dry_result.records[0].disposition is GcSweepDisposition.DRY_RUN

    blocked = GcSweepState.from_report(report)
    blocked_result = blocked.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=False,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )
    assert blocked_result.records[0].disposition is GcSweepDisposition.UNAPPROVED
    assert deleted == []


def test_recovery_precondition_blocks_destructive_delete() -> None:
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=False,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )

    assert result.records[0].disposition is GcSweepDisposition.RECOVERY_BLOCKED
    assert deleted == []


def test_confirmed_orphan_delete_is_idempotent_across_passes() -> None:
    blobs = {"oai2-blobs/a"}
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    first = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
    )
    assert first.records[0].disposition is GcSweepDisposition.DELETED
    assert blobs == set()

    state.start_next_pass()
    second = state.process_batch(
        now=101.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
    )
    assert second.records[0].disposition is GcSweepDisposition.ALREADY_ABSENT


def test_partial_failure_resumes_at_failed_candidate() -> None:
    blobs = {"oai2-blobs/a", "oai2-blobs/b"}
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))
    fail_b = True

    def delete(key: str) -> None:
        nonlocal fail_b
        if key == "oai2-blobs/b" and fail_b:
            fail_b = False
            raise RuntimeError("dependency outage")
        blobs.remove(key)

    first = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        max_items=2,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=delete,
    )

    assert [record.disposition for record in first.records] == [
        GcSweepDisposition.DELETED,
        GcSweepDisposition.FAILED,
    ]
    assert first.next_cursor == 1
    assert state.cursor == 1
    assert blobs == {"oai2-blobs/b"}

    resumed = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        max_items=2,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=delete,
    )
    assert resumed.records[-1].disposition is GcSweepDisposition.DELETED
    assert resumed.complete is True
    assert blobs == set()


def test_snapshot_round_trip_preserves_cursor_and_first_seen() -> None:
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))
    result = state.process_batch(
        now=20.0,
        grace_seconds=100.0,
        max_items=1,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=lambda _key: None,
    )
    assert result.next_cursor == 1

    restored = GcSweepState.from_snapshot(state.to_snapshot())
    assert restored.cursor == 1
    assert restored.candidates[0].first_seen_at == 10.0


def test_snapshot_rejects_cursor_tampering() -> None:
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))
    snapshot = state.to_snapshot()
    snapshot["cursor"] = 1

    with pytest.raises(ValueError, match="snapshot fingerprint is invalid"):
        GcSweepState.from_snapshot(snapshot)


def test_reference_added_after_initial_lookup_prevents_delete() -> None:
    deleted: list[str] = []
    references: tuple[str, ...] = ()
    lookups = 0
    state = GcSweepState.from_report(_report("oai2-blobs/race"))

    def reference_lookup(_key: str) -> tuple[str, ...]:
        nonlocal lookups
        lookups += 1
        return references

    def blob_exists(_key: str) -> bool:
        nonlocal references
        references = ("ko-race",)
        return True

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=reference_lookup,
        blob_exists=blob_exists,
        delete_blob=deleted.append,
    )

    assert lookups == 2
    assert result.records[0].disposition is GcSweepDisposition.RE_REFERENCED
    assert result.records[0].knowledge_ids == ("ko-race",)
    assert deleted == []


def test_rereferenced_candidate_requires_new_dry_run_before_future_delete() -> None:
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/shared"))

    first = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: ("ko-live",),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )
    assert first.records[0].disposition is GcSweepDisposition.RE_REFERENCED

    state.start_next_pass()
    second = state.process_batch(
        now=200.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )

    assert second.complete is True
    assert second.records == ()
    assert deleted == []


def test_non_boolean_blob_existence_result_fails_closed() -> None:
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: "yes",  # type: ignore[return-value]
        delete_blob=deleted.append,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert state.cursor == 0
    assert deleted == []


def test_sweep_without_lease_authority_unchanged() -> None:
    """When ``lease_authority`` is not provided the destructive path is
    unchanged: no D1 lease is queried, delete_blob runs, and the candidate
    is recorded as DELETED.
    """
    blobs = {"oai2-blobs/a"}
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
    )

    assert result.records[0].disposition is GcSweepDisposition.DELETED
    assert blobs == set()


def test_sweep_lease_authority_acquires_before_and_finalizes_after_delete() -> None:
    """With ``lease_authority`` provided the destructive path acquires a
    D1-authoritative lease immediately before delete_blob and finalizes the
    lease after a confirmed absence.
    """
    blobs = {"oai2-blobs/a"}
    state = GcSweepState.from_report(_report("oai2-blobs/a"))
    authority = GcDeleteLeaseAuthority()

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
        lease_authority=authority,
    )

    assert result.records[0].disposition is GcSweepDisposition.DELETED
    assert blobs == set()
    # Lease was finalized: a subsequent reference attempt is blocked as a
    # deleted body (proves finalize_delete ran successfully).
    assert (
        authority.try_add_reference("oai2-blobs/a", "ko-after")
        is GcReferenceDecision.BODY_DELETED
    )


def test_sweep_lease_authority_writer_reference_during_delete_path_denies() -> None:
    """If the lease-authority state changes between the second reference
    lookup and the lease acquire (simulated here by a writer adding a
    reference inside blob_exists), the sweep emits LEASE_DENIED and does
    not call delete_blob.
    """
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/race"))
    authority = GcDeleteLeaseAuthority()

    def reference_lookup(_key: str) -> tuple[str, ...]:
        return ()

    def blob_exists(_key: str) -> bool:
        # Race: a writer adds a reference via the lease authority between
        # the second sweep reference lookup and the lease acquire.
        authority.try_add_reference("oai2-blobs/race", "ko-late")
        return True

    def delete_blob(key: str) -> None:
        deleted.append(key)

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=reference_lookup,
        blob_exists=blob_exists,
        delete_blob=delete_blob,
        lease_authority=authority,
    )

    assert result.records[0].disposition is GcSweepDisposition.LEASE_DENIED
    assert deleted == []
    # No lease is held for this key after a denial.
    assert (
        authority.try_add_reference("oai2-blobs/race", "ko-other")
        is GcReferenceDecision.ADDED
    )


def test_sweep_lease_authority_records_failure_when_delete_fails() -> None:
    """When ``lease_authority`` is provided and delete_blob raises, the sweep
    records a retryable failure on the lease (DELETE_FAILED state) so the
    next attempt can recover the claim and writers stay excluded.
    """
    blobs = {"oai2-blobs/a"}
    state = GcSweepState.from_report(_report("oai2-blobs/a"))
    authority = GcDeleteLeaseAuthority()

    def delete_blob(key: str) -> None:
        raise RuntimeError("simulated R2 outage")

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=delete_blob,
        lease_authority=authority,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert state.cursor == 0
    # The lease stays held and continues to block writers (recoverable).
    assert (
        authority.try_add_reference("oai2-blobs/a", "ko-recover")
        is GcReferenceDecision.BLOCKED_BY_DELETE_LEASE
    )
