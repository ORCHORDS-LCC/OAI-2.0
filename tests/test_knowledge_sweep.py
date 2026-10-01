"""Tests for conservative, resumable R2 orphan sweep behavior."""

from __future__ import annotations

from oai2.knowledge.gc import (
    GcDryRunReport,
    GcObjectDisposition,
    GcObjectRecord,
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
