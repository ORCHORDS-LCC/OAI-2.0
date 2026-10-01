"""Tests for reference-safe dry-run R2 liveness reconciliation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from oai2.knowledge import (
    GcObjectDisposition,
    GcReconciliationState,
    R2InventoryObject,
)


@dataclass(frozen=True)
class _Row:
    knowledge_id: str
    r2_blob_key: str | None


def _records_by_key(report):
    return {record.key: record for record in report.records}


def test_dry_run_classifies_shared_present_missing_and_orphan() -> None:
    rows = [
        _Row("ko-1", "oai2-blobs/shared"),
        _Row("ko-2", "oai2-blobs/shared"),
        _Row("ko-3", "oai2-blobs/missing"),
        _Row("ko-no-body", None),
    ]
    state = GcReconciliationState.from_rows(rows, observed_at=100.0)
    state.consume_page(
        [
            R2InventoryObject(
                key="oai2-blobs/shared", size_bytes=10, uploaded_at=90.0
            ),
            R2InventoryObject(
                key="oai2-blobs/orphan", size_bytes=7, uploaded_at=80.0
            ),
        ],
        next_cursor=None,
    )

    report = state.build_report()
    records = _records_by_key(report)

    assert (
        records["oai2-blobs/shared"].disposition
        is GcObjectDisposition.REFERENCED_PRESENT
    )
    assert records["oai2-blobs/shared"].knowledge_ids == ("ko-1", "ko-2")
    assert (
        records["oai2-blobs/missing"].disposition
        is GcObjectDisposition.REFERENCED_MISSING
    )
    assert records["oai2-blobs/missing"].size_bytes is None
    assert (
        records["oai2-blobs/orphan"].disposition
        is GcObjectDisposition.UNREFERENCED_CANDIDATE
    )
    assert records["oai2-blobs/orphan"].age_seconds == 20.0

    assert report.referenced_present_count == 1
    assert report.referenced_missing_count == 1
    assert report.unreferenced_candidate_count == 1
    assert report.inventory_object_count == 2
    assert report.inventory_bytes_known == 17
    assert report.unreferenced_candidate_bytes_known == 7
    assert report.unknown_size_object_count == 0
    assert report.oldest_unreferenced_candidate_age_seconds == 20.0


def test_incomplete_inventory_refuses_final_report() -> None:
    state = GcReconciliationState.from_rows(
        [_Row("ko-1", "oai2-blobs/a")], observed_at=10.0
    )
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=1)],
        next_cursor="page-2",
    )

    with pytest.raises(RuntimeError, match="inventory scan is incomplete"):
        state.build_report()


def test_snapshot_round_trip_allows_resume() -> None:
    state = GcReconciliationState.from_rows(
        [_Row("ko-1", "oai2-blobs/a")], observed_at=100.0
    )
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=3, uploaded_at=90.0)],
        next_cursor="next-page",
    )

    restored = GcReconciliationState.from_snapshot(state.to_snapshot())
    assert restored.next_cursor == "next-page"
    assert restored.pages_processed == 1

    restored.consume_page(
        [R2InventoryObject(key="oai2-blobs/orphan", size_bytes=5, uploaded_at=95.0)],
        next_cursor=None,
    )
    report = restored.build_report()

    assert report.pages_processed == 2
    assert report.referenced_present_count == 1
    assert report.unreferenced_candidate_count == 1


def test_conflicting_duplicate_inventory_metadata_is_rejected() -> None:
    state = GcReconciliationState.from_rows([], observed_at=100.0)
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=3)],
        next_cursor="next",
    )

    with pytest.raises(ValueError, match="conflicting inventory metadata"):
        state.consume_page(
            [R2InventoryObject(key="oai2-blobs/a", size_bytes=4)],
            next_cursor=None,
        )


def test_inventory_object_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="size_bytes must be non-negative"):
        R2InventoryObject(key="oai2-blobs/a", size_bytes=-1)


def test_unknown_size_is_counted_where_metrics_are_available() -> None:
    state = GcReconciliationState.from_rows([], observed_at=50.0)
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/orphan", size_bytes=None)],
        next_cursor=None,
    )

    report = state.build_report()
    assert report.inventory_bytes_known == 0
    assert report.unreferenced_candidate_bytes_known == 0
    assert report.unknown_size_object_count == 1
