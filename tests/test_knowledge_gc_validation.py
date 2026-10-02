"""Outer-guard-rail boundary contracts for ``oai2.knowledge.gc``.

This file pins the public-safety boundary of the dry-run R2 liveness
reconciliation. The companion module ``oai2/knowledge/gc.py`` derives the
mark set from an authoritative set of retained knowledge rows, consumes
paginated R2 inventory, and produces a deterministic report of present,
missing, and unreferenced object keys.

These tests focus on the *outer guard rails*: enum stability,
dataclass ``__post_init__`` validation, cursor invariants, snapshot
schema/version/fingerprint boundaries, and exhaustive error-path
coverage for ``from_rows`` / ``consume_page`` / ``build_report`` /
``from_snapshot``. They complement (rather than duplicate) the
happy-path and resumability tests in ``test_knowledge_gc.py``.

A refactor that swaps ``isinstance(value, int)`` for the wider
``(int, float)`` test — or that silently coerces non-string
``r2_blob_key`` values, or that weakens the snapshot fingerprint
contract — must trip one of these tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from oai2.knowledge import gc as gc_mod
from oai2.knowledge.gc import (
    GcDryRunReport,
    GcObjectDisposition,
    GcObjectRecord,
    GcReconciliationState,
    R2InventoryObject,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    knowledge_id: object
    r2_blob_key: str | None


def _ref(state: GcReconciliationState) -> dict[str, set[str]]:
    return {key: set(ids) for key, ids in state.references.items()}


def _snap_fingerprint(state: GcReconciliationState) -> str:
    return state.reference_fingerprint


def _build_empty_state() -> GcReconciliationState:
    return GcReconciliationState(observed_at=10.0)


def _build_state_with_pages(
    *objects: R2InventoryObject,
    references: tuple[tuple[str, str], ...] = (),
    observed_at: float = 10.0,
) -> GcReconciliationState:
    rows = [_Row(kid, key) for key, kid in references]
    state = GcReconciliationState.from_rows(rows, observed_at=observed_at)
    if objects:
        state.consume_page(list(objects), next_cursor=None)
    return state


# ---------------------------------------------------------------------------
# 1. GcObjectDisposition enum stability
# ---------------------------------------------------------------------------


def test_disposition_has_three_distinct_members() -> None:
    """The disposition enum is closed; adding a value is a public-API change
    and must be flagged here so the reviewer's contract stays visible."""
    members = list(GcObjectDisposition)
    assert len(members) == 3
    assert len({m.value for m in members}) == 3


@pytest.mark.parametrize(
    "member, expected",
    [
        (GcObjectDisposition.REFERENCED_PRESENT, "referenced_present"),
        (GcObjectDisposition.REFERENCED_MISSING, "referenced_missing"),
        (GcObjectDisposition.UNREFERENCED_CANDIDATE, "unreferenced_candidate"),
    ],
)
def test_disposition_string_values_are_stable(member: GcObjectDisposition, expected: str) -> None:
    """The string forms are persisted in snapshots; do not rename."""
    assert member.value == expected


def test_disposition_lookup_by_value_constructs_member() -> None:
    """The enum is constructible from its string value — used by snapshot
    deserialization."""
    for member in GcObjectDisposition:
        assert GcObjectDisposition(member.value) is member


# ---------------------------------------------------------------------------
# 2. R2InventoryObject field guards
# ---------------------------------------------------------------------------


def test_inventory_object_default_size_and_uploaded_are_none() -> None:
    """Both size_bytes and uploaded_at default to ``None`` — the dry-run
    tolerates objects whose size and age are not yet known."""
    obj = R2InventoryObject(key="oai2-blobs/a")
    assert obj.size_bytes is None
    assert obj.uploaded_at is None


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", " leading", "trailing "],
    ids=["empty", "whitespace", "leading", "trailing"],
)
def test_inventory_object_rejects_non_normalized_key(bad_key: str) -> None:
    """Empty / whitespace / unstripped keys are all rejected — the
    reference set treats the key as a canonical identifier."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        R2InventoryObject(key=bad_key)


def test_inventory_object_rejects_non_string_key() -> None:
    """A non-string key is a structural error — must fail loudly."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        R2InventoryObject(key=123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_size", [-1, True, 1.5, "3"], ids=["negative", "bool", "float", "string"]
)
def test_inventory_object_rejects_invalid_size_bytes(bad_size: object) -> None:
    """size_bytes must be a non-negative int (zero allowed). ``bool``
    is rejected because ``isinstance(True, int) is True`` — a refactor
    that switches to ``isinstance(value, int)`` would silently accept
    ``True`` (=1) as a byte count."""
    with pytest.raises(ValueError, match="size_bytes must be a non-negative integer"):
        R2InventoryObject(key="oai2-blobs/a", size_bytes=bad_size)  # type: ignore[arg-type]


def test_inventory_object_accepts_size_bytes_zero() -> None:
    """Zero is a valid size — the GC accounting allows it explicitly."""
    obj = R2InventoryObject(key="oai2-blobs/a", size_bytes=0)
    assert obj.size_bytes == 0


@pytest.mark.parametrize(
    "bad_time",
    [math.nan, math.inf, -math.inf, -1.0, True],
    ids=["nan", "posinf", "neginf", "negative", "bool"],
)
def test_inventory_object_rejects_non_finite_or_negative_uploaded_at(
    bad_time: object,
) -> None:
    """``uploaded_at`` is a clock value — NaN / ±inf / negative /
    bool all fail. The bool arm is load-bearing (see _Row above)."""
    with pytest.raises(ValueError, match="uploaded_at must be a finite non-negative number"):
        R2InventoryObject(key="oai2-blobs/a", uploaded_at=bad_time)  # type: ignore[arg-type]


def test_inventory_object_accepts_zero_uploaded_at() -> None:
    """Zero is a valid uploaded_at — the epoch is a legitimate clock value."""
    obj = R2InventoryObject(key="oai2-blobs/a", uploaded_at=0.0)
    assert obj.uploaded_at == 0.0


# ---------------------------------------------------------------------------
# 3. GcObjectRecord + GcDryRunReport shape
# ---------------------------------------------------------------------------


def test_object_record_required_field_set() -> None:
    """GcObjectRecord's required fields are exactly ``key`` and
    ``disposition``; the rest default to safe sentinels."""
    rec = GcObjectRecord(key="oai2-blobs/a", disposition=GcObjectDisposition.REFERENCED_PRESENT)
    assert rec.knowledge_ids == ()
    assert rec.size_bytes is None
    assert rec.uploaded_at is None
    assert rec.age_seconds is None


def test_dry_run_report_required_field_set() -> None:
    """GcDryRunReport is fully populated — no defaults to sentinel."""
    report = GcDryRunReport(
        observed_at=100.0,
        pages_processed=1,
        records=(),
        referenced_present_count=0,
        referenced_missing_count=0,
        unreferenced_candidate_count=0,
        inventory_object_count=0,
        inventory_bytes_known=0,
        unreferenced_candidate_bytes_known=0,
        unknown_size_object_count=0,
        oldest_unreferenced_candidate_age_seconds=None,
    )
    assert report.observed_at == 100.0
    assert report.pages_processed == 1
    assert report.records == ()


def test_object_record_and_dry_run_report_are_frozen() -> None:
    """Both public-safe evidence containers are immutable after construction."""
    rec = GcObjectRecord(key="oai2-blobs/a", disposition=GcObjectDisposition.REFERENCED_PRESENT)
    report = GcDryRunReport(
        observed_at=1.0,
        pages_processed=0,
        records=(),
        referenced_present_count=0,
        referenced_missing_count=0,
        unreferenced_candidate_count=0,
        inventory_object_count=0,
        inventory_bytes_known=0,
        unreferenced_candidate_bytes_known=0,
        unknown_size_object_count=0,
        oldest_unreferenced_candidate_age_seconds=None,
    )
    with pytest.raises((AttributeError, Exception)):
        rec.key = "oai2-blobs/other"  # type: ignore[misc]
    with pytest.raises((AttributeError, Exception)):
        report.observed_at = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. GcReconciliationState __post_init__ guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_time",
    [math.nan, math.inf, -math.inf, -1.0, True],
    ids=["nan", "posinf", "neginf", "negative", "bool"],
)
def test_reconciliation_state_rejects_invalid_observed_at(bad_time: object) -> None:
    """observed_at must be a finite non-negative number — same bool / NaN /
    inf rejection as R2InventoryObject.uploaded_at."""
    with pytest.raises(ValueError, match="observed_at must be a finite non-negative number"):
        GcReconciliationState(observed_at=bad_time)  # type: ignore[arg-type]


def test_reconciliation_state_accepts_zero_observed_at() -> None:
    """Zero observed_at is a valid epoch."""
    state = GcReconciliationState(observed_at=0.0)
    assert state.observed_at == 0.0


@pytest.mark.parametrize(
    "bad_pages", [True, -1, 1.5, "3"], ids=["bool", "negative", "float", "string"]
)
def test_reconciliation_state_rejects_invalid_pages_processed(
    bad_pages: object,
) -> None:
    """pages_processed must be a non-negative int — bool is rejected."""
    with pytest.raises(ValueError, match="pages_processed must be a non-negative integer"):
        GcReconciliationState(observed_at=1.0, pages_processed=bad_pages)  # type: ignore[arg-type]


def test_reconciliation_state_rejects_fingerprint_mismatch() -> None:
    """If the supplied ``reference_fingerprint`` does not match the
    canonical fingerprint of the references mapping, the state is
    inconsistent and must be rejected."""
    with pytest.raises(ValueError, match="reference fingerprint does not match references"):
        GcReconciliationState(observed_at=1.0, reference_fingerprint="not-the-real-fp")


def test_reconciliation_state_rejects_completed_with_next_cursor() -> None:
    """inventory_complete=True + next_cursor set is an internal inconsistency
    that the constructor refuses."""
    with pytest.raises(ValueError, match="completed inventory cannot have next_cursor"):
        GcReconciliationState(
            observed_at=1.0,
            pages_processed=2,
            next_cursor="still-more",
            inventory_complete=True,
        )


def test_reconciliation_state_rejects_incomplete_pages_with_no_cursor() -> None:
    """pages_processed > 0 + next_cursor=None + inventory_complete=False is
    an inconsistent mid-scan state."""
    with pytest.raises(
        ValueError,
        match="incomplete inventory with pages must have next_cursor",
    ):
        GcReconciliationState(
            observed_at=1.0,
            pages_processed=2,
            next_cursor=None,
            inventory_complete=False,
        )


def test_reconciliation_state_auto_computes_empty_fingerprint() -> None:
    """Empty references produce the SHA-256 of the empty dict (the canonical
    sentinel) — the fingerprint is auto-populated when the caller omits it."""
    state = GcReconciliationState(observed_at=1.0)
    expected = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    assert state.reference_fingerprint == expected


def test_reconciliation_state_accepts_consistent_fingerprint() -> None:
    """If the caller supplies the canonical fingerprint explicitly, the
    constructor accepts it."""
    state = GcReconciliationState(observed_at=1.0)
    state2 = GcReconciliationState(
        observed_at=1.0,
        reference_fingerprint=state.reference_fingerprint,
    )
    assert state2.reference_fingerprint == state.reference_fingerprint


# ---------------------------------------------------------------------------
# 5. from_rows validation
# ---------------------------------------------------------------------------


def test_from_rows_skips_none_and_empty_keys_silently() -> None:
    """``None`` and empty ``r2_blob_key`` are interpreted as "no body"
    and silently dropped — only structural errors raise."""
    state = GcReconciliationState.from_rows(
        [
            _Row("ko-1", None),
            _Row("ko-2", ""),
            _Row("ko-3", "oai2-blobs/a"),
            _Row("ko-4", "oai2-blobs/a"),
        ],
        observed_at=1.0,
    )
    assert _ref(state) == {"oai2-blobs/a": {"ko-3", "ko-4"}}


def test_from_rows_coerces_non_string_knowledge_id_to_str() -> None:
    """Knowledge ids are str-coerced — int ids round-trip via ``str()``."""
    state = GcReconciliationState.from_rows(
        [_Row(123, "oai2-blobs/a")],
        observed_at=1.0,
    )
    assert _ref(state) == {"oai2-blobs/a": {"123"}}


def test_from_rows_rejects_non_string_blob_key() -> None:
    """A non-string ``r2_blob_key`` (other than ``None``) is a structural
    error — must fail loudly."""
    with pytest.raises(ValueError, match="row r2_blob_key must be a normalized string"):
        GcReconciliationState.from_rows(
            [_Row("ko-1", 123)],  # type: ignore[arg-type]
            observed_at=1.0,
        )


def test_from_rows_rejects_non_normalized_blob_key() -> None:
    """Unstripped keys are rejected because the snapshot canonicalization
    relies on key.strip() == key."""
    with pytest.raises(ValueError, match="row r2_blob_key must be a normalized string"):
        GcReconciliationState.from_rows(
            [_Row("ko-1", " leading")],
            observed_at=1.0,
        )


def test_from_rows_aggregates_multiple_knowledge_ids_per_key() -> None:
    """Multiple rows pointing at the same blob key accumulate into a set."""
    state = GcReconciliationState.from_rows(
        [
            _Row("ko-1", "oai2-blobs/shared"),
            _Row("ko-2", "oai2-blobs/shared"),
            _Row("ko-3", "oai2-blobs/other"),
        ],
        observed_at=1.0,
    )
    assert _ref(state) == {
        "oai2-blobs/shared": {"ko-1", "ko-2"},
        "oai2-blobs/other": {"ko-3"},
    }


# ---------------------------------------------------------------------------
# 6. consume_page guards
# ---------------------------------------------------------------------------


def test_consume_page_rejects_when_inventory_complete() -> None:
    """Once ``inventory_complete`` is True the scan is sealed; further
    pages are rejected with a RuntimeError, not a ValueError, to signal
    the precondition violation."""
    state = _build_empty_state()
    state.consume_page([], next_cursor=None)
    with pytest.raises(RuntimeError, match="inventory scan is already complete"):
        state.consume_page(
            [R2InventoryObject(key="oai2-blobs/a", size_bytes=1)],
            next_cursor=None,
        )


@pytest.mark.parametrize(
    "bad_cursor",
    ["", 0, 123, True, False, ["x"]],
    ids=["empty-string", "int-zero", "int-positive", "bool-true", "bool-false", "list"],
)
def test_consume_page_rejects_invalid_next_cursor(bad_cursor: object) -> None:
    """``next_cursor`` must be either ``None`` (completion signal) or a
    non-empty string (more-pages signal). Every other shape is rejected."""
    state = _build_empty_state()
    with pytest.raises(ValueError, match="next_cursor must be a non-empty string or null"):
        state.consume_page([], next_cursor=bad_cursor)  # type: ignore[arg-type]


def test_consume_page_rejects_conflict_within_same_page() -> None:
    """Two R2InventoryObject entries with the same key but different
    metadata inside the same page are rejected atomically — no inventory
    or cursor state changes."""
    state = _build_empty_state()
    with pytest.raises(
        ValueError,
        match="conflicting inventory metadata for object key 'oai2-blobs/a'",
    ):
        state.consume_page(
            [
                R2InventoryObject(key="oai2-blobs/a", size_bytes=10),
                R2InventoryObject(key="oai2-blobs/a", size_bytes=11),
            ],
            next_cursor=None,
        )
    assert state.pages_processed == 0
    assert dict(state.inventory) == {}


def test_consume_page_rejects_conflict_with_previous_inventory_atomically() -> None:
    """A cross-page conflict must roll back — the page is *not* partially
    merged into the inventory; pages_processed and next_cursor remain
    pinned to the previous successful page."""
    state = _build_empty_state()
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=10)],
        next_cursor="page-2",
    )
    with pytest.raises(
        ValueError,
        match="conflicting inventory metadata for object key 'oai2-blobs/a'",
    ):
        state.consume_page(
            [R2InventoryObject(key="oai2-blobs/a", size_bytes=11)],
            next_cursor=None,
        )
    assert state.pages_processed == 1
    assert state.next_cursor == "page-2"
    assert state.inventory_complete is False
    assert dict(state.inventory) == {
        "oai2-blobs/a": R2InventoryObject(key="oai2-blobs/a", size_bytes=10),
    }


def test_consume_page_accepts_identical_duplicate_metadata() -> None:
    """Repeating the exact same R2InventoryObject for a known key across
    pages is idempotent — it does not raise and the state advances."""
    state = _build_empty_state()
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=10)],
        next_cursor="page-2",
    )
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=10)],
        next_cursor=None,
    )
    assert state.pages_processed == 2
    assert state.inventory_complete is True


def test_consume_page_increments_pages_processed_and_advances_cursor() -> None:
    """Each successful page bumps ``pages_processed`` and re-pins
    ``next_cursor`` to the value the caller passed."""
    state = _build_empty_state()
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=1)],
        next_cursor="page-2",
    )
    assert state.pages_processed == 1
    assert state.next_cursor == "page-2"
    assert state.inventory_complete is False

    state.consume_page([], next_cursor=None)
    assert state.pages_processed == 2
    assert state.next_cursor is None
    assert state.inventory_complete is True


def test_consume_page_accepts_empty_page_when_more_pages_remain() -> None:
    """A page with zero objects is valid mid-scan — it just advances
    the cursor without touching the inventory."""
    state = _build_empty_state()
    state.consume_page([], next_cursor="page-2")
    assert state.pages_processed == 1
    assert dict(state.inventory) == {}


# ---------------------------------------------------------------------------
# 7. build_report edge cases
# ---------------------------------------------------------------------------


def test_build_report_requires_inventory_complete() -> None:
    """Mid-scan reports are rejected — a referenced body could look missing
    if unseen pages still contain it."""
    state = _build_empty_state()
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=1)],
        next_cursor="page-2",
    )
    with pytest.raises(RuntimeError, match="inventory scan is incomplete"):
        state.build_report()


def test_build_report_empty_inventory_with_only_references_yields_missing_count() -> None:
    """When inventory is empty but references exist, every reference is
    REFERENCED_MISSING and there are no records for unknown inventory."""
    state = GcReconciliationState.from_rows(
        [_Row("ko-1", "oai2-blobs/a"), _Row("ko-2", "oai2-blobs/a")],
        observed_at=100.0,
    )
    state.consume_page([], next_cursor=None)
    report = state.build_report()
    assert report.referenced_present_count == 0
    assert report.referenced_missing_count == 1
    assert report.unreferenced_candidate_count == 0
    assert report.inventory_object_count == 0
    assert report.inventory_bytes_known == 0
    assert report.unreferenced_candidate_bytes_known == 0
    assert report.unknown_size_object_count == 0
    assert report.oldest_unreferenced_candidate_age_seconds is None
    assert len(report.records) == 1


def test_build_report_empty_state_yields_zero_counts() -> None:
    """Empty references + empty inventory + complete scan → all counts are 0
    (the completion sentinel page itself bumps ``pages_processed`` to 1)."""
    state = _build_empty_state()
    state.consume_page([], next_cursor=None)
    report = state.build_report()
    assert report.pages_processed == 1
    assert report.referenced_present_count == 0
    assert report.referenced_missing_count == 0
    assert report.unreferenced_candidate_count == 0
    assert report.inventory_object_count == 0
    assert report.inventory_bytes_known == 0
    assert report.unreferenced_candidate_bytes_known == 0
    assert report.unknown_size_object_count == 0
    assert report.oldest_unreferenced_candidate_age_seconds is None
    assert report.records == ()


def test_build_report_clamps_negative_age_to_zero() -> None:
    """``uploaded_at > observed_at`` is clamped to 0 — age cannot go
    negative even if clocks are skewed."""
    state = _build_empty_state()
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/orphan", size_bytes=5, uploaded_at=200.0)],
        next_cursor=None,
    )
    state.observed_at = 100.0
    report = state.build_report()
    assert report.records[0].age_seconds == 0.0
    assert report.oldest_unreferenced_candidate_age_seconds == 0.0


def test_build_report_oldest_unreferenced_age_is_max_across_objects() -> None:
    """``oldest_unreferenced_candidate_age_seconds`` is the maximum age
    across all UNREFERENCED_CANDIDATE records — not the sum or the count."""
    state = _build_empty_state()
    state.observed_at = 100.0
    state.consume_page(
        [
            R2InventoryObject(key="oai2-blobs/a", size_bytes=5, uploaded_at=80.0),
            R2InventoryObject(key="oai2-blobs/b", size_bytes=5, uploaded_at=70.0),
            R2InventoryObject(key="oai2-blobs/c", size_bytes=5, uploaded_at=90.0),
        ],
        next_cursor=None,
    )
    report = state.build_report()
    assert report.oldest_unreferenced_candidate_age_seconds == 30.0


def test_build_report_separates_known_and_unknown_sizes() -> None:
    """Unknown sizes count toward ``unknown_size_object_count`` but do
    not contribute to ``inventory_bytes_known``."""
    state = _build_empty_state()
    state.consume_page(
        [
            R2InventoryObject(key="oai2-blobs/a", size_bytes=5, uploaded_at=80.0),
            R2InventoryObject(key="oai2-blobs/b", size_bytes=None, uploaded_at=70.0),
            R2InventoryObject(key="oai2-blobs/c", size_bytes=7, uploaded_at=90.0),
        ],
        next_cursor=None,
    )
    report = state.build_report()
    assert report.inventory_bytes_known == 12
    assert report.unreferenced_candidate_bytes_known == 12
    assert report.unknown_size_object_count == 1


def test_build_report_records_sorted_by_key() -> None:
    """``records`` is deterministically sorted by key — the report is
    comparable across resumes."""
    state = _build_empty_state()
    state.consume_page(
        [
            R2InventoryObject(key="oai2-blobs/z", size_bytes=1, uploaded_at=95.0),
            R2InventoryObject(key="oai2-blobs/a", size_bytes=1, uploaded_at=95.0),
            R2InventoryObject(key="oai2-blobs/m", size_bytes=1, uploaded_at=95.0),
        ],
        next_cursor=None,
    )
    report = state.build_report()
    assert [r.key for r in report.records] == ["oai2-blobs/a", "oai2-blobs/m", "oai2-blobs/z"]


def test_build_report_present_record_inherits_inventory_metadata() -> None:
    """A REFERENCED_PRESENT record copies size_bytes / uploaded_at /
    age_seconds from the inventory object — it does not synthesize them."""
    state = GcReconciliationState.from_rows(
        [_Row("ko-1", "oai2-blobs/a")],
        observed_at=100.0,
    )
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=10, uploaded_at=90.0)],
        next_cursor=None,
    )
    report = state.build_report()
    record = report.records[0]
    assert record.disposition is GcObjectDisposition.REFERENCED_PRESENT
    assert record.size_bytes == 10
    assert record.uploaded_at == 90.0
    assert record.age_seconds == 10.0


def test_build_report_missing_record_has_no_inventory_metadata() -> None:
    """A REFERENCED_MISSING record has ``size_bytes=None``,
    ``uploaded_at=None``, ``age_seconds=None`` — the object is absent."""
    state = GcReconciliationState.from_rows(
        [_Row("ko-1", "oai2-blobs/missing")],
        observed_at=100.0,
    )
    state.consume_page([], next_cursor=None)
    report = state.build_report()
    record = report.records[0]
    assert record.disposition is GcObjectDisposition.REFERENCED_MISSING
    assert record.size_bytes is None
    assert record.uploaded_at is None
    assert record.age_seconds is None


# ---------------------------------------------------------------------------
# 8. to_snapshot shape
# ---------------------------------------------------------------------------


def test_snapshot_schema_version_is_one() -> None:
    """The on-disk snapshot format is versioned at schema_version=1."""
    state = _build_empty_state()
    assert state.to_snapshot()["schema_version"] == 1


def test_snapshot_references_are_lists_sorted_by_id() -> None:
    """Each value of ``references`` is a sorted list of knowledge id
    strings — sets are not JSON-serializable."""
    state = GcReconciliationState.from_rows(
        [
            _Row("ko-b", "oai2-blobs/a"),
            _Row("ko-a", "oai2-blobs/a"),
            _Row("ko-c", "oai2-blobs/a"),
        ],
        observed_at=1.0,
    )
    snap = state.to_snapshot()
    assert snap["references"] == {"oai2-blobs/a": ["ko-a", "ko-b", "ko-c"]}


def test_snapshot_inventory_keys_sorted() -> None:
    """The inventory mapping is emitted in sorted key order — the
    snapshot is canonical across iterations."""
    state = _build_empty_state()
    state.consume_page(
        [
            R2InventoryObject(key="oai2-blobs/z", size_bytes=1, uploaded_at=99.0),
            R2InventoryObject(key="oai2-blobs/a", size_bytes=1, uploaded_at=99.0),
            R2InventoryObject(key="oai2-blobs/m", size_bytes=1, uploaded_at=99.0),
        ],
        next_cursor=None,
    )
    snap = state.to_snapshot()
    inventory_keys = sorted(snap["inventory"].keys())  # type: ignore[attr-defined]
    assert inventory_keys == ["oai2-blobs/a", "oai2-blobs/m", "oai2-blobs/z"]


def test_snapshot_round_trip_preserves_inventory_metadata() -> None:
    """A snapshot must round-trip through ``from_snapshot`` to a state
    that produces an equivalent report."""
    state = GcReconciliationState.from_rows(
        [
            _Row("ko-1", "oai2-blobs/shared"),
            _Row("ko-2", "oai2-blobs/shared"),
            _Row("ko-3", "oai2-blobs/missing"),
        ],
        observed_at=100.0,
    )
    state.consume_page(
        [
            R2InventoryObject(key="oai2-blobs/shared", size_bytes=10, uploaded_at=90.0),
            R2InventoryObject(key="oai2-blobs/orphan", size_bytes=5, uploaded_at=85.0),
        ],
        next_cursor=None,
    )
    snap = state.to_snapshot()
    restored = GcReconciliationState.from_snapshot(snap)
    assert _snap_fingerprint(restored) == _snap_fingerprint(state)
    assert restored.next_cursor is None
    assert restored.pages_processed == 1
    report = restored.build_report()
    assert report.referenced_present_count == 1
    assert report.referenced_missing_count == 1
    assert report.unreferenced_candidate_count == 1


# ---------------------------------------------------------------------------
# 9. from_snapshot validation
# ---------------------------------------------------------------------------


def test_from_snapshot_rejects_missing_schema_version() -> None:
    """The schema_version key is required — older snapshots without it
    must be refused."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    del snap["schema_version"]
    with pytest.raises(ValueError, match="unsupported or missing snapshot schema_version"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_wrong_schema_version() -> None:
    """Future schema versions are explicitly refused — silent acceptance
    would risk misinterpreting the format."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["schema_version"] = 99
    with pytest.raises(ValueError, match="unsupported or missing snapshot schema_version"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_non_mapping_references() -> None:
    """``references`` must be a mapping of str → list[str] — not a list,
    tuple, or scalar."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["references"] = []
    with pytest.raises(ValueError, match="snapshot references must be a mapping"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_non_list_knowledge_ids() -> None:
    """Each value in ``references`` must be a list of strings — sets and
    scalars are not accepted (they would round-trip incorrectly)."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["references"] = {"oai2-blobs/a": "ko-1"}
    with pytest.raises(ValueError, match="snapshot references contain invalid data"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_empty_or_non_string_knowledge_id() -> None:
    """Empty / non-string ids in the knowledge_ids list are rejected —
    they would violate the structural invariant."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["references"] = {"oai2-blobs/a": [""]}
    with pytest.raises(ValueError, match="snapshot references contain invalid data"):
        GcReconciliationState.from_snapshot(snap)

    snap2 = _build_empty_state().to_snapshot()
    snap2["references"] = {"oai2-blobs/a": [123]}
    with pytest.raises(ValueError, match="snapshot references contain invalid data"):
        GcReconciliationState.from_snapshot(snap2)


def test_from_snapshot_rejects_non_mapping_inventory() -> None:
    """``inventory`` must be a mapping of str → object-mapping — not a list
    or scalar."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["inventory"] = []
    with pytest.raises(ValueError, match="snapshot inventory must be a mapping"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_inventory_key_mismatch() -> None:
    """The outer inventory key must equal the inner object's ``key``
    field — divergence indicates a tampered or corrupted snapshot."""
    state = _build_empty_state()
    state.consume_page(
        [R2InventoryObject(key="oai2-blobs/a", size_bytes=1, uploaded_at=99.0)],
        next_cursor=None,
    )
    snap = state.to_snapshot()
    snap["inventory"] = {
        "oai2-blobs/a": {"key": "oai2-blobs/b", "size_bytes": 1, "uploaded_at": 99.0}
    }
    with pytest.raises(ValueError, match="snapshot inventory key does not match object key"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_missing_reference_fingerprint() -> None:
    """``reference_fingerprint`` is required — without it the snapshot is
    incomplete and could be forged without detection."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    del snap["reference_fingerprint"]
    with pytest.raises(ValueError, match="snapshot reference_fingerprint is required"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_empty_reference_fingerprint() -> None:
    """An empty fingerprint is treated as missing — it provides no
    integrity guarantee."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["reference_fingerprint"] = ""
    with pytest.raises(ValueError, match="snapshot reference_fingerprint is required"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_wrong_reference_fingerprint() -> None:
    """A snapshot whose fingerprint does not match its references is
    rejected — the integrity check is a load-bearing safety property."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["reference_fingerprint"] = "not-the-canonical-fp"
    with pytest.raises(ValueError, match="snapshot reference fingerprint is invalid"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_non_boolean_inventory_complete() -> None:
    """``inventory_complete`` must be a strict bool — int / str are
    rejected because Python's truthiness rules differ."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["inventory_complete"] = "false"
    with pytest.raises(ValueError, match="snapshot inventory_complete must be a boolean"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_non_string_or_empty_next_cursor() -> None:
    """``next_cursor`` must be either ``None`` or a non-empty string."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["next_cursor"] = ""
    with pytest.raises(ValueError, match="snapshot next_cursor must be a non-empty string or null"):
        GcReconciliationState.from_snapshot(snap)

    snap2 = _build_empty_state().to_snapshot()
    snap2["next_cursor"] = 123
    with pytest.raises(ValueError, match="snapshot next_cursor must be a non-empty string or null"):
        GcReconciliationState.from_snapshot(snap2)


def test_from_snapshot_rejects_completed_with_next_cursor() -> None:
    """The cursor-state invariant is re-validated on snapshot restore —
    completed-with-cursor is still inconsistent."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["pages_processed"] = 1
    snap["inventory_complete"] = True
    snap["next_cursor"] = "still-more"
    with pytest.raises(ValueError, match="completed inventory cannot have next_cursor"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_authoritative_reference_drift() -> None:
    """If authoritative_rows is supplied and its fingerprint does not
    match the snapshot's, the snapshot is stale and must be rejected."""
    rows_now = [_Row("ko-1", "oai2-blobs/a")]
    state = GcReconciliationState.from_rows(rows_now, observed_at=10.0)
    snap = state.to_snapshot()
    drifted_rows = [_Row("ko-1", "oai2-blobs/different")]
    with pytest.raises(ValueError, match="authoritative reference set changed since snapshot"):
        GcReconciliationState.from_snapshot(snap, authoritative_rows=drifted_rows)


def test_from_snapshot_accepts_consistent_authoritative_rows() -> None:
    """When authoritative_rows matches the snapshot fingerprint, restore
    succeeds and the fingerprint is preserved."""
    rows = [_Row("ko-1", "oai2-blobs/a")]
    state = GcReconciliationState.from_rows(rows, observed_at=10.0)
    snap = state.to_snapshot()
    restored = GcReconciliationState.from_snapshot(snap, authoritative_rows=rows)
    assert _snap_fingerprint(restored) == _snap_fingerprint(state)


def test_from_snapshot_rejects_inventory_with_negative_size_bytes() -> None:
    """Inventory entries are re-validated through R2InventoryObject — a
    negative size_bytes raises the same boundary error as direct construction."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["inventory"] = {
        "oai2-blobs/a": {"key": "oai2-blobs/a", "size_bytes": -1, "uploaded_at": 99.0}
    }
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        GcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_inventory_with_nan_uploaded_at() -> None:
    """Non-finite ``uploaded_at`` is rejected — JSON does not normally
    allow NaN, but the validator must defend against tampering."""
    state = _build_empty_state()
    snap = state.to_snapshot()
    snap["inventory"] = {
        "oai2-blobs/a": {"key": "oai2-blobs/a", "size_bytes": 1, "uploaded_at": math.nan}
    }
    with pytest.raises(ValueError, match="must be a finite non-negative number"):
        GcReconciliationState.from_snapshot(snap)


# ---------------------------------------------------------------------------
# 10. Dataclass shape + __all__
# ---------------------------------------------------------------------------


def test_reconciliation_state_is_slots_but_not_frozen() -> None:
    """GcReconciliationState is mutable (it advances through
    ``consume_page``) but uses ``__slots__`` — both must hold."""
    import dataclasses

    state = _build_empty_state()
    dataclasses.fields(state)  # ensure dataclass-decorated
    params = type(state).__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is False
    # Mutate next_cursor
    state.next_cursor = "manually-set"


def test_r2_inventory_object_record_and_dry_run_report_are_frozen_with_slots() -> None:
    """The three public-safe evidence containers are immutable + slotted."""
    for cls in (R2InventoryObject, GcObjectRecord, GcDryRunReport):
        params = cls.__dataclass_params__  # type: ignore[union-attr]
        assert params.frozen is True, cls.__name__
        assert params.slots is True, cls.__name__


def test_module_all_lists_six_public_exports() -> None:
    """``__all__`` is the canonical public surface — adding a name is a
    deliberate API change and must be flagged here."""
    assert sorted(gc_mod.__all__) == sorted(
        [
            "KnowledgeBlobRow",
            "GcObjectDisposition",
            "R2InventoryObject",
            "GcObjectRecord",
            "GcDryRunReport",
            "GcReconciliationState",
        ]
    )


def test_knowledge_blob_row_protocol_has_required_attributes() -> None:
    """The KnowledgeBlobRow Protocol declares ``knowledge_id`` and
    ``r2_blob_key`` — the structural contract callers must satisfy."""
    protocol = gc_mod.KnowledgeBlobRow
    annotations = getattr(protocol, "__annotations__", {})
    assert "knowledge_id" in annotations
    assert "r2_blob_key" in annotations
