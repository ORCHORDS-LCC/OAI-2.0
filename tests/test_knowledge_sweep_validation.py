"""Outer guard rails for ``oai2/knowledge/sweep.py`` boundary contracts.

The existing ``test_knowledge_sweep.py`` (sync, 22 tests) and
``test_knowledge_sweep_async.py`` (async, 11 tests) cover the happy paths and
the most important end-to-end dispositions (DEFERRED_GRACE, RE_REFERENCED,
DRY_RUN, UNAPPROVED, RECOVERY_BLOCKED, DELETED, ALREADY_ABSENT, FAILED with
three reasons, LEASE_DENIED with three causes). What is NOT pinned:

* ``GcSweepDisposition`` enum stability and per-value ``.value`` strings.
* ``GcSweepCandidate.__post_init__`` field guards (empty/whitespace key, NaN /
  inf / negative / non-str first_seen_at, bool first_seen_at, negative / bool
  / non-int size_bytes).
* ``GcSweepState.__post_init__`` cursor guards (bool cursor, negative cursor,
  cursor > len(candidates), duplicate keys, retired_keys not subset,
  candidate_fingerprint mismatch).
* ``from_report`` validation (orphan count mismatch, non-UNREFERENCED records
  excluded, NaN observed_at).
* ``process_batch`` argument validation (NaN / ±inf / negative / bool for now,
  grace_seconds, max_items, destructive / authorized / recovery_ready, and
  lease_ttl_seconds with lease_authority).
* ``process_batch`` failure-mode boundaries (``reference_lookup`` raises in
  first OR second call, ``blob_exists`` raises, ``delete_blob`` raises without
  lease, ``delete_blob`` succeeds but blob remains, ``max_items`` ceiling).
* ``aprocess_batch`` async validation (destructive + lease_store=None raises,
  expected_revision non-negative, lease_ttl_seconds positive when lease_store
  present, NaN now, 0 max_items, bool flags).
* ``start_next_pass`` boundary (incomplete cursor raises, cursor=0 with no
  candidates is OK, cursor=len(candidates) resets).
* ``to_snapshot`` / ``from_snapshot`` negative cases (missing schema_version,
  wrong schema_version, missing fingerprint, non-list candidates /
  retired_keys / records, raw record with empty knowledge_ids, invalid
  disposition, etc.).
* Dataclass shape (frozen=True on GcSweepCandidate / GcSweepRecord /
  GcSweepBatchResult; slots=True on GcSweepState; GcSweepState NOT frozen).
* ``__all__`` exports.
* ``_sweep_record_from_d1_result`` exhaustive mapping (4 outcomes + detail=None
  fallback).

Each section pins one or more of these contracts with a small, sharp test that
fails immediately on a regression. The pattern follows the slice-29 / slice-30
/ slice-31 outer-guard-rail files in this campaign.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from oai2.knowledge.gc import (
    GcDryRunReport,
    GcObjectDisposition,
    GcObjectRecord,
)
from oai2.knowledge.gc_delete_d1_runtime import (
    D1DeleteOutcome,
    D1DeleteResult,
)
from oai2.knowledge.gc_lease import GcDeleteLeaseAuthority
from oai2.knowledge.sweep import (
    GcSweepBatchResult,
    GcSweepCandidate,
    GcSweepDisposition,
    GcSweepRecord,
    GcSweepState,
    _snapshot_fingerprint,
)
from oai2.knowledge.sweep import (
    __all__ as sweep_all,
)


def _refingerprint(snapshot: dict[str, object]) -> None:
    """Recompute ``snapshot_fingerprint`` after the payload was tampered with.

    The fingerprint is over the entire payload minus the ``snapshot_fingerprint``
    key itself. Snapshot tests that mutate one field need to keep the fingerprint
    in sync to exercise the field-specific guard rather than the fingerprint guard.
    """
    payload = dict(snapshot)
    payload.pop("snapshot_fingerprint", None)
    snapshot["snapshot_fingerprint"] = _snapshot_fingerprint(payload)


# --- Shared builders ---------------------------------------------------------


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


def _candidate(key: str = "oai2-blobs/a", first_seen_at: float = 10.0) -> GcSweepCandidate:
    return GcSweepCandidate(key=key, first_seen_at=first_seen_at, size_bytes=5)


def _state(*keys: str) -> GcSweepState:
    """Build a state directly from raw keys (bypasses ``from_report`` filtering)."""
    candidates = tuple(_candidate(key=k) for k in keys)
    return GcSweepState(candidates=candidates)


def _forbidden() -> Any:
    raise AssertionError("callback must not be invoked")


# ============================================================================
# Section 1: GcSweepDisposition enum stability
# ============================================================================


def test_sweep_disposition_enum_has_exactly_nine_members() -> None:
    """Closing the surface: exactly 9 distinct disposition outcomes.

    A refactor that adds a 10th (or removes one) without updating the planner
    would silently change the public-safe evidence contract.
    """
    assert len(set(GcSweepDisposition)) == 9


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (GcSweepDisposition.DEFERRED_GRACE, "deferred_grace"),
        (GcSweepDisposition.RE_REFERENCED, "re_referenced"),
        (GcSweepDisposition.DRY_RUN, "dry_run"),
        (GcSweepDisposition.UNAPPROVED, "unapproved"),
        (GcSweepDisposition.RECOVERY_BLOCKED, "recovery_blocked"),
        (GcSweepDisposition.LEASE_DENIED, "lease_denied"),
        (GcSweepDisposition.DELETED, "deleted"),
        (GcSweepDisposition.ALREADY_ABSENT, "already_absent"),
        (GcSweepDisposition.FAILED, "failed"),
    ],
)
def test_sweep_disposition_value_strings_are_stable(member: GcSweepDisposition, value: str) -> None:
    """Each disposition's ``.value`` string is part of the public-safe evidence surface."""
    assert member.value == value
    assert str(member) == value  # StrEnum: str(member) == .value


def test_sweep_disposition_lookup_by_value_constructs_member() -> None:
    """``GcSweepDisposition(value)`` reconstructs the named member — used by
    ``from_snapshot`` to deserialize persisted records. A rename would break
    snapshot round-trips for any already-published checkpoint.
    """
    for value in (
        "deferred_grace",
        "re_referenced",
        "dry_run",
        "unapproved",
        "recovery_blocked",
        "lease_denied",
        "deleted",
        "already_absent",
        "failed",
    ):
        assert GcSweepDisposition(value) is not None
        assert GcSweepDisposition(value).value == value


# ============================================================================
# Section 2: GcSweepCandidate.__post_init__ field guards
# ============================================================================


def test_candidate_with_zero_size_bytes_is_accepted() -> None:
    """``size_bytes=0`` is a non-negative integer and must be accepted."""
    candidate = GcSweepCandidate(key="oai2-blobs/zero", first_seen_at=10.0, size_bytes=0)
    assert candidate.size_bytes == 0


def test_candidate_with_default_size_bytes_is_none() -> None:
    """Default ``size_bytes=None`` is the documented unknown-size path."""
    candidate = GcSweepCandidate(key="oai2-blobs/unknown", first_seen_at=10.0)
    assert candidate.size_bytes is None


@pytest.mark.parametrize("bad_key", ["", "   ", " leading", "trailing "])
def test_candidate_with_empty_or_whitespace_key_raises(bad_key: str) -> None:
    """Empty / whitespace-only / unstripped keys are rejected — the planner
    uses normalized keys as the lookup boundary, so any unstripped variant
    would silently miss the R2 object.
    """
    with pytest.raises(ValueError, match="non-empty normalized string"):
        GcSweepCandidate(key=bad_key, first_seen_at=10.0)


def test_candidate_with_non_string_key_raises() -> None:
    """Non-string keys (int, bytes, None) are rejected."""
    with pytest.raises(ValueError, match="non-empty normalized string"):
        GcSweepCandidate(key=123, first_seen_at=10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", [-1.0, math.nan, math.inf, -math.inf])
def test_candidate_with_non_finite_or_negative_first_seen_at_raises(bad_value: float) -> None:
    """Non-finite or negative ``first_seen_at`` is rejected — the planner
    uses ``now - first_seen_at`` to compute age, and NaN / ±inf would
    propagate into the age comparison and silently produce DEFERRED_GRACE
    for every candidate.
    """
    with pytest.raises(ValueError, match="first_seen_at"):
        GcSweepCandidate(key="oai2-blobs/a", first_seen_at=bad_value)


def test_candidate_with_bool_first_seen_at_raises() -> None:
    """``bool`` is rejected because ``isinstance(True, int) is True`` — a
    refactor that switches to ``isinstance(x, int)`` would silently accept
    ``True`` (=1) as a first_seen_at timestamp.
    """
    with pytest.raises(ValueError, match="first_seen_at"):
        GcSweepCandidate(key="oai2-blobs/a", first_seen_at=True)


def test_candidate_with_negative_size_bytes_raises() -> None:
    """Negative ``size_bytes`` is rejected — the GC accounting assumes
    size_bytes ≥ 0 (zero is allowed).
    """
    with pytest.raises(ValueError, match="size_bytes"):
        GcSweepCandidate(key="oai2-blobs/a", first_seen_at=10.0, size_bytes=-1)


def test_candidate_with_bool_size_bytes_raises() -> None:
    """Bool ``size_bytes`` is rejected (same ``isinstance(True, int) is True``
    reason as ``first_seen_at``).
    """
    with pytest.raises(ValueError, match="size_bytes"):
        GcSweepCandidate(key="oai2-blobs/a", first_seen_at=10.0, size_bytes=True)


def test_candidate_with_non_int_size_bytes_raises() -> None:
    """Float or string ``size_bytes`` is rejected — the planner assumes
    integer byte counts (the R2 inventory uses 64-bit counts).
    """
    with pytest.raises(ValueError, match="size_bytes"):
        GcSweepCandidate(key="oai2-blobs/a", first_seen_at=10.0, size_bytes=1.5)  # type: ignore[arg-type]


def test_candidate_is_frozen() -> None:
    """``GcSweepCandidate`` is ``frozen=True`` — a refactor that drops the
    flag would silently allow mutating the candidate after the planner
    appended it to the candidates tuple (the candidate_fingerprint would
    then no longer match).
    """
    candidate = _candidate()
    with pytest.raises((AttributeError, Exception)):
        candidate.key = "oai2-blobs/other"  # type: ignore[misc]


# ============================================================================
# Section 3: GcSweepState.__post_init__ guards
# ============================================================================


def test_state_with_bool_cursor_raises() -> None:
    """Bool cursor is rejected because ``isinstance(True, int) is True``.

    The ``isinstance(self.cursor, bool) or not isinstance(self.cursor, int)``
    guard is the only thing that distinguishes bool from int — drop the
    bool check and ``True`` silently becomes cursor=1.
    """
    with pytest.raises(ValueError, match="cursor must be an integer"):
        GcSweepState(candidates=(_candidate(),), cursor=True)


def test_state_with_negative_cursor_raises() -> None:
    """Negative cursor is rejected — the planner walks ``[cursor, len)`` and
    a negative cursor would silently re-process the same candidates.
    """
    with pytest.raises(ValueError, match="cursor is outside candidate bounds"):
        GcSweepState(candidates=(_candidate(),), cursor=-1)


def test_state_with_cursor_past_end_raises() -> None:
    """Cursor > len(candidates) is rejected — the planner assumes
    ``cursor <= len(candidates)`` (the ``complete`` condition).
    """
    with pytest.raises(ValueError, match="cursor is outside candidate bounds"):
        GcSweepState(candidates=(_candidate(),), cursor=2)


def test_state_with_duplicate_keys_raises() -> None:
    """Duplicate candidate keys are rejected — the planner retires keys by
    adding them to ``retired_keys``, and a duplicate would silently process
    only the first occurrence.
    """
    with pytest.raises(ValueError, match="duplicate keys"):
        GcSweepState(
            candidates=(
                _candidate(key="oai2-blobs/dup"),
                _candidate(key="oai2-blobs/dup"),
            )
        )


def test_state_with_unknown_retired_keys_raises() -> None:
    """``retired_keys`` must be a subset of candidate keys — otherwise the
    retired set could grow without bound as new candidates arrive.
    """
    with pytest.raises(ValueError, match="retired_keys contain unknown candidates"):
        GcSweepState(
            candidates=(_candidate(key="oai2-blobs/a"),),
            retired_keys={"oai2-blobs/unknown"},
        )


def test_state_with_mismatched_candidate_fingerprint_raises() -> None:
    """A non-empty ``candidate_fingerprint`` that does not match the
    candidates tuple is rejected — this is the tamper guard for
    ``to_snapshot`` round-trips where the candidate list is replaced.
    """
    with pytest.raises(ValueError, match="candidate fingerprint does not match"):
        GcSweepState(
            candidates=(_candidate(key="oai2-blobs/a"),),
            candidate_fingerprint="not-the-real-fingerprint",
        )


def test_state_with_empty_candidate_fingerprint_auto_computes() -> None:
    """Empty ``candidate_fingerprint`` is auto-computed — the
    ``from_snapshot`` path always restores a fingerprint, but a
    freshly-built state without one is also valid.
    """
    state = GcSweepState(candidates=(_candidate(key="oai2-blobs/a"),))
    assert state.candidate_fingerprint  # non-empty


# ============================================================================
# Section 4: from_report validation
# ============================================================================


def test_from_report_filters_non_unreferenced_dispositions() -> None:
    """Only ``UNREFERENCED_CANDIDATE`` records become sweep candidates —
    REFERENCED_PRESENT / REFERENCED_MISSING / UNKNOWN_SIZE are excluded
    (the GC sweep only reconsiders the orphan candidate subset).
    """
    records = (
        GcObjectRecord(
            key="oai2-blobs/orphan",
            disposition=GcObjectDisposition.UNREFERENCED_CANDIDATE,
            size_bytes=5,
            uploaded_at=1.0,
            age_seconds=9.0,
        ),
        GcObjectRecord(
            key="oai2-blobs/referenced",
            disposition=GcObjectDisposition.REFERENCED_PRESENT,
            size_bytes=5,
            uploaded_at=1.0,
            age_seconds=9.0,
        ),
        GcObjectRecord(
            key="oai2-blobs/missing",
            disposition=GcObjectDisposition.REFERENCED_MISSING,
            size_bytes=5,
            uploaded_at=1.0,
            age_seconds=9.0,
        ),
    )
    report = GcDryRunReport(
        observed_at=10.0,
        pages_processed=1,
        records=records,
        referenced_present_count=1,
        referenced_missing_count=1,
        unreferenced_candidate_count=1,
        inventory_object_count=3,
        inventory_bytes_known=15,
        unreferenced_candidate_bytes_known=5,
        unknown_size_object_count=0,
        oldest_unreferenced_candidate_age_seconds=9.0,
    )

    state = GcSweepState.from_report(report)

    assert tuple(c.key for c in state.candidates) == ("oai2-blobs/orphan",)


def test_from_report_rejects_mismatched_orphan_count() -> None:
    """The report's ``unreferenced_candidate_count`` must match the count
    of ``UNREFERENCED_CANDIDATE`` records — the planner uses the count as
    a cross-check, and a mismatch indicates the dry-run report is
    inconsistent.
    """
    records = (
        GcObjectRecord(
            key="oai2-blobs/a",
            disposition=GcObjectDisposition.UNREFERENCED_CANDIDATE,
            size_bytes=5,
            uploaded_at=1.0,
            age_seconds=9.0,
        ),
    )
    report = GcDryRunReport(
        observed_at=10.0,
        pages_processed=1,
        records=records,
        referenced_present_count=0,
        referenced_missing_count=0,
        unreferenced_candidate_count=2,  # lies: 1 record but claim 2
        inventory_object_count=1,
        inventory_bytes_known=5,
        unreferenced_candidate_bytes_known=5,
        unknown_size_object_count=0,
        oldest_unreferenced_candidate_age_seconds=9.0,
    )

    with pytest.raises(ValueError, match="report orphan count does not match"):
        GcSweepState.from_report(report)


def test_from_report_rejects_non_finite_observed_at() -> None:
    """NaN / ±inf ``observed_at`` is rejected at the boundary — the planner
    uses ``observed_at`` as the candidate's ``first_seen_at`` baseline,
    and a non-finite value would propagate into the age comparison.
    """
    with pytest.raises(ValueError, match="observed_at"):
        GcSweepState.from_report(_report("oai2-blobs/a", observed_at=math.nan))


# ============================================================================
# Section 5: process_batch argument validation
# ============================================================================


@pytest.mark.parametrize(
    "bad_now",
    [math.nan, -math.inf, -1.0, -0.0001],
)
def test_process_batch_rejects_non_finite_or_negative_now(bad_now: float) -> None:
    """NaN / -inf / negative ``now`` are rejected — ``+inf`` is intentionally
    accepted (a far-future clock is not malformed; only negative / NaN
    values would produce bogus DEFERRED_GRACE for every candidate).
    """
    state = _state("oai2-blobs/a")
    with pytest.raises(ValueError, match="now must be a finite non-negative number"):
        state.process_batch(
            now=bad_now,
            grace_seconds=0.0,
            reference_lookup=lambda _key: (),
            blob_exists=lambda _key: True,
            delete_blob=lambda _key: None,
        )


@pytest.mark.parametrize("bad_grace", [-1.0, math.nan, -math.inf])
def test_process_batch_rejects_non_finite_or_negative_grace_seconds(bad_grace: float) -> None:
    """Negative / NaN ``grace_seconds`` are rejected — ``0`` and ``+inf``
    are intentionally accepted (zero means "delete immediately"; +inf
    means "never delete"; both are valid operator choices).
    """
    state = _state("oai2-blobs/a")
    with pytest.raises(ValueError, match="grace_seconds must be a finite non-negative number"):
        state.process_batch(
            now=100.0,
            grace_seconds=bad_grace,
            reference_lookup=lambda _key: (),
            blob_exists=lambda _key: True,
            delete_blob=lambda _key: None,
        )


@pytest.mark.parametrize("bad_max_items", [0, -1, 1.5, True])
def test_process_batch_rejects_non_positive_int_max_items(bad_max_items: object) -> None:
    """``max_items`` must be a positive integer — zero would mean "process
    nothing", breaking the planner's contract that ``max_items >= 1`` means
    "process at least one candidate".
    """
    state = _state("oai2-blobs/a")
    with pytest.raises(ValueError, match="max_items must be a positive integer"):
        state.process_batch(
            now=100.0,
            grace_seconds=0.0,
            max_items=bad_max_items,  # type: ignore[arg-type]
            reference_lookup=lambda _key: (),
            blob_exists=lambda _key: True,
            delete_blob=lambda _key: None,
        )


@pytest.mark.parametrize(
    ("attr_name", "bad_value"),
    [
        ("destructive", 1),
        ("destructive", "yes"),
        ("authorized", 2),
        ("authorized", "no"),
        ("recovery_ready", 0),
        ("recovery_ready", "true"),
    ],
)
def test_process_batch_rejects_non_bool_flags(attr_name: str, bad_value: object) -> None:
    """Bool flags must be ``True`` / ``False`` exactly — ``isinstance(True, int)``
    is ``True`` so a refactor that uses ``isinstance(x, int)`` would silently
    accept ``1`` as ``destructive=True``.
    """
    state = _state("oai2-blobs/a")
    kwargs = {
        "now": 100.0,
        "grace_seconds": 0.0,
        "reference_lookup": lambda _key: (),
        "blob_exists": lambda _key: True,
        "delete_blob": lambda _key: None,
        attr_name: bad_value,
    }
    with pytest.raises(ValueError, match=f"{attr_name} must be a boolean"):
        state.process_batch(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_ttl",
    [-1.0, 0.0, math.nan, -math.inf],
)
def test_process_batch_rejects_non_positive_lease_ttl_when_lease_authority_set(
    bad_ttl: float,
) -> None:
    """``lease_ttl_seconds`` must be positive when ``lease_authority`` is
    provided — a zero or negative TTL would let the lease expire
    immediately, silently disabling the writer-fence.
    """
    state = _state("oai2-blobs/a")
    with pytest.raises(ValueError, match="lease_ttl_seconds must be"):
        state.process_batch(
            now=100.0,
            grace_seconds=0.0,
            destructive=True,
            authorized=True,
            recovery_ready=True,
            reference_lookup=lambda _key: (),
            blob_exists=lambda _key: True,
            delete_blob=lambda _key: None,
            lease_authority=GcDeleteLeaseAuthority(),
            lease_ttl_seconds=bad_ttl,
        )


# ============================================================================
# Section 6: process_batch failure-mode boundaries
# ============================================================================


def test_process_batch_first_reference_lookup_failure_keeps_cursor_and_records_failed() -> None:
    """If the first ``reference_lookup`` raises, the candidate is recorded
    as FAILED with detail="authoritative reference lookup failed" and the
    cursor does NOT advance — the planner can retry the same candidate
    on a later call.
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))

    def failing_lookup(_key: str) -> tuple[str, ...]:
        raise RuntimeError("lookup outage")

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=failing_lookup,
        blob_exists=lambda _key: True,
        delete_blob=lambda _key: None,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert result.records[0].detail == "authoritative reference lookup failed"
    assert state.cursor == 0  # did NOT advance
    assert result.complete is False


def test_process_batch_second_reference_lookup_failure_keeps_cursor_and_records_failed() -> None:
    """If the second (final, pre-delete) ``reference_lookup`` raises — after
    the destructive gates pass — the candidate is recorded as FAILED with
    detail="final authoritative reference lookup failed" and the cursor
    does NOT advance.
    """
    blobs = {"oai2-blobs/a"}
    state = GcSweepState.from_report(_report("oai2-blobs/a"))
    lookup_count = {"n": 0}

    def race_lookup(_key: str) -> tuple[str, ...]:
        lookup_count["n"] += 1
        return () if lookup_count["n"] == 1 else ("ko-late",)

    # The second lookup returns a reference — but we want it to RAISE.
    # Re-design: second call raises, first call returns nothing.
    def race_lookup_raises(_key: str) -> tuple[str, ...]:
        lookup_count["n"] += 1
        if lookup_count["n"] == 2:
            raise RuntimeError("lookup outage")
        return ()

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=race_lookup_raises,
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert result.records[0].detail == "final authoritative reference lookup failed"
    assert state.cursor == 0  # did NOT advance
    assert blobs == {"oai2-blobs/a"}  # untouched


def test_process_batch_blob_exists_failure_keeps_cursor_and_records_failed() -> None:
    """If ``blob_exists`` raises on the post-leases existence check, the
    candidate is recorded as FAILED with detail="blob existence check failed"
    and the cursor does NOT advance.
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    def failing_exists(_key: str) -> bool:
        raise RuntimeError("R2 outage")

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=failing_exists,
        delete_blob=lambda _key: None,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert result.records[0].detail == "blob existence check failed"
    assert state.cursor == 0


def test_process_batch_delete_blob_remains_records_failed_and_keeps_cursor() -> None:
    """If ``delete_blob`` returns successfully but the post-delete
    ``blob_exists`` re-check confirms the blob is present, the candidate
    is recorded as FAILED with detail="blob remained present after deletion"
    and the cursor does NOT advance — the planner must NOT record a
    confirmed deletion that did not actually take.
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    # blob_exists always returns True (even after delete_blob "succeeds")
    # — simulates the "R2 delete acknowledged but the object remained" race.
    def persistent_exists(_key: str) -> bool:
        return True

    deleted: list[str] = []

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=persistent_exists,
        delete_blob=deleted.append,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert result.records[0].detail == "blob remained present after deletion"
    assert state.cursor == 0
    assert deleted == ["oai2-blobs/a"]  # delete_blob WAS called


def test_process_batch_max_items_ceiling_advances_exactly_by_max_items() -> None:
    """``max_items=N`` processes exactly AT most ``N`` candidates per call —
    the planner never exceeds the bounded batch size even when more
    candidates remain.
    """
    state = GcSweepState.from_report(
        _report("oai2-blobs/a", "oai2-blobs/b", "oai2-blobs/c", "oai2-blobs/d")
    )

    result = state.process_batch(
        now=100.0,
        grace_seconds=100.0,  # force DEFERRED_GRACE for every candidate
        max_items=2,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=lambda _key: None,
    )

    assert len(result.records) == 2
    assert all(r.disposition is GcSweepDisposition.DEFERRED_GRACE for r in result.records)
    assert state.cursor == 2
    assert result.complete is False
    assert result.next_cursor == 2


def test_process_batch_retires_already_seen_key_on_referenced_disposition() -> None:
    """A candidate whose ``reference_lookup`` returns knowledge_ids is
    retired (added to ``retired_keys``) so the next pass skips it even
    after ``start_next_pass()`` resets the cursor.

    This pins the boundary between a single pass and a multi-pass sweep:
    the retired set is the planner's "permanent skip" memory.
    """
    state = GcSweepState.from_report(_report("oai2-blobs/shared"))

    state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: ("ko-live",),
        blob_exists=lambda _key: True,
        delete_blob=lambda _key: None,
    )

    assert "oai2-blobs/shared" in state.retired_keys

    state.start_next_pass()
    second = state.process_batch(
        now=200.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=lambda _key: None,
    )

    assert second.records == ()
    assert second.complete is True


# ============================================================================
# Section 7: aprocess_batch async validation
# ============================================================================


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


@pytest.mark.asyncio
async def test_aprocess_batch_destructive_without_lease_store_raises() -> None:
    """``destructive=True`` with ``lease_store=None`` is rejected — the
    async path delegates the entire destructive flow to
    ``delete_candidate_with_d1_lease`` which requires a lease_store.
    Failing closed here prevents the planner from silently bypassing
    the D1-authoritative lease.
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    with pytest.raises(
        ValueError, match="destructive async sweeps require an authoritative D1 lease_store"
    ):
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


@pytest.mark.asyncio
async def test_aprocess_batch_non_destructive_without_lease_store_is_ok() -> None:
    """The opposite ``only`` is also pinned: non-destructive async sweeps
    do NOT require a lease_store (the destructive path is the only path
    that touches the D1 lease boundary).
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=False,
        reference_lookup=_async_lookup([]),
        blob_exists=_async_exists(False),
        delete_blob=_async_delete(None),
        lease_store=None,
    )

    assert result.records[0].disposition is GcSweepDisposition.DRY_RUN


@pytest.mark.asyncio
async def test_aprocess_batch_rejects_negative_expected_revision() -> None:
    """``expected_revision`` must be a non-negative integer — a negative
    revision would silently bypass the D1 lease schema's mismatch check.
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    with pytest.raises(ValueError, match="expected_revision"):
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
            expected_revision=-1,
        )


@pytest.mark.asyncio
async def test_aprocess_batch_rejects_bool_expected_revision() -> None:
    """Bool ``expected_revision`` is rejected (same ``isinstance(True, int)``
    reason as ``first_seen_at`` in ``GcSweepCandidate``).
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    with pytest.raises(ValueError, match="expected_revision"):
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
            expected_revision=True,
        )


@pytest.mark.asyncio
async def test_aprocess_batch_first_reference_lookup_failure_records_failed() -> None:
    """The async path translates the first ``reference_lookup`` ``Exception``
    into FAILED with detail="authoritative reference lookup failed" and the
    cursor does NOT advance — same contract as the sync path.

    Uses ``destructive=False`` so the planner never needs a lease_store to
    reach the lookup-failure branch (the failure branch fires BEFORE the
    destructive gates).
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))

    async def failing_lookup(_key: str) -> tuple[str, ...]:
        raise RuntimeError("lookup outage")

    result = await state.aprocess_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=False,
        reference_lookup=failing_lookup,
        blob_exists=_async_exists(False),
        delete_blob=_async_delete(None),
        lease_store=None,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert result.records[0].detail == "authoritative reference lookup failed"
    assert state.cursor == 0
    assert result.complete is False


# ============================================================================
# Section 8: start_next_pass boundary
# ============================================================================


def test_start_next_pass_raises_when_cursor_not_at_end() -> None:
    """``start_next_pass`` requires the current pass to be COMPLETED
    (cursor == len(candidates)) — calling it mid-pass would silently
    re-process the same candidates from cursor=0.
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))

    state.process_batch(
        now=100.0,
        grace_seconds=100.0,
        max_items=1,  # process only 1, leaving cursor=1 < len=2
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=lambda _key: None,
    )
    assert state.cursor == 1  # not yet complete

    with pytest.raises(RuntimeError, match="cannot start next pass before current pass completes"):
        state.start_next_pass()


def test_start_next_pass_on_completed_state_resets_cursor_to_zero() -> None:
    """After all candidates are processed, ``start_next_pass`` resets the
    cursor to 0 so the next pass re-evaluates every candidate.
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = state.process_batch(
        now=100.0,
        grace_seconds=100.0,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=lambda _key: None,
    )
    assert result.complete is True
    assert state.cursor == 1

    state.start_next_pass()
    assert state.cursor == 0


def test_start_next_pass_on_empty_state_is_a_no_op() -> None:
    """An empty state (cursor=0, len(candidates)=0) is already complete —
    ``start_next_pass`` resets to 0 without raising.
    """
    state = GcSweepState(candidates=())
    assert state.cursor == 0

    state.start_next_pass()
    assert state.cursor == 0


# ============================================================================
# Section 9: Snapshot boundary contracts
# ============================================================================


def test_to_snapshot_includes_schema_version_one() -> None:
    """``schema_version=1`` is part of the public-safe wire contract — a
    refactor that bumps to v2 without a migration path would break every
    already-published checkpoint.
    """
    state = _state("oai2-blobs/a")
    snapshot = state.to_snapshot()
    assert snapshot["schema_version"] == 1


def test_to_snapshot_records_sorted_retired_keys() -> None:
    """``retired_keys`` is serialized as a SORTED list — the sort is
    part of the snapshot fingerprint contract (sort_keys=True in
    ``_snapshot_fingerprint``), so any other order would invalidate
    the fingerprint.
    """
    state = GcSweepState(
        candidates=(
            _candidate(key="oai2-blobs/a"),
            _candidate(key="oai2-blobs/b"),
            _candidate(key="oai2-blobs/c"),
        ),
        retired_keys={"oai2-blobs/c", "oai2-blobs/a", "oai2-blobs/b"},
    )
    snapshot = state.to_snapshot()
    assert snapshot["retired_keys"] == [
        "oai2-blobs/a",
        "oai2-blobs/b",
        "oai2-blobs/c",
    ]


def test_from_snapshot_rejects_missing_schema_version() -> None:
    """``schema_version`` is required — without it, the fingerprint
    format is unknown and cannot be revalidated.
    """
    state = _state("oai2-blobs/a")
    snapshot = state.to_snapshot()
    snapshot.pop("schema_version")

    with pytest.raises(ValueError, match="unsupported or missing sweep snapshot schema_version"):
        GcSweepState.from_snapshot(snapshot)


def test_from_snapshot_rejects_wrong_schema_version() -> None:
    """``schema_version != 1`` is rejected — a v2 checkpoint cannot be
    read by a v1 planner without an explicit migration.
    """
    state = _state("oai2-blobs/a")
    snapshot = state.to_snapshot()
    snapshot["schema_version"] = 2

    with pytest.raises(ValueError, match="unsupported or missing sweep snapshot schema_version"):
        GcSweepState.from_snapshot(snapshot)


def test_from_snapshot_rejects_missing_snapshot_fingerprint() -> None:
    """``snapshot_fingerprint`` is required — without it, the payload
    cannot be integrity-checked.
    """
    state = _state("oai2-blobs/a")
    snapshot = state.to_snapshot()
    snapshot.pop("snapshot_fingerprint")

    with pytest.raises(ValueError, match="snapshot fingerprint is required"):
        GcSweepState.from_snapshot(snapshot)


def test_from_snapshot_rejects_non_list_candidates() -> None:
    """``candidates`` must be a list — any other mapping shape (dict /
    tuple / string) would silently confuse the per-candidate loop.
    """
    state = _state("oai2-blobs/a")
    snapshot = state.to_snapshot()
    snapshot["candidates"] = "not-a-list"
    _refingerprint(snapshot)

    with pytest.raises(ValueError, match="snapshot candidates must be a list"):
        GcSweepState.from_snapshot(snapshot)


def test_from_snapshot_rejects_retired_keys_with_empty_string() -> None:
    """``retired_keys`` must contain only non-empty strings — an empty
    string would silently collide with the candidate-key normalization
    that rejects empty keys.
    """
    state = _state("oai2-blobs/a")
    snapshot = state.to_snapshot()
    snapshot["retired_keys"] = [""]
    _refingerprint(snapshot)

    with pytest.raises(ValueError, match="snapshot retired_keys must be a string list"):
        GcSweepState.from_snapshot(snapshot)


def test_from_snapshot_rejects_record_with_empty_knowledge_id() -> None:
    """``record.knowledge_ids`` must contain only non-empty strings — an
    empty knowledge_id would falsely indicate a reference exists.
    """
    state = _state("oai2-blobs/a")
    snapshot = state.to_snapshot()
    snapshot["records"] = [
        {
            "key": "oai2-blobs/a",
            "disposition": "re_referenced",
            "processed_at": 10.0,
            "knowledge_ids": [""],
            "detail": "",
        }
    ]
    _refingerprint(snapshot)

    with pytest.raises(ValueError, match="snapshot record knowledge_ids are invalid"):
        GcSweepState.from_snapshot(snapshot)


def test_from_snapshot_rejects_unknown_disposition_string() -> None:
    """A disposition string that is not a ``GcSweepDisposition`` member
    is rejected — silently passing an unknown disposition would corrupt
    the public-safe evidence surface.
    """
    state = _state("oai2-blobs/a")
    snapshot = state.to_snapshot()
    snapshot["records"] = [
        {
            "key": "oai2-blobs/a",
            "disposition": "not_a_real_disposition",
            "processed_at": 10.0,
            "knowledge_ids": [],
            "detail": "",
        }
    ]
    _refingerprint(snapshot)

    with pytest.raises(ValueError, match="snapshot record disposition is invalid"):
        GcSweepState.from_snapshot(snapshot)


def test_snapshot_round_trip_preserves_records_and_retired_keys() -> None:
    """A full ``to_snapshot`` → ``from_snapshot`` round-trip preserves
    every field: candidates, cursor, records (with disposition /
    processed_at / knowledge_ids / detail), retired_keys, and the
    candidate_fingerprint. This pins the snapshot as a lossless
    resumable checkpoint.
    """
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))
    state.process_batch(
        now=20.0,
        grace_seconds=100.0,
        max_items=1,
        reference_lookup=lambda _key: ("ko-1", "ko-2"),
        blob_exists=lambda _key: True,
        delete_blob=lambda _key: None,
    )

    # The RE_REFERENCED path retired "oai2-blobs/a". Manually retire the
    # second candidate ("oai2-blobs/b") — it IS a candidate key, so the
    # ``retired_keys ⊆ key_set`` guard accepts it.
    state.retired_keys.add("oai2-blobs/b")

    snapshot = state.to_snapshot()
    restored = GcSweepState.from_snapshot(snapshot)

    assert restored.cursor == state.cursor
    assert tuple(c.key for c in restored.candidates) == tuple(c.key for c in state.candidates)
    assert restored.retired_keys == state.retired_keys
    assert len(restored.records) == len(state.records)
    for got, want in zip(restored.records, state.records, strict=True):
        assert got.key == want.key
        assert got.disposition == want.disposition
        assert got.processed_at == want.processed_at
        assert got.knowledge_ids == want.knowledge_ids


# ============================================================================
# Section 10: Dataclass shape + __all__ exports
# ============================================================================


def test_sweep_state_is_not_frozen_but_has_slots() -> None:
    """``GcSweepState`` is intentionally NOT ``frozen=True`` — the planner
    mutates ``cursor``, ``records``, and ``retired_keys`` during
    ``process_batch``. It IS ``slots=True`` (rejects ``__dict__``), so
    a refactor that drops the slots flag would silently allow
    attribute injection outside declared fields.
    """
    state = _state("oai2-blobs/a")
    # Not frozen: cursor can be reassigned.
    state.cursor = 0
    # Slots: __dict__ is not present.
    assert "__dict__" not in dir(state)


def test_sweep_candidate_record_and_result_are_frozen() -> None:
    """``GcSweepCandidate`` / ``GcSweepRecord`` / ``GcSweepBatchResult``
    are all ``frozen=True`` — they are the public-safe evidence surface,
    so they must be immutable after construction.
    """
    candidate = _candidate()
    record = GcSweepRecord(
        key="oai2-blobs/a",
        disposition=GcSweepDisposition.DRY_RUN,
        processed_at=10.0,
    )
    result = GcSweepBatchResult(records=(record,), next_cursor=0, complete=False)

    # Each frozen class has its own field; mutate that specific field to
    # prove the frozen flag is set on every class.
    with pytest.raises((AttributeError, Exception)):
        candidate.key = "oai2-blobs/other"  # type: ignore[misc]
    with pytest.raises((AttributeError, Exception)):
        record.key = "oai2-blobs/other"  # type: ignore[misc]
    with pytest.raises((AttributeError, Exception)):
        result.records = ()  # type: ignore[misc]


def test_sweep_module_exports_all_five_public_names() -> None:
    """``__all__`` lists exactly the 5 public classes — a refactor that
    adds a public name without updating ``__all__`` (or removes one
    without bumping a deprecation) would silently break
    ``from oai2.knowledge.sweep import *`` callers.
    """
    assert set(sweep_all) == {
        "GcSweepDisposition",
        "GcSweepCandidate",
        "GcSweepRecord",
        "GcSweepBatchResult",
        "GcSweepState",
    }


# ============================================================================
# Section 11: _sweep_record_from_d1_result exhaustive mapping
# ============================================================================


def test_sweep_record_from_d1_result_maps_deleted_outcome() -> None:
    """``D1DeleteOutcome.DELETED`` → ``GcSweepDisposition.DELETED``."""
    from oai2.knowledge.sweep import _sweep_record_from_d1_result

    result = D1DeleteResult(
        key="oai2-blobs/a",
        outcome=D1DeleteOutcome.DELETED,
        token="t",
        detail="R2 deletion confirmed",
    )
    record = _sweep_record_from_d1_result(candidate=_candidate(), now=10.0, result=result)

    assert record.disposition is GcSweepDisposition.DELETED
    assert record.detail == "R2 deletion confirmed"


def test_sweep_record_from_d1_result_maps_already_absent_outcome() -> None:
    """``D1DeleteOutcome.ALREADY_ABSENT`` → ``GcSweepDisposition.ALREADY_ABSENT``."""
    from oai2.knowledge.sweep import _sweep_record_from_d1_result

    result = D1DeleteResult(
        key="oai2-blobs/a",
        outcome=D1DeleteOutcome.ALREADY_ABSENT,
        token="t",
        detail="R2 already absent",
    )
    record = _sweep_record_from_d1_result(candidate=_candidate(), now=10.0, result=result)

    assert record.disposition is GcSweepDisposition.ALREADY_ABSENT
    assert record.detail == "R2 already absent"


def test_sweep_record_from_d1_result_maps_lease_denied_outcome() -> None:
    """``D1DeleteOutcome.LEASE_DENIED`` → ``GcSweepDisposition.LEASE_DENIED``."""
    from oai2.knowledge.sweep import _sweep_record_from_d1_result

    result = D1DeleteResult(
        key="oai2-blobs/a",
        outcome=D1DeleteOutcome.LEASE_DENIED,
        token="t",
        detail="R2 lease denied",
    )
    record = _sweep_record_from_d1_result(candidate=_candidate(), now=10.0, result=result)

    assert record.disposition is GcSweepDisposition.LEASE_DENIED
    assert record.detail == "R2 lease denied"


def test_sweep_record_from_d1_result_maps_failed_outcome() -> None:
    """``D1DeleteOutcome.FAILED`` → ``GcSweepDisposition.FAILED``."""
    from oai2.knowledge.sweep import _sweep_record_from_d1_result

    result = D1DeleteResult(
        key="oai2-blobs/a",
        outcome=D1DeleteOutcome.FAILED,
        token="t",
        detail="R2 delete failed",
    )
    record = _sweep_record_from_d1_result(candidate=_candidate(), now=10.0, result=result)

    assert record.disposition is GcSweepDisposition.FAILED
    assert record.detail == "R2 delete failed"


def test_sweep_record_from_d1_result_falls_back_to_default_detail() -> None:
    """``result.detail=None`` (or empty) → ``"d1 delete boundary returned no detail"`` —
    the planner must always produce a non-empty detail string for the
    public-safe evidence surface.
    """
    from oai2.knowledge.sweep import _sweep_record_from_d1_result

    result = D1DeleteResult(
        key="oai2-blobs/a",
        outcome=D1DeleteOutcome.DELETED,
        token="t",
        detail="",
    )
    record = _sweep_record_from_d1_result(candidate=_candidate(), now=10.0, result=result)

    assert record.detail == "d1 delete boundary returned no detail"
