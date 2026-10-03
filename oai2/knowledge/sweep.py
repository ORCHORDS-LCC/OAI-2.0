"""Conservative, resumable sweep planning for confirmed R2 orphan candidates.

This module is safe-by-default: dry-run is the default, destructive execution
requires explicit authorization and recovery readiness, and every candidate is
rechecked against authoritative metadata immediately before deletion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from .gc import GcDryRunReport, GcObjectDisposition
from .gc_delete_d1_runtime import (
    D1DeleteOutcome,
    D1DeleteResult,
    delete_candidate_with_d1_lease,
)
from .gc_lease import GcDeleteLeaseAuthority, GcDeleteLeaseDecision
from .gc_lease_d1_runtime import D1GcLeaseStore


class GcSweepDisposition(StrEnum):
    """Outcome for one sweep candidate in one processing pass."""

    DEFERRED_GRACE = "deferred_grace"
    RE_REFERENCED = "re_referenced"
    DRY_RUN = "dry_run"
    UNAPPROVED = "unapproved"
    RECOVERY_BLOCKED = "recovery_blocked"
    LEASE_DENIED = "lease_denied"
    DELETED = "deleted"
    ALREADY_ABSENT = "already_absent"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class GcSweepCandidate:
    """One liveness-report orphan candidate eligible for later recheck."""

    key: str
    first_seen_at: float
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key or self.key != self.key.strip():
            raise ValueError("candidate key must be a non-empty normalized string")
        _require_non_negative_number(self.first_seen_at, "first_seen_at")
        if self.size_bytes is not None:
            _require_non_negative_int(self.size_bytes, "size_bytes")


@dataclass(slots=True, frozen=True)
class GcSweepRecord:
    """Public-safe evidence for one candidate processing decision."""

    key: str
    disposition: GcSweepDisposition
    processed_at: float
    knowledge_ids: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(slots=True, frozen=True)
class GcSweepBatchResult:
    """Result for one bounded processing call."""

    records: tuple[GcSweepRecord, ...]
    next_cursor: int | None
    complete: bool


@dataclass(slots=True)
class GcSweepState:
    """Resumable state for conservative orphan recheck and optional deletion."""

    candidates: tuple[GcSweepCandidate, ...]
    cursor: int = 0
    records: list[GcSweepRecord] = field(default_factory=list)
    retired_keys: set[str] = field(default_factory=set)
    candidate_fingerprint: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.cursor, bool) or not isinstance(self.cursor, int):
            raise ValueError("cursor must be an integer")
        if self.cursor < 0 or self.cursor > len(self.candidates):
            raise ValueError("cursor is outside candidate bounds")
        keys = [candidate.key for candidate in self.candidates]
        key_set = set(keys)
        if len(key_set) != len(keys):
            raise ValueError("sweep candidates contain duplicate keys")
        if not self.retired_keys <= key_set:
            raise ValueError("retired_keys contain unknown candidates")
        expected = _candidate_fingerprint(self.candidates)
        if not self.candidate_fingerprint:
            self.candidate_fingerprint = expected
        elif self.candidate_fingerprint != expected:
            raise ValueError("candidate fingerprint does not match candidates")

    @classmethod
    def from_report(
        cls, report: GcDryRunReport, *, prior: GcSweepState | None = None
    ) -> Self:
        """Create sweep state from a completed, non-destructive dry-run report.

        ``prior`` carries the grace clock across runs, and without it the grace
        window is unsatisfiable.

        Measured on the previous source, every candidate was stamped
        ``first_seen_at = report.observed_at``. A sweep that re-runs its
        dry-run each cycle -- the natural shape for a scheduled sweep -- then
        restamps the clock every time, so ``age = now - first_seen_at`` is
        always ~0 and every candidate is ``deferred_grace`` forever:

            cycle 1 @   1000  first_seen=1000  -> deferred_grace
            cycle 2 @   5000  first_seen=5000  -> deferred_grace
            cycle 3 @  50000  first_seen=50000 -> deferred_grace

        REQ-GC-021 holds vacuously and AC-GC-023 ("a confirmed old orphan is
        deleted exactly once") is unreachable on that path. Passing the
        previous state as ``prior`` is what makes the window mean elapsed
        time rather than "time since this process started".

        Two rules govern what carries forward:

        * A key in ``prior.retired_keys`` does **not** carry its age. It was
          found referenced, so that period of unreferencedness is over; if it
          is unreferenced again it must earn a fresh window, because the row
          that came and went may have been a different one.
        * A carried ``first_seen_at`` is clamped to ``report.observed_at``.
          A clock that moved backwards must not manufacture age, and clamping
          fails safe: the candidate simply defers a little longer.
        """
        _require_non_negative_number(report.observed_at, "report observed_at")
        carried: dict[str, float] = {}
        if prior is not None:
            for candidate in prior.candidates:
                if candidate.key in prior.retired_keys:
                    continue
                carried[candidate.key] = min(
                    candidate.first_seen_at, report.observed_at
                )
        candidates = tuple(
            GcSweepCandidate(
                key=record.key,
                first_seen_at=carried.get(record.key, report.observed_at),
                size_bytes=record.size_bytes,
            )
            for record in report.records
            if record.disposition is GcObjectDisposition.UNREFERENCED_CANDIDATE
        )
        if len(candidates) != report.unreferenced_candidate_count:
            raise ValueError("report orphan count does not match candidate records")
        return cls(candidates=candidates)

    def process_batch(
        self,
        *,
        now: float,
        grace_seconds: float,
        reference_lookup: Callable[[str], Iterable[str]],
        blob_exists: Callable[[str], bool],
        delete_blob: Callable[[str], None],
        destructive: bool = False,
        authorized: bool = False,
        recovery_ready: bool = False,
        max_items: int = 100,
        lease_authority: GcDeleteLeaseAuthority | None = None,
        lease_ttl_seconds: float = 60.0,
    ) -> GcSweepBatchResult:
        """Process a bounded number of candidates conservatively.

        A failed dependency operation leaves the cursor on the failing
        candidate so a later call can retry it. Non-failing decisions advance
        the cursor and are retained as public-safe evidence records.

        When ``lease_authority`` is provided the destructive path acquires a
        D1-authoritative lease immediately before ``delete_blob`` and either
        finalizes it on a confirmed absence or records a retryable failure on
        the lease when the delete does not confirm. With ``lease_authority``
        ``None`` the destructive path is unchanged.
        """
        _require_non_negative_number(now, "now")
        _require_non_negative_number(grace_seconds, "grace_seconds")
        _require_positive_int(max_items, "max_items")
        _require_bool(destructive, "destructive")
        _require_bool(authorized, "authorized")
        _require_bool(recovery_ready, "recovery_ready")
        if lease_authority is not None:
            _require_positive_number(lease_ttl_seconds, "lease_ttl_seconds")

        emitted: list[GcSweepRecord] = []
        attempted = 0
        while self.cursor < len(self.candidates) and attempted < max_items:
            candidate = self.candidates[self.cursor]
            if candidate.key in self.retired_keys:
                self.cursor += 1
                continue
            attempted += 1
            age = max(0.0, float(now) - candidate.first_seen_at)

            if age < float(grace_seconds):
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.DEFERRED_GRACE,
                    processed_at=float(now),
                    detail="grace period has not elapsed",
                )
                self._record_and_advance(record, emitted)
                continue

            try:
                knowledge_ids = tuple(
                    sorted({str(value) for value in reference_lookup(candidate.key)})
                )
            except Exception:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.FAILED,
                    processed_at=float(now),
                    detail="authoritative reference lookup failed",
                )
                self._record_without_advance(record, emitted)
                break

            if knowledge_ids:
                self.retired_keys.add(candidate.key)
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.RE_REFERENCED,
                    processed_at=float(now),
                    knowledge_ids=knowledge_ids,
                    detail="authoritative reference exists",
                )
                self._record_and_advance(record, emitted)
                continue

            if not destructive:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.DRY_RUN,
                    processed_at=float(now),
                    detail="destructive mode is disabled",
                )
                self._record_and_advance(record, emitted)
                continue

            if not authorized:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.UNAPPROVED,
                    processed_at=float(now),
                    detail="destructive authorization is missing",
                )
                self._record_and_advance(record, emitted)
                continue

            if not recovery_ready:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.RECOVERY_BLOCKED,
                    processed_at=float(now),
                    detail="recovery prerequisite is not satisfied",
                )
                self._record_and_advance(record, emitted)
                continue

            try:
                exists_before = _require_bool_result(
                    blob_exists(candidate.key), "blob_exists"
                )
            except Exception:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.FAILED,
                    processed_at=float(now),
                    detail="blob existence check failed",
                )
                self._record_without_advance(record, emitted)
                break

            if not exists_before:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.ALREADY_ABSENT,
                    processed_at=float(now),
                    detail="blob was already absent",
                )
                self._record_and_advance(record, emitted)
                continue

            try:
                final_knowledge_ids = tuple(
                    sorted({str(value) for value in reference_lookup(candidate.key)})
                )
            except Exception:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.FAILED,
                    processed_at=float(now),
                    detail="final authoritative reference lookup failed",
                )
                self._record_without_advance(record, emitted)
                break

            if final_knowledge_ids:
                self.retired_keys.add(candidate.key)
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.RE_REFERENCED,
                    processed_at=float(now),
                    knowledge_ids=final_knowledge_ids,
                    detail="authoritative reference appeared before deletion",
                )
                self._record_and_advance(record, emitted)
                continue

            lease_token: str | None = None
            if lease_authority is not None:
                lease_revision = lease_authority.revision
                lease_token = f"gc-sweep:{candidate.key}:{lease_revision}"
                acquire = lease_authority.acquire_delete_lease(
                    key=candidate.key,
                    token=lease_token,
                    owner="gc-sweep",
                    now=float(now),
                    ttl_seconds=float(lease_ttl_seconds),
                    expected_revision=lease_revision,
                )
                if acquire.decision is not GcDeleteLeaseDecision.ACQUIRED:
                    record = GcSweepRecord(
                        key=candidate.key,
                        disposition=GcSweepDisposition.LEASE_DENIED,
                        processed_at=float(now),
                        knowledge_ids=acquire.knowledge_ids,
                        detail=(
                            "d1-authoritative lease denied: "
                            f"{acquire.decision.value}"
                        ),
                    )
                    self._record_and_advance(record, emitted)
                    continue

            if lease_authority is not None and lease_token is not None:
                if not lease_authority.validate_delete_lease(
                    candidate.key, lease_token, now=float(now)
                ):
                    record = GcSweepRecord(
                        key=candidate.key,
                        disposition=GcSweepDisposition.LEASE_DENIED,
                        processed_at=float(now),
                        detail="d1-authoritative lease invalid before deletion",
                    )
                    self._record_without_advance(record, emitted)
                    break

            try:
                delete_blob(candidate.key)
                still_exists = _require_bool_result(
                    blob_exists(candidate.key), "blob_exists"
                )
            except Exception:
                _record_lease_failure(lease_authority, candidate.key, lease_token, float(now))
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.FAILED,
                    processed_at=float(now),
                    detail="blob deletion failed",
                )
                self._record_without_advance(record, emitted)
                break

            if still_exists:
                _record_lease_failure(lease_authority, candidate.key, lease_token, float(now))
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.FAILED,
                    processed_at=float(now),
                    detail="blob remained present after deletion",
                )
                self._record_without_advance(record, emitted)
                break

            _finalize_lease(lease_authority, candidate.key, lease_token, float(now))
            record = GcSweepRecord(
                key=candidate.key,
                disposition=GcSweepDisposition.DELETED,
                processed_at=float(now),
                detail="blob deletion confirmed",
            )
            self._record_and_advance(record, emitted)

        complete = self.cursor >= len(self.candidates)
        return GcSweepBatchResult(
            records=tuple(emitted),
            next_cursor=None if complete else self.cursor,
            complete=complete,
        )

    async def aprocess_batch(
        self,
        *,
        now: float,
        grace_seconds: float,
        reference_lookup: Callable[[str], Awaitable[Iterable[str]]],
        blob_exists: Callable[[str], Awaitable[bool]],
        delete_blob: Callable[[str], Awaitable[None]],
        destructive: bool = False,
        authorized: bool = False,
        recovery_ready: bool = False,
        max_items: int = 100,
        lease_store: D1GcLeaseStore | None = None,
        expected_revision: int = 0,
        lease_ttl_seconds: float = 60.0,
    ) -> GcSweepBatchResult:
        """Async counterpart to :meth:`process_batch` over the live D1/R2 boundary.

        Honors the same grace, authorization, and recovery-readiness gates as
        the synchronous path. The destructive path delegates to
        :func:`delete_candidate_with_d1_lease`, so a candidate is deleted only
        while an authoritative D1 lease remains valid and the final reference
        recheck passes through the async adapter. ``lease_store`` is required
        when ``destructive=True``; the call fails closed otherwise.

        ``expected_revision`` MUST match the D1 corpus/reference revision the
        caller intends the sweep to operate under. A mismatch causes the lease
        acquisition to be rejected by the D1 lease schema, which the planner
        surfaces as :attr:`GcSweepDisposition.LEASE_DENIED`.
        """
        _require_non_negative_number(now, "now")
        _require_non_negative_number(grace_seconds, "grace_seconds")
        _require_positive_int(max_items, "max_items")
        _require_bool(destructive, "destructive")
        _require_bool(authorized, "authorized")
        _require_bool(recovery_ready, "recovery_ready")
        _require_non_negative_int(expected_revision, "expected_revision")
        if lease_store is not None:
            _require_positive_number(lease_ttl_seconds, "lease_ttl_seconds")
        if destructive and lease_store is None:
            raise ValueError(
                "destructive async sweeps require an authoritative D1 lease_store"
            )

        emitted: list[GcSweepRecord] = []
        attempted = 0
        while self.cursor < len(self.candidates) and attempted < max_items:
            candidate = self.candidates[self.cursor]
            if candidate.key in self.retired_keys:
                self.cursor += 1
                continue
            attempted += 1
            age = max(0.0, float(now) - candidate.first_seen_at)

            if age < float(grace_seconds):
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.DEFERRED_GRACE,
                    processed_at=float(now),
                    detail="grace period has not elapsed",
                )
                self._record_and_advance(record, emitted)
                continue

            try:
                knowledge_ids = tuple(
                    sorted(
                        {
                            str(value)
                            for value in await reference_lookup(candidate.key)
                        }
                    )
                )
            except Exception:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.FAILED,
                    processed_at=float(now),
                    detail="authoritative reference lookup failed",
                )
                self._record_without_advance(record, emitted)
                break

            if knowledge_ids:
                self.retired_keys.add(candidate.key)
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.RE_REFERENCED,
                    processed_at=float(now),
                    knowledge_ids=knowledge_ids,
                    detail="authoritative reference exists",
                )
                self._record_and_advance(record, emitted)
                continue

            if not destructive:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.DRY_RUN,
                    processed_at=float(now),
                    detail="destructive mode is disabled",
                )
                self._record_and_advance(record, emitted)
                continue

            if not authorized:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.UNAPPROVED,
                    processed_at=float(now),
                    detail="destructive authorization is missing",
                )
                self._record_and_advance(record, emitted)
                continue

            if not recovery_ready:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.RECOVERY_BLOCKED,
                    processed_at=float(now),
                    detail="recovery prerequisite is not satisfied",
                )
                self._record_and_advance(record, emitted)
                continue

            # The destructive path delegates the entire
            # acquire → revalidate → delete → finalize/failure flow to the
            # async boundary so the planner cannot accidentally bypass any step.
            lease_store_for_call = lease_store
            assert lease_store_for_call is not None  # validated above
            result = await delete_candidate_with_d1_lease(
                key=candidate.key,
                lease_store=lease_store_for_call,
                expected_revision=int(expected_revision),
                now=float(now),
                lease_ttl_seconds=float(lease_ttl_seconds),
                blob_exists=blob_exists,
                delete_blob=delete_blob,
                owner="gc-sweep",
            )
            record = _sweep_record_from_d1_result(
                candidate=candidate, now=float(now), result=result
            )
            if record.disposition is GcSweepDisposition.LEASE_DENIED:
                self._record_and_advance(record, emitted)
                continue
            if record.disposition is GcSweepDisposition.FAILED:
                self._record_without_advance(record, emitted)
                break
            self._record_and_advance(record, emitted)

        complete = self.cursor >= len(self.candidates)
        return GcSweepBatchResult(
            records=tuple(emitted),
            next_cursor=None if complete else self.cursor,
            complete=complete,
        )

    def start_next_pass(self) -> None:
        """Restart candidate evaluation after a completed pass."""
        if self.cursor != len(self.candidates):
            raise RuntimeError("cannot start next pass before current pass completes")
        self.cursor = 0

    def to_snapshot(self) -> dict[str, object]:
        """Serialize the bounded sweep state for process-level resume."""
        payload: dict[str, object] = {
            "schema_version": 1,
            "candidate_fingerprint": self.candidate_fingerprint,
            "candidates": [
                {
                    "key": candidate.key,
                    "first_seen_at": candidate.first_seen_at,
                    "size_bytes": candidate.size_bytes,
                }
                for candidate in self.candidates
            ],
            "cursor": self.cursor,
            "retired_keys": sorted(self.retired_keys),
            "records": [
                {
                    "key": record.key,
                    "disposition": record.disposition.value,
                    "processed_at": record.processed_at,
                    "knowledge_ids": list(record.knowledge_ids),
                    "detail": record.detail,
                }
                for record in self.records
            ],
        }
        payload["snapshot_fingerprint"] = _snapshot_fingerprint(payload)
        return payload

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> Self:
        """Restore a strictly validated sweep checkpoint."""
        if snapshot.get("schema_version") != 1:
            raise ValueError("unsupported or missing sweep snapshot schema_version")
        fingerprint = snapshot.get("snapshot_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("snapshot fingerprint is required")
        fingerprint_payload = dict(snapshot)
        fingerprint_payload.pop("snapshot_fingerprint", None)
        if fingerprint != _snapshot_fingerprint(fingerprint_payload):
            raise ValueError("snapshot fingerprint is invalid")
        raw_candidates = snapshot.get("candidates")
        raw_retired_keys = snapshot.get("retired_keys", [])
        raw_records = snapshot.get("records", [])
        if not isinstance(raw_candidates, list):
            raise ValueError("snapshot candidates must be a list")
        if not isinstance(raw_retired_keys, list) or any(
            not isinstance(value, str) or not value for value in raw_retired_keys
        ):
            raise ValueError("snapshot retired_keys must be a string list")
        if not isinstance(raw_records, list):
            raise ValueError("snapshot records must be a list")

        candidates: list[GcSweepCandidate] = []
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                raise ValueError("snapshot candidate contains invalid data")
            candidates.append(
                GcSweepCandidate(
                    key=_required_string(raw.get("key"), "candidate key"),
                    first_seen_at=_required_non_negative_float(
                        raw.get("first_seen_at"), "candidate first_seen_at"
                    ),
                    size_bytes=_optional_non_negative_int(raw.get("size_bytes")),
                )
            )

        records: list[GcSweepRecord] = []
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                raise ValueError("snapshot record contains invalid data")
            raw_knowledge_ids = raw.get("knowledge_ids", [])
            if not isinstance(raw_knowledge_ids, list) or any(
                not isinstance(value, str) or not value
                for value in raw_knowledge_ids
            ):
                raise ValueError("snapshot record knowledge_ids are invalid")
            detail = raw.get("detail")
            if detail is not None and not isinstance(detail, str):
                raise ValueError("snapshot record detail must be a string or null")
            try:
                disposition = GcSweepDisposition(
                    _required_string(raw.get("disposition"), "record disposition")
                )
            except ValueError as exc:
                raise ValueError("snapshot record disposition is invalid") from exc
            records.append(
                GcSweepRecord(
                    key=_required_string(raw.get("key"), "record key"),
                    disposition=disposition,
                    processed_at=_required_non_negative_float(
                        raw.get("processed_at"), "record processed_at"
                    ),
                    knowledge_ids=tuple(raw_knowledge_ids),
                    detail=detail,
                )
            )

        cursor = _required_non_negative_int(snapshot.get("cursor"), "cursor")
        fingerprint = _required_string(
            snapshot.get("candidate_fingerprint"), "candidate_fingerprint"
        )
        return cls(
            candidates=tuple(candidates),
            cursor=cursor,
            records=records,
            retired_keys=set(raw_retired_keys),
            candidate_fingerprint=fingerprint,
        )

    def _record_and_advance(
        self,
        record: GcSweepRecord,
        emitted: list[GcSweepRecord],
    ) -> None:
        self.records.append(record)
        emitted.append(record)
        self.cursor += 1

    def _record_without_advance(
        self,
        record: GcSweepRecord,
        emitted: list[GcSweepRecord],
    ) -> None:
        self.records.append(record)
        emitted.append(record)


def _record_lease_failure(
    authority: GcDeleteLeaseAuthority | None,
    key: str,
    token: str | None,
    now: float,
) -> None:
    """Record a retryable failure on a held sweep lease, if any.

    Best-effort: any authority exception while transitioning the lease to
    DELETE_FAILED must not mask the upstream sweep failure.
    """
    if authority is None or token is None:
        return
    try:
        authority.record_delete_failure(key=key, token=token, now=now)
    except Exception:
        pass


def _finalize_lease(
    authority: GcDeleteLeaseAuthority | None,
    key: str,
    token: str | None,
    now: float,
) -> None:
    """Finalize a held sweep lease after a confirmed R2 absence.

    Best-effort: any authority exception while finalizing must not mask the
    upstream confirmed deletion evidence.
    """
    if authority is None or token is None:
        return
    try:
        authority.finalize_delete(
            key=key, token=token, now=now, already_absent=True
        )
    except Exception:
        pass


def _sweep_record_from_d1_result(
    *,
    candidate: GcSweepCandidate,
    now: float,
    result: D1DeleteResult,
) -> GcSweepRecord:
    """Translate a :class:`D1DeleteResult` into a sweep evidence record.

    The mapping preserves the planner's public-safe vocabulary while pinning the
    detail string to the boundary's authoritative reason. Any unexpected
    outcome is recorded as :attr:`GcSweepDisposition.FAILED` so the planner
    never silently overstates the destructive result.
    """
    detail = result.detail or "d1 delete boundary returned no detail"
    if result.outcome is D1DeleteOutcome.DELETED:
        disposition = GcSweepDisposition.DELETED
    elif result.outcome is D1DeleteOutcome.ALREADY_ABSENT:
        disposition = GcSweepDisposition.ALREADY_ABSENT
    elif result.outcome is D1DeleteOutcome.LEASE_DENIED:
        disposition = GcSweepDisposition.LEASE_DENIED
    elif result.outcome is D1DeleteOutcome.FAILED:
        disposition = GcSweepDisposition.FAILED
    else:  # pragma: no cover - defensive: StrEnum guards exhaustiveness
        disposition = GcSweepDisposition.FAILED
        detail = f"unknown d1 delete outcome: {result.outcome}"
    return GcSweepRecord(
        key=candidate.key,
        disposition=disposition,
        processed_at=float(now),
        detail=detail,
    )


def _snapshot_fingerprint(payload: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("snapshot contains non-serializable data") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_fingerprint(candidates: Iterable[GcSweepCandidate]) -> str:
    payload = [
        {
            "key": candidate.key,
            "first_seen_at": candidate.first_seen_at,
            "size_bytes": candidate.size_bytes,
        }
        for candidate in candidates
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_non_negative_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _require_positive_number(value: object, name: str) -> float:
    result = _require_non_negative_number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_bool_result(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} returned a non-boolean value")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


def _required_non_negative_float(value: object, name: str) -> float:
    return _require_non_negative_number(value, name)


def _required_non_negative_int(value: object, name: str) -> int:
    return _require_non_negative_int(value, name)


def _optional_non_negative_int(value: object) -> int | None:
    if value is None:
        return None
    return _require_non_negative_int(value, "size_bytes")


__all__ = [
    "GcSweepDisposition",
    "GcSweepCandidate",
    "GcSweepRecord",
    "GcSweepBatchResult",
    "GcSweepState",
]
