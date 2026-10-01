"""Conservative, resumable sweep planning for confirmed R2 orphan candidates.

This module is safe-by-default: dry-run is the default, destructive execution
requires explicit authorization and recovery readiness, and every candidate is
rechecked against authoritative metadata immediately before deletion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from .gc import GcDryRunReport, GcObjectDisposition


class GcSweepDisposition(StrEnum):
    """Outcome for one sweep candidate in one processing pass."""

    DEFERRED_GRACE = "deferred_grace"
    RE_REFERENCED = "re_referenced"
    DRY_RUN = "dry_run"
    UNAPPROVED = "unapproved"
    RECOVERY_BLOCKED = "recovery_blocked"
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
    candidate_fingerprint: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.cursor, bool) or not isinstance(self.cursor, int):
            raise ValueError("cursor must be an integer")
        if self.cursor < 0 or self.cursor > len(self.candidates):
            raise ValueError("cursor is outside candidate bounds")
        keys = [candidate.key for candidate in self.candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("sweep candidates contain duplicate keys")
        expected = _candidate_fingerprint(self.candidates)
        if not self.candidate_fingerprint:
            self.candidate_fingerprint = expected
        elif self.candidate_fingerprint != expected:
            raise ValueError("candidate fingerprint does not match candidates")

    @classmethod
    def from_report(cls, report: GcDryRunReport) -> Self:
        """Create sweep state from a completed, non-destructive dry-run report."""
        _require_non_negative_number(report.observed_at, "report observed_at")
        candidates = tuple(
            GcSweepCandidate(
                key=record.key,
                first_seen_at=report.observed_at,
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
    ) -> GcSweepBatchResult:
        """Process a bounded number of candidates conservatively.

        A failed dependency operation leaves the cursor on the failing
        candidate so a later call can retry it. Non-failing decisions advance
        the cursor and are retained as public-safe evidence records.
        """
        _require_non_negative_number(now, "now")
        _require_non_negative_number(grace_seconds, "grace_seconds")
        _require_positive_int(max_items, "max_items")

        emitted: list[GcSweepRecord] = []
        attempted = 0
        while self.cursor < len(self.candidates) and attempted < max_items:
            candidate = self.candidates[self.cursor]
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
                exists_before = bool(blob_exists(candidate.key))
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
                delete_blob(candidate.key)
                still_exists = bool(blob_exists(candidate.key))
            except Exception:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.FAILED,
                    processed_at=float(now),
                    detail="blob deletion failed",
                )
                self._record_without_advance(record, emitted)
                break

            if still_exists:
                record = GcSweepRecord(
                    key=candidate.key,
                    disposition=GcSweepDisposition.FAILED,
                    processed_at=float(now),
                    detail="blob remained present after deletion",
                )
                self._record_without_advance(record, emitted)
                break

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
        raw_records = snapshot.get("records", [])
        if not isinstance(raw_candidates, list):
            raise ValueError("snapshot candidates must be a list")
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


def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
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
