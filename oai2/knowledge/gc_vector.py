"""Reference-safe dry-run reconciliation for Vectorize vectors (#261).

Non-destructive by construction. This module contains no delete path, no
binding reference, and no way to reach a Vectorize index. It derives the mark
set from authoritative ``knowledge_index.vectorize_id`` values, consumes
paginated vector inventory, and produces a deterministic report. Granting
deletion and running a sweep are separate work items requiring separate explicit
authorization (see #261 AC-GC-034).

Two rules here are the whole reason this module is not the R2 reconciler with a
different field name:

* Liveness comes from D1 alone. An unreferenced vector is a candidate because no
  authoritative row names it, never because Vectorize reports it as unused and
  never because a read failed to resolve it.
* No embedding-version filter, ever. A row naming an older-version vector is a
  live reference. Scoping liveness to the current embedding version would let a
  sweep delete a live vector in the middle of a re-embedding (#73), which is
  precisely the state a re-embedding creates.

Legacy rows whose ``vectorize_id`` equals ``knowledge_id`` need no special case:
the mark set is whatever the row says it is. Pinning that is a test, not code.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class KnowledgeVectorRow(Protocol):
    """Structural view required from an authoritative knowledge metadata row."""

    knowledge_id: object
    vectorize_id: str | None


class VectorGcDisposition(StrEnum):
    """Dry-run classification for one unique vector id."""

    REFERENCED_PRESENT = "referenced_present"
    REFERENCED_MISSING = "referenced_missing"
    UNREFERENCED_CANDIDATE = "unreferenced_candidate"


@dataclass(slots=True, frozen=True)
class VectorGcGracePolicy:
    """How long an unreferenced vector must persist before it is even eligible.

    REQ-GC-033: unreferenced vectors are grace-windowed candidates with
    persisted first-seen state, and must survive at least one authoritative
    recheck. Both halves are load-bearing and neither is a formality:

    - ``grace_seconds`` bounds the wall-clock exposure. A concurrent writer
      that is mid-``put()`` has already written the vector but may not have
      committed its D1 row yet; a short window converts that race into data
      loss.
    - ``minimum_rechecks`` bounds the *observations*. Age alone is not
      evidence: a single sighting, however old, is exactly what an in-flight
      adoption looks like. A candidate must be re-observed against fresh D1
      at least ``minimum_rechecks`` times after it was first seen before the
      window can be considered satisfied.

    ``grace_seconds`` is required to be strictly positive and
    ``minimum_rechecks`` at least 1, so a policy cannot be constructed that
    makes a first sighting eligible. That is the whole point of the type.
    """

    grace_seconds: float = 900.0
    minimum_rechecks: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.grace_seconds, bool)
            or not isinstance(self.grace_seconds, (int, float))
            or not math.isfinite(float(self.grace_seconds))
            or float(self.grace_seconds) <= 0.0
        ):
            raise ValueError("grace_seconds must be a finite number > 0")
        if (
            isinstance(self.minimum_rechecks, bool)
            or not isinstance(self.minimum_rechecks, int)
            or self.minimum_rechecks < 1
        ):
            raise ValueError("minimum_rechecks must be an integer >= 1")

    def satisfies(self, *, first_seen_at: float, rechecks: int, now: float) -> bool:
        """Whether a candidate has cleared BOTH halves of the window.

        Returns False on a non-finite or backwards ``now``: a clock that went
        backwards must not manufacture eligibility, and neither must one that
        is not a real time at all.
        """
        if rechecks < self.minimum_rechecks:
            return False
        if not math.isfinite(now) or not math.isfinite(first_seen_at):
            return False
        age = now - first_seen_at
        return age >= self.grace_seconds


@dataclass(slots=True, frozen=True)
class VectorGcGraceState:
    """Persisted first-seen state for one grace-windowed candidate.

    Carried across runs, because the window is a statement about how long a
    vector has been unreferenced, and that spans process boundaries. A
    candidate that forgot its first sighting every run would never accumulate
    age and would never become eligible -- which is the safe direction to
    fail, but it would also mean the sweep could never run.
    """

    vector_id: str
    first_seen_at: float
    rechecks: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.vector_id, str)
            or not self.vector_id
            or self.vector_id != self.vector_id.strip()
        ):
            raise ValueError("vector_id must be a non-empty normalized string")
        if not _is_non_negative_number(self.first_seen_at):
            raise ValueError("first_seen_at must be a finite non-negative number")
        if (
            isinstance(self.rechecks, bool)
            or not isinstance(self.rechecks, int)
            or self.rechecks < 0
        ):
            raise ValueError("rechecks must be a non-negative integer")


@dataclass(slots=True, frozen=True)
class VectorInventoryEntry:
    """Public-safe metadata for one vector in an inventory page.

    Deliberately carries NO embedding ``values``. A vector's float payload is
    large, is never needed to decide liveness, and has no business in a report.
    Only the attribution fields the read contract already defines are kept.
    """

    vector_id: str
    knowledge_id: str | None = None
    content_hash: str | None = None
    embedding_version: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.vector_id, str)
            or not self.vector_id
            or self.vector_id != self.vector_id.strip()
        ):
            raise ValueError("vector_id must be a non-empty normalized string")
        for name in ("knowledge_id", "content_hash", "embedding_version"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be null or a non-empty normalized string")


@dataclass(slots=True, frozen=True)
class VectorGcRecord:
    """One unique vector id in a completed reconciliation report."""

    vector_id: str
    disposition: VectorGcDisposition
    knowledge_ids: tuple[str, ...] = ()
    # Grace-window state. All three are None/0/False for anything that is not
    # an UNREFERENCED_CANDIDATE, so a reader cannot mistake a referenced
    # vector for one that has been accumulating toward deletion.
    first_seen_at: float | None = None
    rechecks: int = 0
    grace_satisfied: bool = False

    @property
    def eligible_for_authorization(self) -> bool:
        """Whether a destructive act could even be *asked* for this vector.

        Both halves of REQ-GC-033 must hold, and this module still has no
        destructive path: satisfying the window earns the right to request
        separate, explicit authorization. It is not authorization, and it
        does not make the vector safe to remove on its own.
        """
        return self.disposition is VectorGcDisposition.UNREFERENCED_CANDIDATE and (
            self.grace_satisfied
        )


@dataclass(slots=True, frozen=True)
class VectorGcDryRunReport:
    """Deterministic, non-destructive result of a completed inventory scan."""

    observed_at: float
    pages_processed: int
    records: tuple[VectorGcRecord, ...]
    referenced_present_count: int
    referenced_missing_count: int
    unreferenced_candidate_count: int
    inventory_vector_count: int
    # How many candidates have cleared BOTH halves of the grace window.
    # Reported separately from the candidate count so a reader can never
    # mistake 'seen unreferenced' for 'eligible for authorization'.
    grace_satisfied_count: int = 0


@dataclass(slots=True)
class VectorGcReconciliationState:
    """Resumable state for a paginated vector inventory reconciliation.

    A final report is unavailable until the inventory source signals completion
    (``next_cursor=None``). Without that, an unseen page would make a referenced
    vector look like an orphan, and orphans are what gets deleted.
    """

    observed_at: float
    references: dict[str, set[str]] = field(default_factory=dict)
    inventory: dict[str, VectorInventoryEntry] = field(default_factory=dict)
    pages_processed: int = 0
    next_cursor: str | None = None
    inventory_complete: bool = False
    # Persisted first-seen state, keyed by vector id. Carried across snapshot
    # round-trips so the window measures elapsed time rather than counting
    # process lifetimes.
    grace_candidates: dict[str, VectorGcGraceState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_non_negative_number(self.observed_at):
            raise ValueError("observed_at must be a finite non-negative number")
        if (
            isinstance(self.pages_processed, bool)
            or not isinstance(self.pages_processed, int)
            or self.pages_processed < 0
        ):
            raise ValueError("pages_processed must be a non-negative integer")
        _validate_cursor_state(
            pages_processed=self.pages_processed,
            next_cursor=self.next_cursor,
            inventory_complete=self.inventory_complete,
        )

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[KnowledgeVectorRow],
        *,
        observed_at: float,
    ) -> VectorGcReconciliationState:
        """Build the authoritative mark set from D1 rows.

        A row with no vectorize_id contributes nothing: it has no semantic index
        entry, so it cannot keep a vector alive and it is not evidence of a
        missing one.
        """
        references: dict[str, set[str]] = {}
        for row in rows:
            vector_id = row.vectorize_id
            if not vector_id:
                continue
            if not isinstance(vector_id, str) or vector_id != vector_id.strip():
                raise ValueError("row vectorize_id must be a normalized string")
            references.setdefault(vector_id, set()).add(str(row.knowledge_id))
        return cls(observed_at=observed_at, references=references)

    def consume_page(
        self,
        vectors: Iterable[VectorInventoryEntry],
        *,
        next_cursor: str | None,
    ) -> None:
        """Consume one inventory page. Never mutates the index."""
        if self.inventory_complete:
            raise RuntimeError("inventory scan is already complete")
        if next_cursor is not None and (
            not isinstance(next_cursor, str) or not next_cursor
        ):
            raise ValueError("next_cursor must be a non-empty string or null")

        page: dict[str, VectorInventoryEntry] = {}
        for entry in vectors:
            previous = page.get(entry.vector_id)
            if previous is not None and previous != entry:
                raise ValueError(
                    f"conflicting inventory metadata for vector {entry.vector_id!r}"
                )
            page[entry.vector_id] = entry

        for vector_id, entry in page.items():
            previous = self.inventory.get(vector_id)
            if previous is not None and previous != entry:
                raise ValueError(
                    f"conflicting inventory metadata for vector {vector_id!r}"
                )

        self.inventory.update(page)
        self.pages_processed += 1
        self.next_cursor = next_cursor
        self.inventory_complete = next_cursor is None

    def to_snapshot(self) -> dict[str, object]:
        """JSON-serializable snapshot for process-level resume."""
        return {
            "schema_version": 2,
            "observed_at": self.observed_at,
            "references": {k: sorted(v) for k, v in sorted(self.references.items())},
            "inventory": {
                k: {
                    "vector_id": e.vector_id,
                    "knowledge_id": e.knowledge_id,
                    "content_hash": e.content_hash,
                    "embedding_version": e.embedding_version,
                }
                for k, e in sorted(self.inventory.items())
            },
            "pages_processed": self.pages_processed,
            "next_cursor": self.next_cursor,
            "inventory_complete": self.inventory_complete,
            "grace_candidates": {
                vector_id: {
                    "vector_id": grace.vector_id,
                    "first_seen_at": grace.first_seen_at,
                    "rechecks": grace.rechecks,
                }
                for vector_id, grace in sorted(self.grace_candidates.items())
            },
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, object],
        *,
        authoritative_rows: Iterable[KnowledgeVectorRow] | None = None,
    ) -> VectorGcReconciliationState:
        """Restore a validated snapshot, optionally re-checking current D1 rows."""
        schema_version = snapshot.get("schema_version")
        if schema_version not in (1, 2):
            raise ValueError("unsupported or missing snapshot schema_version")

        raw_references = snapshot.get("references")
        raw_inventory = snapshot.get("inventory")
        if not isinstance(raw_references, Mapping):
            raise ValueError("snapshot references must be a mapping")
        if not isinstance(raw_inventory, Mapping):
            raise ValueError("snapshot inventory must be a mapping")

        references: dict[str, set[str]] = {}
        for key, raw_ids in raw_references.items():
            if (
                not isinstance(key, str)
                or not key
                or key != key.strip()
                or not isinstance(raw_ids, list)
                or not raw_ids
                or any(not isinstance(i, str) or not i for i in raw_ids)
            ):
                raise ValueError("snapshot references contain invalid data")
            references[key] = set(raw_ids)

        inventory: dict[str, VectorInventoryEntry] = {}
        for key, raw_entry in raw_inventory.items():
            if not isinstance(key, str) or not isinstance(raw_entry, Mapping):
                raise ValueError("snapshot inventory contains invalid data")
            vector_id = raw_entry.get("vector_id")
            if not isinstance(vector_id, str) or vector_id != key:
                raise ValueError("snapshot inventory key does not match vector_id")
            inventory[key] = VectorInventoryEntry(
                vector_id=vector_id,
                knowledge_id=_optional_normalized(raw_entry.get("knowledge_id")),
                content_hash=_optional_normalized(raw_entry.get("content_hash")),
                embedding_version=_optional_normalized(
                    raw_entry.get("embedding_version")
                ),
            )

        grace_candidates: dict[str, VectorGcGraceState] = {}
        if schema_version >= 2:
            raw_grace = snapshot.get("grace_candidates", {})
            if not isinstance(raw_grace, Mapping):
                raise ValueError("snapshot grace_candidates must be a mapping")
            for key, raw_grace_entry in raw_grace.items():
                if not isinstance(key, str) or not isinstance(raw_grace_entry, Mapping):
                    raise ValueError("snapshot grace_candidates contain invalid data")
                vector_id = raw_grace_entry.get("vector_id")
                if not isinstance(vector_id, str) or vector_id != key:
                    raise ValueError("snapshot grace_candidates key does not match vector_id")
                grace_candidates[key] = VectorGcGraceState(
                    vector_id=vector_id,
                    first_seen_at=_as_number(raw_grace_entry.get("first_seen_at")),
                    rechecks=_as_int(raw_grace_entry.get("rechecks", 0)),
                )
        # A v1 snapshot carries no grace history, so it restores with an EMPTY
        # candidate set. That is not a downgrade: it grants no grace credit,
        # so every candidate must re-earn a first sighting. Back-filling
        # first-seen from the snapshot's observed_at would manufacture the
        # very age a v1 snapshot has no evidence for.

        if authoritative_rows is not None:
            current = cls.from_rows(
                authoritative_rows, observed_at=_as_number(snapshot.get("observed_at"))
            )
            if current.references != references:
                raise ValueError("authoritative reference set changed since snapshot")

        return cls(
            observed_at=_as_number(snapshot.get("observed_at")),
            references=references,
            inventory=inventory,
            pages_processed=_as_int(snapshot.get("pages_processed", 0)),
            next_cursor=_optional_cursor(snapshot.get("next_cursor")),
            inventory_complete=bool(snapshot.get("inventory_complete", False)),
            grace_candidates=grace_candidates,
        )

    def recheck_grace(
        self,
        *,
        now: float,
        policy: VectorGcGracePolicy,
    ) -> dict[str, VectorGcGraceState]:
        """Advance the grace window against a completed reconciliation.

        This is the ``RECHK`` step in this issue's own diagram: the first
        ``KEEP / clear candidate`` branch and the first half of the
        ``still unreferenced`` branch. It reads the CURRENT authoritative mark
        set -- the one just built from D1 -- and reconciles it with the
        persisted first-seen state.

        Three transitions, and the order matters:

        1. A vector that is referenced again is **removed** from the candidate
           set. Its grace history is discarded, not merely paused: if it goes
           unreferenced a second time it must earn a fresh window, because the
           thing that made it referenced may have moved.
        2. A vector seen unreferenced for the first time is recorded with
           ``rechecks=0``. It is deliberately **not** eligible, however much
           wall-clock time has passed: a first sighting is indistinguishable
           from a ``put()`` that has written its vector but not yet committed
           its D1 row.
        3. A vector already being tracked accrues one recheck, because being
           re-observed against fresh D1 is precisely what "survived an
           authoritative recheck" means.

        Mutates and returns ``self.grace_candidates``. Requires a completed
        scan: classifying against a partial inventory would age a candidate on
        the strength of a page nobody has read.
        """
        if not self.inventory_complete:
            raise RuntimeError("inventory scan is incomplete")
        if not _is_non_negative_number(now):
            raise ValueError("now must be a finite non-negative number")

        tracked: dict[str, VectorGcGraceState] = {}
        for vector_id in sorted(set(self.references) | set(self.inventory)):
            if vector_id in self.references:
                # Branch 1: KEEP. A reference appeared; the candidate is
                # cleared rather than left to age back into eligibility.
                continue
            prior = self.grace_candidates.get(vector_id)
            if prior is None:
                # Branch 2: first sighting. No grace credit whatsoever.
                tracked[vector_id] = VectorGcGraceState(
                    vector_id=vector_id, first_seen_at=now, rechecks=0
                )
            else:
                # Branch 3: survived another authoritative recheck.
                tracked[vector_id] = VectorGcGraceState(
                    vector_id=vector_id,
                    first_seen_at=prior.first_seen_at,
                    rechecks=prior.rechecks + 1,
                )
        self.grace_candidates = tracked
        return self.grace_candidates

    def build_report(
        self, *, policy: VectorGcGracePolicy | None = None
    ) -> VectorGcDryRunReport:
        """Classify every vector the union of the two sets covers.

        ``policy`` populates the grace-window fields. Without it a report
        still classifies, but every candidate is reported as first-sighting
        and therefore ineligible -- the conservative reading, and the one that
        cannot accidentally authorize anything.
        """
        if not self.inventory_complete:
            raise RuntimeError("inventory scan is incomplete")

        records: list[VectorGcRecord] = []
        referenced_present = 0
        referenced_missing = 0
        candidates = 0

        for vector_id in sorted(set(self.references) | set(self.inventory)):
            knowledge_ids = tuple(sorted(self.references.get(vector_id, set())))
            present = vector_id in self.inventory
            # Membership, not truthiness. A reference record with no ids is a
            # degraded record, not an absent one: branching on ``knowledge_ids``
            # dropped such a key into UNREFERENCED_CANDIDATE and queued
            # referenced vectors for deletion. The authoritative row naming the
            # key is the reason to look, not a licence to remove it.
            referenced = vector_id in self.references
            if referenced and present:
                disposition = VectorGcDisposition.REFERENCED_PRESENT
                referenced_present += 1
            elif referenced:
                # A row names a vector the index does not have. That is an
                # integrity state to report, NOT a delete candidate: the
                # authoritative row is the reason to look, not a licence.
                disposition = VectorGcDisposition.REFERENCED_MISSING
                referenced_missing += 1
            else:
                disposition = VectorGcDisposition.UNREFERENCED_CANDIDATE
                candidates += 1
            # Grace state is reported ONLY for an unreferenced candidate. A
            # referenced vector carrying a first-seen timestamp would read as
            # though it were accumulating toward deletion.
            grace = (
                self.grace_candidates.get(vector_id)
                if disposition is VectorGcDisposition.UNREFERENCED_CANDIDATE
                else None
            )
            records.append(
                VectorGcRecord(
                    vector_id=vector_id,
                    disposition=disposition,
                    knowledge_ids=knowledge_ids,
                    first_seen_at=None if grace is None else grace.first_seen_at,
                    rechecks=0 if grace is None else grace.rechecks,
                    grace_satisfied=(
                        grace is not None
                        and policy is not None
                        and policy.satisfies(
                            first_seen_at=grace.first_seen_at,
                            rechecks=grace.rechecks,
                            now=self.observed_at,
                        )
                    ),
                )
            )

        return VectorGcDryRunReport(
            observed_at=self.observed_at,
            pages_processed=self.pages_processed,
            records=tuple(records),
            referenced_present_count=referenced_present,
            referenced_missing_count=referenced_missing,
            unreferenced_candidate_count=candidates,
            inventory_vector_count=len(self.inventory),
            grace_satisfied_count=sum(1 for r in records if r.grace_satisfied),
        )


def _validate_cursor_state(
    *,
    pages_processed: int,
    next_cursor: str | None,
    inventory_complete: bool,
) -> None:
    if inventory_complete and next_cursor is not None:
        raise ValueError("completed inventory cannot have next_cursor")
    if not inventory_complete and pages_processed > 0 and next_cursor is None:
        raise ValueError("incomplete inventory with pages must have next_cursor")


def _is_non_negative_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _as_number(value: object) -> float:
    if not _is_non_negative_number(value):
        raise ValueError("observed_at must be a finite non-negative number")
    assert isinstance(value, (int, float))  # narrowed by _is_non_negative_number
    return float(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("pages_processed must be a non-negative integer")
    return value


def _optional_normalized(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("expected null or a non-empty normalized string")
    return value


def _optional_cursor(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("next_cursor must be a non-empty string or null")
    return value


__all__ = [
    "KnowledgeVectorRow",
    "VectorGcDisposition",
    "VectorGcGracePolicy",
    "VectorGcGraceState",
    "VectorInventoryEntry",
    "VectorGcRecord",
    "VectorGcDryRunReport",
    "VectorGcReconciliationState",
]
