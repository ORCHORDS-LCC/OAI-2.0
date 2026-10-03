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
            "schema_version": 1,
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
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, object],
        *,
        authoritative_rows: Iterable[KnowledgeVectorRow] | None = None,
    ) -> VectorGcReconciliationState:
        """Restore a validated snapshot, optionally re-checking current D1 rows."""
        if snapshot.get("schema_version") != 1:
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
        )

    def build_report(self) -> VectorGcDryRunReport:
        """Classify every vector the union of the two sets covers."""
        if not self.inventory_complete:
            raise RuntimeError("inventory scan is incomplete")

        records: list[VectorGcRecord] = []
        referenced_present = 0
        referenced_missing = 0
        candidates = 0

        for vector_id in sorted(set(self.references) | set(self.inventory)):
            knowledge_ids = tuple(sorted(self.references.get(vector_id, set())))
            present = vector_id in self.inventory
            if knowledge_ids and present:
                disposition = VectorGcDisposition.REFERENCED_PRESENT
                referenced_present += 1
            elif knowledge_ids:
                # A row names a vector the index does not have. That is an
                # integrity state to report, NOT a delete candidate: the
                # authoritative row is the reason to look, not a licence.
                disposition = VectorGcDisposition.REFERENCED_MISSING
                referenced_missing += 1
            else:
                disposition = VectorGcDisposition.UNREFERENCED_CANDIDATE
                candidates += 1
            records.append(
                VectorGcRecord(
                    vector_id=vector_id,
                    disposition=disposition,
                    knowledge_ids=knowledge_ids,
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
    "VectorInventoryEntry",
    "VectorGcRecord",
    "VectorGcDryRunReport",
    "VectorGcReconciliationState",
]
