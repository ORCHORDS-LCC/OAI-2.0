"""Reference-safe dry-run reconciliation for content-addressed R2 bodies.

This module is intentionally non-destructive. It derives the mark set from
an authoritative set of retained knowledge rows, consumes paginated R2
inventory, and produces a deterministic report of present, missing, and
unreferenced object keys. Deletion/grace-period policy belongs to a separate
work item.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, Self


class KnowledgeBlobRow(Protocol):
    """Structural view required from an authoritative knowledge metadata row."""

    knowledge_id: object
    r2_blob_key: str | None


class GcObjectDisposition(StrEnum):
    """Dry-run classification for one unique R2 object key."""

    REFERENCED_PRESENT = "referenced_present"
    REFERENCED_MISSING = "referenced_missing"
    UNREFERENCED_CANDIDATE = "unreferenced_candidate"


@dataclass(slots=True, frozen=True)
class R2InventoryObject:
    """Public-safe metadata returned by an R2 inventory page."""

    key: str
    size_bytes: int | None = None
    uploaded_at: float | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must not be empty")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


@dataclass(slots=True, frozen=True)
class GcObjectRecord:
    """One unique key in a completed reconciliation report."""

    key: str
    disposition: GcObjectDisposition
    knowledge_ids: tuple[str, ...] = ()
    size_bytes: int | None = None
    uploaded_at: float | None = None
    age_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class GcDryRunReport:
    """Deterministic, non-destructive result of a completed inventory scan."""

    observed_at: float
    pages_processed: int
    records: tuple[GcObjectRecord, ...]
    referenced_present_count: int
    referenced_missing_count: int
    unreferenced_candidate_count: int
    inventory_object_count: int
    inventory_bytes_known: int
    unreferenced_candidate_bytes_known: int
    unknown_size_object_count: int
    oldest_unreferenced_candidate_age_seconds: float | None


@dataclass(slots=True)
class GcReconciliationState:
    """Resumable state for a paginated R2 inventory reconciliation.

    The state is serializable through :meth:`to_snapshot` and can be restored
    with :meth:`from_snapshot`. A final report is intentionally unavailable
    until the inventory source signals completion with ``next_cursor=None``;
    otherwise unseen pages could make a referenced body look missing.
    """

    observed_at: float
    references: dict[str, set[str]] = field(default_factory=dict)
    inventory: dict[str, R2InventoryObject] = field(default_factory=dict)
    pages_processed: int = 0
    next_cursor: str | None = None
    inventory_complete: bool = False

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[KnowledgeBlobRow],
        *,
        observed_at: float,
    ) -> Self:
        references: dict[str, set[str]] = {}
        for row in rows:
            key = row.r2_blob_key
            if not key:
                continue
            references.setdefault(key, set()).add(str(row.knowledge_id))
        return cls(observed_at=float(observed_at), references=references)

    def consume_page(
        self,
        objects: Iterable[R2InventoryObject],
        *,
        next_cursor: str | None,
    ) -> None:
        """Consume one inventory page without deleting or mutating R2."""
        if self.inventory_complete:
            raise RuntimeError("inventory scan is already complete")

        for obj in objects:
            previous = self.inventory.get(obj.key)
            if previous is not None and previous != obj:
                raise ValueError(
                    f"conflicting inventory metadata for object key {obj.key!r}"
                )
            self.inventory[obj.key] = obj

        self.pages_processed += 1
        self.next_cursor = next_cursor
        self.inventory_complete = next_cursor is None

    def to_snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for process-level resume."""
        return {
            "observed_at": self.observed_at,
            "references": {
                key: sorted(ids) for key, ids in sorted(self.references.items())
            },
            "inventory": {
                key: {
                    "key": obj.key,
                    "size_bytes": obj.size_bytes,
                    "uploaded_at": obj.uploaded_at,
                }
                for key, obj in sorted(self.inventory.items())
            },
            "pages_processed": self.pages_processed,
            "next_cursor": self.next_cursor,
            "inventory_complete": self.inventory_complete,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> Self:
        """Restore a state previously produced by :meth:`to_snapshot`."""
        raw_references = snapshot.get("references", {})
        raw_inventory = snapshot.get("inventory", {})
        if not isinstance(raw_references, Mapping):
            raise ValueError("snapshot references must be a mapping")
        if not isinstance(raw_inventory, Mapping):
            raise ValueError("snapshot inventory must be a mapping")

        references: dict[str, set[str]] = {}
        for key, raw_ids in raw_references.items():
            if not isinstance(key, str) or not isinstance(raw_ids, list):
                raise ValueError("snapshot references contain invalid data")
            references[key] = {str(item) for item in raw_ids}

        inventory: dict[str, R2InventoryObject] = {}
        for key, raw_obj in raw_inventory.items():
            if not isinstance(key, str) or not isinstance(raw_obj, Mapping):
                raise ValueError("snapshot inventory contains invalid data")
            obj = R2InventoryObject(
                key=str(raw_obj.get("key", key)),
                size_bytes=_optional_int(raw_obj.get("size_bytes")),
                uploaded_at=_optional_float(raw_obj.get("uploaded_at")),
            )
            if obj.key != key:
                raise ValueError("snapshot inventory key does not match object key")
            inventory[key] = obj

        next_cursor_value = snapshot.get("next_cursor")
        if next_cursor_value is not None and not isinstance(next_cursor_value, str):
            raise ValueError("snapshot next_cursor must be a string or null")

        return cls(
            observed_at=float(snapshot["observed_at"]),
            references=references,
            inventory=inventory,
            pages_processed=int(snapshot.get("pages_processed", 0)),
            next_cursor=next_cursor_value,
            inventory_complete=bool(snapshot.get("inventory_complete", False)),
        )

    def build_report(self) -> GcDryRunReport:
        """Build a final report after all inventory pages have been consumed."""
        if not self.inventory_complete:
            raise RuntimeError("inventory scan is incomplete")

        records: list[GcObjectRecord] = []
        referenced_present = 0
        referenced_missing = 0
        orphan_candidates = 0
        inventory_bytes_known = 0
        orphan_bytes_known = 0
        unknown_size_objects = 0
        oldest_orphan_age: float | None = None

        for obj in self.inventory.values():
            if obj.size_bytes is None:
                unknown_size_objects += 1
            else:
                inventory_bytes_known += obj.size_bytes

        all_keys = sorted(set(self.references) | set(self.inventory))
        for key in all_keys:
            knowledge_ids = tuple(sorted(self.references.get(key, set())))
            obj = self.inventory.get(key)

            if knowledge_ids and obj is not None:
                disposition = GcObjectDisposition.REFERENCED_PRESENT
                referenced_present += 1
            elif knowledge_ids:
                disposition = GcObjectDisposition.REFERENCED_MISSING
                referenced_missing += 1
            else:
                disposition = GcObjectDisposition.UNREFERENCED_CANDIDATE
                orphan_candidates += 1

            age_seconds = _age_seconds(self.observed_at, obj.uploaded_at) if obj else None
            if disposition is GcObjectDisposition.UNREFERENCED_CANDIDATE:
                if obj is not None and obj.size_bytes is not None:
                    orphan_bytes_known += obj.size_bytes
                if age_seconds is not None:
                    oldest_orphan_age = (
                        age_seconds
                        if oldest_orphan_age is None
                        else max(oldest_orphan_age, age_seconds)
                    )

            records.append(
                GcObjectRecord(
                    key=key,
                    disposition=disposition,
                    knowledge_ids=knowledge_ids,
                    size_bytes=obj.size_bytes if obj else None,
                    uploaded_at=obj.uploaded_at if obj else None,
                    age_seconds=age_seconds,
                )
            )

        return GcDryRunReport(
            observed_at=self.observed_at,
            pages_processed=self.pages_processed,
            records=tuple(records),
            referenced_present_count=referenced_present,
            referenced_missing_count=referenced_missing,
            unreferenced_candidate_count=orphan_candidates,
            inventory_object_count=len(self.inventory),
            inventory_bytes_known=inventory_bytes_known,
            unreferenced_candidate_bytes_known=orphan_bytes_known,
            unknown_size_object_count=unknown_size_objects,
            oldest_unreferenced_candidate_age_seconds=oldest_orphan_age,
        )


def _age_seconds(observed_at: float, uploaded_at: float | None) -> float | None:
    if uploaded_at is None:
        return None
    return max(0.0, observed_at - uploaded_at)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid integer")
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid float")
    return float(value)


__all__ = [
    "KnowledgeBlobRow",
    "GcObjectDisposition",
    "R2InventoryObject",
    "GcObjectRecord",
    "GcDryRunReport",
    "GcReconciliationState",
]
