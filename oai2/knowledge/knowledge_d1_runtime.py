"""Async D1 writer for authoritative knowledge metadata and corpus revision.

The writer keeps the knowledge_index mutation and corpus revision advance inside
one D1 batch transaction. Both statements use the same expected revision and
deletion-lease exclusion conditions, so a write is accepted only when the
object key is not protected by an active lease.
"""

from __future__ import annotations

from collections.abc import Mapping

from .cloudflare import CFRow
from .gc_lease_d1_runtime import D1DatabaseBinding
from .knowledge_d1 import (
    KNOWLEDGE_CORPUS_ADVANCE_SQL,
    KNOWLEDGE_CORPUS_REVISION_SQL,
    KNOWLEDGE_WRITER_UPSERT_SQL,
    knowledge_schema_statements,
)


class D1KnowledgeWriter:
    def __init__(self, database: D1DatabaseBinding) -> None:
        self._db = database

    async def ensure_schema(self) -> None:
        statements = [self._db.prepare(sql) for sql in knowledge_schema_statements()]
        results = await self._db.batch(statements)
        if len(results) != len(statements):
            raise RuntimeError("D1 knowledge schema batch returned an unexpected result count")
        for result in results:
            if not _result_success(result):
                raise RuntimeError("D1 knowledge schema batch reported an unsuccessful statement")

    async def corpus_revision(self) -> int:
        value = await self._db.prepare(KNOWLEDGE_CORPUS_REVISION_SQL).first("revision")
        return _non_negative_int(value, "revision")

    async def write_metadata(
        self,
        row: CFRow,
        *,
        expected_revision: int,
        now: float,
    ) -> int | None:
        """Write metadata and advance revision atomically, or return None if denied."""
        revision = _non_negative_int(expected_revision, "expected_revision")
        timestamp = _non_negative_number(now, "now")
        next_revision = revision + 1

        upsert = self._db.prepare(KNOWLEDGE_WRITER_UPSERT_SQL).bind(
            str(row.knowledge_id),
            row.topic,
            row.content_hash,
            float(row.authority),
            row.status.value,
            row.source_uri,
            float(row.retrieved_at),
            row.r2_blob_key,
            row.vectorize_id,
            revision,
            timestamp,
        )
        advance = self._db.prepare(KNOWLEDGE_CORPUS_ADVANCE_SQL).bind(
            revision,
            row.r2_blob_key,
            timestamp,
        )

        results = await self._db.batch([upsert, advance])
        if len(results) != 2:
            raise RuntimeError("D1 knowledge writer batch returned an unexpected result count")
        for result in results:
            if not _result_success(result):
                raise RuntimeError("D1 knowledge writer batch reported an unsuccessful statement")

        write_changes = _result_changes(results[0])
        revision_changes = _result_changes(results[1])
        if (write_changes, revision_changes) == (0, 0):
            return None
        if (write_changes, revision_changes) != (1, 1):
            raise RuntimeError(
                "D1 knowledge writer batch produced inconsistent mutation counts"
            )
        return next_revision


def _result_success(result: object) -> bool:
    if isinstance(result, Mapping):
        value = result.get("success")
    else:
        value = getattr(result, "success", None)
    if not isinstance(value, bool):
        raise RuntimeError("D1 result does not expose a boolean success field")
    return value


def _result_changes(result: object) -> int:
    if isinstance(result, Mapping):
        meta = result.get("meta")
    else:
        meta = getattr(result, "meta", None)
    if isinstance(meta, Mapping):
        changes = meta.get("changes")
    else:
        changes = getattr(meta, "changes", None)
    return _non_negative_int(changes, "changes")


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _non_negative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


__all__ = ["D1KnowledgeWriter"]
