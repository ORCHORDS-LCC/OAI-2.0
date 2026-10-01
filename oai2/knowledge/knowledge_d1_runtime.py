"""Async D1 writer for authoritative knowledge metadata and corpus revision.

The writer keeps the knowledge_index mutation and corpus revision advance inside
one D1 batch transaction. Both statements use the same expected revision and
deletion-lease exclusion conditions, so a write is accepted only when the
object key is not protected by an active lease.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..core import KnowledgeId, Status
from .abstraction import RetrievalRequest
from .cloudflare import CFRow
from .gc_lease_d1_runtime import D1DatabaseBinding
from .knowledge_d1 import (
    KNOWLEDGE_CORPUS_ADVANCE_SQL,
    KNOWLEDGE_CORPUS_REVISION_SQL,
    KNOWLEDGE_GET_SQL,
    KNOWLEDGE_WRITER_UPSERT_SQL,
    knowledge_query_sql,
    knowledge_schema_statements,
)


class D1KnowledgeReader:
    """Async prepared-statement reader for authoritative D1 knowledge metadata."""

    def __init__(self, database: D1DatabaseBinding) -> None:
        self._db = database

    async def get_row(self, knowledge_id: KnowledgeId) -> CFRow | None:
        statement = self._db.prepare(KNOWLEDGE_GET_SQL).bind(str(knowledge_id))
        result = await statement.run()
        rows = _result_rows(result)
        if not rows:
            return None
        if len(rows) != 1:
            raise RuntimeError("D1 knowledge get returned multiple rows")
        return _row_from_mapping(rows[0])

    async def query_rows(self, request: RetrievalRequest) -> list[CFRow]:
        if not request.topic:
            raise ValueError("retrieval topic must be non-empty")
        if request.limit <= 0:
            raise ValueError("retrieval limit must be positive")
        if not request.include_status:
            return []
        sql = knowledge_query_sql(len(request.include_status))
        bind_values: list[object] = [
            request.topic,
            float(request.min_authority),
            *(status.value for status in request.include_status),
            int(request.limit),
        ]
        result = await self._db.prepare(sql).bind(*bind_values).run()
        return [_row_from_mapping(row) for row in _result_rows(result)]


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


def _result_rows(result: object) -> list[Mapping[str, object]]:
    if isinstance(result, Mapping):
        rows = result.get("results")
    else:
        rows = getattr(result, "results", None)
    if hasattr(rows, "to_py"):
        rows = rows.to_py()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise RuntimeError("D1 result does not expose a row sequence")
    out: list[Mapping[str, object]] = []
    for row in rows:
        if hasattr(row, "to_py"):
            row = row.to_py()
        if not isinstance(row, Mapping):
            raise RuntimeError("D1 result contains a non-mapping row")
        out.append(row)
    return out


def _row_from_mapping(row: Mapping[str, object]) -> CFRow:
    try:
        knowledge_id = row["knowledge_id"]
        topic = row["topic"]
        content_hash = row["content_hash"]
        authority = row["authority"]
        status = row["status"]
        retrieved_at = row["retrieved_at"]
    except KeyError as exc:
        raise RuntimeError(f"D1 knowledge row is missing field: {exc.args[0]}") from exc
    if not isinstance(knowledge_id, str) or not knowledge_id:
        raise RuntimeError("D1 knowledge row has invalid knowledge_id")
    if not isinstance(topic, str) or not topic:
        raise RuntimeError("D1 knowledge row has invalid topic")
    if not isinstance(content_hash, str) or not content_hash:
        raise RuntimeError("D1 knowledge row has invalid content_hash")
    if isinstance(authority, bool) or not isinstance(authority, (int, float)):
        raise RuntimeError("D1 knowledge row has invalid authority")
    if isinstance(retrieved_at, bool) or not isinstance(retrieved_at, (int, float)):
        raise RuntimeError("D1 knowledge row has invalid retrieved_at")
    try:
        lifecycle = Status(str(status))
    except ValueError as exc:
        raise RuntimeError("D1 knowledge row has invalid status") from exc
    source_uri = row.get("source_uri")
    r2_blob_key = row.get("r2_blob_key")
    vectorize_id = row.get("vectorize_id")
    for name, value in (
        ("source_uri", source_uri),
        ("r2_blob_key", r2_blob_key),
        ("vectorize_id", vectorize_id),
    ):
        if value is not None and not isinstance(value, str):
            raise RuntimeError(f"D1 knowledge row has invalid {name}")
    return CFRow(
        knowledge_id=KnowledgeId(knowledge_id),
        topic=topic,
        content_hash=content_hash,
        authority=float(authority),
        status=lifecycle,
        source_uri=source_uri,
        retrieved_at=float(retrieved_at),
        r2_blob_key=r2_blob_key,
        vectorize_id=vectorize_id,
    )


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


__all__ = ["D1KnowledgeReader", "D1KnowledgeWriter"]
