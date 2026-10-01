from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import CFRow
from oai2.knowledge.knowledge_d1 import (
    KNOWLEDGE_CORPUS_ADVANCE_SQL,
    KNOWLEDGE_CORPUS_REVISION_SQL,
    KNOWLEDGE_WRITER_UPSERT_SQL,
)
from oai2.knowledge.knowledge_d1_runtime import D1KnowledgeWriter


@dataclass
class FakeMeta:
    changes: int = 0


@dataclass
class FakeResult:
    success: bool = True
    meta: FakeMeta = field(default_factory=FakeMeta)


@dataclass
class FakeStatement:
    query: str
    first_value: object | None = None
    bound: tuple[object, ...] = ()

    def bind(self, *values: object) -> FakeStatement:
        self.bound = values
        return self

    async def first(self, column_name: str | None = None) -> object | None:
        return self.first_value


class FakeDatabase:
    def __init__(self) -> None:
        self.prepared: list[FakeStatement] = []
        self.first_values: dict[str, object] = {}
        self.batch_results: list[object] = []
        self.batched: list[FakeStatement] = []

    def prepare(self, query: str) -> FakeStatement:
        stmt = FakeStatement(query=query, first_value=self.first_values.get(query))
        self.prepared.append(stmt)
        return stmt

    async def batch(self, statements: list[FakeStatement]) -> list[object]:
        self.batched = list(statements)
        return list(self.batch_results)


def _row() -> CFRow:
    return CFRow(
        knowledge_id=KnowledgeId("ko_writer_1"),
        topic="writer",
        content_hash="a" * 64,
        authority=0.9,
        status=Status.EXPERIMENTAL,
        source_uri="https://example.test/source",
        retrieved_at=10.0,
        r2_blob_key="oai2-blobs/" + "a" * 64,
        vectorize_id="ko_writer_1",
    )


@pytest.mark.asyncio
async def test_corpus_revision_reads_authoritative_singleton() -> None:
    db = FakeDatabase()
    db.first_values[KNOWLEDGE_CORPUS_REVISION_SQL] = 7
    writer = D1KnowledgeWriter(db)
    assert await writer.corpus_revision() == 7


@pytest.mark.asyncio
async def test_write_metadata_batches_upsert_and_revision_advance() -> None:
    db = FakeDatabase()
    db.batch_results = [
        FakeResult(meta=FakeMeta(changes=1)),
        FakeResult(meta=FakeMeta(changes=1)),
    ]
    writer = D1KnowledgeWriter(db)

    assert await writer.write_metadata(_row(), expected_revision=7, now=12.0) == 8
    assert [stmt.query for stmt in db.batched] == [
        KNOWLEDGE_WRITER_UPSERT_SQL,
        KNOWLEDGE_CORPUS_ADVANCE_SQL,
    ]
    assert db.batched[0].bound[-2:] == (7, 12.0)
    assert db.batched[1].bound == (7, _row().r2_blob_key, 12.0)


@pytest.mark.asyncio
async def test_write_metadata_returns_none_when_revision_or_lease_gate_denies() -> None:
    db = FakeDatabase()
    db.batch_results = [
        FakeResult(meta=FakeMeta(changes=0)),
        FakeResult(meta=FakeMeta(changes=0)),
    ]
    writer = D1KnowledgeWriter(db)
    assert await writer.write_metadata(_row(), expected_revision=7, now=12.0) is None


@pytest.mark.asyncio
async def test_write_metadata_fails_closed_on_inconsistent_batch_counts() -> None:
    db = FakeDatabase()
    db.batch_results = [
        FakeResult(meta=FakeMeta(changes=1)),
        FakeResult(meta=FakeMeta(changes=0)),
    ]
    writer = D1KnowledgeWriter(db)
    with pytest.raises(RuntimeError, match="inconsistent mutation counts"):
        await writer.write_metadata(_row(), expected_revision=7, now=12.0)



def test_d1_knowledge_writer_exports() -> None:
    from oai2.knowledge import (
        D1KnowledgeWriter as ExportedWriter,
        KNOWLEDGE_CORPUS_ADVANCE_SQL as ExportedAdvance,
        KNOWLEDGE_CORPUS_REVISION_SQL as ExportedRevision,
        KNOWLEDGE_SCHEMA_VERSION as ExportedVersion,
        KNOWLEDGE_WRITER_UPSERT_SQL as ExportedUpsert,
    )
    from oai2.knowledge.knowledge_d1 import (
        KNOWLEDGE_CORPUS_ADVANCE_SQL,
        KNOWLEDGE_CORPUS_REVISION_SQL,
        KNOWLEDGE_SCHEMA_VERSION,
        KNOWLEDGE_WRITER_UPSERT_SQL,
    )

    assert ExportedWriter is D1KnowledgeWriter
    assert ExportedAdvance is KNOWLEDGE_CORPUS_ADVANCE_SQL
    assert ExportedRevision is KNOWLEDGE_CORPUS_REVISION_SQL
    assert ExportedUpsert is KNOWLEDGE_WRITER_UPSERT_SQL
    assert ExportedVersion == KNOWLEDGE_SCHEMA_VERSION == 1
