from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import CFRow, RetrievalRequest
from oai2.knowledge.gc_lease_d1 import gc_lease_schema_statements
from oai2.knowledge.knowledge_d1 import (
    KNOWLEDGE_CORPUS_ADVANCE_SQL,
    KNOWLEDGE_CORPUS_REVISION_SQL,
    KNOWLEDGE_GET_SQL,
    KNOWLEDGE_WRITER_UPSERT_SQL,
    knowledge_query_sql,
    knowledge_schema_statements,
)
from oai2.knowledge.knowledge_d1_runtime import D1KnowledgeReader, D1KnowledgeWriter


@dataclass
class FakeMeta:
    changes: int = 0


@dataclass
class FakeResult:
    success: bool = True
    meta: FakeMeta = field(default_factory=FakeMeta)
    results: object = field(default_factory=list)


@dataclass
class FakeStatement:
    query: str
    first_value: object | None = None
    run_result: object = field(default_factory=FakeResult)
    bound: tuple[object, ...] = ()

    def bind(self, *values: object) -> FakeStatement:
        self.bound = values
        return self

    async def first(self, column_name: str | None = None) -> object | None:
        return self.first_value

    async def run(self) -> object:
        return self.run_result


class FakeDatabase:
    def __init__(self) -> None:
        self.prepared: list[FakeStatement] = []
        self.first_values: dict[str, object] = {}
        self.batch_results: list[object] = []
        self.run_results: dict[str, object] = {}
        self.batched: list[FakeStatement] = []

    def prepare(self, query: str) -> FakeStatement:
        stmt = FakeStatement(
            query=query,
            first_value=self.first_values.get(query),
            run_result=self.run_results.get(query, FakeResult()),
        )
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
        KNOWLEDGE_CORPUS_ADVANCE_SQL as ExportedAdvance,
    )
    from oai2.knowledge import (
        KNOWLEDGE_CORPUS_REVISION_SQL as ExportedRevision,
    )
    from oai2.knowledge import (
        KNOWLEDGE_SCHEMA_VERSION as ExportedVersion,
    )
    from oai2.knowledge import (
        KNOWLEDGE_WRITER_UPSERT_SQL as ExportedUpsert,
    )
    from oai2.knowledge import (
        D1KnowledgeWriter as ExportedWriter,
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



def test_knowledge_d1_table_and_schema_exports() -> None:
    from oai2.knowledge import (
        KNOWLEDGE_CORPUS_STATE_TABLE,
        KNOWLEDGE_INDEX_TABLE,
        KNOWLEDGE_SCHEMA_SQL,
        knowledge_schema_statements,
    )
    from oai2.knowledge.knowledge_d1 import (
        KNOWLEDGE_CORPUS_STATE_TABLE as SourceCorpusStateTable,
    )
    from oai2.knowledge.knowledge_d1 import (
        KNOWLEDGE_INDEX_TABLE as SourceIndexTable,
    )
    from oai2.knowledge.knowledge_d1 import (
        KNOWLEDGE_SCHEMA_SQL as SourceSchemaSql,
    )
    from oai2.knowledge.knowledge_d1 import (
        knowledge_schema_statements as SourceStatements,
    )

    assert KNOWLEDGE_INDEX_TABLE == SourceIndexTable == "knowledge_index"
    assert (
        KNOWLEDGE_CORPUS_STATE_TABLE
        == SourceCorpusStateTable
        == "knowledge_corpus_state"
    )
    assert KNOWLEDGE_SCHEMA_SQL is SourceSchemaSql
    assert knowledge_schema_statements is SourceStatements



def _row_mapping(
    *,
    knowledge_id: str = "ko_reader_1",
    topic: str = "reader/topic",
    authority: float = 0.8,
    status: str = "EXPERIMENTAL",
) -> dict[str, object]:
    return {
        "knowledge_id": knowledge_id,
        "topic": topic,
        "content_hash": "b" * 64,
        "authority": authority,
        "status": status,
        "source_uri": "https://example.test/read",
        "retrieved_at": 12.0,
        "r2_blob_key": "oai2-blobs/" + "b" * 64,
        "vectorize_id": knowledge_id,
        "corpus_revision": 5,
    }


@pytest.mark.asyncio
async def test_reader_get_row_uses_prepared_run_results() -> None:
    db = FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = FakeResult(results=[_row_mapping()])
    reader = D1KnowledgeReader(db)

    row = await reader.get_row(KnowledgeId("ko_reader_1"))

    assert row is not None
    assert row.knowledge_id == "ko_reader_1"
    assert row.topic == "reader/topic"
    assert row.status is Status.EXPERIMENTAL
    stmt = db.prepared[-1]
    assert stmt.query == KNOWLEDGE_GET_SQL
    assert stmt.bound == ("ko_reader_1",)


@pytest.mark.asyncio
async def test_reader_get_row_returns_none_for_missing_id() -> None:
    db = FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = {"results": []}
    reader = D1KnowledgeReader(db)

    assert await reader.get_row(KnowledgeId("missing")) is None


@pytest.mark.asyncio
async def test_reader_query_rows_binds_topic_authority_statuses_and_limit() -> None:
    db = FakeDatabase()
    request = RetrievalRequest(
        topic="reader",
        limit=3,
        min_authority=0.5,
        include_status=(Status.IMPLEMENTED, Status.EXPERIMENTAL),
    )
    sql = knowledge_query_sql(2)
    db.run_results[sql] = {
        "results": [
            _row_mapping(knowledge_id="ko_a", authority=0.9),
            _row_mapping(knowledge_id="ko_b", authority=0.7),
        ]
    }
    reader = D1KnowledgeReader(db)

    rows = await reader.query_rows(request)

    assert [str(row.knowledge_id) for row in rows] == ["ko_a", "ko_b"]
    stmt = db.prepared[-1]
    assert stmt.query == sql
    assert stmt.bound == (
        "reader",
        0.5,
        "IMPLEMENTED",
        "EXPERIMENTAL",
        3,
    )


@pytest.mark.asyncio
async def test_reader_fails_closed_on_invalid_row_status() -> None:
    db = FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = FakeResult(
        results=[_row_mapping(status="NOT_A_STATUS")]
    )
    reader = D1KnowledgeReader(db)

    with pytest.raises(RuntimeError, match="invalid status"):
        await reader.get_row(KnowledgeId("ko_reader_1"))


def test_knowledge_query_sql_rejects_empty_status_set() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        knowledge_query_sql(0)


# ---------------------------------------------------------------------------
# #205 REQ-CFOPS-015 — the D1 CONTRACT, not fake orchestration.
#
# The orchestration tests elsewhere use a write RECORDER, which proves the same
# identity was targeted but NOT that the authoritative table holds one row.
# These execute the real KNOWLEDGE_WRITER_UPSERT_SQL / CORPUS_ADVANCE_SQL
# against a real SQLite database, following the existing sqlite harness in
# tests/test_gc_lease_d1.py, so persistence cardinality is established rather
# than inferred.
# ---------------------------------------------------------------------------


def _authoritative_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    for statement in knowledge_schema_statements():
        db.execute(statement)
    for statement in gc_lease_schema_statements():
        db.execute(statement)
    # The schema already creates the corpus-state singleton, so seed it rather
    # than inserting a second row.
    db.execute(
        "INSERT INTO knowledge_corpus_state(singleton, revision) VALUES (1, 4) "
        "ON CONFLICT(singleton) DO UPDATE SET revision = 4"
    )
    return db


def _upsert(
    db: sqlite3.Connection,
    *,
    knowledge_id: str,
    content_hash: str,
    expected_revision: int,
) -> bool:
    # The SQL uses anonymous numbered parameters (?1..?n), so sqlite3 binds
    # them positionally.
    db.execute(
        KNOWLEDGE_WRITER_UPSERT_SQL,
        (
            knowledge_id,
            "runtime",
            content_hash,
            0.9,
            "experimental",
            "https://example.test/source",
            10.0,
            f"oai2-blobs/{content_hash}",
            knowledge_id,
            expected_revision,
            0,
        ),
    )
    db.execute(
        KNOWLEDGE_CORPUS_ADVANCE_SQL,
        (expected_revision, f"oai2-blobs/{content_hash}", 0),
    )
    db.commit()
    return (
        db.execute(
            "SELECT COUNT(*) FROM knowledge_index WHERE knowledge_id = ?",
            (knowledge_id,),
        ).fetchone()[0]
        == 1
    )


def test_repeated_upsert_leaves_exactly_one_authoritative_row() -> None:
    """D1 CONTRACT: repeated upsert of the same knowledge_id is one row."""
    db = _authoritative_db()

    assert _upsert(db, knowledge_id="ko_d1", content_hash="a" * 64, expected_revision=4)
    assert _upsert(db, knowledge_id="ko_d1", content_hash="a" * 64, expected_revision=5)
    assert _upsert(db, knowledge_id="ko_d1", content_hash="a" * 64, expected_revision=6)

    rows = db.execute(
        "SELECT knowledge_id, content_hash, corpus_revision FROM knowledge_index"
    ).fetchall()
    assert len(rows) == 1, f"expected one authoritative row, got {rows}"
    assert rows[0][0] == "ko_d1"
    # The latest content wins; the row is replaced, never duplicated.
    assert rows[0][1] == "a" * 64
    assert rows[0][2] == 7, "the row records the revision it was written at"
    db.close()


def test_upsert_replaces_content_hash_rather_than_adding_a_row() -> None:
    """A changed body under a stable id updates in place."""
    db = _authoritative_db()

    _upsert(db, knowledge_id="ko_d1", content_hash="a" * 64, expected_revision=4)
    _upsert(db, knowledge_id="ko_d1", content_hash="b" * 64, expected_revision=5)

    rows = db.execute(
        "SELECT content_hash FROM knowledge_index WHERE knowledge_id = 'ko_d1'"
    ).fetchall()
    assert rows == [("b" * 64,)], "content change must update the single row"
    db.close()


def test_distinct_ids_produce_distinct_rows() -> None:
    """The control: the keying is real, so a different id is a different row."""
    db = _authoritative_db()

    _upsert(db, knowledge_id="ko_one", content_hash="a" * 64, expected_revision=4)
    _upsert(db, knowledge_id="ko_two", content_hash="b" * 64, expected_revision=5)

    count = db.execute("SELECT COUNT(*) FROM knowledge_index").fetchone()[0]
    assert count == 2, "distinct knowledge_ids must be distinct authoritative rows"
    db.close()


def test_a_stale_expected_revision_is_denied_and_writes_nothing() -> None:
    """A conflicting/stale revision must not claim the request committed."""
    db = _authoritative_db()

    _upsert(db, knowledge_id="ko_d1", content_hash="a" * 64, expected_revision=4)
    assert db.execute(
        "SELECT revision FROM knowledge_corpus_state WHERE singleton = 1"
    ).fetchone()[0] == 5, "one successful upsert advanced the corpus by one"

    # Expected revision 4 is now stale: the corpus is at 5.
    _upsert(db, knowledge_id="ko_d1", content_hash="c" * 64, expected_revision=4)

    rows = db.execute(
        "SELECT content_hash FROM knowledge_index WHERE knowledge_id = 'ko_d1'"
    ).fetchall()
    assert rows == [("a" * 64,)], "a stale-revision write must not overwrite the row"
    revision = db.execute(
        "SELECT revision FROM knowledge_corpus_state WHERE singleton = 1"
    ).fetchone()[0]
    assert revision == 5, "a denied write must not advance the corpus revision"
    db.close()
