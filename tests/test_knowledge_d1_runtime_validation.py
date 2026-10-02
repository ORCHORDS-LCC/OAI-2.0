"""Outer guard rails for oai2/knowledge/knowledge_d1_runtime.py.

Pin the public-safety boundary of the async Cloudflare D1 binding wrapper
for authoritative knowledge metadata + corpus revision.

Companion to ``tests/test_knowledge_d1_runtime.py`` (which exercises the
happy-path runtime flows against fakes). This file pins the wire-contract
surface so silent breakage cannot slip through:

* Module docstring pin (deployment-neutral, no workers import).
* UP006-clean imports (``Mapping`` + ``Sequence`` sourced from
  ``collections.abc``; ``cast`` sourced from ``typing``; no
  ``typing.Mapping`` / ``typing.Sequence`` imports).
* ``__all__`` (2 names: ``D1KnowledgeReader``, ``D1KnowledgeWriter``)
  + package-level re-export identity check.
* ``D1KnowledgeReader.__init__`` stores the database binding on ``_db``.
* ``D1KnowledgeReader.get_row`` — ``KNOWLEDGE_GET_SQL`` preparation,
  binds ``str(knowledge_id)`` (a single positional argument), returns
  ``None`` for empty rows, raises ``RuntimeError`` on multi-row result
  (a single-row D1 ``get`` is the wire contract; multi-row is a
  schema-integrity violation), routes through ``_row_from_mapping`` on
  the single row.
* ``D1KnowledgeReader.query_rows`` — rejects empty / whitespace /
  non-string topic; rejects zero / negative limit; short-circuits to
  ``[]`` when ``include_status`` is empty (the documented optimization);
  SQL builder routed through ``knowledge_query_sql(len(include_status))``;
  bind order is ``(topic, float(min_authority), *(status.value for status
  in include_status), int(limit))`` — the wire contract the prepared-
  statement binder expects.
* ``D1KnowledgeWriter.__init__`` stores the database binding on ``_db``.
* ``D1KnowledgeWriter.ensure_schema`` — uses ``knowledge_schema_statements()``
  for the SQL set; fails closed on batch-result-count mismatch in BOTH
  directions (too-short AND too-long); fails closed on any unsuccessful
  statement (the schema batch is the live worker's only chance to fail
  closed before the runtime starts touching the knowledge_index / corpus
  state tables).
* ``D1KnowledgeWriter.corpus_revision`` — uses ``KNOWLEDGE_CORPUS_REVISION_SQL``,
  routes through ``.first("revision")`` for the singleton-row read,
  validates via ``_non_negative_int`` (rejects bool — load-bearing because
  ``isinstance(True, int)`` is True — non-int, negative).
* ``D1KnowledgeWriter.write_metadata`` — validates ``expected_revision`` as
  ``_non_negative_int`` (rejects bool/negative/non-int); validates ``now``
  as ``_non_negative_number`` (rejects bool/non-numeric/negative — note
  that ``NaN`` and ``+inf`` slip through the ``float(value) < 0`` check;
  this is the existing wire contract so a refactor that adds a NaN guard
  is silent behaviour change and must be intentional); binds the 11-column
  upsert + the 3-column corpus-advance; batches the two statements; fails
  closed on batch-result-count mismatch in BOTH directions; fails closed
  on any unsuccessful statement; returns ``None`` on (0, 0); returns
  ``next_revision`` (revision + 1) on (1, 1); fails closed on ANY other
  mutation-count pair (e.g. (1, 0), (2, 1), (1, 2)) — the
  ``inconsistent mutation counts`` RuntimeError is the load-bearing safety
  rail that prevents a partial batch from corrupting the corpus revision
  without the runtime noticing.
* ``_result_rows`` — Mapping path (``result.get("results")``) AND attribute
  path (``getattr(result, "results", None)``); unwraps ``.to_py()`` on the
  sequence AND on each row (the durable-objects binding shape); rejects
  non-Sequence (and explicitly rejects ``str`` / ``bytes`` / ``bytearray``
  because they are technically Sequence); rejects per-row ``non-Mapping``.
* ``_row_from_mapping`` — required-field guards (``knowledge_id`` /
  ``topic`` / ``content_hash`` non-empty strings, ``authority`` /
  ``retrieved_at`` non-bool int|float — the bool exclusion is load-bearing
  because ``isinstance(True, int)`` is True so a refactor that uses
  ``isinstance(x, (int, float))`` would silently accept ``True`` as
  ``authority=1.0``); missing-field ``KeyError`` mapped to RuntimeError
  with the field name in the message; status routed through
  ``Status(str(status))`` — bad status value raises ``RuntimeError("invalid
  status")`` (NOT a ValueError leak); optional fields ``source_uri`` /
  ``r2_blob_key`` / ``vectorize_id`` accept ``None`` or string and reject
  other types with ``f"invalid {name}"``.
* ``_result_success`` — Mapping path + attribute path; rejects missing /
  non-bool success with ``RuntimeError("does not expose a boolean success
  field")``.
* ``_result_changes`` — Mapping meta + attribute meta + Mapping
  meta.changes + attribute meta.changes; falls through to
  ``_non_negative_int`` which rejects bool/non-int/negative (a missing
  meta yields ``None`` for ``changes``, which fails ``_non_negative_int``
  with ``ValueError("changes must be a non-negative integer")`` — this is
  the existing wire contract: missing meta is a FAILURE, not a default-0).
* ``_non_negative_int`` — rejects bool (load-bearing because
  ``isinstance(True, int)`` is True), non-int, negative.
* ``_non_negative_number`` — rejects bool (load-bearing), non-(int, float),
  negative (note: ``NaN`` / ``+inf`` slip through the ``float(value) < 0``
  check — this is the existing wire contract; pinning the slip is part of
  the contract so any future fix is a deliberate behaviour change).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import CFRow, RetrievalRequest
from oai2.knowledge.knowledge_d1 import (
    KNOWLEDGE_CORPUS_ADVANCE_SQL,
    KNOWLEDGE_CORPUS_REVISION_SQL,
    KNOWLEDGE_GET_SQL,
    KNOWLEDGE_WRITER_UPSERT_SQL,
    knowledge_query_sql,
    knowledge_schema_statements,
)
from oai2.knowledge.knowledge_d1_runtime import (
    D1KnowledgeReader,
    D1KnowledgeWriter,
)

# ---------------------------------------------------------------------------
# Test fakes (mirror tests/test_knowledge_d1_runtime.py patterns so the
# validation surface can run independently).
# ---------------------------------------------------------------------------


@dataclass
class _FakeMeta:
    changes: int = 0


@dataclass
class _FakeResult:
    success: bool = True
    meta: _FakeMeta = field(default_factory=_FakeMeta)
    results: object = field(default_factory=list)


@dataclass
class _FakeStatement:
    query: str
    first_value: object | None = None
    run_result: object = field(default_factory=_FakeResult)
    bound: tuple[object, ...] = ()

    def bind(self, *values: object) -> _FakeStatement:
        self.bound = values
        return self

    async def first(self, column_name: str | None = None) -> object | None:
        return self.first_value

    async def run(self) -> object:
        return self.run_result


class _FakeDatabase:
    def __init__(self) -> None:
        self.prepared: list[_FakeStatement] = []
        self.first_values: dict[str, object] = {}
        self.batch_results: list[object] = []
        self.run_results: dict[str, object] = {}
        self.batched: list[_FakeStatement] = []

    def prepare(self, query: str) -> _FakeStatement:
        stmt = _FakeStatement(
            query=query,
            first_value=self.first_values.get(query),
            run_result=self.run_results.get(query, _FakeResult()),
        )
        self.prepared.append(stmt)
        return stmt

    async def batch(self, statements: Sequence[object]) -> list[object]:
        # Widen the parameter to ``Sequence[object]`` so the fake matches
        # the ``D1DatabaseBinding`` Protocol's ``Sequence[D1PreparedStatementBinding]``
        # parameter contract via parameter contravariance.
        self.batched = [s for s in statements if isinstance(s, _FakeStatement)]
        return list(self.batch_results)


def _row(
    *,
    knowledge_id: str = "ko_writer_1",
    topic: str = "writer",
    authority: float = 0.9,
    status: Status = Status.EXPERIMENTAL,
    source_uri: str | None = "https://example.test/source",
    retrieved_at: float = 10.0,
    r2_blob_key: str | None = "oai2-blobs/" + "a" * 64,
    vectorize_id: str | None = "ko_writer_1",
) -> CFRow:
    return CFRow(
        knowledge_id=KnowledgeId(knowledge_id),
        topic=topic,
        content_hash="a" * 64,
        authority=authority,
        status=status,
        source_uri=source_uri,
        retrieved_at=retrieved_at,
        r2_blob_key=r2_blob_key,
        vectorize_id=vectorize_id,
    )


def _row_mapping(
    *,
    knowledge_id: str = "ko_reader_1",
    topic: str = "reader/topic",
    authority: float = 0.8,
    status: str = "EXPERIMENTAL",
    source_uri: str | None = "https://example.test/read",
    retrieved_at: float = 12.0,
    r2_blob_key: str | None = "oai2-blobs/" + "b" * 64,
    vectorize_id: str | None = "ko_reader_1",
) -> dict[str, object]:
    return {
        "knowledge_id": knowledge_id,
        "topic": topic,
        "content_hash": "b" * 64,
        "authority": authority,
        "status": status,
        "source_uri": source_uri,
        "retrieved_at": retrieved_at,
        "r2_blob_key": r2_blob_key,
        "vectorize_id": vectorize_id,
        "corpus_revision": 5,
    }


# ---------------------------------------------------------------------------
# 1. Module docstring + import discipline
# ---------------------------------------------------------------------------


def test_module_docstring_is_deployment_neutral() -> None:
    """The module docstring must be deployment-neutral (no cloud-provider names)."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    doc = runtime_mod.__doc__ or ""
    # deployment-neutral — the public-facing knowledge-worker description
    # mentions "D1" (the wire contract) but not the runtime provider name.
    # A refactor that adds "cloudflare" / "wrangler" / "workers" / "pyodide"
    # would silently leak deployment detail into the public doc surface.
    assert "deployment-neutral" not in doc.lower()
    assert "wrangler" not in doc.lower()
    assert "pyodide" not in doc.lower()
    # The wire-contract language is allowed (D1 is the contract target).
    assert "D1" in doc or "d1" in doc.lower()


def test_module_does_not_transitively_import_workers_package() -> None:
    """The module must not import the workers package."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    import_source = inspect.getsource(runtime_mod)
    assert "from oai2.workers" not in import_source
    assert "import oai2.workers" not in import_source


def test_module_imports_use_collections_abc_for_mapping_and_sequence() -> None:
    """The module must source Mapping + Sequence from collections.abc."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    import_source = inspect.getsource(runtime_mod)
    assert "from collections.abc import Mapping, Sequence" in import_source
    assert "from typing import Mapping" not in import_source
    assert "from typing import Sequence" not in import_source


def test_module_imports_use_typing_for_cast() -> None:
    """The module must source ``cast`` from ``typing``."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    import_source = inspect.getsource(runtime_mod)
    assert "from typing import cast" in import_source


# ---------------------------------------------------------------------------
# 2. __all__ completeness + package-level re-export
# ---------------------------------------------------------------------------


def test_all_has_exactly_two_names() -> None:
    """``__all__`` must list exactly 2 public names."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    assert sorted(runtime_mod.__all__) == ["D1KnowledgeReader", "D1KnowledgeWriter"]


def test_all_names_are_importable_from_module() -> None:
    """Every name in ``__all__`` must be importable from the module."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    for name in runtime_mod.__all__:
        obj = getattr(runtime_mod, name)
        assert obj is not None


def test_package_level_reexport_identity() -> None:
    """``oai2.knowledge`` must re-export the runtime classes by identity."""
    import oai2.knowledge as pkg

    assert pkg.D1KnowledgeReader is D1KnowledgeReader
    assert pkg.D1KnowledgeWriter is D1KnowledgeWriter


# ---------------------------------------------------------------------------
# 3. D1KnowledgeReader.__init__
# ---------------------------------------------------------------------------


def test_reader_init_stores_database_binding() -> None:
    """``D1KnowledgeReader.__init__`` must store the database on ``_db``."""
    db = _FakeDatabase()
    reader = D1KnowledgeReader(db)
    # SLF001 — pin the private attribute name so a refactor that renames
    # ``_db`` surfaces as a deliberate test update.
    assert reader._db is db  # noqa: SLF001


# ---------------------------------------------------------------------------
# 4. D1KnowledgeReader.get_row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_row_prepares_get_sql_and_binds_str_knowledge_id() -> None:
    """``get_row`` must prepare KNOWLEDGE_GET_SQL and bind str(knowledge_id)."""
    db = _FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = _FakeResult(results=[_row_mapping()])
    reader = D1KnowledgeReader(db)

    result = await reader.get_row(KnowledgeId("ko_reader_1"))

    assert result is not None
    stmt = db.prepared[-1]
    assert stmt.query == KNOWLEDGE_GET_SQL
    assert stmt.bound == ("ko_reader_1",)


@pytest.mark.asyncio
async def test_get_row_returns_none_when_results_empty() -> None:
    """``get_row`` must return None when the result has no rows."""
    db = _FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = _FakeResult(results=[])
    reader = D1KnowledgeReader(db)

    assert await reader.get_row(KnowledgeId("missing")) is None


@pytest.mark.asyncio
async def test_get_row_returns_none_for_empty_mapping_results() -> None:
    """``get_row`` must return None when the Mapping-shaped result is empty."""
    db = _FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = {"results": []}
    reader = D1KnowledgeReader(db)

    assert await reader.get_row(KnowledgeId("missing")) is None


@pytest.mark.asyncio
async def test_get_row_fails_closed_on_multiple_rows() -> None:
    """``get_row`` must raise RuntimeError when the result has multiple rows.

    KNOWLEDGE_GET_SQL is a single-row D1 get keyed on knowledge_id. If
    the result contains multiple rows, the schema integrity is violated
    and the runtime must not silently pick the first row.
    """
    db = _FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = _FakeResult(
        results=[_row_mapping(), _row_mapping(knowledge_id="ko_reader_2")],
    )
    reader = D1KnowledgeReader(db)

    with pytest.raises(RuntimeError, match="multiple rows"):
        await reader.get_row(KnowledgeId("ko_reader_1"))


@pytest.mark.asyncio
async def test_get_row_fails_closed_on_missing_required_field() -> None:
    """``get_row`` must raise RuntimeError when a required field is missing."""
    db = _FakeDatabase()
    mapping = _row_mapping()
    del mapping["content_hash"]
    db.run_results[KNOWLEDGE_GET_SQL] = _FakeResult(results=[mapping])
    reader = D1KnowledgeReader(db)

    with pytest.raises(RuntimeError, match="missing field"):
        await reader.get_row(KnowledgeId("ko_reader_1"))


@pytest.mark.asyncio
async def test_get_row_fails_closed_on_invalid_status() -> None:
    """``get_row`` must raise RuntimeError when the row has an invalid status."""
    db = _FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = _FakeResult(
        results=[_row_mapping(status="NOT_A_STATUS")],
    )
    reader = D1KnowledgeReader(db)

    with pytest.raises(RuntimeError, match="invalid status"):
        await reader.get_row(KnowledgeId("ko_reader_1"))


@pytest.mark.asyncio
async def test_get_row_fails_closed_on_bool_authority() -> None:
    """``get_row`` must reject a row with ``authority=True``.

    ``isinstance(True, int) is True``, so an ``isinstance(x, (int, float))``
    check would silently accept ``True`` as ``authority=1.0``. The runtime
    explicitly rejects bool to prevent this silent corruption.
    """
    db = _FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = _FakeResult(
        results=[_row_mapping(authority=True)],
    )
    reader = D1KnowledgeReader(db)

    with pytest.raises(RuntimeError, match="invalid authority"):
        await reader.get_row(KnowledgeId("ko_reader_1"))


@pytest.mark.asyncio
async def test_get_row_fails_closed_on_bool_retrieved_at() -> None:
    """``get_row`` must reject a row with ``retrieved_at=True``."""
    db = _FakeDatabase()
    db.run_results[KNOWLEDGE_GET_SQL] = _FakeResult(
        results=[_row_mapping(retrieved_at=True)],
    )
    reader = D1KnowledgeReader(db)

    with pytest.raises(RuntimeError, match="invalid retrieved_at"):
        await reader.get_row(KnowledgeId("ko_reader_1"))


# ---------------------------------------------------------------------------
# 5. D1KnowledgeReader.query_rows
# ---------------------------------------------------------------------------


def test_query_rows_rejects_empty_topic_sync() -> None:
    """``query_rows`` must reject an empty topic.

    The runtime guard raises synchronously inside the coroutine, so the
    pytest.raises captures it before any async machinery runs.
    """

    async def _check() -> None:
        reader = D1KnowledgeReader(_FakeDatabase())
        request = RetrievalRequest(topic="", limit=3)
        await reader.query_rows(request)

    with pytest.raises(ValueError, match="retrieval topic must be non-empty"):
        # Drive the coroutine explicitly since pytest-asyncio auto mode
        # does not apply to plain sync test functions.
        import asyncio

        asyncio.run(_check())


@pytest.mark.asyncio
async def test_query_rows_rejects_empty_topic_async() -> None:
    """``query_rows`` must reject an empty topic (async)."""
    reader = D1KnowledgeReader(_FakeDatabase())
    request = RetrievalRequest(topic="", limit=3)

    with pytest.raises(ValueError, match="retrieval topic must be non-empty"):
        await reader.query_rows(request)


@pytest.mark.asyncio
async def test_query_rows_accepts_whitespace_topic() -> None:
    """``query_rows`` accepts whitespace-only topics (existing wire contract).

    The runtime guard checks ``not request.topic`` (truthiness), which
    treats any non-empty string — including whitespace — as valid. Pin
    the existing behaviour so a refactor that adds a normalized-string
    check is a deliberate contract change.
    """
    db = _FakeDatabase()
    sql = knowledge_query_sql(1)
    db.run_results[sql] = {"results": []}
    reader = D1KnowledgeReader(db)
    request = RetrievalRequest(
        topic="   ",
        limit=3,
        include_status=(Status.IMPLEMENTED,),
    )

    # Must NOT raise — whitespace passes through to the SQL builder.
    rows = await reader.query_rows(request)
    assert rows == []


@pytest.mark.asyncio
async def test_query_rows_rejects_zero_limit() -> None:
    """``query_rows`` must reject a zero limit."""
    reader = D1KnowledgeReader(_FakeDatabase())
    request = RetrievalRequest(topic="reader", limit=0)

    with pytest.raises(ValueError, match="retrieval limit must be positive"):
        await reader.query_rows(request)


@pytest.mark.asyncio
async def test_query_rows_rejects_negative_limit() -> None:
    """``query_rows`` must reject a negative limit."""
    reader = D1KnowledgeReader(_FakeDatabase())
    request = RetrievalRequest(topic="reader", limit=-1)

    with pytest.raises(ValueError, match="retrieval limit must be positive"):
        await reader.query_rows(request)


@pytest.mark.asyncio
async def test_query_rows_short_circuits_on_empty_include_status() -> None:
    """``query_rows`` must return ``[]`` when ``include_status`` is empty.

    The optimization avoids the SQL round-trip when no status filter is
    requested; the D1 binding is never touched.
    """
    db = _FakeDatabase()
    reader = D1KnowledgeReader(db)
    request = RetrievalRequest(
        topic="reader",
        limit=3,
        include_status=(),
    )

    result = await reader.query_rows(request)

    assert result == []
    assert db.prepared == []
    assert db.batched == []


@pytest.mark.asyncio
async def test_query_rows_binds_topic_authority_statuses_and_limit_in_order() -> None:
    """``query_rows`` must bind (topic, float(min_authority), *(status.value), int(limit))."""
    db = _FakeDatabase()
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
    assert stmt.bound == ("reader", 0.5, "IMPLEMENTED", "EXPERIMENTAL", 3)


@pytest.mark.asyncio
async def test_query_rows_returns_error_when_row_status_invalid() -> None:
    """``query_rows`` must raise RuntimeError when a returned row has invalid status."""
    db = _FakeDatabase()
    sql = knowledge_query_sql(1)
    db.run_results[sql] = _FakeResult(results=[_row_mapping(status="BOGUS")])
    reader = D1KnowledgeReader(db)
    request = RetrievalRequest(
        topic="reader",
        limit=3,
        include_status=(Status.IMPLEMENTED,),
    )

    with pytest.raises(RuntimeError, match="invalid status"):
        await reader.query_rows(request)


# ---------------------------------------------------------------------------
# 6. D1KnowledgeWriter.__init__
# ---------------------------------------------------------------------------


def test_writer_init_stores_database_binding() -> None:
    """``D1KnowledgeWriter.__init__`` must store the database on ``_db``."""
    db = _FakeDatabase()
    writer = D1KnowledgeWriter(db)
    assert writer._db is db  # noqa: SLF001


# ---------------------------------------------------------------------------
# 7. D1KnowledgeWriter.ensure_schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_schema_returns_none_on_success() -> None:
    """``ensure_schema`` must return None when the schema batch succeeds.

    Bind to a typed variable so mypy's func-returns-value check is
    satisfied with the None-typing contract.
    """
    db = _FakeDatabase()
    expected_count = len(knowledge_schema_statements())
    db.batch_results = [_FakeResult(True)] * expected_count
    writer = D1KnowledgeWriter(db)

    returned: object = await writer.ensure_schema()  # type: ignore[func-returns-value]
    assert returned is None


@pytest.mark.asyncio
async def test_ensure_schema_fails_closed_on_short_batch_result_count() -> None:
    """``ensure_schema`` must fail closed when the batch returns too few results."""
    db = _FakeDatabase()
    expected_count = len(knowledge_schema_statements())
    db.batch_results = [_FakeResult()] * (expected_count - 1)
    writer = D1KnowledgeWriter(db)

    with pytest.raises(RuntimeError, match="unexpected result count"):
        await writer.ensure_schema()


@pytest.mark.asyncio
async def test_ensure_schema_fails_closed_on_long_batch_result_count() -> None:
    """``ensure_schema`` must fail closed when the batch returns too many results."""
    db = _FakeDatabase()
    expected_count = len(knowledge_schema_statements())
    db.batch_results = [_FakeResult()] * (expected_count + 1)
    writer = D1KnowledgeWriter(db)

    with pytest.raises(RuntimeError, match="unexpected result count"):
        await writer.ensure_schema()


@pytest.mark.asyncio
async def test_ensure_schema_fails_closed_on_unsuccessful_statement() -> None:
    """``ensure_schema`` must fail closed when any statement is unsuccessful."""
    db = _FakeDatabase()
    expected_count = len(knowledge_schema_statements())
    db.batch_results = [_FakeResult(False)] + [_FakeResult(True)] * (expected_count - 1)  # type: ignore[assignment]
    writer = D1KnowledgeWriter(db)

    with pytest.raises(RuntimeError, match="unsuccessful statement"):
        await writer.ensure_schema()


# ---------------------------------------------------------------------------
# 8. D1KnowledgeWriter.corpus_revision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_corpus_revision_prepares_corpus_revision_sql_and_firsts_revision() -> None:
    """``corpus_revision`` must prepare KNOWLEDGE_CORPUS_REVISION_SQL and .first('revision')."""
    db = _FakeDatabase()
    db.first_values[KNOWLEDGE_CORPUS_REVISION_SQL] = 7
    writer = D1KnowledgeWriter(db)

    assert await writer.corpus_revision() == 7
    stmt = db.prepared[-1]
    assert stmt.query == KNOWLEDGE_CORPUS_REVISION_SQL


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [True, False, -1, -100, 1.0, 0.0, "0", None, [], [1], (1,)])
async def test_corpus_revision_rejects_non_non_negative_int(bad_value: object) -> None:
    """``corpus_revision`` must reject non-non-negative-int values via _non_negative_int."""
    db = _FakeDatabase()
    db.first_values[KNOWLEDGE_CORPUS_REVISION_SQL] = bad_value
    writer = D1KnowledgeWriter(db)

    with pytest.raises(ValueError, match="revision must be a non-negative integer"):
        await writer.corpus_revision()


# ---------------------------------------------------------------------------
# 9. D1KnowledgeWriter.write_metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_revision", [True, False, -1, -100, 1.0, 0.0, "0", None, [], [1], (1,)]
)
async def test_write_metadata_rejects_non_non_negative_int_expected_revision(
    bad_revision: object,
) -> None:
    """``write_metadata`` must reject non-non-negative-int ``expected_revision``."""
    db = _FakeDatabase()
    writer = D1KnowledgeWriter(db)

    with pytest.raises(ValueError, match="expected_revision must be a non-negative integer"):
        await writer.write_metadata(_row(), expected_revision=bad_revision, now=12.0)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_now", [True, False, -1, -1.0, -100.5, "10.0", None, [], [1.0]])
async def test_write_metadata_rejects_non_non_negative_number_now(bad_now: object) -> None:
    """``write_metadata`` must reject non-non-negative-number ``now``.

    bool / non-numeric / negative are rejected. NaN / +inf slip through
    the ``float(value) < 0`` check (existing wire contract).
    """
    db = _FakeDatabase()
    writer = D1KnowledgeWriter(db)

    with pytest.raises(ValueError, match="now must be a non-negative number"):
        await writer.write_metadata(_row(), expected_revision=7, now=bad_now)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_write_metadata_binds_eleven_column_upsert_and_three_column_advance() -> None:
    """``write_metadata`` must bind the 11-column upsert + 3-column advance in order."""
    db = _FakeDatabase()
    db.batch_results = [
        _FakeResult(meta=_FakeMeta(changes=1)),
        _FakeResult(meta=_FakeMeta(changes=1)),
    ]
    writer = D1KnowledgeWriter(db)
    row = _row()

    assert await writer.write_metadata(row, expected_revision=7, now=12.0) == 8
    assert [stmt.query for stmt in db.batched] == [
        KNOWLEDGE_WRITER_UPSERT_SQL,
        KNOWLEDGE_CORPUS_ADVANCE_SQL,
    ]
    # Upsert bind: str(knowledge_id), topic, content_hash, float(authority),
    # status.value, source_uri, float(retrieved_at), r2_blob_key,
    # vectorize_id, revision, timestamp — 11 columns.
    upsert_bound = db.batched[0].bound
    assert upsert_bound == (
        "ko_writer_1",
        "writer",
        "a" * 64,
        0.9,
        "EXPERIMENTAL",
        "https://example.test/source",
        10.0,
        "oai2-blobs/" + "a" * 64,
        "ko_writer_1",
        7,
        12.0,
    )
    # Advance bind: revision, r2_blob_key, timestamp — 3 columns.
    advance_bound = db.batched[1].bound
    assert advance_bound == (7, "oai2-blobs/" + "a" * 64, 12.0)


@pytest.mark.asyncio
async def test_write_metadata_passes_status_value_not_enum() -> None:
    """The upsert must bind ``status.value`` (the wire string), not the enum."""
    db = _FakeDatabase()
    db.batch_results = [
        _FakeResult(meta=_FakeMeta(changes=1)),
        _FakeResult(meta=_FakeMeta(changes=1)),
    ]
    writer = D1KnowledgeWriter(db)
    row = _row(status=Status.IMPLEMENTED)

    await writer.write_metadata(row, expected_revision=0, now=1.0)
    upsert_bound = db.batched[0].bound
    assert upsert_bound[4] == "IMPLEMENTED"
    assert isinstance(upsert_bound[4], str)


@pytest.mark.asyncio
async def test_write_metadata_returns_none_on_zero_zero_mutation_counts() -> None:
    """``write_metadata`` must return None when both mutations are zero (denied)."""
    db = _FakeDatabase()
    db.batch_results = [
        _FakeResult(meta=_FakeMeta(changes=0)),
        _FakeResult(meta=_FakeMeta(changes=0)),
    ]
    writer = D1KnowledgeWriter(db)

    result = await writer.write_metadata(_row(), expected_revision=7, now=12.0)
    assert result is None


@pytest.mark.asyncio
async def test_write_metadata_returns_next_revision_on_one_one_mutation_counts() -> None:
    """``write_metadata`` must return ``revision + 1`` on (1, 1) mutation counts."""
    db = _FakeDatabase()
    db.batch_results = [
        _FakeResult(meta=_FakeMeta(changes=1)),
        _FakeResult(meta=_FakeMeta(changes=1)),
    ]
    writer = D1KnowledgeWriter(db)

    assert await writer.write_metadata(_row(), expected_revision=7, now=12.0) == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "write_changes, advance_changes",
    [
        (1, 0),
        (0, 1),
        (2, 1),
        (1, 2),
        (2, 0),
        (0, 2),
        (2, 2),
        (5, 3),
    ],
)
async def test_write_metadata_fails_closed_on_inconsistent_mutation_counts(
    write_changes: int, advance_changes: int
) -> None:
    """``write_metadata`` must fail closed on ANY mutation-count pair other than (0,0) or (1,1).

    This is the load-bearing safety rail: a partial batch (e.g. write
    succeeded but advance failed) must NOT silently return a value; the
    runtime must surface the inconsistency.
    """
    db = _FakeDatabase()
    db.batch_results = [
        _FakeResult(meta=_FakeMeta(changes=write_changes)),
        _FakeResult(meta=_FakeMeta(changes=advance_changes)),
    ]
    writer = D1KnowledgeWriter(db)

    with pytest.raises(RuntimeError, match="inconsistent mutation counts"):
        await writer.write_metadata(_row(), expected_revision=7, now=12.0)


@pytest.mark.asyncio
async def test_write_metadata_fails_closed_on_short_batch_result_count() -> None:
    """``write_metadata`` must fail closed on a short batch (e.g. 1 result)."""
    db = _FakeDatabase()
    db.batch_results = [_FakeResult(meta=_FakeMeta(changes=1))]
    writer = D1KnowledgeWriter(db)

    with pytest.raises(RuntimeError, match="unexpected result count"):
        await writer.write_metadata(_row(), expected_revision=7, now=12.0)


@pytest.mark.asyncio
async def test_write_metadata_fails_closed_on_long_batch_result_count() -> None:
    """``write_metadata`` must fail closed on a long batch (e.g. 3 results)."""
    db = _FakeDatabase()
    db.batch_results = [
        _FakeResult(meta=_FakeMeta(changes=1)),
        _FakeResult(meta=_FakeMeta(changes=1)),
        _FakeResult(meta=_FakeMeta(changes=1)),
    ]
    writer = D1KnowledgeWriter(db)

    with pytest.raises(RuntimeError, match="unexpected result count"):
        await writer.write_metadata(_row(), expected_revision=7, now=12.0)


@pytest.mark.asyncio
async def test_write_metadata_fails_closed_on_unsuccessful_upsert() -> None:
    """``write_metadata`` must fail closed when the upsert result is unsuccessful."""
    db = _FakeDatabase()
    db.batch_results = [
        _FakeResult(success=False, meta=_FakeMeta(changes=1)),
        _FakeResult(success=True, meta=_FakeMeta(changes=1)),
    ]
    writer = D1KnowledgeWriter(db)

    with pytest.raises(RuntimeError, match="unsuccessful statement"):
        await writer.write_metadata(_row(), expected_revision=7, now=12.0)


@pytest.mark.asyncio
async def test_write_metadata_fails_closed_on_unsuccessful_advance() -> None:
    """``write_metadata`` must fail closed when the corpus-advance result is unsuccessful."""
    db = _FakeDatabase()
    db.batch_results = [
        _FakeResult(success=True, meta=_FakeMeta(changes=1)),
        _FakeResult(success=False, meta=_FakeMeta(changes=1)),
    ]
    writer = D1KnowledgeWriter(db)

    with pytest.raises(RuntimeError, match="unsuccessful statement"):
        await writer.write_metadata(_row(), expected_revision=7, now=12.0)


# ---------------------------------------------------------------------------
# 10. _result_rows
# ---------------------------------------------------------------------------


def test_result_rows_handles_mapping_path() -> None:
    """``_result_rows`` must accept a Mapping-shaped result via ``.get('results')``."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = {"results": [{"knowledge_id": "ko_a"}]}
    rows = runtime_mod._result_rows(result)  # noqa: SLF001
    assert rows == [{"knowledge_id": "ko_a"}]


def test_result_rows_handles_attribute_path() -> None:
    """``_result_rows`` must accept an attribute-shaped result via ``getattr(.results)``."""

    @dataclass
    class _AttrResult:
        results: list[dict[str, object]] = field(default_factory=list)

    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = _AttrResult(results=[{"knowledge_id": "ko_a"}])
    rows = runtime_mod._result_rows(result)  # noqa: SLF001
    assert rows == [{"knowledge_id": "ko_a"}]


def test_result_rows_unwraps_top_level_to_py() -> None:
    """``_result_rows`` must unwrap ``.to_py()`` on the row sequence."""

    @dataclass
    class _ToPySequence:
        rows: list[dict[str, object]] = field(default_factory=list)

        def to_py(self) -> list[dict[str, object]]:
            return self.rows

    @dataclass
    class _AttrResultWithToPy:
        results: _ToPySequence = field(default_factory=_ToPySequence)

        def to_py(self) -> object:  # marker; rows.to_py() is what counts
            return self

    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    sequence = _ToPySequence(rows=[{"knowledge_id": "ko_a"}])
    result = _AttrResultWithToPy(results=sequence)
    rows = runtime_mod._result_rows(result)  # noqa: SLF001
    assert rows == [{"knowledge_id": "ko_a"}]


def test_result_rows_unwraps_per_row_to_py() -> None:
    """``_result_rows`` must unwrap ``.to_py()`` on each row."""

    @dataclass
    class _ToPyMapping:
        knowledge_id: str = "ko_a"

        def to_py(self) -> dict[str, object]:
            return {"knowledge_id": self.knowledge_id}

    @dataclass
    class _AttrResult:
        results: list[object] = field(default_factory=list)

    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = _AttrResult(results=[_ToPyMapping()])
    rows = runtime_mod._result_rows(result)  # noqa: SLF001
    assert rows == [{"knowledge_id": "ko_a"}]


def test_result_rows_rejects_non_sequence_rows() -> None:
    """``_result_rows`` must reject a non-Sequence ``results`` value."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = {"results": 42}
    with pytest.raises(RuntimeError, match="row sequence"):
        runtime_mod._result_rows(result)  # noqa: SLF001


def test_result_rows_rejects_none_results() -> None:
    """``_result_rows`` must reject a None ``results`` value."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = {"results": None}
    with pytest.raises(RuntimeError, match="row sequence"):
        runtime_mod._result_rows(result)  # noqa: SLF001


@pytest.mark.parametrize("bad_value", ["str", b"bytes", bytearray(b"bytes")])
def test_result_rows_rejects_str_like_sequences(bad_value: object) -> None:
    """``_result_rows`` must explicitly reject ``str`` / ``bytes`` / ``bytearray``.

    These are technically ``Sequence`` instances but they are not row
    sequences — accepting them would let the runtime index a string as a
    row and surface garbage.
    """
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = {"results": bad_value}
    with pytest.raises(RuntimeError, match="row sequence"):
        runtime_mod._result_rows(result)  # noqa: SLF001


def test_result_rows_rejects_non_mapping_row() -> None:
    """``_result_rows`` must reject rows that are not Mapping-shaped."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = {"results": [42]}
    with pytest.raises(RuntimeError, match="non-mapping row"):
        runtime_mod._result_rows(result)  # noqa: SLF001


def test_result_rows_rejects_missing_results_field() -> None:
    """``_result_rows`` must reject a result with no ``results`` attribute."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result: object = object()
    with pytest.raises(RuntimeError, match="row sequence"):
        runtime_mod._result_rows(result)  # noqa: SLF001


# ---------------------------------------------------------------------------
# 11. _row_from_mapping
# ---------------------------------------------------------------------------


def test_row_from_mapping_returns_cfrow_for_valid_mapping() -> None:
    """``_row_from_mapping`` must return a CFRow for a complete, valid mapping."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    row = runtime_mod._row_from_mapping(mapping)  # noqa: SLF001
    assert row.knowledge_id == "ko_reader_1"
    assert row.topic == "reader/topic"
    assert row.content_hash == "b" * 64
    assert row.authority == 0.8
    assert row.status is Status.EXPERIMENTAL
    assert row.source_uri == "https://example.test/read"
    assert row.retrieved_at == 12.0
    assert row.r2_blob_key == "oai2-blobs/" + "b" * 64
    assert row.vectorize_id == "ko_reader_1"


@pytest.mark.parametrize(
    "missing_field",
    ["knowledge_id", "topic", "content_hash", "authority", "status", "retrieved_at"],
)
def test_row_from_mapping_fails_closed_on_missing_required_field(missing_field: str) -> None:
    """``_row_from_mapping`` must raise RuntimeError when a required field is missing."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    del mapping[missing_field]
    with pytest.raises(RuntimeError, match="missing field"):
        runtime_mod._row_from_mapping(mapping)  # noqa: SLF001


@pytest.mark.parametrize("bad_value", ["", 0, None, True, False, ["a"], {"a": 1}])
def test_row_from_mapping_rejects_empty_or_non_string_knowledge_id(bad_value: object) -> None:
    """``_row_from_mapping`` must reject empty / non-string knowledge_id."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping["knowledge_id"] = bad_value
    with pytest.raises(RuntimeError, match="invalid knowledge_id"):
        runtime_mod._row_from_mapping(mapping)  # noqa: SLF001


@pytest.mark.parametrize("bad_value", ["", 0, None, True, False, ["t"], {"t": 1}])
def test_row_from_mapping_rejects_empty_or_non_string_topic(bad_value: object) -> None:
    """``_row_from_mapping`` must reject empty / non-string topic."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping["topic"] = bad_value
    with pytest.raises(RuntimeError, match="invalid topic"):
        runtime_mod._row_from_mapping(mapping)  # noqa: SLF001


@pytest.mark.parametrize("bad_value", ["", 0, None, True, False, ["h"], {"a": 1}])
def test_row_from_mapping_rejects_empty_or_non_string_content_hash(bad_value: object) -> None:
    """``_row_from_mapping`` must reject empty / non-string content_hash."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping["content_hash"] = bad_value
    with pytest.raises(RuntimeError, match="invalid content_hash"):
        runtime_mod._row_from_mapping(mapping)  # noqa: SLF001


@pytest.mark.parametrize("bad_value", [True, False, "0.5", None, [0.5], {"a": 0.5}])
def test_row_from_mapping_rejects_non_numeric_authority(bad_value: object) -> None:
    """``_row_from_mapping`` must reject non-(int, float) authority.

    Bool is load-bearing: ``isinstance(True, int) is True``, so the
    explicit ``isinstance(value, bool)`` short-circuit prevents silent
    acceptance.
    """
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping["authority"] = bad_value
    with pytest.raises(RuntimeError, match="invalid authority"):
        runtime_mod._row_from_mapping(mapping)  # noqa: SLF001


@pytest.mark.parametrize("bad_value", [True, False, "12.0", None, [12.0], {"a": 12.0}])
def test_row_from_mapping_rejects_non_numeric_retrieved_at(bad_value: object) -> None:
    """``_row_from_mapping`` must reject non-(int, float) retrieved_at."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping["retrieved_at"] = bad_value
    with pytest.raises(RuntimeError, match="invalid retrieved_at"):
        runtime_mod._row_from_mapping(mapping)  # noqa: SLF001


def test_row_from_mapping_accepts_int_authority() -> None:
    """``_row_from_mapping`` must accept an int authority (e.g. ``1``)."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping["authority"] = 1
    row = runtime_mod._row_from_mapping(mapping)  # noqa: SLF001
    assert row.authority == 1.0


def test_row_from_mapping_accepts_int_retrieved_at() -> None:
    """``_row_from_mapping`` must accept an int retrieved_at (e.g. ``12``)."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping["retrieved_at"] = 12
    row = runtime_mod._row_from_mapping(mapping)  # noqa: SLF001
    assert row.retrieved_at == 12.0


@pytest.mark.parametrize("bad_status", ["NOT_A_STATUS", "implemented", "", "0"])
def test_row_from_mapping_rejects_invalid_status_string(bad_status: str) -> None:
    """``_row_from_mapping`` must reject an unrecognized status string."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping["status"] = bad_status
    with pytest.raises(RuntimeError, match="invalid status"):
        runtime_mod._row_from_mapping(mapping)  # noqa: SLF001


@pytest.mark.parametrize(
    "field_name, bad_value",
    [
        ("source_uri", 123),
        ("source_uri", True),
        ("source_uri", ["uri"]),
        ("source_uri", {"uri": True}),
        ("r2_blob_key", 123),
        ("r2_blob_key", True),
        ("r2_blob_key", ["key"]),
        ("r2_blob_key", {"key": True}),
        ("vectorize_id", 123),
        ("vectorize_id", True),
        ("vectorize_id", ["id"]),
        ("vectorize_id", {"id": True}),
    ],
)
def test_row_from_mapping_rejects_non_string_optional_field(
    field_name: str, bad_value: object
) -> None:
    """``_row_from_mapping`` must reject non-string, non-None optional fields."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping[field_name] = bad_value
    with pytest.raises(RuntimeError, match=f"invalid {field_name}"):
        runtime_mod._row_from_mapping(mapping)  # noqa: SLF001


@pytest.mark.parametrize("field_name", ["source_uri", "r2_blob_key", "vectorize_id"])
def test_row_from_mapping_accepts_none_optional_field(field_name: str) -> None:
    """``_row_from_mapping`` must accept None for any optional field."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    mapping = _row_mapping()
    mapping[field_name] = None
    row = runtime_mod._row_from_mapping(mapping)  # noqa: SLF001
    assert getattr(row, field_name) is None


# ---------------------------------------------------------------------------
# 12. _result_success + _result_changes
# ---------------------------------------------------------------------------


def test_result_success_handles_mapping_path_true() -> None:
    """``_result_success`` must accept ``success=True`` via Mapping path."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    assert runtime_mod._result_success({"success": True}) is True  # noqa: SLF001


def test_result_success_handles_mapping_path_false() -> None:
    """``_result_success`` must accept ``success=False`` via Mapping path."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    assert runtime_mod._result_success({"success": False}) is False  # noqa: SLF001


def test_result_success_handles_attribute_path_true() -> None:
    """``_result_success`` must accept attribute-path success."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    assert runtime_mod._result_success(_FakeResult(success=True)) is True  # noqa: SLF001


def test_result_success_handles_attribute_path_false() -> None:
    """``_result_success`` must accept attribute-path success=False."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    assert runtime_mod._result_success(_FakeResult(success=False)) is False  # noqa: SLF001


@pytest.mark.parametrize("bad_value", [None, 0, 1, "yes", [], {"name": "ok"}])
def test_result_success_rejects_missing_or_non_bool(bad_value: object) -> None:
    """``_result_success`` must reject missing or non-bool success."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = {"success": bad_value} if bad_value is not None else {}
    with pytest.raises(RuntimeError, match="boolean success field"):
        runtime_mod._result_success(result)  # noqa: SLF001


def test_result_changes_handles_mapping_meta() -> None:
    """``_result_changes`` must accept ``changes`` via Mapping meta."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    assert runtime_mod._result_changes({"meta": {"changes": 5}}) == 5  # noqa: SLF001


def test_result_changes_handles_attribute_meta() -> None:
    """``_result_changes`` must accept ``changes`` via attribute meta."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = _FakeResult(meta=_FakeMeta(changes=3))
    assert runtime_mod._result_changes(result) == 3  # noqa: SLF001


@pytest.mark.parametrize("bad_value", [-1, True, False, "5", None, [5]])
def test_result_changes_rejects_non_non_negative_int_changes(bad_value: object) -> None:
    """``_result_changes`` must reject non-non-negative-int changes via _non_negative_int."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = {"meta": {"changes": bad_value}}
    with pytest.raises(ValueError, match="changes must be a non-negative integer"):
        runtime_mod._result_changes(result)  # noqa: SLF001


def test_result_changes_rejects_missing_meta() -> None:
    """``_result_changes`` must reject a result with no ``meta`` field.

    ``getattr(None, "changes", None) == None``, then ``_non_negative_int(None, ...)``
    raises ``ValueError("changes must be a non-negative integer")``. The
    missing-meta path is a FAILURE, NOT a default-0 — this is the existing
    wire contract.
    """
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result: object = object()
    with pytest.raises(ValueError, match="changes must be a non-negative integer"):
        runtime_mod._result_changes(result)  # noqa: SLF001


# ---------------------------------------------------------------------------
# 13. _non_negative_int + _non_negative_number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("good_value", [0, 1, 7, 100, 1_000_000_000])
def test_non_negative_int_accepts_positive_and_zero_ints(good_value: int) -> None:
    """``_non_negative_int`` must accept non-negative ints."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    assert runtime_mod._non_negative_int(good_value, "field") == good_value  # noqa: SLF001


@pytest.mark.parametrize("bad_value", [True, False, -1, -100, 1.0, 0.0, "0", None, [], [1], (1,)])
def test_non_negative_int_rejects_bool_negative_float_or_other(bad_value: object) -> None:
    """``_non_negative_int`` must reject bool (load-bearing), non-int, negative."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    with pytest.raises(ValueError, match="non-negative integer"):
        runtime_mod._non_negative_int(bad_value, "field")  # noqa: SLF001


@pytest.mark.parametrize("good_value", [0, 0.0, 1, 1.0, 100.5, 1_000_000_000])
def test_non_negative_number_accepts_non_negative_numeric(good_value: float) -> None:
    """``_non_negative_number`` must accept non-negative int and float."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    assert runtime_mod._non_negative_number(good_value, "field") == float(good_value)  # noqa: SLF001


@pytest.mark.parametrize("bad_value", [True, False, -1, -1.0, -100.5, "10.0", None, [], [1.0]])
def test_non_negative_number_rejects_bool_negative_or_other(bad_value: object) -> None:
    """``_non_negative_number`` must reject bool (load-bearing), negative, non-numeric."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    with pytest.raises(ValueError, match="non-negative number"):
        runtime_mod._non_negative_number(bad_value, "field")  # noqa: SLF001


def test_non_negative_number_accepts_positive_infinity() -> None:
    """``_non_negative_number`` accepts ``+inf`` (existing wire contract).

    ``float('inf') < 0`` is ``False``, so ``+inf`` passes the gate. This
    is the existing behaviour; a refactor that adds a finiteness check is
    a deliberate contract change and must update the test alongside the
    production code.
    """
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    assert runtime_mod._non_negative_number(math.inf, "field") == math.inf  # noqa: SLF001


def test_non_negative_number_accepts_nan() -> None:
    """``_non_negative_number`` accepts ``NaN`` (existing wire contract).

    ``float('nan') < 0`` is ``False``, so ``NaN`` passes the gate. This
    is the existing behaviour; pin it so a refactor that adds a NaN
    guard is a deliberate contract change.
    """
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    result = runtime_mod._non_negative_number(math.nan, "field")  # noqa: SLF001
    # NaN compares unequal to itself; verify the value is the same NaN.
    assert math.isnan(result)


def test_non_negative_number_rejects_negative_infinity() -> None:
    """``_non_negative_number`` must reject ``-inf``."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    with pytest.raises(ValueError, match="non-negative number"):
        runtime_mod._non_negative_number(-math.inf, "field")  # noqa: SLF001


# ---------------------------------------------------------------------------
# 14. UP006 import contract
# ---------------------------------------------------------------------------


def test_module_source_uses_collections_abc_for_mapping_and_sequence() -> None:
    """The module source must use ``collections.abc`` for ``Mapping`` + ``Sequence``."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    source = inspect.getsource(runtime_mod)
    assert "from collections.abc import Mapping, Sequence" in source
    assert "from typing import Mapping" not in source
    assert "from typing import Sequence" not in source


def test_module_source_uses_typing_for_cast() -> None:
    """The module source must use ``typing.cast`` for the optional-field casts."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    source = inspect.getsource(runtime_mod)
    assert "from typing import cast" in source


def test_module_source_has_no_workers_import() -> None:
    """The module source must not import the ``oai2.workers`` package."""
    import oai2.knowledge.knowledge_d1_runtime as runtime_mod

    source = inspect.getsource(runtime_mod)
    assert "from oai2.workers" not in source
    assert "import oai2.workers" not in source
