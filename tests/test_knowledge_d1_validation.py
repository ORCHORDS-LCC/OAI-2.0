"""Outer-guard-rail boundary contracts for ``oai2.knowledge.knowledge_d1``.

This file pins the public-safety boundary of the D1 schema and
transactional writer contract for authoritative knowledge metadata. The
companion module ``oai2/knowledge/knowledge_d1.py`` defines the wire
contract every Cloudflare D1 deployment consumes — the schema constants
(``KNOWLEDGE_INDEX_TABLE``, ``KNOWLEDGE_CORPUS_STATE_TABLE``,
``KNOWLEDGE_SCHEMA_VERSION``) are referenced by the live D1 migrations,
and the SQL statements (``KNOWLEDGE_SCHEMA_SQL``, ``KNOWLEDGE_GET_SQL``,
``KNOWLEDGE_WRITER_UPSERT_SQL``, ``KNOWLEDGE_CORPUS_ADVANCE_SQL``,
``KNOWLEDGE_CORPUS_REVISION_SQL``) are persisted in D1 prepared/batch
calls. A rename or content change would silently corrupt the production
schema.

These tests focus on the *outer guard rails* the existing
``tests/test_knowledge_d1_runtime.py`` (12 tests) do NOT pin:

- ``KNOWLEDGE_SCHEMA_VERSION`` integer wire-contract pinning
  (value ``1`` — a bump without coordinating with the D1 migration log
  would corrupt historical records).
- Table-name constants (``KNOWLEDGE_INDEX_TABLE``,
  ``KNOWLEDGE_CORPUS_STATE_TABLE``) pinned as the literal strings the
  live D1 migrations expect.
- ``knowledge_query_sql()`` argument validation (the existing test
  covers only the empty-status-set path; this file pins the full
  ``bool / non-int / negative / None`` rejection matrix and the
  ``bool-subclasses-int`` load-bearing case) + placeholder-index
  stability for ``status_count ∈ {1, 2, 5}`` (the ``?3``-through-``?3+N``
  range and the ``limit_index = 3 + N`` are the wire contract the
  prepared-statement binder expects).
- ``knowledge_schema_statements()`` behaviour (tuple type, non-empty
  filtered output, no trailing semicolons, includes CREATE TABLE +
  INSERT statements).
- SQL-content pinning for the load-bearing substrings the live D1
  migrations depend on (``STRICT``, ``CHECK(authority >= 0 AND authority
  <= 1)``, ``CHECK(retrieved_at >= 0)``, ``CHECK(corpus_revision >=
  0)``, ``ON CONFLICT(knowledge_id) DO UPDATE``, ``revision + 1``,
  ``singleton = 1``, ``ORDER BY authority DESC, retrieved_at DESC``).
- ``__all__`` (10 names) + package-level re-exports identity check.

A refactor that renames ``knowledge_index`` -> ``knowledge_table``, that
replaces ```` (kg <= 1`` keyword in place of ``kg = 0`` keyword in
``knowledge_query_sql``, or that drops the GC lease-not-joined-check
``knowledge_gc_delete_lease`` reference in the upsert SQL must trip one
of these tests.
"""

from __future__ import annotations

import pytest

from oai2.knowledge import knowledge_d1 as knowledge_d1_mod
from oai2.knowledge.knowledge_d1 import (
    KNOWLEDGE_CORPUS_ADVANCE_SQL,
    KNOWLEDGE_CORPUS_REVISION_SQL,
    KNOWLEDGE_CORPUS_STATE_TABLE,
    KNOWLEDGE_GET_SQL,
    KNOWLEDGE_INDEX_TABLE,
    KNOWLEDGE_SCHEMA_SQL,
    KNOWLEDGE_SCHEMA_VERSION,
    KNOWLEDGE_WRITER_UPSERT_SQL,
    knowledge_query_sql,
    knowledge_schema_statements,
)

# ---------------------------------------------------------------------------
# 1. KNOWLEDGE_SCHEMA_VERSION wire-contract stability
# ---------------------------------------------------------------------------


def test_knowledge_schema_version_is_pinned_int_one() -> None:
    """``KNOWLEDGE_SCHEMA_VERSION`` is the wire contract for D1 migrations.
    A refactor that bumps it without coordinating with the D1 migration log
    would silently corrupt historical records."""
    assert KNOWLEDGE_SCHEMA_VERSION == 1
    assert isinstance(KNOWLEDGE_SCHEMA_VERSION, int)


# ---------------------------------------------------------------------------
# 2. Table-name constants wire-contract stability
# ---------------------------------------------------------------------------


def test_knowledge_index_table_is_pinned_string() -> None:
    """``KNOWLEDGE_INDEX_TABLE`` is the live D1 table name the upsert +
    get + query paths reference — pinned as the literal string."""
    assert KNOWLEDGE_INDEX_TABLE == "knowledge_index"
    assert isinstance(KNOWLEDGE_INDEX_TABLE, str)


def test_knowledge_corpus_state_table_is_pinned_string() -> None:
    """``KNOWLEDGE_CORPUS_STATE_TABLE`` is the live D1 table the
    revision-authority singleton occupies — pinned as the literal
    string."""
    assert KNOWLEDGE_CORPUS_STATE_TABLE == "knowledge_corpus_state"
    assert isinstance(KNOWLEDGE_CORPUS_STATE_TABLE, str)


# ---------------------------------------------------------------------------
# 3. knowledge_query_sql() argument validation
# ---------------------------------------------------------------------------


def test_knowledge_query_sql_rejects_empty_status_set() -> None:
    """``status_count=0`` is rejected — the existing test in
    test_knowledge_d1_runtime.py covers this path; pinned here for
    symmetry with the rest of the negative matrix."""
    with pytest.raises(ValueError, match="positive integer"):
        knowledge_query_sql(0)


@pytest.mark.parametrize(
    "bad_value",
    [
        -1,
        -999,
    ],
)
def test_knowledge_query_sql_rejects_negative_status_count(bad_value: int) -> None:
    """Negative status counts are rejected — the retrieval pipeline never
    emits a negative status set; pinned for symmetry with the zero
    case."""
    with pytest.raises(ValueError, match="positive integer"):
        knowledge_query_sql(bad_value)


@pytest.mark.parametrize("bad_value", [True, False])
def test_knowledge_query_sql_rejects_bool_status_count(bad_value: bool) -> None:
    """``bool`` is rejected because ``isinstance(True, int) is True`` —
    a refactor that drops the bool guard would silently accept True /
    False as a status count of 1 / 0."""
    with pytest.raises(ValueError, match="positive integer"):
        knowledge_query_sql(bad_value)


@pytest.mark.parametrize(
    "bad_value",
    ["1", 1.5, [1], (1,), {1}, None],
)
def test_knowledge_query_sql_rejects_non_int_status_count(
    bad_value: object,
) -> None:
    """Non-int status counts are rejected — str / float / list / tuple /
    set / None are never valid. ``None`` falls back to the standard
    ``positive integer`` message because the guard rejects ``None``
    before reaching the value-rejection branch."""
    with pytest.raises(ValueError, match="positive integer"):
        knowledge_query_sql(bad_value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. knowledge_query_sql() placeholder generation (wire contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_count", [1, 2, 5])
def test_knowledge_query_sql_status_placeholders_indexed_at_three(
    status_count: int,
) -> None:
    """The status-set IN(...) placeholders are ``?3`` through
    ``?{3+N-1}`` — the prepared-statement binder depends on this exact
    indexing. ``?1`` (topic) and ``?2`` (min_authority) come before;
    ``?{3+N}`` is the LIMIT."""
    sql = knowledge_query_sql(status_count)
    expected_status_placeholders = ", ".join(f"?{index}" for index in range(3, 3 + status_count))
    expected_limit_index = 3 + status_count
    assert expected_status_placeholders in sql
    assert f"LIMIT ?{expected_limit_index}" in sql


@pytest.mark.parametrize("status_count", [1, 2, 5])
def test_knowledge_query_sql_includes_topic_and_authority_filters(
    status_count: int,
) -> None:
    """The ``WHERE`` clause always includes ``instr(lower(topic),
    lower(?1)) > 0`` and ``authority >= ?2`` — pinned so a refactor that
    reorders the WHERE clause (changing placeholder indexes) trips here
    instead of silently breaking the binder."""
    sql = knowledge_query_sql(status_count)
    assert "instr(lower(topic), lower(?1)) > 0" in sql
    assert "authority >= ?2" in sql


@pytest.mark.parametrize("status_count", [1, 2, 5])
def test_knowledge_query_sql_orders_results_by_authority_then_retrieved(
    status_count: int,
) -> None:
    """``ORDER BY authority DESC, retrieved_at DESC`` is the documented
    retrieval ordering — pinned because downstream prompt-code branches
    on the rank order."""
    sql = knowledge_query_sql(status_count)
    assert "ORDER BY authority DESC, retrieved_at DESC" in sql


# ---------------------------------------------------------------------------
# 5. knowledge_schema_statements() behaviour
# ---------------------------------------------------------------------------


def test_knowledge_schema_statements_returns_tuple_of_strings() -> None:
    """``knowledge_schema_statements()`` returns a tuple of strings —
    the Cloudflare D1 prepared/batch API consumes a list of SQL
    statements; tuple-of-strings is the documented wire shape."""
    statements = knowledge_schema_statements()
    assert isinstance(statements, tuple)
    assert all(isinstance(statement, str) for statement in statements)


def test_knowledge_schema_statements_includes_create_table_statements() -> None:
    """The schema-statements output must include the
    ``CREATE TABLE IF NOT EXISTS knowledge_index`` and
    ``CREATE TABLE IF NOT EXISTS knowledge_corpus_state`` statements —
    pinned so the live D1 migration creates both tables."""
    statements = knowledge_schema_statements()
    joined = "\n".join(statements)
    assert "CREATE TABLE IF NOT EXISTS knowledge_index" in joined
    assert "CREATE TABLE IF NOT EXISTS knowledge_corpus_state" in joined


def test_knowledge_schema_statements_includes_corpus_state_seed() -> None:
    """The schema-statements output must include the
    ``INSERT OR IGNORE INTO knowledge_corpus_state(singleton, revision)
    VALUES (1, 0)`` seed — pinned because the revision authority
    singleton must exist for the upsert path to commit."""
    statements = knowledge_schema_statements()
    joined = "\n".join(statements)
    assert "INSERT OR IGNORE INTO knowledge_corpus_state" in joined
    assert "VALUES (1, 0)" in joined


def test_knowledge_schema_statements_strips_empty_segments() -> None:
    """Empty SQL segments (from trailing ``;`` or ``;;``) are filtered
    out — pinned so a refactor that removes the ``if statement.strip()``
    filter does not silently pass empty statements to D1."""
    statements = knowledge_schema_statements()
    assert all(statement.strip() for statement in statements)
    assert all("\n" not in statement or statement.strip() for statement in statements)


def test_knowledge_schema_statements_at_least_three_statements() -> None:
    """The schema-statements output is at least 3 statements (CREATE
    TABLE knowledge_index + CREATE INDEX + CREATE TABLE
    knowledge_corpus_state + INSERT seed) — pinned so a refactor that
    silently drops one of the schema statements fails here."""
    statements = knowledge_schema_statements()
    assert len(statements) >= 3


# ---------------------------------------------------------------------------
# 6. SQL-content pinning for load-bearing substrings
# ---------------------------------------------------------------------------


def test_knowledge_schema_sql_uses_strict_mode() -> None:
    """``STRICT`` mode is set on both tables — pinned because removing
    STRICT would let D1 silently coerce non-TEXT values into TEXT and
    corrupt the knowledge_index."""
    assert "STRICT" in KNOWLEDGE_SCHEMA_SQL


def test_knowledge_schema_sql_enforces_authority_in_range() -> None:
    """``CHECK(authority >= 0 AND authority <= 1)`` is the documented
    authority-range invariant — pinned because the authority field is
    the trust signal the prompt scorer depends on."""
    assert "CHECK(authority >= 0 AND authority <= 1)" in KNOWLEDGE_SCHEMA_SQL


def test_knowledge_schema_sql_enforces_non_negative_retrieved_at() -> None:
    """``CHECK(retrieved_at >= 0)`` is the documented
    retrieved-at-invariant — pinned because negative timestamps would
    corrupt the ORDER BY retrieved_at DESC path."""
    assert "CHECK(retrieved_at >= 0)" in KNOWLEDGE_SCHEMA_SQL


def test_knowledge_schema_sql_enforces_non_negative_corpus_revision() -> None:
    """``CHECK(corpus_revision >= 0)`` is the documented
    corpus-revision-invariant — pinned because the upsert path uses
    ``corpus_revision + 1`` to advance the authority revision."""
    assert "CHECK(corpus_revision >= 0)" in KNOWLEDGE_SCHEMA_SQL


def test_knowledge_schema_sql_enforces_corpus_singleton() -> None:
    """``CHECK(singleton = 1)`` on the corpus_state table pins the
    singleton-row invariant — pinned because the upsert path assumes
    exactly one corpus_state row exists."""
    assert "CHECK(singleton = 1)" in KNOWLEDGE_SCHEMA_SQL


def test_knowledge_schema_sql_indexes_r2_blob_key() -> None:
    """The ``idx_knowledge_index_r2_blob_key`` index on ``r2_blob_key``
    is the GC-sweep query optimization — pinned because dropping the
    index would make sweep queries slow as the index grows."""
    assert "idx_knowledge_index_r2_blob_key" in KNOWLEDGE_SCHEMA_SQL


def test_knowledge_writer_upsert_sql_includes_on_conflict_clause() -> None:
    """``ON CONFLICT(knowledge_id) DO UPDATE`` is the documented
    idempotent-upsert path — pinned because removing it would make
    re-ingestion silently fail on duplicate knowledge_ids."""
    assert "ON CONFLICT(knowledge_id) DO UPDATE" in KNOWLEDGE_WRITER_UPSERT_SQL


def test_knowledge_writer_upsert_sql_includes_revision_guard() -> None:
    """The upsert SQL gates on
    ``knowledge_corpus_state.revision = ?10`` — pinned because removing
    the guard would let ingestion bypass the corpus-revision authority."""
    assert "knowledge_corpus_state" in KNOWLEDGE_WRITER_UPSERT_SQL
    assert "revision = ?10" in KNOWLEDGE_WRITER_UPSERT_SQL


def test_knowledge_writer_upsert_sql_includes_gc_lease_gate() -> None:
    """The upsert SQL gates on the ``knowledge_gc_delete_lease`` table
    to block writes against active GC leases — pinned because removing
    the gate would let ingestion resurrect a deleted blob key."""
    assert "knowledge_gc_delete_lease" in KNOWLEDGE_WRITER_UPSERT_SQL


def test_knowledge_writer_upsert_sql_advances_corpus_revision() -> None:
    """``SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10 + 1`` advances
    the corpus revision by 1 on every successful upsert — pinned
    because a refactor that drops the ``+ 1`` would silently stall the
    corpus revision authority."""
    assert "?10 + 1" in KNOWLEDGE_WRITER_UPSERT_SQL


def test_knowledge_corpus_advance_sql_increments_revision() -> None:
    """``SET revision = revision + 1`` is the corpus-revision-advance
    contract — pinned because a refactor that replaces ``+ 1`` with
    ``+ 0`` would stall the corpus revision authority."""
    assert "SET revision = revision + 1" in KNOWLEDGE_CORPUS_ADVANCE_SQL


def test_knowledge_corpus_advance_sql_gates_on_gc_lease() -> None:
    """The corpus-advance SQL gates on ``knowledge_gc_delete_lease`` to
    block revision-advance while a GC lease is active — pinned because
    removing the gate would let a new revision be applied while a
    deletion is still pending."""
    assert "knowledge_gc_delete_lease" in KNOWLEDGE_CORPUS_ADVANCE_SQL


def test_knowledge_corpus_advance_sql_pins_singleton() -> None:
    """``WHERE singleton = 1`` pins the singleton-row contract — pinned
    because the upsert path assumes exactly one corpus_state row
    exists."""
    assert "WHERE singleton = 1" in KNOWLEDGE_CORPUS_ADVANCE_SQL


def test_knowledge_get_sql_includes_all_required_columns() -> None:
    """The get-by-id SQL selects all 10 documented columns from the
    knowledge_index table — pinned because a refactor that drops a
    column would silently truncate the reader's return value."""
    for column in [
        "knowledge_id",
        "topic",
        "content_hash",
        "authority",
        "status",
        "source_uri",
        "retrieved_at",
        "r2_blob_key",
        "vectorize_id",
        "corpus_revision",
    ]:
        assert f"{column}," in KNOWLEDGE_GET_SQL or f"{column}\n" in KNOWLEDGE_GET_SQL


def test_knowledge_corpus_revision_sql_pins_singleton() -> None:
    """``WHERE singleton = 1`` pins the singleton-row contract — pinned
    because the revision-authority read depends on exactly one
    corpus_state row."""
    assert "WHERE singleton = 1" in KNOWLEDGE_CORPUS_REVISION_SQL


# ---------------------------------------------------------------------------
# 7. __all__ exports + package-level identity
# ---------------------------------------------------------------------------


def test_module_all_lists_ten_public_exports() -> None:
    """``__all__`` is the canonical public surface — adding a name is a
    deliberate API change and must be flagged here."""
    expected = {
        "KNOWLEDGE_SCHEMA_VERSION",
        "KNOWLEDGE_INDEX_TABLE",
        "KNOWLEDGE_CORPUS_STATE_TABLE",
        "KNOWLEDGE_SCHEMA_SQL",
        "KNOWLEDGE_CORPUS_REVISION_SQL",
        "KNOWLEDGE_GET_SQL",
        "knowledge_query_sql",
        "KNOWLEDGE_WRITER_UPSERT_SQL",
        "KNOWLEDGE_CORPUS_ADVANCE_SQL",
        "knowledge_schema_statements",
    }
    assert set(knowledge_d1_mod.__all__) == expected
    assert len(knowledge_d1_mod.__all__) == 10


def test_knowledge_d1_symbols_are_exported_from_knowledge_package() -> None:
    """All 10 public symbols are re-exported at the ``oai2.knowledge``
    package level — pinned against accidental re-export removal."""
    from oai2.knowledge import (
        KNOWLEDGE_CORPUS_ADVANCE_SQL as ExportedAdvance,
    )
    from oai2.knowledge import (
        KNOWLEDGE_CORPUS_REVISION_SQL as ExportedRevision,
    )
    from oai2.knowledge import (
        KNOWLEDGE_CORPUS_STATE_TABLE as ExportedCorpusStateTable,
    )
    from oai2.knowledge import KNOWLEDGE_GET_SQL as ExportedGet
    from oai2.knowledge import KNOWLEDGE_INDEX_TABLE as ExportedIndexTable
    from oai2.knowledge import KNOWLEDGE_SCHEMA_SQL as ExportedSchemaSql
    from oai2.knowledge import KNOWLEDGE_SCHEMA_VERSION as ExportedVersion
    from oai2.knowledge import KNOWLEDGE_WRITER_UPSERT_SQL as ExportedUpsert
    from oai2.knowledge import knowledge_query_sql as ExportedQuerySql
    from oai2.knowledge import knowledge_schema_statements as ExportedStatements

    assert ExportedAdvance is KNOWLEDGE_CORPUS_ADVANCE_SQL
    assert ExportedRevision is KNOWLEDGE_CORPUS_REVISION_SQL
    assert ExportedCorpusStateTable == KNOWLEDGE_CORPUS_STATE_TABLE
    assert ExportedGet is KNOWLEDGE_GET_SQL
    assert ExportedIndexTable == KNOWLEDGE_INDEX_TABLE
    assert ExportedSchemaSql is KNOWLEDGE_SCHEMA_SQL
    assert ExportedVersion == KNOWLEDGE_SCHEMA_VERSION
    assert ExportedUpsert is KNOWLEDGE_WRITER_UPSERT_SQL
    assert ExportedQuerySql is knowledge_query_sql
    assert ExportedStatements is knowledge_schema_statements
