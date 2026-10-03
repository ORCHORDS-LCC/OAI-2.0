"""D1 supersession gate, executed against a real sqlite3 database.

The point of this file is that it runs the REAL `knowledge_query_sql` and the
REAL `KNOWLEDGE_WRITER_UPSERT_SQL` against sqlite3, not a hand-written
approximation. A test double that merely looks like the filter proves nothing
about the predicate that actually runs in production.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager

import pytest

from oai2.knowledge.knowledge_d1 import (
    KNOWLEDGE_CORPUS_ADVANCE_SQL,
    KNOWLEDGE_INDEX_TABLE,
    KNOWLEDGE_SCHEMA_MIGRATION_SQL,
    KNOWLEDGE_SCHEMA_SQL,
    KNOWLEDGE_SCHEMA_VERSION,
    KNOWLEDGE_SUPERSESSION_COLUMNS,
    KNOWLEDGE_WRITER_UPSERT_SQL,
    knowledge_query_sql,
    knowledge_schema_statements,
    missing_supersession_migration,
)

KID = "ko_1"


@contextmanager
def _fresh_db() -> Iterator[sqlite3.Connection]:
    """Yield a migrated in-memory database and ALWAYS close it.

    The connection used to be handed back bare and no caller closed it, so
    every test in this file leaked a handle. Under ``pytest -W error`` the
    resulting ``ResourceWarning: unclosed database`` was raised as an
    unraisable exception during garbage collection, and pytest attributed it
    to whatever test happened to be running at that moment -- which is how a
    leak in this file came to be reported as a knowledge-trace failure in an
    unrelated file.
    """
    db = sqlite3.connect(":memory:")
    try:
        db.execute(
            "CREATE TABLE knowledge_gc_delete_lease (object_key TEXT, state TEXT, expires_at REAL)"
        )
        db.execute(
            "CREATE TABLE knowledge_gc_resource_lease ("
            "resource_type TEXT, resource_key TEXT, state TEXT, expires_at REAL)"
        )
        for statement in knowledge_schema_statements():
            db.execute(statement)
        yield db
    finally:
        db.close()


def _upsert(
    db: sqlite3.Connection,
    *,
    kid: str,
    topic: str = "deploy",
    status: str = "EXPERIMENTAL",
    superseded_by: str | None = None,
    superseded_at: float | None = None,
    revision: int = 0,
) -> None:
    db.execute(
        KNOWLEDGE_WRITER_UPSERT_SQL,
        (
            kid, topic, "c" * 64, 0.5, status, None, 1.0,
            f"blob-{kid}", f"vec-{kid}", superseded_by, superseded_at,
            revision, 10.0,
        ),
    )
    db.execute(
        KNOWLEDGE_CORPUS_ADVANCE_SQL, (revision, f"blob-{kid}", 10.0, f"vec-{kid}")
    )
    db.commit()


def _query(db: sqlite3.Connection, topic: str = "deploy") -> list[str]:
    sql = knowledge_query_sql(1)
    return [r[0] for r in db.execute(sql, (topic, 0.0, "EXPERIMENTAL", 50))]


# ---------------------------------------------------------------------------
# 1. The schema actually has the columns
# ---------------------------------------------------------------------------


def test_fresh_schema_carries_the_supersession_columns() -> None:
    with _fresh_db() as db:
        columns = {r[1] for r in db.execute(f"PRAGMA table_info({KNOWLEDGE_INDEX_TABLE})")}
        assert set(KNOWLEDGE_SUPERSESSION_COLUMNS) <= columns


def test_schema_sql_itself_declares_the_columns() -> None:
    assert "superseded_by TEXT" in KNOWLEDGE_SCHEMA_SQL
    assert "superseded_at REAL" in KNOWLEDGE_SCHEMA_SQL


# ---------------------------------------------------------------------------
# 2. Editing CREATE TABLE alone is NOT sufficient -- the additive migration
# ---------------------------------------------------------------------------


def test_editing_create_table_alone_does_not_migrate_a_provisioned_database() -> None:
    """Why KNOWLEDGE_SCHEMA_MIGRATION_SQL exists, demonstrated.

    A database provisioned from the PREVIOUS schema has no supersession
    columns. Re-running the updated `CREATE TABLE IF NOT EXISTS` leaves it
    untouched, so without an explicit ALTER the new columns never appear and
    every query selecting them fails.
    """
    # This test deliberately builds the PREVIOUS schema, so it cannot use
    # _fresh_db(). The connection is still owned and closed here; leaking it
    # was what made an unrelated test fail under -W error.
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute(
            f"CREATE TABLE {KNOWLEDGE_INDEX_TABLE} ("
            "knowledge_id TEXT PRIMARY KEY NOT NULL, topic TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, authority REAL NOT NULL, status TEXT NOT NULL, "
            "source_uri TEXT, retrieved_at REAL NOT NULL, r2_blob_key TEXT, "
            "vectorize_id TEXT, corpus_revision INTEGER NOT NULL)"
        )
        before = {r[1] for r in db.execute(f"PRAGMA table_info({KNOWLEDGE_INDEX_TABLE})")}
        assert "superseded_by" not in before

        # The updated schema statement is a no-op here.
        for statement in knowledge_schema_statements():
            if statement.upper().startswith("CREATE TABLE IF NOT EXISTS KNOWLEDGE_INDEX"):
                db.execute(statement)
        after_create = {r[1] for r in db.execute(f"PRAGMA table_info({KNOWLEDGE_INDEX_TABLE})")}
        assert after_create == before, "CREATE TABLE IF NOT EXISTS must be a no-op here"

        # The additive migration is what actually adds them.
        for statement in missing_supersession_migration(after_create):
            db.execute(statement)
        migrated = {r[1] for r in db.execute(f"PRAGMA table_info({KNOWLEDGE_INDEX_TABLE})")}
        assert set(KNOWLEDGE_SUPERSESSION_COLUMNS) <= migrated


def test_migration_is_idempotent() -> None:
    assert len(missing_supersession_migration([])) == len(KNOWLEDGE_SCHEMA_MIGRATION_SQL)
    assert missing_supersession_migration(list(KNOWLEDGE_SUPERSESSION_COLUMNS)) == ()
    # Partially migrated: only the missing one is emitted.
    assert len(missing_supersession_migration(["superseded_by"])) == 1


def test_migration_statements_are_additive_alter_add_column() -> None:
    for statement in KNOWLEDGE_SCHEMA_MIGRATION_SQL:
        assert statement.upper().startswith("ALTER TABLE")
        assert "ADD COLUMN" in statement.upper()


# ---------------------------------------------------------------------------
# 3. The REAL query excludes superseded rows
# ---------------------------------------------------------------------------


def test_real_query_excludes_a_superseded_row() -> None:
    with _fresh_db() as db:
        _upsert(db, kid="ko_old", superseded_by="qpipe:deploy:new", superseded_at=5.0)
        _upsert(db, kid="ko_new", revision=1)
        assert _query(db) == ["ko_new"]


def test_real_query_excludes_a_superseded_row_even_when_status_is_default() -> None:
    """The filter cannot be delegated to the status set.

    EXPERIMENTAL is in the default include set, so a status-based filter
    would return this row. The row is excluded anyway, which is the
    whole point of filtering on the field.
    """
    with _fresh_db() as db:
        _upsert(db, kid="ko_old", status="IMPLEMENTED", superseded_by="qpipe:deploy:new")
        _upsert(db, kid="ko_new", status="IMPLEMENTED", revision=1)
        sql = knowledge_query_sql(1)
        found = [r[0] for r in db.execute(sql, ("deploy", 0.0, "IMPLEMENTED", 50))]
        assert found == ["ko_new"]


def test_real_query_keeps_rows_with_a_superseded_at_but_no_link() -> None:
    """superseded_at alone must not exclude.

    The timestamp records when a supersession happened; only a non-null
    superseded_by asserts that this record IS superseded. Conflating them
    would drop live rows that merely carry a timestamp.
    """
    with _fresh_db() as db:
        _upsert(db, kid="ko_live", superseded_at=5.0)
        assert _query(db) == ["ko_live"]


def test_upsert_overwrites_a_previously_set_supersession_link() -> None:
    """Re-ingesting a record as live must clear the stale link."""
    with _fresh_db() as db:
        _upsert(db, kid="ko_x", superseded_by="qpipe:deploy:old", superseded_at=5.0)
        _upsert(db, kid="ko_x", revision=1)
        assert _query(db) == ["ko_x"]
        row = db.execute(
            f"SELECT superseded_by, superseded_at FROM {KNOWLEDGE_INDEX_TABLE} WHERE knowledge_id = ?",
            ("ko_x",),
        ).fetchone()
        assert row == (None, None)


def test_schema_version_is_unchanged_by_this_migration() -> None:
    """An additive, nullable column is not a v1 -> v2 break.

    KNOWLEDGE_SCHEMA_VERSION is the wire contract the live D1 migrations
    coordinate against, so it is deliberately NOT bumped here. Existing rows
    read NULL, which is the correct "never superseded" value.
    """
    assert KNOWLEDGE_SCHEMA_VERSION == 1


def test_placeholder_contract_is_unchanged() -> None:
    """Adding columns must not shift the binder's placeholder indices."""
    for count, expected_limit in ((1, 4), (2, 5), (5, 8)):
        sql = knowledge_query_sql(count)
        assert f"LIMIT ?{expected_limit}" in sql
        assert "superseded_by IS NULL" in sql


@pytest.mark.parametrize("count", [0, -1, True, None, 1.0, "1"])
def test_query_sql_still_rejects_invalid_status_count(count: object) -> None:
    with pytest.raises(ValueError):
        knowledge_query_sql(count)  # type: ignore[arg-type]
