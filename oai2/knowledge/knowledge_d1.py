"""D1 schema and transactional writer contract for authoritative knowledge metadata.

Public-safe SQL only: no Cloudflare account IDs, resource IDs, endpoints, or
credentials. The statements are designed for the D1 prepared/batch API.
"""

from __future__ import annotations

from collections.abc import Iterable

from .gc_resource_lease_d1 import GC_RESOURCE_LEASE_TABLE

KNOWLEDGE_SCHEMA_VERSION = 1
KNOWLEDGE_INDEX_TABLE = "knowledge_index"
KNOWLEDGE_CORPUS_STATE_TABLE = "knowledge_corpus_state"

# #261: the vector reclamation path asks "does any row reference this vector?",
# which is a lookup on vectorize_id. Without an index on that column the
# predicate is a full table scan on every acquire, revalidate and finalize, and
# a sweep runs those against the whole authoritative table. The index below is
# symmetric with the existing blob-key index. Applying it to a live D1 database
# is a schema migration and is NOT performed by this source change.
#
# Kept as a Python comment rather than an SQL one: the schema is split on
# semicolons into individual statements, and a comment containing one would
# silently split a statement in two.
KNOWLEDGE_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {KNOWLEDGE_INDEX_TABLE} (
    knowledge_id TEXT PRIMARY KEY NOT NULL,
    topic TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    authority REAL NOT NULL CHECK(authority >= 0 AND authority <= 1),
    status TEXT NOT NULL,
    source_uri TEXT,
    retrieved_at REAL NOT NULL CHECK(retrieved_at >= 0),
    r2_blob_key TEXT,
    vectorize_id TEXT,
    superseded_by TEXT,
    superseded_at REAL,
    corpus_revision INTEGER NOT NULL CHECK(corpus_revision >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_knowledge_index_r2_blob_key
ON {KNOWLEDGE_INDEX_TABLE}(r2_blob_key);

CREATE INDEX IF NOT EXISTS idx_knowledge_index_vectorize_id
ON {KNOWLEDGE_INDEX_TABLE}(vectorize_id);

CREATE TABLE IF NOT EXISTS {KNOWLEDGE_CORPUS_STATE_TABLE} (
    singleton INTEGER PRIMARY KEY NOT NULL CHECK(singleton = 1),
    revision INTEGER NOT NULL CHECK(revision >= 0)
) STRICT;

INSERT OR IGNORE INTO {KNOWLEDGE_CORPUS_STATE_TABLE}(singleton, revision)
VALUES (1, 0);
""".strip()

# Adding the supersession columns to KNOWLEDGE_SCHEMA_SQL is necessary but NOT
# sufficient. That statement is `CREATE TABLE IF NOT EXISTS`, so on a database
# that was already provisioned the table exists, the statement is a no-op, and
# the new columns never appear -- while every query that now selects them fails.
# That is the silent half-migration this pair exists to prevent.
#
# So the schema change ships with an explicit additive migration. SQLite has no
# `ADD COLUMN IF NOT EXISTS`, so each ALTER is emitted on its own and applied
# only when the column is genuinely absent -- see
# `missing_supersession_migration()` for the detection the provisioning step
# uses. Both are additive: nullable columns with no default, so an existing row
# is unaffected and no backfill is required.
KNOWLEDGE_SCHEMA_MIGRATION_SQL: tuple[str, ...] = (
    f"ALTER TABLE {KNOWLEDGE_INDEX_TABLE} ADD COLUMN superseded_by TEXT",
    f"ALTER TABLE {KNOWLEDGE_INDEX_TABLE} ADD COLUMN superseded_at REAL",
)

KNOWLEDGE_SUPERSESSION_COLUMNS: tuple[str, ...] = (
    "superseded_by",
    "superseded_at",
)

KNOWLEDGE_CORPUS_REVISION_SQL = f"""
SELECT revision
FROM {KNOWLEDGE_CORPUS_STATE_TABLE}
WHERE singleton = 1
""".strip()


KNOWLEDGE_GET_SQL = f"""
SELECT
    knowledge_id,
    topic,
    content_hash,
    authority,
    status,
    source_uri,
    retrieved_at,
    r2_blob_key,
    vectorize_id,
    superseded_by,
    superseded_at,
    corpus_revision
FROM {KNOWLEDGE_INDEX_TABLE}
WHERE knowledge_id = ?1
LIMIT 1
""".strip()


def knowledge_query_sql(status_count: int) -> str:
    """Build the bounded filtered retrieval SQL for an explicit status set."""
    if isinstance(status_count, bool) or not isinstance(status_count, int) or status_count <= 0:
        raise ValueError("status_count must be a positive integer")
    status_placeholders = ", ".join(f"?{index}" for index in range(3, 3 + status_count))
    limit_index = 3 + status_count
    return f"""
SELECT
    knowledge_id,
    topic,
    content_hash,
    authority,
    status,
    source_uri,
    retrieved_at,
    r2_blob_key,
    vectorize_id,
    superseded_by,
    superseded_at,
    corpus_revision
FROM {KNOWLEDGE_INDEX_TABLE}
WHERE instr(lower(topic), lower(?1)) > 0
  AND authority >= ?2
  AND status IN ({status_placeholders})
  AND superseded_by IS NULL
ORDER BY authority DESC, retrieved_at DESC
LIMIT ?{limit_index}
""".strip()

# #261: the resource-typed writer fence, defined ONCE.
#
# A put adopts a body AND a vector together, so either one being under lease
# must stop it. This is written as a single template used by BOTH the metadata
# upsert and the corpus-revision advance, for two reasons:
#
#   1. Correctness. The two statements run in one D1 batch and the writer
#      requires them to agree: a write of 1 with a revision of 0 (or the
#      reverse) is an inconsistent mutation count and fails closed. If the
#      fence were written twice, the copies could drift and break that
#      agreement in exactly the way the invariant exists to catch.
#   2. It is not a preflight. `gc_resource_lease_d1.GC_RESOURCE_WRITER_BLOCK_SQL`
#      can express the same question, but consuming it as
#      `blocked = SELECT ...; if blocked: return; write()` is a TOCTOU race:
#      a sweeper can acquire the lease in the gap and the write still lands.
#      The predicates below therefore live INSIDE the same compare-and-set
#      that performs the write, so the lease state and the mutation are
#      decided by one statement against one snapshot.
#
# NULL handling: when a row is not vectorized, `vectorize_id` binds as NULL and
# `resource_key = NULL` evaluates to NULL, never true, so the vector predicate
# selects no row and the NOT EXISTS holds. A non-vectorized write is therefore
# never blocked by a vector lease. `resource_key` is NOT NULL in the lease
# table, so no stored lease can match a NULL key either.
_TYPED_RESOURCE_FENCE = """
AND NOT EXISTS (
    SELECT 1
    FROM {lease_table}
    WHERE resource_type = 'r2_blob'
      AND resource_key = {blob}
      AND state IN ('active', 'delete_failed')
      AND expires_at > {now}
)
AND NOT EXISTS (
    SELECT 1
    FROM {lease_table}
    WHERE resource_type = 'vector'
      AND resource_key = {vector}
      AND state IN ('active', 'delete_failed')
      AND expires_at > {now}
)
"""

_TYPED_FENCE_UP = _TYPED_RESOURCE_FENCE.format(
    lease_table=GC_RESOURCE_LEASE_TABLE, blob="?8", vector="?9", now="?13"
)
_TYPED_FENCE_CONFLICT = _TYPED_RESOURCE_FENCE.format(
    lease_table=GC_RESOURCE_LEASE_TABLE,
    blob="excluded.r2_blob_key",
    vector="excluded.vectorize_id",
    now="?13",
)
_TYPED_FENCE_ADVANCE = _TYPED_RESOURCE_FENCE.format(
    lease_table=GC_RESOURCE_LEASE_TABLE, blob="?2", vector="?4", now="?3"
)

KNOWLEDGE_WRITER_UPSERT_SQL = f"""
INSERT INTO {KNOWLEDGE_INDEX_TABLE} (
    knowledge_id,
    topic,
    content_hash,
    authority,
    status,
    source_uri,
    retrieved_at,
    r2_blob_key,
    vectorize_id,
    superseded_by,
    superseded_at,
    corpus_revision
)
SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12 + 1
WHERE EXISTS (
    SELECT 1
    FROM {KNOWLEDGE_CORPUS_STATE_TABLE}
    WHERE singleton = 1
      AND revision = ?12
)
AND NOT EXISTS (
    SELECT 1
    FROM knowledge_gc_delete_lease
    WHERE object_key = ?8
      AND state IN ('active', 'delete_failed')
      AND expires_at > ?13
)
{_TYPED_FENCE_UP}
ON CONFLICT(knowledge_id) DO UPDATE SET
    topic = excluded.topic,
    content_hash = excluded.content_hash,
    authority = excluded.authority,
    status = excluded.status,
    source_uri = excluded.source_uri,
    retrieved_at = excluded.retrieved_at,
    r2_blob_key = excluded.r2_blob_key,
    vectorize_id = excluded.vectorize_id,
    superseded_by = excluded.superseded_by,
    superseded_at = excluded.superseded_at,
    corpus_revision = excluded.corpus_revision
WHERE EXISTS (
    SELECT 1
    FROM {KNOWLEDGE_CORPUS_STATE_TABLE}
    WHERE singleton = 1
      AND revision = ?12
)
AND NOT EXISTS (
    SELECT 1
    FROM knowledge_gc_delete_lease
    WHERE object_key = excluded.r2_blob_key
      AND state IN ('active', 'delete_failed')
      AND expires_at > ?13
)
{_TYPED_FENCE_CONFLICT}
""".strip()

# The corpus-revision advance carries the SAME fence, on the SAME parameters,
# so a fenced write cannot advance the revision without writing the row, nor
# write the row without advancing the revision. ?4 is the vectorize_id the
# write was adopting; it is new to this statement and is bound by the writer.
KNOWLEDGE_CORPUS_ADVANCE_SQL = f"""
UPDATE {KNOWLEDGE_CORPUS_STATE_TABLE}
SET revision = revision + 1
WHERE singleton = 1
  AND revision = ?1
  AND NOT EXISTS (
      SELECT 1
      FROM knowledge_gc_delete_lease
      WHERE object_key = ?2
        AND state IN ('active', 'delete_failed')
        AND expires_at > ?3
  )
{_TYPED_FENCE_ADVANCE}
""".strip()


def knowledge_schema_statements() -> tuple[str, ...]:
    return tuple(
        statement.strip()
        for statement in KNOWLEDGE_SCHEMA_SQL.split(";")
        if statement.strip()
    )


def missing_supersession_migration(existing_columns: Iterable[str]) -> tuple[str, ...]:
    """Return the ALTER statements a database still needs.

    ``existing_columns`` is the result of ``PRAGMA table_info(knowledge_index)``.
    A column already present is skipped, which is what makes provisioning
    re-runnable: a second pass applies nothing and does not raise
    "duplicate column name".
    """
    present = {str(column) for column in existing_columns}
    # strict=True so a future edit that adds a column without a matching
    # ALTER fails loudly here instead of silently dropping a migration.
    by_column = dict(
        zip(KNOWLEDGE_SUPERSESSION_COLUMNS, KNOWLEDGE_SCHEMA_MIGRATION_SQL, strict=True)
    )
    return tuple(
        by_column[column]
        for column in KNOWLEDGE_SUPERSESSION_COLUMNS
        if column not in present
    )


__all__ = [
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
    "KNOWLEDGE_SCHEMA_MIGRATION_SQL",
    "KNOWLEDGE_SUPERSESSION_COLUMNS",
    "missing_supersession_migration",
]
