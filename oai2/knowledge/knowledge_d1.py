"""D1 schema and transactional writer contract for authoritative knowledge metadata.

Public-safe SQL only: no Cloudflare account IDs, resource IDs, endpoints, or
credentials. The statements are designed for the D1 prepared/batch API.
"""

from __future__ import annotations

KNOWLEDGE_SCHEMA_VERSION = 1
KNOWLEDGE_INDEX_TABLE = "knowledge_index"
KNOWLEDGE_CORPUS_STATE_TABLE = "knowledge_corpus_state"

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
    corpus_revision INTEGER NOT NULL CHECK(corpus_revision >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_knowledge_index_r2_blob_key
ON {KNOWLEDGE_INDEX_TABLE}(r2_blob_key);

CREATE TABLE IF NOT EXISTS {KNOWLEDGE_CORPUS_STATE_TABLE} (
    singleton INTEGER PRIMARY KEY NOT NULL CHECK(singleton = 1),
    revision INTEGER NOT NULL CHECK(revision >= 0)
) STRICT;

INSERT OR IGNORE INTO {KNOWLEDGE_CORPUS_STATE_TABLE}(singleton, revision)
VALUES (1, 0);
""".strip()

KNOWLEDGE_CORPUS_REVISION_SQL = f"""
SELECT revision
FROM {KNOWLEDGE_CORPUS_STATE_TABLE}
WHERE singleton = 1
""".strip()

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
    corpus_revision
)
SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10 + 1
WHERE EXISTS (
    SELECT 1
    FROM {KNOWLEDGE_CORPUS_STATE_TABLE}
    WHERE singleton = 1
      AND revision = ?10
)
AND NOT EXISTS (
    SELECT 1
    FROM knowledge_gc_delete_lease
    WHERE object_key = ?8
      AND state IN ('active', 'delete_failed')
      AND expires_at > ?11
)
ON CONFLICT(knowledge_id) DO UPDATE SET
    topic = excluded.topic,
    content_hash = excluded.content_hash,
    authority = excluded.authority,
    status = excluded.status,
    source_uri = excluded.source_uri,
    retrieved_at = excluded.retrieved_at,
    r2_blob_key = excluded.r2_blob_key,
    vectorize_id = excluded.vectorize_id,
    corpus_revision = excluded.corpus_revision
WHERE EXISTS (
    SELECT 1
    FROM {KNOWLEDGE_CORPUS_STATE_TABLE}
    WHERE singleton = 1
      AND revision = ?10
)
AND NOT EXISTS (
    SELECT 1
    FROM knowledge_gc_delete_lease
    WHERE object_key = excluded.r2_blob_key
      AND state IN ('active', 'delete_failed')
      AND expires_at > ?11
)
""".strip()

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
""".strip()


def knowledge_schema_statements() -> tuple[str, ...]:
    return tuple(
        statement.strip()
        for statement in KNOWLEDGE_SCHEMA_SQL.split(";")
        if statement.strip()
    )


__all__ = [
    "KNOWLEDGE_SCHEMA_VERSION",
    "KNOWLEDGE_INDEX_TABLE",
    "KNOWLEDGE_CORPUS_STATE_TABLE",
    "KNOWLEDGE_SCHEMA_SQL",
    "KNOWLEDGE_CORPUS_REVISION_SQL",
    "KNOWLEDGE_WRITER_UPSERT_SQL",
    "KNOWLEDGE_CORPUS_ADVANCE_SQL",
    "knowledge_schema_statements",
]
