"""Public-safe D1 schema contract for GC deletion leases.

This module defines the durable D1 representation required by WI-GC-003.
It intentionally contains no account IDs, database IDs, bucket names,
endpoints, or credentials. Live Worker code is expected to apply the
statements through the bound D1Database using prepared/batched async calls.
"""

from __future__ import annotations

GC_LEASE_SCHEMA_VERSION = 1

GC_LEASE_TABLE = "knowledge_gc_delete_lease"

GC_LEASE_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {GC_LEASE_TABLE} (
    object_key TEXT PRIMARY KEY NOT NULL,
    token TEXT NOT NULL,
    owner TEXT NOT NULL,
    acquired_at REAL NOT NULL CHECK(acquired_at >= 0),
    expires_at REAL NOT NULL CHECK(expires_at > acquired_at),
    state TEXT NOT NULL CHECK(
        state IN ('active', 'delete_failed', 'deleted', 'released', 'replaced')
    ),
    failure_count INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
    finalized_decision TEXT CHECK(
        finalized_decision IS NULL OR
        finalized_decision IN (
            'deleted',
            'already_absent',
            'already_finalized',
            'retryable_failure_recorded',
            'invalid_lease'
        )
    ),
    authority_revision INTEGER NOT NULL CHECK(authority_revision >= 0),
    updated_at REAL NOT NULL CHECK(updated_at >= 0)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_gc_delete_lease_token
ON {GC_LEASE_TABLE}(token);
""".strip()

GC_LEASE_SELECT_SQL = f"""
SELECT
    object_key,
    token,
    owner,
    acquired_at,
    expires_at,
    state,
    failure_count,
    finalized_decision,
    authority_revision,
    updated_at
FROM {GC_LEASE_TABLE}
WHERE object_key = ?1
""".strip()

GC_LEASE_REFERENCE_COUNT_SQL = """
SELECT COUNT(*) AS retained_reference_count
FROM knowledge_index
WHERE r2_blob_key = ?1
""".strip()

GC_LEASE_UPSERT_SQL = f"""
INSERT INTO {GC_LEASE_TABLE} (
    object_key,
    token,
    owner,
    acquired_at,
    expires_at,
    state,
    failure_count,
    finalized_decision,
    authority_revision,
    updated_at
)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
ON CONFLICT(object_key) DO UPDATE SET
    token = excluded.token,
    owner = excluded.owner,
    acquired_at = excluded.acquired_at,
    expires_at = excluded.expires_at,
    state = excluded.state,
    failure_count = excluded.failure_count,
    finalized_decision = excluded.finalized_decision,
    authority_revision = excluded.authority_revision,
    updated_at = excluded.updated_at
""".strip()


GC_LEASE_ACQUIRE_SQL = f"""
INSERT INTO {GC_LEASE_TABLE} (
    object_key,
    token,
    owner,
    acquired_at,
    expires_at,
    state,
    failure_count,
    finalized_decision,
    authority_revision,
    updated_at
)
SELECT
    ?1, ?2, ?3, ?4, ?5, 'active', 0, NULL, ?6, ?4
WHERE NOT EXISTS (
    SELECT 1
    FROM knowledge_index
    WHERE r2_blob_key = ?1
)
ON CONFLICT(object_key) DO UPDATE SET
    token = excluded.token,
    owner = excluded.owner,
    acquired_at = excluded.acquired_at,
    expires_at = excluded.expires_at,
    state = 'active',
    failure_count = 0,
    finalized_decision = NULL,
    authority_revision = excluded.authority_revision,
    updated_at = excluded.updated_at
WHERE
    (
        {GC_LEASE_TABLE}.state IN ('released', 'replaced')
        OR (
            {GC_LEASE_TABLE}.state IN ('active', 'delete_failed')
            AND {GC_LEASE_TABLE}.expires_at <= excluded.acquired_at
        )
    )
    AND NOT EXISTS (
        SELECT 1
        FROM knowledge_index
        WHERE r2_blob_key = excluded.object_key
    )
""".strip()

GC_LEASE_WRITER_BLOCK_SQL = f"""
SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM {GC_LEASE_TABLE}
        WHERE object_key = ?1
          AND state IN ('active', 'delete_failed')
          AND expires_at > ?2
    )
    THEN 1
    ELSE 0
END AS writer_blocked
""".strip()


GC_LEASE_VALIDATE_SQL = f"""
SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM {GC_LEASE_TABLE}
        WHERE object_key = ?1
          AND token = ?2
          AND state IN ('active', 'delete_failed')
          AND expires_at > ?3
    )
    AND NOT EXISTS (
        SELECT 1
        FROM knowledge_index
        WHERE r2_blob_key = ?1
    )
    THEN 1
    ELSE 0
END AS lease_valid
""".strip()


GC_LEASE_RECORD_FAILURE_SQL = f"""
UPDATE {GC_LEASE_TABLE}
SET
    state = 'delete_failed',
    failure_count = failure_count + 1,
    finalized_decision = 'retryable_failure_recorded',
    authority_revision = ?4,
    updated_at = ?3
WHERE object_key = ?1
  AND token = ?2
  AND state IN ('active', 'delete_failed')
  AND expires_at > ?3
  AND NOT EXISTS (
      SELECT 1
      FROM knowledge_index
      WHERE r2_blob_key = ?1
  )
""".strip()

GC_LEASE_FINALIZE_SQL = f"""
UPDATE {GC_LEASE_TABLE}
SET
    state = 'deleted',
    finalized_decision = ?4,
    authority_revision = ?5,
    updated_at = ?3
WHERE object_key = ?1
  AND token = ?2
  AND state IN ('active', 'delete_failed')
  AND expires_at > ?3
  AND ?4 IN ('deleted', 'already_absent')
  AND NOT EXISTS (
      SELECT 1
      FROM knowledge_index
      WHERE r2_blob_key = ?1
  )
""".strip()

GC_LEASE_RELEASE_SQL = f"""
UPDATE {GC_LEASE_TABLE}
SET
    state = 'released',
    finalized_decision = NULL,
    authority_revision = ?4,
    updated_at = ?3
WHERE object_key = ?1
  AND token = ?2
  AND state IN ('active', 'delete_failed')
  AND expires_at > ?3
""".strip()


def gc_lease_schema_statements() -> tuple[str, ...]:
    """Return schema statements in deterministic application order."""
    return tuple(
        statement.strip()
        for statement in GC_LEASE_SCHEMA_SQL.split(";")
        if statement.strip()
    )


__all__ = [
    "GC_LEASE_SCHEMA_VERSION",
    "GC_LEASE_TABLE",
    "GC_LEASE_SCHEMA_SQL",
    "GC_LEASE_SELECT_SQL",
    "GC_LEASE_REFERENCE_COUNT_SQL",
    "GC_LEASE_UPSERT_SQL",
    "GC_LEASE_ACQUIRE_SQL",
    "GC_LEASE_WRITER_BLOCK_SQL",
    "GC_LEASE_VALIDATE_SQL",
    "GC_LEASE_RECORD_FAILURE_SQL",
    "GC_LEASE_FINALIZE_SQL",
    "GC_LEASE_RELEASE_SQL",
    "gc_lease_schema_statements",
]
