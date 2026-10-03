"""Resource-typed GC deletion leases, including Vectorize vectors.

Why this module exists (#261, parent #209, WI-GC-004)
-------------------------------------------------------
#209's deletion lease is R2-only in three separate ways, and all three matter:

  1. The lease table has no resource type. ``object_key`` is the primary key, so
     an R2 body and a Vectorize vector share one namespace and cannot be
     distinguished.
  2. Every reference predicate is ``knowledge_index.r2_blob_key = ?``. A vector
     is never consulted, so "is this still unreferenced?" asks about the wrong
     column for a vector.
  3. The writer fence is consulted with the blob key only. ``put()`` adopts a
     ``vectorize_id`` without any check against a lease on that vector.

(3) is the dangerous one, and the existing R2 sweep's own re-check cannot catch
it: the R2 finalize re-verifies ``NOT EXISTS (... r2_blob_key = ?)``, which for
a vector key is trivially true, so the finalize succeeds for a vector that a
concurrent writer has just made authoritative.

The fix is to make the lease resource-typed, so ONE state machine, ONE
idempotency path and ONE fence apply to every kind of deletable object. This is
deliberately a separate table from ``knowledge_gc_delete_lease`` rather than a
migration of it: #233 owns the live R2 lease, its tests, and its recovery
semantics, and an in-place PK change on a table that holds live lease state is a
migration that needs its own plan and authorization. Nothing here rewrites or
invalidates an existing R2 lease row.

The reference predicate is expressed once, as a CASE over the resource type,
rather than as a pair of near-identical statements, so the two resource types
cannot drift apart in how they answer "is this still referenced?".

Public-safe: no account ids, database ids, bucket names, index names, endpoints
or credentials.
"""

from __future__ import annotations

GC_RESOURCE_LEASE_SCHEMA_VERSION = 1

GC_RESOURCE_LEASE_TABLE = "knowledge_gc_resource_lease"

GC_RESOURCE_TYPE_R2_BLOB = "r2_blob"
GC_RESOURCE_TYPE_VECTOR = "vector"
GC_RESOURCE_TYPES = (GC_RESOURCE_TYPE_R2_BLOB, GC_RESOURCE_TYPE_VECTOR)

# A resource key is namespaced by type in exactly one place. The CASE below is
# that place: it is the ONLY statement that decides which knowledge_index column
# a resource type lives in.
_REFERENCE_PREDICATE = """
NOT EXISTS (
    SELECT 1
    FROM knowledge_index
    WHERE CASE ?1
        WHEN 'r2_blob' THEN r2_blob_key
        WHEN 'vector' THEN vectorize_id
    END = ?2
)
"""

GC_RESOURCE_LEASE_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {GC_RESOURCE_LEASE_TABLE} (
    resource_type TEXT NOT NULL CHECK(resource_type IN ('r2_blob', 'vector')),
    resource_key TEXT NOT NULL,
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
    updated_at REAL NOT NULL CHECK(updated_at >= 0),
    PRIMARY KEY (resource_type, resource_key)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_gc_resource_lease_token
ON {GC_RESOURCE_LEASE_TABLE}(resource_type, token);
""".strip()

GC_RESOURCE_LEASE_SELECT_SQL = f"""
SELECT
    resource_type,
    resource_key,
    token,
    owner,
    acquired_at,
    expires_at,
    state,
    failure_count,
    finalized_decision,
    authority_revision,
    updated_at
FROM {GC_RESOURCE_LEASE_TABLE}
WHERE resource_type = ?1 AND resource_key = ?2
""".strip()

# How many authoritative rows currently reference this resource. For a vector
# that is any row naming it in vectorize_id, INCLUDING a row at a different
# embedding_version: during a re-embedding a row legitimately references an
# older-version vector, and scoping this to the current version would let a
# sweep delete a live vector mid-migration.
GC_RESOURCE_REFERENCE_COUNT_SQL = """
SELECT COUNT(*) AS retained_reference_count
FROM knowledge_index
WHERE CASE ?1
    WHEN 'r2_blob' THEN r2_blob_key
    WHEN 'vector' THEN vectorize_id
END = ?2
""".strip()

# Acquire only when the authoritative revision is unchanged AND nothing
# references the resource. The reference test is evaluated at acquire time so a
# lease can never begin life already stale.
GC_RESOURCE_LEASE_ACQUIRE_SQL = f"""
INSERT INTO {GC_RESOURCE_LEASE_TABLE} (
    resource_type, resource_key, token, owner, acquired_at, expires_at,
    state, failure_count, finalized_decision, authority_revision, updated_at
)
SELECT
    ?1, ?2, ?3, ?4, ?5, ?6, 'active', 0, NULL, ?7, ?5
WHERE EXISTS (
    SELECT 1 FROM knowledge_corpus_state WHERE singleton = 1 AND revision = ?7
)
AND {_REFERENCE_PREDICATE}
ON CONFLICT(resource_type, resource_key) DO UPDATE SET
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
        {GC_RESOURCE_LEASE_TABLE}.state IN ('released', 'replaced')
        OR (
            {GC_RESOURCE_LEASE_TABLE}.state IN ('active', 'delete_failed')
            AND {GC_RESOURCE_LEASE_TABLE}.expires_at <= excluded.acquired_at
        )
    )
    AND EXISTS (
        SELECT 1 FROM knowledge_corpus_state
        WHERE singleton = 1 AND revision = excluded.authority_revision
    )
    AND {_REFERENCE_PREDICATE}
""".strip()

# THE FENCE. A writer must not adopt a resource a sweeper is holding. This
# takes BOTH resources a put touches, because a single put adopts a body and a
# vector together and either one being leased must stop it.
GC_RESOURCE_WRITER_BLOCK_SQL = f"""
SELECT CASE
    WHEN EXISTS (
        SELECT 1 FROM {GC_RESOURCE_LEASE_TABLE}
        WHERE resource_type = 'r2_blob' AND resource_key = ?1
          AND state IN ('active', 'delete_failed') AND expires_at > ?3
    )
    OR EXISTS (
        SELECT 1 FROM {GC_RESOURCE_LEASE_TABLE}
        WHERE resource_type = 'vector' AND resource_key = ?2
          AND state IN ('active', 'delete_failed') AND expires_at > ?3
    )
    THEN 1
    ELSE 0
END AS writer_blocked
""".strip()

# Revalidate immediately before acting. This re-checks the reference predicate
# for the resource TYPE, which is the re-check that makes the R2 sweep safe and
# which the R2-only lease silently cannot perform for a vector.
GC_RESOURCE_LEASE_VALIDATE_SQL = f"""
SELECT CASE
    WHEN EXISTS (
        SELECT 1 FROM {GC_RESOURCE_LEASE_TABLE}
        WHERE resource_type = ?1 AND resource_key = ?2
          AND token = ?3
          AND state IN ('active', 'delete_failed')
          AND expires_at > ?4
    )
    AND {_REFERENCE_PREDICATE}
    THEN 1
    ELSE 0
END AS lease_valid
""".strip()

GC_RESOURCE_LEASE_RECORD_FAILURE_SQL = f"""
UPDATE {GC_RESOURCE_LEASE_TABLE}
SET
    state = 'delete_failed',
    failure_count = failure_count + 1,
    finalized_decision = 'retryable_failure_recorded',
    authority_revision = ?5,
    updated_at = ?4
WHERE resource_type = ?1 AND resource_key = ?2
  AND token = ?3
  AND state IN ('active', 'delete_failed')
  AND expires_at > ?4
  AND {_REFERENCE_PREDICATE}
""".strip()

GC_RESOURCE_LEASE_FINALIZE_SQL = f"""
UPDATE {GC_RESOURCE_LEASE_TABLE}
SET
    state = 'deleted',
    finalized_decision = ?5,
    authority_revision = ?6,
    updated_at = ?4
WHERE resource_type = ?1 AND resource_key = ?2
  AND token = ?3
  AND state IN ('active', 'delete_failed')
  AND expires_at > ?4
  AND ?5 IN ('deleted', 'already_absent')
  AND {_REFERENCE_PREDICATE}
""".strip()

GC_RESOURCE_LEASE_RELEASE_SQL = f"""
UPDATE {GC_RESOURCE_LEASE_TABLE}
SET
    state = 'released',
    finalized_decision = NULL,
    authority_revision = ?5,
    updated_at = ?4
WHERE resource_type = ?1 AND resource_key = ?2
  AND token = ?3
  AND state IN ('active', 'delete_failed')
  AND expires_at > ?4
""".strip()


def gc_resource_lease_schema_statements() -> tuple[str, ...]:
    """Return schema statements in deterministic application order."""
    return tuple(
        statement.strip()
        for statement in GC_RESOURCE_LEASE_SCHEMA_SQL.split(";")
        if statement.strip()
    )


__all__ = [
    "GC_RESOURCE_LEASE_SCHEMA_VERSION",
    "GC_RESOURCE_LEASE_TABLE",
    "GC_RESOURCE_LEASE_SCHEMA_SQL",
    "GC_RESOURCE_LEASE_SELECT_SQL",
    "GC_RESOURCE_REFERENCE_COUNT_SQL",
    "GC_RESOURCE_LEASE_ACQUIRE_SQL",
    "GC_RESOURCE_WRITER_BLOCK_SQL",
    "GC_RESOURCE_LEASE_VALIDATE_SQL",
    "GC_RESOURCE_LEASE_RECORD_FAILURE_SQL",
    "GC_RESOURCE_LEASE_FINALIZE_SQL",
    "GC_RESOURCE_LEASE_RELEASE_SQL",
    "GC_RESOURCE_TYPE_R2_BLOB",
    "GC_RESOURCE_TYPE_VECTOR",
    "GC_RESOURCE_TYPES",
    "gc_resource_lease_schema_statements",
]
