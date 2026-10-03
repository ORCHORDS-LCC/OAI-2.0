"""#261 / WI-GC-004 — the lease must fence the resource a put actually adopts.

Two groups here, and the distinction matters.

``TestR2OnlyLeaseCannotFenceVectorAdoption`` documents the GAP, using the
existing ``knowledge_gc_delete_lease`` SQL unchanged. Those statements are
#233's, they are not modified by this work, so these tests keep describing the
real R2-only behaviour after the design below is adopted. They are the evidence
that the gap is real rather than imagined.

``TestResourceTypedLeaseInvariants`` proves the required properties of the
resource-typed design in ``gc_resource_lease_d1`` against a real SQLite D1
model, so nothing here is asserted about a schema that has not been executed.

No live binding, no live delete, no network.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from oai2.knowledge.gc_lease_d1 import (
    GC_LEASE_ACQUIRE_SQL,
    GC_LEASE_FINALIZE_SQL,
    GC_LEASE_SCHEMA_VERSION,
    GC_LEASE_VALIDATE_SQL,
    GC_LEASE_WRITER_BLOCK_SQL,
    gc_lease_schema_statements,
)
from oai2.knowledge.gc_resource_lease_d1 import (
    GC_RESOURCE_LEASE_ACQUIRE_SQL,
    GC_RESOURCE_LEASE_FINALIZE_SQL,
    GC_RESOURCE_LEASE_RECORD_FAILURE_SQL,
    GC_RESOURCE_LEASE_RELEASE_SQL,
    GC_RESOURCE_LEASE_SCHEMA_VERSION,
    GC_RESOURCE_LEASE_VALIDATE_SQL,
    GC_RESOURCE_REFERENCE_COUNT_SQL,
    GC_RESOURCE_TYPE_R2_BLOB,
    GC_RESOURCE_TYPE_VECTOR,
    GC_RESOURCE_WRITER_BLOCK_SQL,
    gc_resource_lease_schema_statements,
)
from oai2.knowledge.knowledge_d1 import (
    KNOWLEDGE_WRITER_UPSERT_SQL,
    knowledge_schema_statements,
)

BLOB_A = "oai2-blobs/" + "a" * 64
BLOB_B = "oai2-blobs/" + "b" * 64
VEC_OLD = "oai2v1-" + "1" * 56
VEC_NEW = "oai2v1-" + "2" * 56
KID = "ko_lease"
NOW = 1000.0
EXPIRES = 2000.0
REV = 7


@contextmanager
def _authoritative_db() -> Iterator[sqlite3.Connection]:
    """A D1 model built from the REAL knowledge schema.

    The authoritative statements are used verbatim rather than a hand-rolled
    table, so a writer path that needs a column the real schema lacks fails
    here rather than passing against a convenient stub.
    """
    db = sqlite3.connect(":memory:")
    try:
        for statement in knowledge_schema_statements():
            db.execute(statement)
        db.execute("UPDATE knowledge_corpus_state SET revision = ?", (REV,))
        for statement in gc_lease_schema_statements():
            db.execute(statement)
        for statement in gc_resource_lease_schema_statements():
            db.execute(statement)
        yield db
    finally:
        db.close()


def _writer_upsert(
    db: sqlite3.Connection,
    *,
    blob_key: str,
    vector_id: str,
    at: float = NOW,
) -> int:
    """Run the CURRENT writer upsert.

    Returns the committed row's corpus_revision, or -1 when the write was
    refused. The corpus STATE is advanced by a separate statement, so it is the
    row that shows whether an adoption actually committed.
    """
    db.execute(
        KNOWLEDGE_WRITER_UPSERT_SQL,
        (KID, "topic", "c" * 64, 0.5, "active", "https://example.test/x", NOW,
         blob_key, vector_id, REV, at),
    )
    row = db.execute(
        "SELECT corpus_revision FROM knowledge_index WHERE knowledge_id = ?", (KID,)
    ).fetchone()
    return -1 if row is None else int(row[0])


def _acquire_r2_lease(db: sqlite3.Connection, object_key: str) -> None:
    db.execute(
        GC_LEASE_ACQUIRE_SQL,
        (object_key, "tok-1", "gc-sweep", NOW, EXPIRES, REV),
    )


def _scalar(db: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> int:
    row = db.execute(sql, params).fetchone()
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# THE GAP, against the real R2-only statements
# ---------------------------------------------------------------------------


def test_r2_lease_schema_has_no_resource_type() -> None:
    """A lease cannot say WHICH KIND of object it holds, so it cannot hold a
    vector as distinct from a body."""
    with _authoritative_db() as db:
        cols = {
            row[1] for row in db.execute("PRAGMA table_info(knowledge_gc_delete_lease)")
        }
        assert "resource_type" not in cols
        # The primary key is a bare key, so a body key and a vector id collide.
        pk = [row[1] for row in db.execute("PRAGMA table_info(knowledge_gc_delete_lease)")
              if row[5]]
        assert pk == ["object_key"]


def test_reference_count_for_a_vector_id_is_always_zero() -> None:
    """The R2 sweep's liveness question, asked about a vector.

    A row HAS adopted the vector. The R2 reference query still reports zero,
    because it only ever looks at r2_blob_key.
    """
    with _authoritative_db() as db:
        _writer_upsert(db, blob_key=BLOB_A, vector_id=VEC_OLD)
        from oai2.knowledge.gc_lease_d1 import GC_LEASE_REFERENCE_COUNT_SQL

        assert _scalar(db, GC_LEASE_REFERENCE_COUNT_SQL, (VEC_OLD,)) == 0, (
            "a vector that D1 references reads as unreferenced to the R2 sweep"
        )
        # ...while the body key of the same row is seen correctly.
        assert _scalar(db, GC_LEASE_REFERENCE_COUNT_SQL, (BLOB_A,)) == 1


def test_writer_fence_is_consulted_for_the_vector_it_adopts() -> None:
    """A lease is held on the vector, and the writer must not adopt it.

    HISTORY, kept deliberately: this test used to be named
    ``test_writer_fence_is_not_consulted_for_the_vector_it_adopts`` and asserted
    the opposite. It was the evidence that the #261 gap was real — the fence
    was only ever asked about the blob key, and a put adopted the vector id with
    no check at all. The gap is now closed by moving the fence INSIDE the
    writer's compare-and-set, so the assertion is inverted here.

    The pre-fix behaviour is still pinned, as the RED record, in
    ``tests/test_knowledge_writer_fence.py``.
    """
    with _authoritative_db() as db:
        # The scenario that matters in #261 is a TYPED vector lease, which is
        # what the sweeper now takes out on a vector.
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW)
        # The typed fence answers the question, for the vector as well as the
        # body key.
        assert _scalar(db, GC_RESOURCE_WRITER_BLOCK_SQL, (BLOB_B, VEC_NEW, NOW)) == 1

        revision = _writer_upsert(db, blob_key=BLOB_B, vector_id=VEC_NEW)
        row = db.execute(
            "SELECT vectorize_id FROM knowledge_index WHERE knowledge_id = ?", (KID,)
        ).fetchone()
        assert row is None, "a put adopted a vector under an active lease"
        assert revision == -1, "the refused write still advanced the revision"

        # Structurally, the vector predicate is now inside the writer SQL.
        assert "vectorize_id" in KNOWLEDGE_WRITER_UPSERT_SQL


def test_a_legacy_r2_lease_keyed_on_a_vector_still_does_not_fence() -> None:
    """An unsupported, degenerate state, recorded rather than papered over.

    The #233 table is R2-only and keyed on a bare ``object_key``; nothing
    prevents a caller putting a vector id in that column. The writer consults
    that table with the BODY key, so such a row matches nothing and the write
    proceeds.

    This is left visible on purpose. It is a malformed lease, not a supported
    one, and closing it would mean changing #233's table semantics — which owns
    live lease state and is explicitly out of scope here. The real vector path
    is covered by the typed lease above.
    """
    with _authoritative_db() as db:
        _acquire_r2_lease(db, VEC_NEW)
        # The legacy fence asked about the vector does notice...
        assert _scalar(db, GC_LEASE_WRITER_BLOCK_SQL, (VEC_NEW, NOW)) == 1
        # ...but the writer asks about the body key, which is not that key.
        assert _scalar(db, GC_LEASE_WRITER_BLOCK_SQL, (BLOB_B, NOW)) == 0


def test_r2_finalize_cannot_detect_that_a_vector_became_authoritative() -> None:
    """The exact sequence that loses live data.

    Sweeper validates, a writer adopts, the sweeper finalizes. The R2 finalize
    re-checks the reference predicate as a last safety net — and for a vector
    key that predicate is vacuously true, so it authorizes the delete of a
    vector that is now the authoritative one.
    """
    with _authoritative_db() as db:
        # A previous row references the OLD vector; the new one is unreferenced.
        _writer_upsert(db, blob_key=BLOB_A, vector_id=VEC_OLD)
        # A sweeper reaches a conclusion about VEC_NEW under an R2 lease.
        db.execute(
            GC_LEASE_ACQUIRE_SQL, (VEC_NEW, "tok-vec", "gc-sweep", NOW, EXPIRES, REV)
        )
        # It validates: no row references VEC_NEW by r2_blob_key. True, and wrong.
        assert _scalar(
            db, GC_LEASE_VALIDATE_SQL, (VEC_NEW, "tok-vec", NOW + 10)
        ) == 1, "the R2 validate cannot express a vector reference at all"

        # A writer adopts VEC_NEW in between.
        _writer_upsert(db, blob_key=BLOB_B, vector_id=VEC_NEW)

        # The sweeper re-validates and finalizes. The net does not catch it.
        assert _scalar(
            db, GC_LEASE_VALIDATE_SQL, (VEC_NEW, "tok-vec", NOW + 20)
        ) == 1, "re-validation still cannot see the new reference"
        db.execute(
            GC_LEASE_FINALIZE_SQL, (VEC_NEW, "tok-vec", NOW + 20, "deleted", REV)
        )
        state = db.execute(
            "SELECT state, finalized_decision FROM knowledge_gc_delete_lease "
            "WHERE object_key = ?",
            (VEC_NEW,),
        ).fetchone()
        assert state == ("deleted", "deleted"), (
            "the R2 lease authorized a delete of a now-authoritative vector"
        )


# ---------------------------------------------------------------------------
# THE DESIGN: one resource-typed lease, proven against executed SQLite
# ---------------------------------------------------------------------------


def test_resource_typed_schema_is_versioned_and_separate() -> None:
    # Separate table, so #233's live R2 lease rows and recovery semantics are
    # untouched and no migration is implied by this source change.
    with _authoritative_db() as db:
        assert GC_RESOURCE_LEASE_SCHEMA_VERSION == 1
        assert GC_LEASE_SCHEMA_VERSION == 1
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "knowledge_gc_delete_lease" in tables
        assert "knowledge_gc_resource_lease" in tables
        cols = {
            row[1]
            for row in db.execute("PRAGMA table_info(knowledge_gc_resource_lease)")
        }
        assert {"resource_type", "resource_key"} <= cols


def _acquire(
    db: sqlite3.Connection,
    resource_type: str,
    resource_key: str,
    *,
    token: str = "tok-1",
    at: float = NOW,
    expires: float | None = None,
) -> None:
    db.execute(
        GC_RESOURCE_LEASE_ACQUIRE_SQL,
        (resource_type, resource_key, token, "gc-sweep", at,
         EXPIRES if expires is None else expires, REV),
    )


def test_same_key_string_in_two_resource_types_is_two_different_leases() -> None:
    """A body key and a vector id can never share a lease.

    They occupy the same namespace as strings, so the type must be part of the
    identity. Without it, leasing a body key would also lease an unrelated
    vector that happens to have the same key.
    """
    with _authoritative_db() as db:
        _acquire(db, GC_RESOURCE_TYPE_R2_BLOB, "shared-key")
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, "shared-key", token="tok-2")
        rows = db.execute(
            "SELECT resource_type, token FROM knowledge_gc_resource_lease "
            "ORDER BY resource_type"
        ).fetchall()
        assert rows == [("r2_blob", "tok-1"), ("vector", "tok-2")]


def test_reference_count_is_answered_per_resource_type() -> None:
    with _authoritative_db() as db:
        _writer_upsert(db, blob_key=BLOB_A, vector_id=VEC_OLD)
        assert _scalar(
            db, GC_RESOURCE_REFERENCE_COUNT_SQL, (GC_RESOURCE_TYPE_R2_BLOB, BLOB_A)
        ) == 1
        assert _scalar(
            db, GC_RESOURCE_REFERENCE_COUNT_SQL, (GC_RESOURCE_TYPE_VECTOR, VEC_OLD)
        ) == 1
        assert _scalar(
            db, GC_RESOURCE_REFERENCE_COUNT_SQL, (GC_RESOURCE_TYPE_VECTOR, VEC_NEW)
        ) == 0


def test_a_vector_referenced_at_any_embedding_version_is_not_an_orphan() -> None:
    """The migration guard (REQ-GC-031).

    A row naming an older-version vector is still a live reference. Scoping
    liveness to the current embedding version would let a sweep delete a live
    vector mid-re-embedding, so no version filter may enter this predicate.
    """
    with _authoritative_db() as db:
        db.execute(
            "INSERT INTO knowledge_index"
            "(knowledge_id, topic, content_hash, authority, status, source_uri,"
            " retrieved_at, r2_blob_key, vectorize_id, corpus_revision) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (KID, "t", "c" * 64, 0.5, "active", "https://example.test/x",
             NOW, BLOB_A, VEC_OLD, REV),
        )
        # No embedding version is part of the reference question at all.
        assert _scalar(
            db, GC_RESOURCE_REFERENCE_COUNT_SQL, (GC_RESOURCE_TYPE_VECTOR, VEC_OLD)
        ) == 1
        # Acquiring a lease on a referenced vector is refused.
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_OLD)
        assert db.execute(
            "SELECT COUNT(*) FROM knowledge_gc_resource_lease"
        ).fetchone()[0] == 0, "a lease was acquired on a referenced vector"


def test_writer_fence_blocks_adoption_of_a_leased_vector() -> None:
    """The fence the R2 lease could not provide."""
    with _authoritative_db() as db:
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW)
        assert _scalar(
            db, GC_RESOURCE_WRITER_BLOCK_SQL, (BLOB_B, VEC_NEW, NOW)
        ) == 1, "adoption of a leased vector is not blocked"
        # An unrelated vector is unaffected.
        assert _scalar(
            db, GC_RESOURCE_WRITER_BLOCK_SQL, (BLOB_B, VEC_OLD, NOW)
        ) == 0
        # A released or expired lease stops blocking.
        db.execute(
            GC_RESOURCE_LEASE_RELEASE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", NOW + 1, REV),
        )
        assert _scalar(
            db, GC_RESOURCE_WRITER_BLOCK_SQL, (BLOB_B, VEC_NEW, NOW + 2)
        ) == 0


def test_writer_fence_also_blocks_a_leased_body_key() -> None:
    """A put adopts a body and a vector together; either lease stops it."""
    with _authoritative_db() as db:
        _acquire(db, GC_RESOURCE_TYPE_R2_BLOB, BLOB_B)
        assert _scalar(
            db, GC_RESOURCE_WRITER_BLOCK_SQL, (BLOB_B, VEC_NEW, NOW)
        ) == 1
        assert _scalar(
            db, GC_RESOURCE_WRITER_BLOCK_SQL, (BLOB_A, VEC_NEW, NOW)
        ) == 0


def test_the_losing_writer_is_refused_and_the_winner_keeps_its_vector() -> None:
    """End-to-end: sweeper leases, writer is blocked, sweep completes, then a
    later writer may adopt the reclaimed id."""
    with _authoritative_db() as db:
        _writer_upsert(db, blob_key=BLOB_A, vector_id=VEC_OLD)
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW)

        # The writer consults the fence and is told to stop.
        blocked = _scalar(
            db, GC_RESOURCE_WRITER_BLOCK_SQL, (BLOB_B, VEC_NEW, NOW)
        )
        assert blocked == 1
        assert _scalar(
            db, GC_RESOURCE_REFERENCE_COUNT_SQL, (GC_RESOURCE_TYPE_VECTOR, VEC_NEW)
        ) == 0
        assert db.execute(
            "SELECT vectorize_id FROM knowledge_index WHERE knowledge_id = ?", (KID,)
        ).fetchone()[0] == VEC_OLD, "the authoritative vector changed while leased"

        # Sweep finalizes cleanly, because nothing ever adopted the vector.
        assert _scalar(
            db,
            GC_RESOURCE_LEASE_VALIDATE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", NOW + 5),
        ) == 1
        db.execute(
            GC_RESOURCE_LEASE_FINALIZE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", NOW + 5, "deleted", REV),
        )
        # After the sweep the id is free, and a new put may adopt it.
        assert _scalar(
            db, GC_RESOURCE_WRITER_BLOCK_SQL, (BLOB_B, VEC_NEW, NOW + 6)
        ) == 0


def test_an_adoption_that_races_a_sweeper_still_cannot_be_deleted() -> None:
    """The unsafe sequence, caught by the typed re-check.

    The writer fence is the FIRST line of defence. This tests the SECOND: a
    sweeper re-validates immediately before acting, and the typed predicate
    looks at ``vectorize_id``, so a reference created in the window is seen and
    the lease refuses to validate.

    The adoption is written DIRECTLY, not through ``_writer_upsert``, because
    the writer is now fenced and can no longer perform the unsafe step this
    test needs to simulate. A row inserted directly represents a writer that
    bypassed the fence, or a row that predates the lease — the sweeper must be
    safe against both.
    """
    with _authoritative_db() as db:
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW)
        # An adoption that got past the fence anyway.
        db.execute(
            "INSERT INTO knowledge_index ("
            "knowledge_id, topic, content_hash, authority, status, source_uri,"
            "retrieved_at, r2_blob_key, vectorize_id, corpus_revision) "
            "VALUES (?, 'topic', ?, 0.5, 'active', NULL, ?, ?, ?, ?)",
            (KID, "c" * 64, NOW, BLOB_B, VEC_NEW, REV),
        )

        # The sweeper re-validates and the typed predicate catches the race.
        assert _scalar(
            db,
            GC_RESOURCE_LEASE_VALIDATE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", NOW + 5),
        ) == 0, "the re-check failed to see the new reference"

        # Finalize is refused for the same reason, so no delete is authorized.
        db.execute(
            GC_RESOURCE_LEASE_FINALIZE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", NOW + 5, "deleted", REV),
        )
        state = db.execute(
            "SELECT state FROM knowledge_gc_resource_lease "
            "WHERE resource_type = ? AND resource_key = ?",
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW),
        ).fetchone()
        assert state == ("active",), "finalize authorized a delete of a live vector"


def test_finalize_is_idempotent_and_a_repeat_does_not_report_false_failure() -> None:
    with _authoritative_db() as db:
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW)
        db.execute(
            GC_RESOURCE_LEASE_FINALIZE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", NOW + 1, "deleted", REV),
        )
        cur = db.execute(
            GC_RESOURCE_LEASE_FINALIZE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", NOW + 2, "already_absent", REV),
        )
        assert cur.rowcount == 0, "a second finalize must be a no-op"
        assert db.execute(
            "SELECT finalized_decision FROM knowledge_gc_resource_lease"
        ).fetchone()[0] == "deleted"


def test_failure_recording_is_bounded_and_then_refused_on_re_adoption() -> None:
    """Failure recording stops as soon as the resource is referenced again.

    The re-adoption here is written DIRECTLY rather than through the writer.
    That is deliberate: the writer is now fenced, so it can no longer adopt a
    vector under an unexpired lease, and driving this through it would assert
    the bug rather than the lease invariant. The invariant under test belongs
    to the lease statement — "a referenced resource's lease is not mutated" —
    so the authoritative row is inserted as a row that predates the lease.
    """
    with _authoritative_db() as db:
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW)
        db.execute(
            GC_RESOURCE_LEASE_RECORD_FAILURE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", NOW + 1, REV),
        )
        row = db.execute(
            "SELECT state, failure_count, finalized_decision "
            "FROM knowledge_gc_resource_lease"
        ).fetchone()
        assert row == ("delete_failed", 1, "retryable_failure_recorded")
        # The vector becomes authoritative, as a prior row would have.
        db.execute(
            "INSERT INTO knowledge_index ("
            "knowledge_id, topic, content_hash, authority, status, source_uri,"
            "retrieved_at, r2_blob_key, vectorize_id, corpus_revision) "
            "VALUES (?, 't', ?, 0.5, 'active', NULL, ?, ?, ?, ?)",
            (KID, "c" * 64, NOW, BLOB_B, VEC_NEW, REV),
        )
        db.execute(
            GC_RESOURCE_LEASE_RECORD_FAILURE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", NOW + 2, REV),
        )
        assert db.execute(
            "SELECT failure_count FROM knowledge_gc_resource_lease"
        ).fetchone()[0] == 1


def test_a_stale_authority_revision_blocks_acquisition() -> None:
    with _authoritative_db() as db:
        db.execute(
            GC_RESOURCE_LEASE_ACQUIRE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, VEC_NEW, "tok-1", "gc-sweep", NOW, EXPIRES, REV + 5),
        )
        assert db.execute(
            "SELECT COUNT(*) FROM knowledge_gc_resource_lease"
        ).fetchone()[0] == 0


def test_expired_lease_can_be_reacquired_and_live_one_cannot() -> None:
    with _authoritative_db() as db:
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW, at=NOW)
        # A different owner cannot steal a live lease.
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW, token="tok-2", at=NOW + 1)
        assert db.execute(
            "SELECT token FROM knowledge_gc_resource_lease"
        ).fetchone()[0] == "tok-1"
        # Once expired it can.
        _acquire(
            db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW, token="tok-2",
            at=EXPIRES + 1, expires=EXPIRES + 1000,
        )
        assert db.execute(
            "SELECT token FROM knowledge_gc_resource_lease"
        ).fetchone()[0] == "tok-2"


def test_the_lease_stores_no_credential_or_topology() -> None:
    with _authoritative_db() as db:
        _acquire(db, GC_RESOURCE_TYPE_VECTOR, VEC_NEW)
        cols = {row[1] for row in db.execute("PRAGMA table_info(knowledge_gc_resource_lease)")}
        for forbidden in ("account_id", "database_id", "bucket", "index_name",
                          "namespace", "token_secret", "api_key"):
            assert forbidden not in cols
        # The owner is a logical name, not a credential.
        assert db.execute("SELECT owner FROM knowledge_gc_resource_lease").fetchone()[0] == (
            "gc-sweep"
        )


@pytest.mark.parametrize(
    "bad_type", ["", "body", "R2_BLOB", "VECTOR", "r2", "vectors"]
)
def test_only_declared_resource_types_are_storable(bad_type: str) -> None:
    """An unrecognised type must not silently behave like an unreferenced
    resource, which would make every resource of that type deletable."""
    with _authoritative_db() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                GC_RESOURCE_LEASE_ACQUIRE_SQL,
                (bad_type, "k", "tok", "gc-sweep", NOW, EXPIRES, REV),
            )
