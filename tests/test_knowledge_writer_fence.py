"""#261 — the writer fence must live INSIDE the compare-and-set, not beside it.

THE DEFECT THIS FILE REPRODUCES
-------------------------------
``gc_resource_lease_d1`` defines ``GC_RESOURCE_WRITER_BLOCK_SQL``, a fence that
checks a resource-typed lease. Nothing consumes it. ``D1KnowledgeWriter`` calls
``KNOWLEDGE_WRITER_UPSERT_SQL`` and ``KNOWLEDGE_CORPUS_ADVANCE_SQL``, and both
of those consult only the old #233 R2-only ``knowledge_gc_delete_lease``. A
writer can therefore adopt a vector that a sweeper holds an active lease on, and
commit it, while the corpus revision advances to match.

The tempting one-line fix — SELECT the fence, then write — is a TOCTOU race and
is explicitly rejected: the check and the write must be ONE authoritative D1
compare-and-set, or a sweeper can acquire the lease in the gap and the write
still lands. So the fence predicates are placed inside both CAS statements.

No live binding, no live delete, no network, no migration.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from oai2.knowledge.gc_lease_d1 import gc_lease_schema_statements
from oai2.knowledge.gc_resource_lease_d1 import (
    GC_RESOURCE_LEASE_ACQUIRE_SQL,
    gc_resource_lease_schema_statements,
)
from oai2.knowledge.knowledge_d1 import (
    KNOWLEDGE_CORPUS_ADVANCE_SQL,
    KNOWLEDGE_WRITER_UPSERT_SQL,
    knowledge_schema_statements,
)

BLOB_A = "oai2-blobs/" + "a" * 64
BLOB_B = "oai2-blobs/" + "b" * 64
VEC_NEW = "oai2v1-" + "2" * 56
VEC_OTHER = "oai2v1-" + "3" * 56
KID = "ko_fence"
NOW = 1000.0
EXPIRES = 2000.0
REV = 7


@contextmanager
def _db(revision: int = REV) -> Iterator[sqlite3.Connection]:
    """A D1 model built from the REAL schema statements, verbatim.

    Hand-rolled tables would let a fence predicate pass against a column the
    real schema does not have, so the real statements are used unchanged.
    """
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for statement in knowledge_schema_statements():
            connection.execute(statement)
        connection.execute(
            "UPDATE knowledge_corpus_state SET revision = ?", (revision,)
        )
        for statement in gc_lease_schema_statements():
            connection.execute(statement)
        for statement in gc_resource_lease_schema_statements():
            connection.execute(statement)
        yield connection
    finally:
        connection.close()


def _acquire_resource_lease(
    db: sqlite3.Connection,
    resource_type: str,
    resource_key: str,
    *,
    acquired: float = NOW,
    expires: float = EXPIRES,
) -> int:
    """Acquire a typed lease through the REAL acquire statement."""
    cursor = db.execute(
        GC_RESOURCE_LEASE_ACQUIRE_SQL,
        (resource_type, resource_key, "tok-1", "gc-sweep", acquired, expires, REV),
    )
    return cursor.rowcount


def _writer_batch(
    db: sqlite3.Connection,
    *,
    blob_key: str,
    vector_id: str | None,
    expected_revision: int = REV,
    now: float = NOW,
) -> tuple[int, int]:
    """Run the REAL writer batch: upsert + corpus advance.

    Returns ``(write_changes, revision_changes)``, which is what
    ``D1KnowledgeWriter`` inspects to decide denied vs committed. Both
    statements run in one transaction so the pair is the real atomic unit.
    """
    db.execute("BEGIN")
    try:
        upsert = db.execute(
            KNOWLEDGE_WRITER_UPSERT_SQL,
            (KID, "topic", "c" * 64, 0.5, "active", "https://example.test/x",
             NOW, blob_key, vector_id, expected_revision, now),
        )
        advance = db.execute(
            KNOWLEDGE_CORPUS_ADVANCE_SQL,
            (expected_revision, blob_key, now, vector_id),
        )
        write_changes = upsert.rowcount
        revision_changes = advance.rowcount
    finally:
        db.commit()
    return write_changes, revision_changes


def _revision(db: sqlite3.Connection) -> int:
    row = db.execute(
        "SELECT revision FROM knowledge_corpus_state WHERE singleton = 1"
    ).fetchone()
    assert row is not None
    return int(row[0])


def _adopted_vector(db: sqlite3.Connection) -> str | None:
    row = db.execute(
        "SELECT vectorize_id FROM knowledge_index WHERE knowledge_id = ?", (KID,)
    ).fetchone()
    return None if row is None else row[0]


# ---------------------------------------------------------------------------
# THE GAP — what the real SQL does today, before the fix
# ---------------------------------------------------------------------------


def test_an_active_vector_lease_stops_the_writer() -> None:
    """A sweeper holds VEC_NEW. The writer must not adopt it.

    The race #261 describes: a sweeper has decided VEC_NEW is unreferenced and
    is deleting it; a concurrent put makes it authoritative; the finalize then
    re-checks the R2-only reference predicate, which never looks at
    ``vectorize_id``, finds nothing, and the vector is destroyed while a live
    row points at it.

    RED STATE, recorded deliberately: before the fix this statement pair
    returned ``(1, 1)`` here — the row was written and the corpus revision
    advanced, with an active lease on the very vector being adopted. It is kept
    in the same form so the fix is visible as a change in outcome, not a change
    in assertion.
    """
    with _db() as db:
        assert _acquire_resource_lease(db, "vector", VEC_NEW) == 1

        write_changes, revision_changes = _writer_batch(
            db, blob_key=BLOB_A, vector_id=VEC_NEW
        )

        assert (write_changes, revision_changes) == (0, 0), (
            "the writer adopted a vector under an active lease: "
            f"{write_changes=}, {revision_changes=}"
        )
        assert _adopted_vector(db) is None
        assert _revision(db) == REV, "the corpus revision advanced anyway"


def test_an_active_typed_r2_lease_stops_the_writer() -> None:
    """The same on the R2 side, for the NEW typed table.

    RED STATE, as above: the old #233 table was consulted, so this specific
    case returned ``(1, 1)`` because the writer SQL never mentioned the typed
    table at all.
    """
    with _db() as db:
        assert _acquire_resource_lease(db, "r2_blob", BLOB_A) == 1

        write_changes, revision_changes = _writer_batch(
            db, blob_key=BLOB_A, vector_id=VEC_NEW
        )

        assert (write_changes, revision_changes) == (0, 0), (
            "the writer adopted a body under an active typed lease: "
            f"{write_changes=}, {revision_changes=}"
        )
        assert _revision(db) == REV


def test_the_cas_statements_consult_the_resource_lease_table() -> None:
    """Structural proof, independent of any test data.

    A preflight SELECT could satisfy a behavioural test while still being a
    TOCTOU race, so the requirement is that the CAS statements THEMSELVES
    consult the typed table — for both the body and the vector, in BOTH
    statements.
    """
    from oai2.knowledge.gc_resource_lease_d1 import GC_RESOURCE_LEASE_TABLE

    for statement in (KNOWLEDGE_WRITER_UPSERT_SQL, KNOWLEDGE_CORPUS_ADVANCE_SQL):
        assert GC_RESOURCE_LEASE_TABLE in statement
        assert "resource_type = 'vector'" in statement
        assert "resource_type = 'r2_blob'" in statement
        # The old #233 fence must survive alongside it, not be replaced.
        assert "knowledge_gc_delete_lease" in statement
    # Both the insert branch and the conflict branch must be fenced, or a
    # repeated put of the same knowledge_id would walk straight past the fence.
    upsert = KNOWLEDGE_WRITER_UPSERT_SQL
    assert upsert.count("FROM " + GC_RESOURCE_LEASE_TABLE) == 4, (
        "expected both resource types in both the INSERT and DO UPDATE branches"
    )
    assert upsert.count("FROM knowledge_gc_delete_lease") == 2
    assert KNOWLEDGE_CORPUS_ADVANCE_SQL.count(
        "FROM " + GC_RESOURCE_LEASE_TABLE
    ) == 2


def test_the_fence_is_not_a_preflight_select() -> None:
    """The rejected one-line fix must stay rejected.

    ``GC_RESOURCE_WRITER_BLOCK_SQL`` answers the right question as a bare
    SELECT. Consuming it as a preflight is the TOCTOU shape: check, then write,
    with a window in which a sweeper can acquire the lease. So the statement may
    exist, but the writer must not be reachable through it.
    """
    import inspect

    from oai2.knowledge import gc_resource_lease_d1, knowledge_d1_runtime

    writer_source = inspect.getsource(knowledge_d1_runtime)
    assert "GC_RESOURCE_WRITER_BLOCK_SQL" not in writer_source, (
        "the writer must not preflight the fence; that reintroduces the race"
    )
    # It is still exported and usable by an operator-facing tool if needed.
    assert hasattr(gc_resource_lease_d1, "GC_RESOURCE_WRITER_BLOCK_SQL")
    assert gc_resource_lease_d1.GC_RESOURCE_WRITER_BLOCK_SQL.lstrip().upper().startswith(
        "SELECT"
    )


# ---------------------------------------------------------------------------
# RACE ACCEPTANCE A-K
#
# Every direction, not just the one that was reported. A fence that only blocks
# the case in the bug report is not a fence.
# ---------------------------------------------------------------------------


def test_A_a_vector_lease_first_denies_the_writer() -> None:
    """A. Lease first, writer adopts that vector -> (0, 0), denied."""
    with _db() as db:
        assert _acquire_resource_lease(db, "vector", VEC_NEW) == 1
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=VEC_NEW) == (0, 0)


def test_B_a_typed_r2_lease_first_denies_the_writer() -> None:
    """B. Typed body lease first, writer adopts that body -> (0, 0), denied."""
    with _db() as db:
        assert _acquire_resource_lease(db, "r2_blob", BLOB_A) == 1
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=VEC_NEW) == (0, 0)


def test_C_the_legacy_r2_lease_still_blocks() -> None:
    """C. #233's own lease is untouched and still fences the writer."""
    from oai2.knowledge.gc_lease_d1 import GC_LEASE_ACQUIRE_SQL

    with _db() as db:
        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (BLOB_A, "tok-legacy", "gc-sweep", NOW, EXPIRES, REV),
        )
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=VEC_NEW) == (0, 0)


def test_D_a_writer_that_commits_first_makes_acquisition_fail() -> None:
    """D. Writer first -> the resource is authoritative, so no lease can start.

    The other direction of the same race. Fencing the writer alone would be
    half a fix: without this, a sweeper could take a lease on a resource a
    committed row already references.
    """
    with _db() as db:
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=VEC_NEW) == (1, 1)
        assert _revision(db) == REV + 1
        assert _acquire_resource_lease(db, "vector", VEC_NEW) == 0, (
            "a lease was acquired on a resource the writer already owns"
        )


def test_E_an_unrelated_vector_lease_does_not_block() -> None:
    """E. Lease on a different vector -> the write proceeds."""
    with _db() as db:
        assert _acquire_resource_lease(db, "vector", VEC_OTHER) == 1
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=VEC_NEW) == (1, 1)


def test_F_an_unrelated_body_lease_does_not_block() -> None:
    """F. Lease on a different body -> the write proceeds."""
    with _db() as db:
        assert _acquire_resource_lease(db, "r2_blob", BLOB_B) == 1
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=VEC_NEW) == (1, 1)


def test_G_an_expired_vector_lease_permits_the_writer() -> None:
    """G. Expired lease -> not blocking, matching the #233 expiry policy."""
    with _db() as db:
        assert _acquire_resource_lease(
            db, "vector", VEC_NEW, acquired=NOW, expires=NOW + 10
        ) == 1
        # Evaluate the fence well after expiry.
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=VEC_NEW, now=NOW + 50) == (1, 1)


def test_H_a_released_vector_lease_permits_the_writer() -> None:
    """H. 'released' is not a blocking state."""
    from oai2.knowledge.gc_resource_lease_d1 import GC_RESOURCE_LEASE_RELEASE_SQL

    with _db() as db:
        assert _acquire_resource_lease(db, "vector", VEC_NEW) == 1
        db.execute(
            GC_RESOURCE_LEASE_RELEASE_SQL,
            ("vector", VEC_NEW, "tok-1", NOW + 1, REV),
        )
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=VEC_NEW, now=NOW + 2) == (1, 1)


def test_I_a_delete_failed_unexpired_vector_lease_blocks() -> None:
    """I. 'delete_failed' IS blocking: the object is still mid-sweep."""
    from oai2.knowledge.gc_resource_lease_d1 import (
        GC_RESOURCE_LEASE_RECORD_FAILURE_SQL,
    )

    with _db() as db:
        assert _acquire_resource_lease(db, "vector", VEC_NEW) == 1
        db.execute(
            GC_RESOURCE_LEASE_RECORD_FAILURE_SQL,
            ("vector", VEC_NEW, "tok-1", NOW + 1, REV),
        )
        state = db.execute(
            "SELECT state FROM knowledge_gc_resource_lease"
        ).fetchone()
        assert state[0] == "delete_failed"
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=VEC_NEW, now=NOW + 2) == (0, 0)


def test_J_a_null_vectorize_id_cannot_be_blocked_by_a_vector_lease() -> None:
    """J. A non-vectorized write is never blocked by some other vector's lease.

    This is the NULL trap. `resource_key = NULL` is never true in SQL, so the
    predicate matches nothing; the risk is a future edit that makes the write
    depend on it in the other direction, or an IS NULL comparison that would
    match every unvectorized row against every lease.
    """
    with _db() as db:
        assert _acquire_resource_lease(db, "vector", VEC_NEW) == 1
        # A lease exists on a vector, and the row adopts NO vector at all.
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=None) == (1, 1)
        row = db.execute(
            "SELECT vectorize_id FROM knowledge_index WHERE knowledge_id = ?", (KID,)
        ).fetchone()
        assert row is not None and row[0] is None


def test_K_the_type_namespace_keeps_identical_keys_independent() -> None:
    """K. The same string as a body key and as a vector key does not collide.

    This is the whole reason the lease table is resource-typed: #233 keys on a
    bare object_key, so one string cannot be both.
    """
    shared = "oai2v1-" + "9" * 56
    with _db() as db:
        # A lease on it as a VECTOR must not block a write that uses it as a
        # body key, and vice versa.
        assert _acquire_resource_lease(db, "vector", shared) == 1
        assert _writer_batch(db, blob_key=shared, vector_id=VEC_OTHER) == (1, 1)

    with _db() as db:
        assert _acquire_resource_lease(db, "r2_blob", shared) == 1
        assert _writer_batch(db, blob_key=BLOB_A, vector_id=shared) == (1, 1)

    # And the primary key genuinely namespaces by type.
    with _db() as db:
        _acquire_resource_lease(db, "vector", shared)
        _acquire_resource_lease(db, "r2_blob", shared, expires=EXPIRES)
        rows = db.execute(
            "SELECT resource_type FROM knowledge_gc_resource_lease ORDER BY resource_type"
        ).fetchall()
        assert rows == [("r2_blob",), ("vector",)]


def test_the_two_statements_never_disagree_on_the_fence() -> None:
    """The mutation-count invariant must stay fail-closed.

    ``D1KnowledgeWriter`` treats (0,0) as denied and (1,1) as committed, and
    raises on anything else. That safety depends on the upsert and the advance
    evaluating the SAME fence against the SAME parameters. Any state that
    splits them would show up here as a count that is neither.
    """
    scenarios = {
        "vector lease": ("vector", VEC_NEW, VEC_NEW),
        "body lease": ("r2_blob", BLOB_A, VEC_NEW),
        "no lease": None,
        "unrelated lease": ("vector", VEC_OTHER, VEC_NEW),
    }
    for label, lease in scenarios.items():
        with _db() as db:
            if lease is not None:
                _acquire_resource_lease(db, lease[0], lease[1])
            write_changes, revision_changes = _writer_batch(
                db, blob_key=BLOB_A, vector_id=lease[2] if lease else VEC_NEW
            )
            assert (write_changes, revision_changes) in ((0, 0), (1, 1)), label
            assert write_changes == revision_changes, (
                f"{label}: upsert and advance disagreed, which the writer "
                "turns into an inconsistent-mutation-count failure"
            )
