from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from oai2.knowledge.gc_lease_d1 import (
    GC_LEASE_ACQUIRE_SQL,
    GC_LEASE_FINALIZE_SQL,
    GC_LEASE_RECORD_FAILURE_SQL,
    GC_LEASE_REFERENCE_COUNT_SQL,
    GC_LEASE_RELEASE_SQL,
    GC_LEASE_SCHEMA_VERSION,
    GC_LEASE_SELECT_SQL,
    GC_LEASE_UPSERT_SQL,
    GC_LEASE_WRITER_BLOCK_SQL,
    gc_lease_schema_statements,
)


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(":memory:")
    try:
        db.execute(
            """
            CREATE TABLE knowledge_index (
                knowledge_id TEXT PRIMARY KEY NOT NULL,
                r2_blob_key TEXT
            ) STRICT
            """
        )
        for statement in gc_lease_schema_statements():
            db.execute(statement)
        yield db
    finally:
        db.close()


def test_gc_lease_schema_is_versioned_and_applies_to_sqlite() -> None:
    with _db() as db:
        assert GC_LEASE_SCHEMA_VERSION == 1
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        assert ("knowledge_gc_delete_lease",) in rows


def test_gc_lease_round_trip_and_writer_blocking() -> None:
    with _db() as db:
        db.execute(
            GC_LEASE_UPSERT_SQL,
            (
                "oai2-blobs/abc",
                "lease-token-1",
                "gc-sweep",
                10.0,
                20.0,
                "active",
                0,
                None,
                7,
                10.0,
            ),
        )
        row = db.execute(GC_LEASE_SELECT_SQL, ("oai2-blobs/abc",)).fetchone()
        assert row is not None
        assert row[1] == "lease-token-1"
        assert row[5] == "active"
        assert db.execute(
            GC_LEASE_WRITER_BLOCK_SQL, ("oai2-blobs/abc", 15.0)
        ).fetchone() == (1,)
        assert db.execute(
            GC_LEASE_WRITER_BLOCK_SQL, ("oai2-blobs/abc", 21.0)
        ).fetchone() == (0,)


def test_reference_count_uses_authoritative_knowledge_index() -> None:
    with _db() as db:
        db.executemany(
            "INSERT INTO knowledge_index(knowledge_id, r2_blob_key) VALUES (?, ?)",
            (
                ("ko_1", "oai2-blobs/shared"),
                ("ko_2", "oai2-blobs/shared"),
                ("ko_3", "oai2-blobs/other"),
            ),
        )
        assert db.execute(
            GC_LEASE_REFERENCE_COUNT_SQL, ("oai2-blobs/shared",)
        ).fetchone() == (2,)


def test_schema_rejects_invalid_lease_state_and_negative_failure_count() -> None:
    with _db() as db:
        bad_state = (
            "oai2-blobs/bad-state",
            "lease-token-2",
            "gc-sweep",
            10.0,
            20.0,
            "not-a-state",
            0,
            None,
            1,
            10.0,
        )
        try:
            db.execute(GC_LEASE_UPSERT_SQL, bad_state)
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("invalid lease state must be rejected")

        bad_failures = (
            "oai2-blobs/bad-failures",
            "lease-token-3",
            "gc-sweep",
            10.0,
            20.0,
            "active",
            -1,
            None,
            1,
            10.0,
        )
        try:
            db.execute(GC_LEASE_UPSERT_SQL, bad_failures)
        except sqlite3.IntegrityError:
            return
        raise AssertionError("negative failure_count must be rejected")



def test_conditional_acquire_blocks_referenced_object_and_live_lease() -> None:
    with _db() as db:
        key = "oai2-blobs/shared"
        db.execute(
            "INSERT INTO knowledge_index(knowledge_id, r2_blob_key) VALUES (?, ?)",
            ("ko_1", key),
        )
        before = db.total_changes
        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (key, "lease-1", "gc-sweep", 10.0, 20.0, 7),
        )
        assert db.total_changes == before

        db.execute(
            "DELETE FROM knowledge_index WHERE knowledge_id = ?",
            ("ko_1",),
        )
        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (key, "lease-1", "gc-sweep", 10.0, 20.0, 7),
        )
        row = db.execute(GC_LEASE_SELECT_SQL, (key,)).fetchone()
        assert row is not None
        assert row[1] == "lease-1"

        before = db.total_changes
        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (key, "lease-2", "gc-sweep", 15.0, 25.0, 8),
        )
        assert db.total_changes == before
        row = db.execute(GC_LEASE_SELECT_SQL, (key,)).fetchone()
        assert row is not None
        assert row[1] == "lease-1"


def test_conditional_acquire_allows_expired_takeover_but_rechecks_references() -> None:
    with _db() as db:
        key = "oai2-blobs/a"
        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (key, "lease-old", "gc-sweep", 10.0, 15.0, 7),
        )
        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (key, "lease-new", "gc-sweep", 20.0, 30.0, 8),
        )
        row = db.execute(GC_LEASE_SELECT_SQL, (key,)).fetchone()
        assert row is not None
        assert row[1] == "lease-new"
        assert row[8] == 8

        db.execute(
            "INSERT INTO knowledge_index(knowledge_id, r2_blob_key) VALUES (?, ?)",
            ("ko_2", key),
        )
        before = db.total_changes
        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (key, "lease-third", "gc-sweep", 40.0, 50.0, 9),
        )
        assert db.total_changes == before
        row = db.execute(GC_LEASE_SELECT_SQL, (key,)).fetchone()
        assert row is not None
        assert row[1] == "lease-new"



def test_failure_finalize_and_release_transitions_are_token_guarded() -> None:
    with _db() as db:
        key = "oai2-blobs/a"
        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (key, "lease-1", "gc-sweep", 10.0, 30.0, 7),
        )

        before = db.total_changes
        db.execute(
            GC_LEASE_RECORD_FAILURE_SQL,
            (key, "wrong-token", 12.0, 8),
        )
        assert db.total_changes == before

        db.execute(
            GC_LEASE_RECORD_FAILURE_SQL,
            (key, "lease-1", 12.0, 8),
        )
        row = db.execute(GC_LEASE_SELECT_SQL, (key,)).fetchone()
        assert row is not None
        assert row[5] == "delete_failed"
        assert row[6] == 1
        assert row[7] == "retryable_failure_recorded"

        db.execute(
            GC_LEASE_RELEASE_SQL,
            (key, "lease-1", 13.0, 9),
        )
        row = db.execute(GC_LEASE_SELECT_SQL, (key,)).fetchone()
        assert row is not None
        assert row[5] == "released"
        assert row[7] is None

        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (key, "lease-2", "gc-sweep", 14.0, 30.0, 10),
        )
        db.execute(
            GC_LEASE_FINALIZE_SQL,
            (key, "lease-2", 15.0, "deleted", 11),
        )
        row = db.execute(GC_LEASE_SELECT_SQL, (key,)).fetchone()
        assert row is not None
        assert row[5] == "deleted"
        assert row[7] == "deleted"


def test_finalize_fails_when_reference_reappears() -> None:
    with _db() as db:
        key = "oai2-blobs/a"
        db.execute(
            GC_LEASE_ACQUIRE_SQL,
            (key, "lease-1", "gc-sweep", 10.0, 30.0, 7),
        )
        db.execute(
            "INSERT INTO knowledge_index(knowledge_id, r2_blob_key) VALUES (?, ?)",
            ("ko_race", key),
        )
        before = db.total_changes
        db.execute(
            GC_LEASE_FINALIZE_SQL,
            (key, "lease-1", 15.0, "deleted", 8),
        )
        assert db.total_changes == before
        row = db.execute(GC_LEASE_SELECT_SQL, (key,)).fetchone()
        assert row is not None
        assert row[5] == "active"
