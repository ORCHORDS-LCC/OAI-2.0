from __future__ import annotations

import sqlite3

from oai2.knowledge.gc_lease_d1 import (
    GC_LEASE_REFERENCE_COUNT_SQL,
    GC_LEASE_SCHEMA_VERSION,
    GC_LEASE_SELECT_SQL,
    GC_LEASE_UPSERT_SQL,
    GC_LEASE_WRITER_BLOCK_SQL,
    gc_lease_schema_statements,
)


def _db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
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
    return db


def test_gc_lease_schema_is_versioned_and_applies_to_sqlite() -> None:
    db = _db()
    assert GC_LEASE_SCHEMA_VERSION == 1
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    assert ("knowledge_gc_delete_lease",) in rows


def test_gc_lease_round_trip_and_writer_blocking() -> None:
    db = _db()
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
    db = _db()
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
    db = _db()
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
