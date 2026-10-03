"""#261 / WI-GC-004 — dry-run reconciliation matrix, written BEFORE any
destructive capability exists.

Nothing here deletes anything, and the reconciler has no delete path to call.
That is the point of doing the matrix first: a classification that is wrong
about liveness is far cheaper to find now than after a live sweep exists.

Every case runs through the real D1 model, so the mark set is what D1 actually
says rather than a hand-written set.
"""

from __future__ import annotations

import dataclasses
import inspect
import pathlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from oai2.knowledge import gc_vector
from oai2.knowledge.gc_resource_lease_d1 import (
    GC_RESOURCE_LEASE_ACQUIRE_SQL,
    GC_RESOURCE_TYPE_VECTOR,
)
from oai2.knowledge.gc_vector import (
    VectorGcDisposition,
    VectorGcReconciliationState,
    VectorInventoryEntry,
)
from oai2.knowledge.knowledge_d1 import knowledge_schema_statements

OBSERVED_AT = 5000.0
HASH = "a" * 64


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(":memory:")
    try:
        for statement in knowledge_schema_statements():
            db.execute(statement)
        yield db
    finally:
        db.close()


@dataclasses.dataclass(frozen=True)
class _Row:
    knowledge_id: str
    vectorize_id: str | None
    embedding_version: str | None = None


def _row(
    db: sqlite3.Connection,
    *,
    knowledge_id: str,
    vector_id: str | None,
    embedding_version: str | None = "embed-v1",
) -> _Row:
    db.execute(
        "INSERT INTO knowledge_index"
        "(knowledge_id, topic, content_hash, authority, status, source_uri,"
        " retrieved_at, r2_blob_key, vectorize_id, corpus_revision)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (knowledge_id, "t", HASH, 0.5, "active", None, 1.0,
         f"oai2-blobs/{HASH}", vector_id, 1),
    )
    return _Row(knowledge_id=knowledge_id, vectorize_id=vector_id,
                embedding_version=embedding_version)


def _reconcile(
    db: sqlite3.Connection,
    inventory: list[str],
    *,
    pages: int = 1,
) -> dict[str, VectorGcDisposition]:
    state = VectorGcReconciliationState.from_rows(
        _rows(db), observed_at=OBSERVED_AT
    )
    entries = [
        VectorInventoryEntry(
            vector_id=vector_id,
            knowledge_id=None,
            content_hash=None,
            embedding_version="embed-v1",
        )
        for vector_id in inventory
    ]
    if pages == 1:
        state.consume_page(entries, next_cursor=None)
    else:
        # Split across pages so the completeness gate is exercised.
        for i in range(pages):
            chunk = entries[i::pages]
            state.consume_page(
                chunk, next_cursor=None if i == pages - 1 else f"cursor-{i}"
            )
    return {r.vector_id: r.disposition for r in state.build_report().records}


def _rows(db: sqlite3.Connection) -> list[_Row]:
    return [
        _Row(knowledge_id=r[0], vectorize_id=r[1], embedding_version=r[2])
        for r in db.execute(
            "SELECT knowledge_id, vectorize_id, corpus_revision FROM knowledge_index"
        )
    ]


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def test_referenced_and_present_is_kept() -> None:
    with _db() as db:
        _row(db, knowledge_id="ko_a", vector_id="v-present")
        assert _reconcile(db, ["v-present"]) == {
            "v-present": VectorGcDisposition.REFERENCED_PRESENT
        }


def test_referenced_but_absent_is_an_integrity_state_not_a_candidate() -> None:
    """A row naming a vector the index lacks is a reportable integrity state.

    It is emphatically NOT a delete candidate: the authoritative row is the
    reason to look, and the index is not the authority on what should exist.
    """
    with _db() as db:
        _row(db, knowledge_id="ko_a", vector_id="v-missing")
        result = _reconcile(db, [])
        assert result == {"v-missing": VectorGcDisposition.REFERENCED_MISSING}
        state = VectorGcReconciliationState.from_rows(_rows(db), observed_at=OBSERVED_AT)
        state.consume_page([], next_cursor=None)
        report = state.build_report()
        assert report.referenced_missing_count == 1
        assert report.unreferenced_candidate_count == 0, (
            "a missing referenced vector was classified as deletable"
        )


def test_unreferenced_and_present_is_a_candidate() -> None:
    with _db() as db:
        _row(db, knowledge_id="ko_a", vector_id="v-live")
        result = _reconcile(db, ["v-live", "v-orphan"])
        assert result["v-live"] is VectorGcDisposition.REFERENCED_PRESENT
        assert result["v-orphan"] is VectorGcDisposition.UNREFERENCED_CANDIDATE


def test_a_row_without_a_vector_neither_keeps_nor_breaks_anything() -> None:
    with _db() as db:
        _row(db, knowledge_id="ko_a", vector_id=None)
        state = VectorGcReconciliationState.from_rows(_rows(db), observed_at=OBSERVED_AT)
        state.consume_page(
            [VectorInventoryEntry(vector_id="v-orphan")], next_cursor=None
        )
        report = state.build_report()
        assert [r.vector_id for r in report.records] == ["v-orphan"]
        assert report.unreferenced_candidate_count == 1


def test_legacy_id_is_referenced_and_never_a_candidate() -> None:
    """REQ-GC-032: a pre-#19 row names its vector by knowledge_id.

    It needs no special case because the mark set is whatever D1 says. Pinned
    here because a migration that "normalises" ids could quietly break it.
    """
    with _db() as db:
        legacy = "ko_legacy"
        _row(db, knowledge_id=legacy, vector_id=legacy)
        result = _reconcile(db, [legacy])
        assert result[legacy] is VectorGcDisposition.REFERENCED_PRESENT


def test_a_vector_referenced_at_an_older_embedding_version_is_not_a_candidate() -> None:
    """REQ-GC-031: the migration guard.

    A half-migrated corpus references older-version vectors. A version filter in
    the mark set would classify every one of them as an orphan and a sweep
    would delete live data.
    """
    with _db() as db:
        _row(db, knowledge_id="ko_a", vector_id="v-v0", embedding_version="embed-v0")
        result = _reconcile(db, ["v-v0", "v-v1"])
        assert result["v-v0"] is VectorGcDisposition.REFERENCED_PRESENT
        assert result["v-v1"] is VectorGcDisposition.UNREFERENCED_CANDIDATE
        # The reference set itself carries no version, so nothing can filter on it.
        state = VectorGcReconciliationState.from_rows(_rows(db), observed_at=OBSERVED_AT)
        assert set(state.references) == {"v-v0"}


def test_an_inventory_entry_the_index_volunteers_is_never_promoted() -> None:
    """Extra inventory entries are candidates, not keeps, and never candidates
    that were not in the inventory."""
    with _db() as db:
        _row(db, knowledge_id="ko_a", vector_id="v-live")
        result = _reconcile(db, ["v-live", "v-extra", "v-other"])
        assert result["v-extra"] is VectorGcDisposition.UNREFERENCED_CANDIDATE
        assert result["v-other"] is VectorGcDisposition.UNREFERENCED_CANDIDATE
        assert "v-unseen" not in result, "a vector absent from inventory appeared"


def test_an_incomplete_scan_refuses_to_produce_a_report() -> None:
    """The safety gate.

    If pages are left unconsumed, unseen vectors would be classified
    REFERENCED_MISSING and unseen bodies absent from the orphan set — and
    orphans are what gets deleted. So there is no report until the source says
    it is done.
    """
    with _db() as db:
        _row(db, knowledge_id="ko_a", vector_id="v-live")
        state = VectorGcReconciliationState.from_rows(_rows(db), observed_at=OBSERVED_AT)
        state.consume_page([VectorInventoryEntry(vector_id="v-1")], next_cursor="more")

        # No report while a page is outstanding: unseen vectors would be
        # classified REFERENCED_MISSING and unseen ones would simply be absent.
        with pytest.raises(RuntimeError, match="incomplete"):
            state.build_report()
        assert not state.inventory_complete

        # Finishing the scan produces a report.
        state.consume_page([], next_cursor=None)
        assert state.inventory_complete
        state.build_report()

        # ...and the scan cannot be extended afterwards.
        with pytest.raises(RuntimeError, match="already complete"):
            state.consume_page([], next_cursor=None)

    # An empty scan with no pages at all cannot report either.
    empty = VectorGcReconciliationState(observed_at=OBSERVED_AT)
    with pytest.raises(RuntimeError, match="incomplete"):
        empty.build_report()


def test_paginated_scan_classifies_the_union_correctly() -> None:
    with _db() as db:
        _row(db, knowledge_id="ko_a", vector_id="v-1")
        _row(db, knowledge_id="ko_b", vector_id="v-2")
        result = _reconcile(db, ["v-1", "v-2", "v-3", "v-4"], pages=3)
        assert result["v-1"] is VectorGcDisposition.REFERENCED_PRESENT
        assert result["v-2"] is VectorGcDisposition.REFERENCED_PRESENT
        assert result["v-3"] is VectorGcDisposition.UNREFERENCED_CANDIDATE
        assert result["v-4"] is VectorGcDisposition.UNREFERENCED_CANDIDATE


def test_conflicting_inventory_metadata_for_one_vector_is_refused() -> None:
    with _db() as db:
        _row(db, knowledge_id="ko_a", vector_id="v-1")
        state = VectorGcReconciliationState.from_rows(_rows(db), observed_at=OBSERVED_AT)
        with pytest.raises(ValueError, match="conflicting inventory metadata"):
            state.consume_page(
                [
                    VectorInventoryEntry(vector_id="v-1", embedding_version="a"),
                    VectorInventoryEntry(vector_id="v-1", embedding_version="b"),
                ],
                next_cursor=None,
            )


def test_inventory_never_carries_embedding_values() -> None:
    """A vector's float payload is large, unnecessary for liveness, and has no
    business in a report."""
    fields = {f.name for f in dataclasses.fields(VectorInventoryEntry)}
    assert "values" not in fields
    assert fields == {
        "vector_id", "knowledge_id", "content_hash", "embedding_version"
    }


def test_the_reconciler_module_has_no_destructive_surface() -> None:
    """Structural guarantee, not a convention.

    If deletion is ever added, this fails and the addition has to be an
    explicit, reviewed act rather than something that appeared alongside a
    classification fix.
    """
    source = inspect.getsource(gc_vector)
    for forbidden in (
        "deleteByIds", "delete_by_ids", "delete(",
        "CloudflareVectorizeStore", "VectorizeIndexBinding", "requests", "httpx",
    ):
        assert forbidden not in source, f"{forbidden!r} appears in the dry-run module"

    exported = set(gc_vector.__all__)
    assert not any("delete" in name.lower() for name in exported)
    for name in exported:
        member = getattr(gc_vector, name)
        if inspect.isclass(member) or inspect.isfunction(member):
            doc = inspect.getdoc(member) or ""
            assert "delete" not in doc.lower().split("\n")[0]


# ---------------------------------------------------------------------------
# §12 — a FAILED PUT must not delete anything
# ---------------------------------------------------------------------------


def test_a_failed_put_yields_a_candidate_and_never_a_delete() -> None:
    """The regression #261 asks for, and the reason the design is safe.

    A put whose D1 write is denied leaves a vector that no row references. That
    is the ONLY new object this system creates by failing, and the correct
    handling is: classify it, grace-window it, and reclaim it through
    reconciliation under authorization. It is never rolled back inline, because
    a compensating delete would race a concurrent writer that legitimately
    adopted the same generation.
    """
    from oai2.knowledge.cloudflare_runtime import (
        AsyncCloudflareKnowledgeRuntime,
        KnowledgeConflictError,
    )
    from oai2.knowledge.transport import vector_id_for

    class _Reader:
        def __init__(self) -> None:
            self.rows: dict[str, object] = {}

        async def get_row(self, knowledge_id: str) -> object | None:
            return self.rows.get(str(knowledge_id))

    class _Writer:
        def __init__(self) -> None:
            self.revision = 4
            self.writes: list[object] = []
            self.deny_write = False

        async def corpus_revision(self) -> int:
            return self.revision

        async def write_metadata(
            self, row: object, *, expected_revision: int, now: float
        ) -> int | None:
            if self.deny_write:
                return None
            self.writes.append(row)
            return self.revision + 1

    class _R2:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}

        async def put_text(self, key: str, value: str) -> None:
            self.values[key] = value

        async def get_text(self, key: str) -> str | None:
            return self.values.get(key)

    class _Vectorize:
        def __init__(self) -> None:
            self.vectors: dict[str, dict[str, object]] = {}
            self.deleted: list[str] = []

        async def upsert(self, vector_id: str, values: object, *, metadata: object) -> None:
            self.vectors[vector_id] = {"values": list(values), "metadata": metadata}

        # Present only to PROVE it is never called by the put path.
        async def deleteByIds(self, ids: list[str]) -> None:  # pragma: no cover
            self.deleted.extend(ids)

    class _Kv:
        async def get(self, key: str) -> object | None:
            return None

    class _Obj:
        def __init__(self, content: str) -> None:
            self.knowledge_id = "ko_failed"
            self.topic = "t"
            self.content_hash = __import__("hashlib").sha256(
                content.encode()
            ).hexdigest()
            self.authority = 0.5
            self.status = "IMPLEMENTED"
            self.source_uri = None
            self.retrieved_at = 1.0
            self.content = content
            # object_to_row maps the supersession fields; this stub stands in
            # for a KnowledgeObject, so it must carry them too.
            self.superseded_by = None
            self.superseded_at = None

    async def scenario() -> tuple[list[str], _Vectorize]:
        reader, writer, r2, vectorize, kv = _Reader(), _Writer(), _R2(), _Vectorize(), _Kv()
        runtime = AsyncCloudflareKnowledgeRuntime(
            reader=reader, writer=writer, r2=r2, vectorize=vectorize, kv=kv,
            embedding_version="embed-v1", embedding_digest="d1",
        )
        committed = _Obj("first generation")
        await runtime.put(committed, vector=[1.0, 0.0], now=1.0)
        committed_row = writer.writes[-1]
        reader.rows["ko_failed"] = committed_row

        writer.deny_write = True
        denied = _Obj("denied generation")
        with pytest.raises(KnowledgeConflictError):
            await runtime.put(denied, vector=[0.0, 1.0], now=2.0)
        return sorted(vectorize.vectors), vectorize

    import asyncio

    inventory, vectorize = asyncio.run(scenario())

    # Two vectors exist; only one is referenced.
    assert len(inventory) == 2
    # No delete was issued as rollback, by any path.
    assert vectorize.deleted == [], "a failed put issued a compensating delete"

    # The denied generation is a CANDIDATE, and the committed one is not.
    with _db() as db:
        orphan = vector_id_for(
            knowledge_id="ko_failed",
            content_hash=__import__("hashlib").sha256(
                b"denied generation"
            ).hexdigest(),
            embedding_version="embed-v1",
        )
        live = vector_id_for(
            knowledge_id="ko_failed",
            content_hash=__import__("hashlib").sha256(
                b"first generation"
            ).hexdigest(),
            embedding_version="embed-v1",
        )
        db.execute(
            "INSERT INTO knowledge_index"
            "(knowledge_id, topic, content_hash, authority, status, source_uri,"
            " retrieved_at, r2_blob_key, vectorize_id, corpus_revision)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("ko_failed", "t", HASH, 0.5, "active", None, 1.0, f"oai2-blobs/{HASH}",
             live, 2),
        )
        result = _reconcile(db, inventory)
        assert result[live] is VectorGcDisposition.REFERENCED_PRESENT
        assert result[orphan] is VectorGcDisposition.UNREFERENCED_CANDIDATE

        # And it can be leased as a candidate, which is the only route to a
        # reclaimable state.
        db.execute(
            "INSERT INTO knowledge_corpus_state(singleton, revision) VALUES (1, 2) "
            "ON CONFLICT(singleton) DO UPDATE SET revision = 2"
        )
        from oai2.knowledge.gc_resource_lease_d1 import (
            gc_resource_lease_schema_statements,
        )

        for statement in gc_resource_lease_schema_statements():
            db.execute(statement)
        db.execute(
            GC_RESOURCE_LEASE_ACQUIRE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, orphan, "tok", "gc-sweep",
             100.0, 200.0, 2),
        )
        assert db.execute(
            "SELECT COUNT(*) FROM knowledge_gc_resource_lease"
        ).fetchone()[0] == 1
        # A lease can never be taken on the LIVE vector.
        db.execute(
            GC_RESOURCE_LEASE_ACQUIRE_SQL,
            (GC_RESOURCE_TYPE_VECTOR, live, "tok2", "gc-sweep", 100.0, 200.0, 2),
        )
        assert db.execute(
            "SELECT COUNT(*) FROM knowledge_gc_resource_lease WHERE resource_key = ?",
            (live,),
        ).fetchone()[0] == 0, "a lease was acquired on an authoritative vector"


# ---------------------------------------------------------------------------
# A reference record with no ids is a degraded record, not an absent one
# ---------------------------------------------------------------------------


def _referenced_vector_state(ids: object) -> VectorGcReconciliationState:
    """A state whose single key ``vec-a`` carries ``ids`` as its reference set."""
    return VectorGcReconciliationState(
        observed_at=OBSERVED_AT,
        references={"vec-a": ids},  # type: ignore[arg-type]
        inventory={
            "vec-a": VectorInventoryEntry(
                vector_id="vec-a",
                knowledge_id="ko-1",
                content_hash=HASH,
                embedding_version="v1",
            )
        },
        pages_processed=1,
        next_cursor=None,
        inventory_complete=True,
    )


def test_empty_reference_set_is_not_a_delete_candidate() -> None:
    """An empty id set must not reclassify a referenced vector as an orphan.

    The module states the hazard directly: "an unseen page would make a
    referenced vector look like an orphan, and orphans are what gets deleted."
    Branching on ``knowledge_ids`` made an empty set indistinguishable from no
    reference at all, so a key the authoritative rows still name was queued for
    deletion.
    """
    state = _referenced_vector_state(set())
    dispositions = {r.vector_id: r.disposition for r in state.build_report().records}
    assert dispositions["vec-a"] is VectorGcDisposition.REFERENCED_PRESENT
    assert dispositions["vec-a"] is not VectorGcDisposition.UNREFERENCED_CANDIDATE


def test_empty_reference_set_cannot_be_persisted_and_restored() -> None:
    """The snapshot a degenerate state writes must not be restorable.

    This module has no reference fingerprint to fall back on, so the empty
    list round-tripped unguarded and misclassified on restore.
    """
    state = _referenced_vector_state(set())
    snap = state.to_snapshot()
    with pytest.raises(ValueError, match="snapshot references contain invalid data"):
        VectorGcReconciliationState.from_snapshot(snap)


def test_from_snapshot_rejects_empty_reference_list() -> None:
    """A key mapped to an empty list asserts nothing and must be refused."""
    state = _referenced_vector_state({"ko-1"})
    snap = state.to_snapshot()
    snap["references"] = {"vec-a": []}
    with pytest.raises(ValueError, match="snapshot references contain invalid data"):
        VectorGcReconciliationState.from_snapshot(snap)


def test_populated_reference_set_is_still_classified_as_before() -> None:
    """Guard the opposite failure: a real reference must be unaffected."""
    state = _referenced_vector_state({"ko-1", "ko-2"})
    dispositions = {r.vector_id: r.disposition for r in state.build_report().records}
    assert dispositions["vec-a"] is VectorGcDisposition.REFERENCED_PRESENT


def test_truly_unreferenced_vector_is_still_a_delete_candidate() -> None:
    """A key absent from ``references`` entirely is still a real orphan."""
    state = VectorGcReconciliationState(
        observed_at=OBSERVED_AT,
        references={},
        inventory={
            "vec-a": VectorInventoryEntry(
                vector_id="vec-a", knowledge_id="ko-1",
                content_hash=HASH, embedding_version="v1",
            )
        },
        pages_processed=1,
        next_cursor=None,
        inventory_complete=True,
    )
    report = state.build_report()
    dispositions = {r.vector_id: r.disposition for r in report.records}
    assert dispositions["vec-a"] is VectorGcDisposition.UNREFERENCED_CANDIDATE
    assert report.unreferenced_candidate_count == 1


# ---------------------------------------------------------------------------
# REQ-GC-033: grace-windowed candidates with persisted first-seen state that
# survive at least one authoritative recheck.
#
# The gap this covers: on the previous source, `unreferenced_candidate` was a
# complete answer. A vector seen for the first time thirty seconds ago and a
# vector unreferenced for years produced byte-identical records, and nothing
# required a second look at D1 before a destructive act could consume them.
# ---------------------------------------------------------------------------

GRACE = gc_vector.VectorGcGracePolicy(grace_seconds=900.0, minimum_rechecks=1)


def _entries(vector_ids: list[str]) -> list[VectorInventoryEntry]:
    return [
        VectorInventoryEntry(
            vector_id=v,
            knowledge_id=None,
            content_hash=None,
            embedding_version="embed-v1",
        )
        for v in vector_ids
    ]


def _state(
    references: dict[str, list[str]],
    inventory: list[str],
    *,
    now: float,
    pages: int = 1,
) -> VectorGcReconciliationState:
    rows = [
        _Row(knowledge_id=kid, vectorize_id=vid)
        for vid, kids in references.items()
        for kid in kids
    ]
    state = VectorGcReconciliationState.from_rows(rows, observed_at=now)
    entries = _entries(inventory)
    if pages == 1:
        state.consume_page(entries, next_cursor=None)
    else:
        for i in range(pages):
            state.consume_page(
                entries[i::pages], next_cursor=None if i == pages - 1 else f"c-{i}"
            )
    return state


def _record(state: VectorGcReconciliationState, vector_id: str, policy=GRACE):
    report = state.build_report(policy=policy)
    return next(r for r in report.records if r.vector_id == vector_id)


class TestGracePolicyValidation:
    """A policy that could make a first sighting eligible is not constructible."""

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan"), True])
    def test_zero_or_worse_grace_window_is_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError):
            gc_vector.VectorGcGracePolicy(grace_seconds=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5])
    def test_zero_or_worse_recheck_floor_is_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError):
            gc_vector.VectorGcGracePolicy(minimum_rechecks=bad)  # type: ignore[arg-type]


class TestGraceWindow:
    def test_first_sighting_is_never_eligible_however_old(self) -> None:
        """A first sighting is what an in-flight put looks like.

        A `put()` that has written its vector but not yet committed its D1 row
        is indistinguishable from a real orphan, for exactly as long as that
        put takes. Age alone must never be enough.
        """
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        rec = _record(s, "vec-a")
        assert rec.disposition is VectorGcDisposition.UNREFERENCED_CANDIDATE
        assert rec.first_seen_at == 1_000.0
        assert rec.rechecks == 0
        assert rec.grace_satisfied is False
        assert rec.eligible_for_authorization is False

        # Same state, evaluated an eternity later, still not eligible.
        s.observed_at = 10**12
        assert _record(s, "vec-a").grace_satisfied is False

    def test_candidate_becomes_eligible_only_after_a_recheck_and_the_window(
        self,
    ) -> None:
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        # Window elapsed, but only one sighting so far: NOT eligible.
        s.observed_at = 1_000.0 + 901.0
        assert _record(s, "vec-a").grace_satisfied is False
        # Second authoritative recheck against fresh D1.
        s.recheck_grace(now=1_000.0 + 901.0, policy=GRACE)
        rec = _record(s, "vec-a")
        assert rec.rechecks == 1
        assert rec.grace_satisfied is True
        assert rec.eligible_for_authorization is True

    def test_recheck_before_the_window_still_does_not_satisfy(self) -> None:
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        s.recheck_grace(now=1_100.0, policy=GRACE)
        s.observed_at = 1_100.0
        assert _record(s, "vec-a").rechecks == 1
        assert _record(s, "vec-a").grace_satisfied is False

    def test_first_seen_is_not_reset_by_further_sightings(self) -> None:
        """The window measures elapsed time, not process lifetimes."""
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        for i in range(2, 6):
            s.recheck_grace(now=1_000.0 + i * 100.0, policy=GRACE)
        rec = _record(s, "vec-a")
        assert rec.first_seen_at == 1_000.0
        assert rec.rechecks == 4

    def test_higher_recheck_floor_is_honoured(self) -> None:
        strict = gc_vector.VectorGcGracePolicy(grace_seconds=10.0, minimum_rechecks=3)
        s = _state({}, ["vec-a"], now=0.0)
        s.recheck_grace(now=0.0, policy=strict)
        for _ in range(2):
            s.recheck_grace(now=100.0, policy=strict)
        s.observed_at = 100.0
        assert _record(s, "vec-a", strict).rechecks == 2
        assert _record(s, "vec-a", strict).grace_satisfied is False
        s.recheck_grace(now=100.0, policy=strict)
        assert _record(s, "vec-a", strict).grace_satisfied is True

    def test_a_backwards_clock_does_not_satisfy_the_window(self) -> None:
        s = _state({}, ["vec-a"], now=10_000.0)
        s.recheck_grace(now=10_000.0, policy=GRACE)
        s.recheck_grace(now=10_000.0, policy=GRACE)
        s.observed_at = 1.0  # clock went backwards
        assert _record(s, "vec-a").grace_satisfied is False

    def test_non_finite_now_is_rejected(self) -> None:
        s = _state({}, ["vec-a"], now=1.0)
        with pytest.raises(ValueError):
            s.recheck_grace(now=float("inf"), policy=GRACE)

    def test_recheck_requires_a_completed_scan(self) -> None:
        """Aging a candidate on the strength of a page nobody read is a hazard."""
        rows = [_Row(knowledge_id="ko-1", vectorize_id=None)]
        s = VectorGcReconciliationState.from_rows(rows, observed_at=1.0)
        s.consume_page(_entries(["vec-a"]), next_cursor="c-1")
        with pytest.raises(RuntimeError):
            s.recheck_grace(now=1.0, policy=GRACE)


class TestGraceKeepBranch:
    """REQ-GC-033's diagram: a reference appearing KEEPS the vector."""

    def test_a_reappearing_reference_clears_the_candidate(self) -> None:
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        s.recheck_grace(now=2_000.0, policy=GRACE)
        assert "vec-a" in s.grace_candidates

        # A put adopts it: D1 now names the vector.
        s.references = {"vec-a": {"ko-1"}}
        s.recheck_grace(now=3_000.0, policy=GRACE)
        assert "vec-a" not in s.grace_candidates
        rec = _record(s, "vec-a")
        assert rec.disposition is VectorGcDisposition.REFERENCED_PRESENT
        assert rec.first_seen_at is None
        assert rec.rechecks == 0
        assert rec.grace_satisfied is False

    def test_going_unreferenced_again_starts_a_fresh_window(self) -> None:
        """History is discarded, not paused.

        The reference that came and went may have been a different row, so the
        old age says nothing about this new period of unreferencedness.
        """
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        s.recheck_grace(now=2_000.0, policy=GRACE)
        s.references = {"vec-a": {"ko-1"}}
        s.recheck_grace(now=3_000.0, policy=GRACE)
        s.references = {}
        s.recheck_grace(now=4_000.0, policy=GRACE)
        rec = _record(s, "vec-a")
        assert rec.first_seen_at == 4_000.0
        assert rec.rechecks == 0
        assert rec.grace_satisfied is False

    def test_a_referenced_vector_never_carries_grace_fields(self) -> None:
        """A referenced vector must not read as though it were aging out."""
        s = _state({"vec-a": ["ko-1"]}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        rec = _record(s, "vec-a")
        assert rec.disposition is VectorGcDisposition.REFERENCED_PRESENT
        assert (rec.first_seen_at, rec.rechecks, rec.grace_satisfied) == (None, 0, False)
        assert rec.eligible_for_authorization is False


class TestGraceWithoutPolicy:
    def test_report_without_a_policy_authorizes_nothing(self) -> None:
        """The conservative default: classify, but grant no grace credit."""
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        s.recheck_grace(now=5_000.0, policy=GRACE)
        s.observed_at = 5_000.0
        rec = _record(s, "vec-a", policy=None)
        assert rec.rechecks == 1
        assert rec.grace_satisfied is False
        assert rec.eligible_for_authorization is False

    def test_grace_satisfied_is_counted_separately_from_candidates(self) -> None:
        """'Seen unreferenced' and 'eligible for authorization' are not the same."""
        s = _state({"vec-live": ["ko-1"]}, ["vec-live", "vec-old", "vec-new"], now=0.0)
        s.recheck_grace(now=0.0, policy=GRACE)
        s.recheck_grace(now=10_000.0, policy=GRACE)
        s.observed_at = 10_000.0
        report = s.build_report(policy=GRACE)
        assert report.unreferenced_candidate_count == 2
        assert report.grace_satisfied_count == 2
        assert report.referenced_present_count == 1


class TestGraceSnapshotPersistence:
    def test_first_seen_survives_a_snapshot_round_trip(self) -> None:
        """REQ-GC-033 says PERSISTED first-seen state."""
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        snapshot = s.to_snapshot()
        assert snapshot["schema_version"] == 2

        restored = VectorGcReconciliationState.from_snapshot(snapshot)
        assert restored.grace_candidates["vec-a"].first_seen_at == 1_000.0
        assert restored.grace_candidates["vec-a"].rechecks == 0

        # And the window keeps accruing across the process boundary.
        restored.observed_at = 2_000.0
        restored.recheck_grace(now=2_000.0, policy=GRACE)
        rec = _record(restored, "vec-a")
        assert rec.first_seen_at == 1_000.0
        assert rec.rechecks == 1
        assert rec.grace_satisfied is True

    def test_a_v1_snapshot_grants_no_grace_credit(self) -> None:
        """An old snapshot must not manufacture age it has no evidence for."""
        s = _state({}, ["vec-a"], now=1_000.0)
        snapshot = dict(s.to_snapshot())
        snapshot["schema_version"] = 1
        snapshot.pop("grace_candidates")
        restored = VectorGcReconciliationState.from_snapshot(snapshot)
        assert restored.grace_candidates == {}
        rec = _record(restored, "vec-a")
        assert rec.first_seen_at is None
        assert rec.grace_satisfied is False

    def test_an_unknown_schema_version_is_still_rejected(self) -> None:
        s = _state({}, ["vec-a"], now=1.0)
        snapshot = dict(s.to_snapshot())
        snapshot["schema_version"] = 99
        with pytest.raises(ValueError):
            VectorGcReconciliationState.from_snapshot(snapshot)

    @pytest.mark.parametrize(
        "bad_entry",
        [
            {"first_seen_at": 1.0, "rechecks": 0},  # no vector_id
            {"vector_id": "other", "first_seen_at": 1.0, "rechecks": 0},  # key mismatch
            {"vector_id": "vec-a", "first_seen_at": -1.0, "rechecks": 0},  # negative
            {"vector_id": "vec-a", "first_seen_at": 1.0, "rechecks": -1},  # negative
            {"vector_id": "vec-a", "first_seen_at": 1.0, "rechecks": 1.5},  # not int
        ],
    )
    def test_degenerate_grace_state_cannot_be_restored(
        self, bad_entry: dict
    ) -> None:
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        snapshot = dict(s.to_snapshot())
        snapshot["grace_candidates"] = {"vec-a": bad_entry}
        with pytest.raises(ValueError):
            VectorGcReconciliationState.from_snapshot(snapshot)


class TestGraceNegativeControls:
    """Mutate the implementation and prove the guards above actually bite.

    Each control breaks one behaviour of ``recheck_grace`` or
    ``to_snapshot`` at the SOURCE level and asserts the corresponding guard
    then fails. Patching an attribute is not enough here: the behaviours under
    test are the control flow inside those two functions, so the mutation has
    to be made in the text and reloaded.
    """

    @staticmethod
    def _mutant(tmp_path, mutate):  # type: ignore[no-untyped-def]
        import importlib.util
        import sys

        path = pathlib.Path(gc_vector.__file__)
        source = path.read_text()
        mutated = mutate(source)
        assert mutated != source, "mutation did not apply -- control is vacuous"
        target = tmp_path / "mutant_gc_vector.py"
        target.write_text(mutated)
        name = "oai2.knowledge._mutant_gc_vector"
        spec = importlib.util.spec_from_file_location(name, target)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
        return module

    def test_control_first_sighting_grants_credit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A policy that ignores the recheck floor would promote a first sighting."""
        monkeypatch.setattr(
            gc_vector.VectorGcGracePolicy,
            "satisfies",
            lambda self, *, first_seen_at, rechecks, now: rechecks >= 0,
        )
        s = _state({}, ["vec-a"], now=1_000.0)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        s.observed_at = 10**12
        with pytest.raises(AssertionError):
            assert _record(s, "vec-a").grace_satisfied is False

    def test_control_first_seen_reset_on_every_recheck(
        self, tmp_path
    ) -> None:
        """A window that restarts each run can never mature."""
        module = self._mutant(
            tmp_path,
            lambda s: s.replace(
                "                    first_seen_at=prior.first_seen_at,\n",
                "                    first_seen_at=now,\n",
                1,
            ),
        )
        s = module.VectorGcReconciliationState.from_rows([], observed_at=1_000.0)
        s.consume_page(_entries(["vec-a"]), next_cursor=None)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        s.recheck_grace(now=5_000.0, policy=GRACE)
        s.recheck_grace(now=9_000.0, policy=GRACE)
        s.observed_at = 9_000.0
        rec = s.build_report(policy=GRACE).records[0]
        assert rec.rechecks == 2
        with pytest.raises(AssertionError):
            assert rec.first_seen_at == 1_000.0

    def test_control_candidate_never_cleared_on_a_returning_reference(
        self, tmp_path
    ) -> None:
        """Dropping the KEEP branch would age a referenced vector toward deletion."""
        module = self._mutant(
            tmp_path,
            lambda s: s.replace(
                "            if vector_id in self.references:\n"
                "                # Branch 1: KEEP. A reference appeared; the candidate is\n"
                "                # cleared rather than left to age back into eligibility.\n"
                "                continue\n",
                "",
                1,
            ),
        )
        s = module.VectorGcReconciliationState.from_rows([], observed_at=1_000.0)
        s.consume_page(_entries(["vec-a"]), next_cursor=None)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        s.references = {"vec-a": {"ko-1"}}
        s.recheck_grace(now=2_000.0, policy=GRACE)
        with pytest.raises(AssertionError):
            assert "vec-a" not in s.grace_candidates

    def test_control_grace_state_dropped_from_the_snapshot(
        self, tmp_path
    ) -> None:
        """Without persistence the window measures process lifetimes, not time."""
        module = self._mutant(
            tmp_path,
            lambda s: s.replace(
                '            "grace_candidates": {\n'
                "                vector_id: {\n"
                '                    "vector_id": grace.vector_id,\n'
                '                    "first_seen_at": grace.first_seen_at,\n'
                '                    "rechecks": grace.rechecks,\n'
                "                }\n"
                "                for vector_id, grace in sorted(self.grace_candidates.items())\n"
                "            },\n",
                '            "grace_candidates": {},\n',
                1,
            ),
        )
        s = module.VectorGcReconciliationState.from_rows([], observed_at=1_000.0)
        s.consume_page(_entries(["vec-a"]), next_cursor=None)
        s.recheck_grace(now=1_000.0, policy=GRACE)
        snapshot = s.to_snapshot()
        assert snapshot["grace_candidates"] == {}, "mutation stripped the state"
        restored = module.VectorGcReconciliationState.from_snapshot(snapshot)
        assert restored.grace_candidates == {}
        with pytest.raises(AssertionError):
            assert (
                "vec-a" in restored.grace_candidates
                and restored.grace_candidates["vec-a"].first_seen_at == 1_000.0
            )
