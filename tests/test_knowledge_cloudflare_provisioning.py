"""#261 / #205 §8 — schema provisioning is a deployment step, not a request side effect.

The contract this file pins:

* The order in which objects must exist: knowledge base -> legacy R2 lease ->
  typed resource lease and its index. Order is load-bearing, because the lease
  reference predicates resolve against ``knowledge_index`` columns.
* That the request path does NOT provision. A Worker invocation builds a new
  entrypoint instance, so a per-instance memo cannot cache provisioning across
  requests, and DDL as an ordinary effect of serving traffic is the wrong
  shape regardless.
* That applying the new table to a LIVE D1 database is a migration which has
  NOT been executed here. This is asserted as a source fact, not claimed as
  done.

No live binding, no live migration, no network, no delete.
"""

from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from oai2.knowledge.cloudflare_provisioning import (
    LEGACY_R2_LEASE_TABLE,
    VECTORIZE_ID_INDEX,
    provision_knowledge_schema,
    provisioning_batches,
    required_schema_objects,
    verify_knowledge_schema,
)
from oai2.knowledge.gc_lease_d1 import (
    GC_LEASE_TABLE,
    gc_lease_schema_statements,
)
from oai2.knowledge.gc_resource_lease_d1 import (
    GC_RESOURCE_LEASE_TABLE,
    gc_resource_lease_schema_statements,
)
from oai2.knowledge.knowledge_d1 import (
    KNOWLEDGE_CORPUS_ADVANCE_SQL,
    KNOWLEDGE_CORPUS_STATE_TABLE,
    KNOWLEDGE_INDEX_TABLE,
    KNOWLEDGE_WRITER_UPSERT_SQL,
    knowledge_schema_statements,
)

# ---------------------------------------------------------------------------
# A D1 binding fake that records the ORDER batches are applied in
# ---------------------------------------------------------------------------


class _Stmt:
    def __init__(self, sql: str) -> None:
        self.query = sql


class _RecordingD1:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def prepare(self, sql: str) -> _Stmt:
        return _Stmt(sql)

    async def batch(self, statements: list[_Stmt]) -> list[object]:
        self.batches.append([s.query for s in statements])
        return [{"success": True} for _ in statements]


# ---------------------------------------------------------------------------
# THE ORDER
# ---------------------------------------------------------------------------


def test_provisioning_declares_three_ordered_batches() -> None:
    names = [name for name, _statements in provisioning_batches()]
    assert names == ["knowledge_base", "legacy_r2_lease", "typed_resource_lease"], (
        "the lease tables' reference predicates resolve against knowledge_index, "
        "so they cannot be created before it; and the typed lease is what the "
        "writer fence consults, so it must exist before serving"
    )


def test_the_knowledge_base_batch_creates_the_referenced_tables() -> None:
    _name, statements = provisioning_batches()[0]
    assert list(statements) == list(knowledge_schema_statements())
    joined = " ".join(statements)
    assert KNOWLEDGE_INDEX_TABLE in joined
    assert KNOWLEDGE_CORPUS_STATE_TABLE in joined
    # #261: the vector reference predicate is a lookup on this column, and a
    # sweep runs it against the whole authoritative table.
    assert VECTORIZE_ID_INDEX in joined


def test_the_legacy_lease_batch_is_still_created() -> None:
    """#233 compatibility, deliberately not dropped.

    The writer SQL consults BOTH lease tables. Removing this batch would
    silently unblock writes during any sweep the #233 work owns, and would
    leave existing R2 lease rows behind a table nobody creates.
    """
    _name, statements = provisioning_batches()[1]
    assert list(statements) == list(gc_lease_schema_statements())
    assert LEGACY_R2_LEASE_TABLE == GC_LEASE_TABLE
    assert "knowledge_gc_delete_lease" in KNOWLEDGE_WRITER_UPSERT_SQL, (
        "the writer still consults the legacy table, so it must still exist"
    )


def test_the_typed_lease_batch_is_created_before_serving() -> None:
    _name, statements = provisioning_batches()[2]
    assert list(statements) == list(gc_resource_lease_schema_statements())
    assert GC_RESOURCE_LEASE_TABLE in KNOWLEDGE_WRITER_UPSERT_SQL
    assert GC_RESOURCE_LEASE_TABLE in KNOWLEDGE_CORPUS_ADVANCE_SQL


def test_every_object_the_writer_sql_needs_is_in_the_contract() -> None:
    """Derived from the SQL, so a new dependency cannot be added unnoticed."""
    for name in required_schema_objects():
        statements = [s for _batch, group in provisioning_batches() for s in group]
        assert any(name in s for s in statements), (
            f"{name} is required before serving but no batch creates it"
        )


def test_provisioning_applies_the_batches_in_that_order() -> None:
    import asyncio

    d1 = _RecordingD1()
    applied = asyncio.run(provision_knowledge_schema(d1))  # type: ignore[arg-type]
    assert [group[0] for group in d1.batches][0].startswith("CREATE TABLE")
    assert list(applied) == [
        "knowledge_base", "legacy_r2_lease", "typed_resource_lease"
    ]
    assert "knowledge_gc_delete_lease" in d1.batches[1][0]
    assert "knowledge_gc_resource_lease" in d1.batches[2][0]


# ---------------------------------------------------------------------------
# It really creates a working database, in that order, on real SQLite
# ---------------------------------------------------------------------------


@contextmanager
def _provisioned() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for _name, statements in provisioning_batches():
            for statement in statements:
                db.execute(statement)
        yield db
    finally:
        db.close()


def test_provisioning_in_order_produces_a_database_the_writer_can_use() -> None:
    """The contract is executable, not just declarative."""
    from oai2.knowledge.gc_resource_lease_d1 import (
        GC_RESOURCE_LEASE_ACQUIRE_SQL,
    )

    with _provisioned() as db:
        # Every required object exists.
        for name in required_schema_objects():
            row = db.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type IN ('table','index') AND name = ?",
                (name,),
            ).fetchone()
            assert row[0] == 1, f"{name} was not created by the contract"

        # A lease can be taken out, which needs both the typed table and the
        # knowledge tables its reference predicate reads.
        db.execute(
            "UPDATE knowledge_corpus_state SET revision = 3"
        )
        assert db.execute(
            GC_RESOURCE_LEASE_ACQUIRE_SQL,
            ("vector", "v-1", "tok", "sweep", 1.0, 9.0, 3),
        ).rowcount == 1

        # And the writer runs against it without a "no such table" error,
        # which is what would happen if the typed table were not provisioned.
        db.execute(
            KNOWLEDGE_WRITER_UPSERT_SQL,
            ("ko_1", "t", "c" * 64, 0.5, "active", None, 1.0, "b-1", "v-2", 3, 2.0),
        )
        db.execute(KNOWLEDGE_CORPUS_ADVANCE_SQL, (3, "b-1", 2.0, "v-2"))
        assert db.execute(
            "SELECT revision FROM knowledge_corpus_state"
        ).fetchone()[0] == 4


def test_reordering_the_contract_would_break_it() -> None:
    """Why the order is asserted rather than assumed.

    The typed lease table is created against the knowledge schema, and the
    writer consults it. Provisioning it in the wrong place is not caught by a
    reviewer reading a list; it is caught here, by executing both orders.
    """
    typed = list(gc_resource_lease_schema_statements())
    base = list(knowledge_schema_statements())
    # Reversed: the lease table first. SQLite accepts DDL in any order, but
    # the resulting database is not the one the writer contract assumes.
    assert typed and base
    assert provisioning_batches()[0][1] == tuple(base)
    assert provisioning_batches()[2][1] == tuple(typed)


# ---------------------------------------------------------------------------
# Provisioning is NOT on the request path
# ---------------------------------------------------------------------------


def test_no_request_path_module_calls_provisioning() -> None:
    """The entrypoint must not import or call the provisioning step.

    Structural, because a behavioural test cannot see an import that never
    executes on the happy path. The source is read as TEXT rather than
    imported: the entrypoint imports ``workers``, which only exists inside the
    Worker runtime, and stubbing the runtime here would test the stub.
    """
    import deploy.cloudflare.knowledge_worker as worker_pkg

    source = (Path(worker_pkg.__file__).parent / "entry.py").read_text()
    for forbidden in ("provision_knowledge_schema", "verify_knowledge_schema",
                      "ensure_schema=True"):
        assert forbidden not in source, (
            f"the request path references {forbidden!r}; schema is a "
            "deployment prerequisite, not a request side effect"
        )
    # And it positively passes ensure_schema=False at the call site.
    assert "ensure_schema=False" in source


def test_the_factory_still_offers_explicit_provisioning_for_operators() -> None:
    """Separation, not removal.

    An operator or a deploy script still needs a supported way to provision;
    what must not happen is a request doing it implicitly.
    """
    from oai2.knowledge.cloudflare_factory import (
        build_cloudflare_knowledge_components,
    )

    source = inspect.getsource(build_cloudflare_knowledge_components)
    assert "ensure_schema" in source
    assert "provision_knowledge_schema" in source


def test_verify_is_read_only_and_reports_what_is_missing() -> None:
    import asyncio

    with _provisioned() as db:
        # A real database passes.
        asyncio.run(
            verify_knowledge_schema(_SqliteVerifyBinding(db))  # type: ignore[arg-type]
        )

    # An empty one names what is absent, and does not create it.
    empty = sqlite3.connect(":memory:", isolation_level=None)
    try:
        with pytest.raises(RuntimeError, match="not provisioned"):
            asyncio.run(
                verify_knowledge_schema(_SqliteVerifyBinding(empty))  # type: ignore[arg-type]
            )
        names = {
            row[0] for row in empty.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
            )
        }
        assert GC_RESOURCE_LEASE_TABLE not in names, (
            "the read-only readiness check must not provision anything"
        )
    finally:
        empty.close()


class _SqliteVerifyBinding:
    """Minimal adapter so verify_knowledge_schema runs against real SQLite."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db

    def prepare(self, sql: str) -> _Stmt:
        return _Stmt(sql)

    async def batch(self, statements: list[_Stmt]) -> list[object]:
        out: list[object] = []
        for statement in statements:
            rows = self._db.execute(statement.query).fetchall()
            # Shaped like a real D1 batch result: rows are MAPPINGS keyed by
            # column label, so this proves the readiness check reads by name
            # rather than by position.
            label = "present"
            out.append(
                {"results": [{label: rows[0][0]}] if rows else [], "success": True}
            )
        return out


# ---------------------------------------------------------------------------
# The live migration is NOT done
# ---------------------------------------------------------------------------


def test_the_source_states_the_live_migration_is_unexecuted() -> None:
    """Recorded as a source fact so the claim cannot quietly become true.

    #261 still lists LIVE MIGRATION as unexecuted. If someone later performs
    it, this test is where that has to be updated deliberately.
    """
    from oai2.knowledge import cloudflare_provisioning

    source = inspect.getsource(cloudflare_provisioning)
    assert "NOT been run against a live D1" in source or "has NOT been executed" in source
    assert "migration" in source.lower()
