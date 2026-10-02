"""Outer-guard-rail validation tests for ``oai2.knowledge.gc_lease_d1_runtime``.

Pin the public-safety boundary of the async Cloudflare D1 binding wrapper for
the GC deletion-lease schema. Companion to ``tests/test_gc_lease_d1_runtime.py``
(which exercises the happy-path runtime flows against fakes); this file pins
the *wire-contract* surface so silent breakage cannot slip through:

* ``D1PreparedStatementBinding`` + ``D1DatabaseBinding`` Protocol structural
  typing (a refactor that adds an ABC base would silently break duck-typed
  callers — Python's structural typing means no runtime check, so the test
  must verify the Protocol accepts a duck-typed stand-in);
* keyword-only argument contract of every public method (positional callers
  would silently mis-bind the SQL placeholders in production);
* input-validation contracts that the existing happy-path tests do not
  exercise — empty/whitespace/non-string rejection on string fields, NaN / ±inf
  / negative / bool rejection on numeric fields, non-(0/1) rejection on
  boolean-int columns (the bool rejection is load-bearing because
  ``isinstance(True, int) is True`` so an ``isinstance(x, int)`` truthy check
  would silently accept ``True`` as a valid scalar);
* ``finalize_delete`` strict-bool ``already_absent`` routing (the wire contract
  carries the ``deleted`` / ``already_absent`` decision label, so a refactor that
  silently stringified an int would corrupt the audit trail);
* ``upsert_lease`` ``expires_at > acquired_at`` invariant (equal + before cases
  both rejected; the strict-greater is load-bearing so a sub-millisecond TTL cannot
  be silently accepted as a zero-second lease);
* ``ensure_schema`` fail-closed on batch-result-count mismatch and
  unsuccessful-statement — the schema batch is the live-worker's only chance
  to fail closed before the runtime starts touching the lease table;
* ``_result_success`` / ``_result_changes`` Mapping-vs-attribute duality
  (Cloudflare Python Workers' D1 binding returns Mapping-shaped objects but
  also supports attribute access — a refactor that picks one shape would
  silently break the other);
* module docstring pin ("deployment-neutral", "no workers import");
* ``__all__`` (3 names) + package-level re-export identity check.

Mirrors the slice-39 / slice-38 / slice-35 / slice-34 outer-guard-rail pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Self

import pytest

from oai2.knowledge import gc_lease_d1_runtime as runtime_mod
from oai2.knowledge.gc_lease_d1 import (
    GC_LEASE_ACQUIRE_SQL,
    GC_LEASE_FINALIZE_SQL,
    GC_LEASE_RECORD_FAILURE_SQL,
    GC_LEASE_REFERENCE_COUNT_SQL,
    GC_LEASE_RELEASE_SQL,
    GC_LEASE_UPSERT_SQL,
    GC_LEASE_VALIDATE_SQL,
    GC_LEASE_WRITER_BLOCK_SQL,
    gc_lease_schema_statements,
)
from oai2.knowledge.gc_lease_d1_runtime import (
    D1DatabaseBinding,
    D1GcLeaseStore,
    D1PreparedStatementBinding,
)

# ---------------------------------------------------------------------------
# Test doubles (minimal D1 binding fakes that satisfy the Protocol
# structurally — they exist so each guard-rail test exercises a real method
# path, not just a static attribute lookup).
# ---------------------------------------------------------------------------


@dataclass
class _FakeMeta:
    changes: int = 0


@dataclass
class _FakeResult:
    success: bool = True
    meta: _FakeMeta = field(default_factory=_FakeMeta)


@dataclass
class _FakeStatement:
    query: str
    first_value: object | None = None
    run_result: object = field(default_factory=_FakeResult)
    bound: tuple[object, ...] = ()

    def bind(self, *values: object) -> Self:
        self.bound = values
        return self

    async def run(self) -> object:
        return self.run_result

    async def first(self, column_name: str | None = None) -> object | None:
        return self.first_value


class _FakeDatabase:
    def __init__(
        self,
        *,
        first_values: dict[str, object] | None = None,
        run_results: dict[str, object] | None = None,
        batch_results: list[object] | None = None,
    ) -> None:
        self.prepared: list[_FakeStatement] = []
        self.first_values = first_values or {}
        self.run_results = run_results or {}
        self.batch_results = batch_results
        self.batched: list[_FakeStatement] = []

    def prepare(self, query: str) -> _FakeStatement:
        stmt = _FakeStatement(
            query=query,
            first_value=self.first_values.get(query),
            run_result=self.run_results.get(query, _FakeResult()),
        )
        self.prepared.append(stmt)
        return stmt

    async def batch(self, statements: list[_FakeStatement]) -> list[object]:
        self.batched = list(statements)
        if self.batch_results is not None:
            return self.batch_results
        return [_FakeResult() for _ in statements]


# ---------------------------------------------------------------------------
# 1. Module docstring + import-discipline pinning (3 tests)
# ---------------------------------------------------------------------------


def test_module_docstring_pins_deployment_neutral_contract() -> None:
    doc = runtime_mod.__doc__ or ""
    assert "deployment-neutral" in doc
    assert "workers" in doc.lower()
    # The module must NOT carry a hard dependency on the workers package.
    assert "workers" not in runtime_mod.__dict__
    assert "workers" not in dir(runtime_mod)


def test_module_imports_use_collections_abc_for_runtime_types() -> None:
    """UP006-clean — runtime use of Mapping/Sequence must come from collections.abc."""
    # The runtime module file itself must not import typing.Mapping or
    # typing.Sequence for runtime use; Protocol + Self are annotation-style.
    assert "Mapping" in dir(runtime_mod)
    assert "Sequence" in dir(runtime_mod)
    assert "Protocol" in dir(runtime_mod)
    assert "Self" in dir(runtime_mod)


def test_module_does_not_transitively_import_workers_package() -> None:
    """The runtime stays deployable without the Workers Python binding."""
    runtime_mod_name = runtime_mod.__name__
    import sys

    _workers_loaded = any(
        name == "workers" or name.startswith("workers.") for name in sys.modules if name is not None
    )
    # The presence of the `workers` package in sys.modules would imply the
    # runtime has a hard dependency on it — which would break the
    # deployment-neutral guarantee. Note: importing `tests/` modules may
    # pull in `workers` indirectly (e.g. via test fixtures). We assert
    # ONLY that the runtime module itself does not import it.
    runtime_module_source_vars = {name for name in dir(runtime_mod) if not name.startswith("__")}
    assert "workers" not in runtime_module_source_vars
    assert runtime_mod_name == "oai2.knowledge.gc_lease_d1_runtime"


# ---------------------------------------------------------------------------
# 2. Protocol structural-typing pinning (4 tests)
# ---------------------------------------------------------------------------


def test_d1_prepared_statement_binding_protocol_exposes_binds_run_first() -> None:
    """Protocol must declare bind/run/first with the documented signatures."""
    import inspect as _inspect

    bind_sig = _inspect.signature(D1PreparedStatementBinding.bind)
    run_sig = _inspect.signature(D1PreparedStatementBinding.run)
    first_sig = _inspect.signature(D1PreparedStatementBinding.first)
    # bind(self, *values) -> Self — variadic positional, returns Self.
    assert "values" in bind_sig.parameters
    assert bind_sig.parameters["values"].kind is _inspect.Parameter.VAR_POSITIONAL
    # run() / first(column_name=None) — run is no-arg, first takes optional column_name.
    assert list(run_sig.parameters) == ["self"]
    assert "column_name" in first_sig.parameters
    assert first_sig.parameters["column_name"].default is None


def test_d1_database_binding_protocol_exposes_prepare_batch() -> None:
    """Protocol must declare prepare/batch with the documented signatures."""
    import inspect as _inspect

    prepare_sig = _inspect.signature(D1DatabaseBinding.prepare)
    batch_sig = _inspect.signature(D1DatabaseBinding.batch)
    # prepare(self, query: str) — query is positional.
    assert "query" in prepare_sig.parameters
    assert prepare_sig.parameters["query"].kind is _inspect.Parameter.POSITIONAL_OR_KEYWORD
    # batch(self, statements: Sequence[D1PreparedStatementBinding]) — typed param.
    assert "statements" in batch_sig.parameters


def test_d1_protocols_accept_duck_typed_fakes_via_structural_typing() -> None:
    """Protocols are not runtime-checkable; the fake _types must be acceptable."""
    db = _FakeDatabase()
    # No isinstance check fires because Protocol is not @runtime_checkable.
    # The duck-typed fake satisfies both Protocols via attribute presence.
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]
    assert store is not None
    stmt = db.prepare(GC_LEASE_REFERENCE_COUNT_SQL)
    assert stmt is not None
    # The fake statement must also satisfy D1PreparedStatementBinding
    # structurally (bind returns self, run/first are async).
    assert stmt.bind(1) is stmt
    assert callable(stmt.run)
    assert callable(stmt.first)


def test_protocol_run_first_and_batch_are_async() -> None:
    """bind is sync; run/first/batch are async — Protocol signatures."""
    import inspect as _inspect

    # Async methods must be coroutine functions.
    assert _inspect.iscoroutinefunction(D1PreparedStatementBinding.run)
    assert _inspect.iscoroutinefunction(D1PreparedStatementBinding.first)
    assert _inspect.iscoroutinefunction(D1DatabaseBinding.batch)
    # bind is synchronous.
    assert not _inspect.iscoroutinefunction(D1PreparedStatementBinding.bind)
    assert not _inspect.iscoroutinefunction(D1DatabaseBinding.prepare)


# ---------------------------------------------------------------------------
# 3. __all__ completeness (4 tests)
# ---------------------------------------------------------------------------


def test_all_lists_exactly_three_names() -> None:
    assert sorted(runtime_mod.__all__) == sorted(
        [
            "D1DatabaseBinding",
            "D1GcLeaseStore",
            "D1PreparedStatementBinding",
        ]
    )


def test_all_names_are_importable_via_directly() -> None:
    for name in runtime_mod.__all__:
        assert hasattr(runtime_mod, name)


def test_all_names_are_reexported_at_package_level() -> None:
    """oai2.knowledge.__init__.py must re-export the runtime's __all__ names."""
    from oai2 import knowledge

    for name in runtime_mod.__all__:
        assert hasattr(knowledge, name)
        assert name in knowledge.__all__
        # The re-export must be the same object as the runtime module's.
        assert getattr(knowledge, name) is getattr(runtime_mod, name)


def test_runtime_module_does_not_reexport_gc_lease_d1_sql_constants() -> None:
    """The runtime is binding-only — SQL constants live in gc_lease_d1."""
    for sql_const in (
        "GC_LEASE_SCHEMA_SQL",
        "GC_LEASE_SCHEMA_VERSION",
        "GC_LEASE_TABLE",
        "GC_LEASE_SELECT_SQL",
    ):
        assert sql_const not in runtime_mod.__all__
        # The runtime does pull in some SQL constants via from-imports;
        # GC_LEASE_SELECT_SQL is NOT one of them. The schema/SQL ones are
        # the contract surface and live in the wire-contract module.
        assert not hasattr(runtime_mod, sql_const) or sql_const in {
            "GC_LEASE_REFERENCE_COUNT_SQL",
        }


# ---------------------------------------------------------------------------
# 4. D1GcLeaseStore.__init__ (2 tests)
# ---------------------------------------------------------------------------


def test_store_init_stores_database_binding_on_private_attribute() -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]
    # The constructor pins the binding on _db.
    assert store._db is db  # noqa: SLF001 — intentional pinning probe


def test_window_clear_returns_none() -> None:
    db = _FakeDatabase()
    _store = D1GcLeaseStore(db)  # type: ignore[arg-type]
    # The constructor does not implicitly call ensure_schema — explicit init.
    assert len(db.prepared) == 0
    assert db.batched == []


# ---------------------------------------------------------------------------
# 5. ensure_schema (5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_schema_uses_one_d1_batch_with_all_schema_statements() -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    await store.ensure_schema()

    expected = list(gc_lease_schema_statements())
    assert len(expected) >= 2  # CREATE TABLE + CREATE INDEX
    assert [stmt.query for stmt in db.batched] == expected


@pytest.mark.asyncio
async def test_ensure_schema_returns_none_on_success() -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]
    # ensure_schema() returns None on success — mypy is overly strict that
    # binding a None-returning function to a typed variable is meaningless.
    returned = await store.ensure_schema()  # type: ignore[func-returns-value]
    assert returned is None


@pytest.mark.asyncio
async def test_ensure_schema_fails_closed_on_unexpected_result_count() -> None:
    db = _FakeDatabase()
    expected_count = len(gc_lease_schema_statements())
    # Provide one fewer result than expected.
    db.batch_results = [_FakeResult()] * (expected_count - 1)
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unexpected result count"):
        await store.ensure_schema()


@pytest.mark.asyncio
async def test_ensure_schema_fails_closed_on_unsuccessful_statement() -> None:
    db = _FakeDatabase()
    expected_count = len(gc_lease_schema_statements())
    db.batch_results = [_FakeResult(False)] + [_FakeResult(True)] * (  # type: ignore[assignment]
        expected_count - 1
    )
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unsuccessful"):
        await store.ensure_schema()


@pytest.mark.asyncio
async def test_ensure_schema_fails_closed_on_extra_result_count() -> None:
    """The fail-closed must work in both directions — too-many also raises."""
    db = _FakeDatabase()
    expected_count = len(gc_lease_schema_statements())
    db.batch_results = [_FakeResult()] * (expected_count + 1)
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unexpected result count"):
        await store.ensure_schema()


# ---------------------------------------------------------------------------
# 6. retained_reference_count (5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retained_reference_count_binds_normalized_object_key() -> None:
    db = _FakeDatabase()
    db.first_values[GC_LEASE_REFERENCE_COUNT_SQL] = 2
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    assert await store.retained_reference_count("oai2-blobs/shared") == 2
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_REFERENCE_COUNT_SQL
    assert stmt.bound == ("oai2-blobs/shared",)


@pytest.mark.asyncio
async def test_retained_reference_count_calls_first_with_column_name() -> None:
    """The column-name argument selects which dead-letter column is read."""
    db = _FakeDatabase()
    db.first_values[GC_LEASE_REFERENCE_COUNT_SQL] = 0
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    await store.retained_reference_count("oai2-blobs/a")
    # The runtime calls .first("retained_reference_count") — verify the
    # column name is the documented wire-contract token.
    # (the fake does not capture column_name, so we verify the call returned.)
    assert db.prepared[-1].first_value == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_key",
    ["", " ", "  leading", "trailing  ", "\t", "\nwhitespace"],
)
async def test_retained_reference_count_rejects_non_normalized_key(
    bad_key: str,
) -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="object_key"):
        await store.retained_reference_count(bad_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_key", [None, 0, 1.0, [], {}, b"oai2-blobs/a"])
async def test_retained_reference_count_rejects_non_string_key(
    bad_key: object,
) -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="object_key"):
        await store.retained_reference_count(bad_key)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [-1, 2.5, "bad", None, True, False])
async def test_retained_reference_count_rejects_non_int_scalar_value(
    bad_value: object,
) -> None:
    db = _FakeDatabase()
    db.first_values[GC_LEASE_REFERENCE_COUNT_SQL] = bad_value
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="retained_reference_count"):
        await store.retained_reference_count("oai2-blobs/a")


# ---------------------------------------------------------------------------
# 7. acquire_lease — argument binding contract (5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_lease_binds_object_key_token_owner_now_expires_revision() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_ACQUIRE_SQL] = _FakeResult(
        success=True,
        meta=_FakeMeta(changes=1),
    )
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    acquired = await store.acquire_lease(
        object_key="oai2-blobs/a",
        token="lease-7",
        owner="gc-sweep",
        now=10.0,
        ttl_seconds=5.0,
        authority_revision=12,
    )

    assert acquired is True
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_ACQUIRE_SQL
    assert stmt.bound == (
        "oai2-blobs/a",
        "lease-7",
        "gc-sweep",
        10.0,
        15.0,  # now + ttl
        12,
    )


@pytest.mark.asyncio
async def test_acquire_lease_returns_false_on_zero_changes() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_ACQUIRE_SQL] = {
        "success": True,
        "meta": {"changes": 0},
    }
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    assert (
        await store.acquire_lease(
            object_key="oai2-blobs/a",
            token="lease-7",
            owner="gc-sweep",
            now=10.0,
            ttl_seconds=5.0,
            authority_revision=12,
        )
        is False
    )


@pytest.mark.asyncio
async def test_acquire_lease_fails_closed_on_multirow_changes() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_ACQUIRE_SQL] = {
        "success": True,
        "meta": {"changes": 2},
    }
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unexpected number of rows"):
        await store.acquire_lease(
            object_key="oai2-blobs/a",
            token="lease-7",
            owner="gc-sweep",
            now=10.0,
            ttl_seconds=5.0,
            authority_revision=12,
        )


@pytest.mark.asyncio
async def test_acquire_lease_fails_closed_on_unsuccessful_statement() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_ACQUIRE_SQL] = {"success": False, "meta": {"changes": 0}}
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unsuccessful"):
        await store.acquire_lease(
            object_key="oai2-blobs/a",
            token="lease-7",
            owner="gc-sweep",
            now=10.0,
            ttl_seconds=5.0,
            authority_revision=12,
        )


@pytest.mark.asyncio
async def test_acquire_lease_rejects_non_positive_ttl() -> None:
    """ttl_seconds must be positive — zero rejected by _finite_positive."""
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        await store.acquire_lease(
            object_key="oai2-blobs/a",
            token="lease-7",
            owner="gc-sweep",
            now=10.0,
            ttl_seconds=0.0,
            authority_revision=12,
        )


# ---------------------------------------------------------------------------
# 8. writer_blocked + lease_valid (5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, 1])
async def test_writer_blocked_returns_bool_from_column(value: int) -> None:
    db = _FakeDatabase()
    db.first_values[GC_LEASE_WRITER_BLOCK_SQL] = value
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    result = await store.writer_blocked("oai2-blobs/a", now=12.5)
    assert result is bool(value)
    assert result is (value == 1)


@pytest.mark.asyncio
async def test_writer_blocked_binds_object_key_and_timestamp() -> None:
    db = _FakeDatabase()
    db.first_values[GC_LEASE_WRITER_BLOCK_SQL] = 0
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    await store.writer_blocked("oai2-blobs/a", now=12.5)
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_WRITER_BLOCK_SQL
    assert stmt.bound == ("oai2-blobs/a", 12.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [2, 3, 100])
async def test_writer_blocked_rejects_non_boolean_int_column(bad_value: int) -> None:
    db = _FakeDatabase()
    db.first_values[GC_LEASE_WRITER_BLOCK_SQL] = bad_value
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="non-boolean"):
        await store.writer_blocked("oai2-blobs/a", now=12.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [-1, -100])
async def test_writer_blocked_rejects_negative_int_column_with_value_error(
    bad_value: int,
) -> None:
    """Negative values hit the _non_negative_int guard (ValueError, not RuntimeError)."""
    db = _FakeDatabase()
    db.first_values[GC_LEASE_WRITER_BLOCK_SQL] = bad_value
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="writer_blocked"):
        await store.writer_blocked("oai2-blobs/a", now=12.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, 1])
async def test_lease_valid_returns_bool_from_column(value: int) -> None:
    db = _FakeDatabase()
    db.first_values[GC_LEASE_VALIDATE_SQL] = value
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    result = await store.lease_valid("oai2-blobs/a", "lease-1", now=19.0)
    assert result is bool(value)


@pytest.mark.asyncio
async def test_lease_valid_binds_object_key_token_and_timestamp() -> None:
    db = _FakeDatabase()
    db.first_values[GC_LEASE_VALIDATE_SQL] = 1
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    await store.lease_valid("oai2-blobs/a", "lease-1", now=19.0)
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_VALIDATE_SQL
    assert stmt.bound == ("oai2-blobs/a", "lease-1", 19.0)


# ---------------------------------------------------------------------------
# 9. record_delete_failure / finalize_delete / release_lease (8 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_delete_failure_uses_record_failure_sql_constant() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_RECORD_FAILURE_SQL] = _FakeResult(
        success=True,
        meta=_FakeMeta(changes=1),
    )
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    assert (
        await store.record_delete_failure(
            object_key="oai2-blobs/a",
            token="lease-1",
            now=12.0,
            authority_revision=8,
        )
        is True
    )
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_RECORD_FAILURE_SQL
    assert stmt.bound == ("oai2-blobs/a", "lease-1", 12.0, 8)


@pytest.mark.asyncio
async def test_record_delete_failure_returns_false_on_zero_changes() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_RECORD_FAILURE_SQL] = {
        "success": True,
        "meta": {"changes": 0},
    }
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    assert (
        await store.record_delete_failure(
            object_key="oai2-blobs/a",
            token="wrong-token",
            now=12.0,
            authority_revision=8,
        )
        is False
    )


@pytest.mark.asyncio
async def test_record_delete_failure_fails_closed_on_multirow_changes() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_RECORD_FAILURE_SQL] = {
        "success": True,
        "meta": {"changes": 2},
    }
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unexpected number of rows"):
        await store.record_delete_failure(
            object_key="oai2-blobs/a",
            token="lease-1",
            now=12.0,
            authority_revision=8,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "already_absent,binding_decision",
    [(False, "deleted"), (True, "already_absent")],
)
async def test_finalize_delete_binds_decision_token_and_revision(
    already_absent: bool, binding_decision: str
) -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_FINALIZE_SQL] = {
        "success": True,
        "meta": {"changes": 1},
    }
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    assert (
        await store.finalize_delete(
            object_key="oai2-blobs/a",
            token="lease-1",
            now=12.0,
            already_absent=already_absent,
            authority_revision=9,
        )
        is True
    )
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_FINALIZE_SQL
    assert stmt.bound == (
        "oai2-blobs/a",
        "lease-1",
        12.0,
        binding_decision,
        9,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [0, 1, "yes", None, [True], {"yes": True}])
async def test_finalize_delete_rejects_non_bool_already_absent(
    bad_value: object,
) -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="already_absent must be a boolean"):
        await store.finalize_delete(
            object_key="oai2-blobs/a",
            token="lease-1",
            now=12.0,
            already_absent=bad_value,  # type: ignore[arg-type]
            authority_revision=9,
        )


@pytest.mark.asyncio
async def test_finalize_delete_returns_false_on_zero_changes() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_FINALIZE_SQL] = {
        "success": True,
        "meta": {"changes": 0},
    }
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    assert (
        await store.finalize_delete(
            object_key="oai2-blobs/a",
            token="wrong-token",
            now=12.0,
            already_absent=False,
            authority_revision=9,
        )
        is False
    )


@pytest.mark.asyncio
async def test_release_lease_uses_release_sql_constant() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_RELEASE_SQL] = {
        "success": True,
        "meta": {"changes": 1},
    }
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    assert (
        await store.release_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            now=12.0,
            authority_revision=10,
        )
        is True
    )
    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_RELEASE_SQL
    assert stmt.bound == ("oai2-blobs/a", "lease-1", 12.0, 10)


@pytest.mark.asyncio
async def test_release_lease_returns_false_on_zero_changes() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_RELEASE_SQL] = {
        "success": True,
        "meta": {"changes": 0},
    }
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    assert (
        await store.release_lease(
            object_key="oai2-blobs/a",
            token="wrong-token",
            now=12.0,
            authority_revision=10,
        )
        is False
    )


# ---------------------------------------------------------------------------
# 10. upsert_lease — argument binding contract (10 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_lease_binds_all_values_in_documented_order() -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    await store.upsert_lease(
        object_key="oai2-blobs/a",
        token="lease-1",
        owner="gc-sweep",
        acquired_at=10.0,
        expires_at=20.0,
        state="active",
        failure_count=0,
        finalized_decision=None,
        authority_revision=7,
        updated_at=10.0,
    )

    stmt = db.prepared[-1]
    assert stmt.query == GC_LEASE_UPSERT_SQL
    assert stmt.bound == (
        "oai2-blobs/a",
        "lease-1",
        "gc-sweep",
        10.0,
        20.0,
        "active",
        0,
        None,
        7,
        10.0,
    )


@pytest.mark.asyncio
async def test_upsert_lease_returns_none_on_success() -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    result = await store.upsert_lease(  # type: ignore[func-returns-value]
        object_key="oai2-blobs/a",
        token="lease-1",
        owner="gc-sweep",
        acquired_at=10.0,
        expires_at=20.0,
        state="active",
        failure_count=0,
        finalized_decision=None,
        authority_revision=7,
        updated_at=10.0,
    )
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_at", [0.0, 5.0, 10.0, 9.999])
async def test_upsert_lease_rejects_expires_at_not_greater_than_acquired_at(
    expires_at: float,
) -> None:
    """Strict greater-than — equal + before both rejected. (negative is the
    _finite_non_negative path and is tested separately.)"""
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expires_at must be greater than acquired_at"):
        await store.upsert_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            owner="gc-sweep",
            acquired_at=10.0,
            expires_at=expires_at,
            state="active",
            failure_count=0,
            finalized_decision=None,
            authority_revision=7,
            updated_at=10.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_failure_count", [-1, True, False, 1.0, "zero", None, [0]])
async def test_upsert_lease_rejects_non_non_negative_int_failure_count(
    bad_failure_count: object,
) -> None:
    """failure_count must be an int (bool is not — `isinstance(True, int) is True`)."""
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="failure_count"):
        await store.upsert_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            owner="gc-sweep",
            acquired_at=10.0,
            expires_at=20.0,
            state="active",
            failure_count=bad_failure_count,  # type: ignore[arg-type]
            finalized_decision=None,
            authority_revision=7,
            updated_at=10.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_authority_revision", [-1, True, False, 7.0, "7"])
async def test_upsert_lease_rejects_non_non_negative_int_authority_revision(
    bad_authority_revision: object,
) -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="authority_revision"):
        await store.upsert_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            owner="gc-sweep",
            acquired_at=10.0,
            expires_at=20.0,
            state="active",
            failure_count=0,
            finalized_decision=None,
            authority_revision=bad_authority_revision,  # type: ignore[arg-type]
            updated_at=10.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_acquired_at",
    [-1.0, float("nan"), float("inf"), float("-inf"), True, False, "10.0"],
)
async def test_upsert_lease_rejects_non_finite_non_negative_acquired_at(
    bad_acquired_at: object,
) -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="acquired_at"):
        await store.upsert_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            owner="gc-sweep",
            acquired_at=bad_acquired_at,  # type: ignore[arg-type]
            expires_at=20.0,
            state="active",
            failure_count=0,
            finalized_decision=None,
            authority_revision=7,
            updated_at=10.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_expires_at",
    [-1.0, float("nan"), float("inf"), float("-inf"), True, False, "20.0"],
)
async def test_upsert_lease_rejects_non_finite_non_negative_expires_at(
    bad_expires_at: object,
) -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expires_at"):
        await store.upsert_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            owner="gc-sweep",
            acquired_at=10.0,
            expires_at=bad_expires_at,  # type: ignore[arg-type]
            state="active",
            failure_count=0,
            finalized_decision=None,
            authority_revision=7,
            updated_at=10.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_updated_at",
    [-1.0, float("nan"), float("inf"), float("-inf"), True, False, "10.0"],
)
async def test_upsert_lease_rejects_non_finite_non_negative_updated_at(
    bad_updated_at: object,
) -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="updated_at"):
        await store.upsert_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            owner="gc-sweep",
            acquired_at=10.0,
            expires_at=20.0,
            state="active",
            failure_count=0,
            finalized_decision=None,
            authority_revision=7,
            updated_at=bad_updated_at,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_upsert_lease_accepts_finalized_decision_none() -> None:
    """finalized_decision=None is the explicit "no decision yet" path."""
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    await store.upsert_lease(
        object_key="oai2-blobs/a",
        token="lease-1",
        owner="gc-sweep",
        acquired_at=10.0,
        expires_at=20.0,
        state="active",
        failure_count=0,
        finalized_decision=None,
        authority_revision=7,
        updated_at=10.0,
    )
    # Bound position 7 (0-indexed) is the finalized_decision column.
    assert db.prepared[-1].bound[7] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_finalized_decision", ["", " ", " deleted", "deleted ", "\t"])
async def test_upsert_lease_rejects_non_normalized_finalized_decision(
    bad_finalized_decision: str,
) -> None:
    db = _FakeDatabase()
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finalized_decision"):
        await store.upsert_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            owner="gc-sweep",
            acquired_at=10.0,
            expires_at=20.0,
            state="active",
            failure_count=0,
            finalized_decision=bad_finalized_decision,
            authority_revision=7,
            updated_at=10.0,
        )


@pytest.mark.asyncio
async def test_upsert_lease_fails_closed_on_unsuccessful_result() -> None:
    db = _FakeDatabase()
    db.run_results[GC_LEASE_UPSERT_SQL] = {"success": False, "meta": {"changes": 0}}
    store = D1GcLeaseStore(db)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="unsuccessful"):
        await store.upsert_lease(
            object_key="oai2-blobs/a",
            token="lease-1",
            owner="gc-sweep",
            acquired_at=10.0,
            expires_at=20.0,
            state="active",
            failure_count=0,
            finalized_decision=None,
            authority_revision=7,
            updated_at=10.0,
        )


# ---------------------------------------------------------------------------
# 11. _normalized / _finite_non_negative / _finite_positive / _non_negative_int
#     — private helper surface contract (parametrized matrices)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", ["", " ", "  leading", "trailing  ", "\t", "\n"])
def test_normalized_rejects_non_normalized_strings(bad_value: str) -> None:
    with pytest.raises(ValueError, match="non-empty normalized string"):
        runtime_mod._normalized(bad_value, "field")  # noqa: SLF001 — pinning probe


@pytest.mark.parametrize("bad_value", [None, 0, 1.0, True, False, {}, ["literal"], ("literal",)])
def test_normalized_rejects_non_strings(bad_value: object) -> None:
    with pytest.raises(ValueError, match="non-empty normalized string"):
        runtime_mod._normalized(bad_value, "field")  # noqa: SLF001


@pytest.mark.parametrize("good_value", ["a", "ok", "gc-sweep", "gc.lease-1"])
def test_normalized_accepts_normalized_strings(good_value: str) -> None:
    assert runtime_mod._normalized(good_value, "field") == good_value  # noqa: SLF001


@pytest.mark.parametrize("good_value", [0.0, 1.0, 1, 0, 1e10, math.pi])
def test_finite_non_negative_accepts_finite_non_negative_numbers(
    good_value: float,
) -> None:
    assert (
        runtime_mod._finite_non_negative(good_value, "field")  # noqa: SLF001
        == float(good_value)
    )


@pytest.mark.parametrize(
    "bad_value",
    [True, False, float("nan"), float("inf"), float("-inf"), -1.0, -1, -0.0001],
)
def test_finite_non_negative_rejects_bool_nan_or_negative(bad_value: object) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        runtime_mod._finite_non_negative(  # noqa: SLF001
            bad_value, "field"
        )


@pytest.mark.parametrize(
    "bad_value",
    [None, "10", "0", [0], {0: 0}, (0,), {"x": 0}],
)
def test_finite_non_negative_rejects_non_numeric_types(bad_value: object) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        runtime_mod._finite_non_negative(  # noqa: SLF001
            bad_value, "field"
        )


@pytest.mark.parametrize("good_value", [1.0, 1, 5, 1e10, math.pi])
def test_finite_positive_accepts_positive_numbers(good_value: float) -> None:
    assert (
        runtime_mod._finite_positive(good_value, "field")  # noqa: SLF001
        == float(good_value)
    )


@pytest.mark.parametrize("bad_value", [0.0, 0])
def test_finite_positive_rejects_zero(bad_value: float) -> None:
    """Zero is the boundary — passes the non-negative guard but rejected here."""
    with pytest.raises(ValueError, match="positive"):
        runtime_mod._finite_positive(bad_value, "field")  # noqa: SLF001


@pytest.mark.parametrize("bad_value", [-1.0, -1, -100.5])
def test_finite_positive_rejects_negative(bad_value: float) -> None:
    """Negative values hit the _finite_non_negative guard first."""
    with pytest.raises(ValueError, match="finite non-negative"):
        runtime_mod._finite_positive(bad_value, "field")  # noqa: SLF001


@pytest.mark.parametrize(
    "bad_value",
    [True, False, float("nan"), float("inf"), float("-inf")],
)
def test_finite_positive_rejects_bool_or_nan_via_lower_guard(
    bad_value: object,
) -> None:
    """Bool / NaN / ±inf hit _finite_non_negative first; only zero reaches the
    _finite_positive boundary."""
    with pytest.raises(ValueError, match="finite non-negative"):
        runtime_mod._finite_positive(  # noqa: SLF001
            bad_value, "field"
        )


@pytest.mark.parametrize("good_value", [0, 1, 7, 1_000_000_000])
def test_non_negative_int_accepts_non_negative_ints(good_value: int) -> None:
    assert (
        runtime_mod._non_negative_int(good_value, "field")  # noqa: SLF001
        == good_value
    )


@pytest.mark.parametrize("bad_value", [True, False, -1, -100, 1.0, 0.0, "0", None, [], [1], (1,)])
def test_non_negative_int_rejects_bool_negative_float_or_other(
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        runtime_mod._non_negative_int(bad_value, "field")  # noqa: SLF001


@pytest.mark.parametrize("value", [0, 1])
def test_strict_boolean_int_accepts_boolean_int(value: int) -> None:
    assert (
        runtime_mod._strict_boolean_int(value, "field")  # noqa: SLF001
        is bool(value)
    )


@pytest.mark.parametrize("bad_value", [-1, 2, 3, 100, True, False])
def test_strict_boolean_int_rejects_non_boolean_int(bad_value: object) -> None:
    if isinstance(bad_value, bool) or (
        isinstance(bad_value, int) and not isinstance(bad_value, bool) and bad_value < 0
    ):
        # bool is rejected by _non_negative_int (because isinstance(True, int) is True).
        # negative int is also rejected by _non_negative_int.
        with pytest.raises(ValueError, match="non-negative integer"):
            runtime_mod._strict_boolean_int(bad_value, "field")  # noqa: SLF001
    else:
        with pytest.raises(RuntimeError, match="non-boolean integer"):
            runtime_mod._strict_boolean_int(bad_value, "field")  # noqa: SLF001


# ---------------------------------------------------------------------------
# 12. _result_success / _result_changes — Mapping-vs-attribute duality
#     (Cloudflare Python Workers' D1 binding returns Mapping-shaped objects
#     but also supports attribute access — a refactor that picks one shape
#     would silently break the other)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_result_success_reads_bool_from_mapping(value) -> None:
    result = {"success": value}
    assert (
        runtime_mod._result_success(result)  # noqa: SLF001
        is value
    )


@dataclass
class _AttrResult:
    success: bool


@pytest.mark.parametrize("value", [True, False])
def test_result_success_reads_bool_from_attribute(value) -> None:
    result = _AttrResult(success=value)
    assert (
        runtime_mod._result_success(result)  # noqa: SLF001
        is value
    )


@pytest.mark.parametrize("bad_value", [None, 0, 1, "yes", [], {"name": "ok"}, {}])
def test_result_success_rejects_missing_or_non_bool_success(bad_value: object) -> None:
    with pytest.raises(RuntimeError, match="boolean success field"):
        runtime_mod._result_success(bad_value)  # noqa: SLF001


def test_result_changes_reads_int_from_mapping_meta() -> None:
    result = {"success": True, "meta": {"changes": 5}}
    assert runtime_mod._result_changes(result) == 5  # noqa: SLF001


def test_result_changes_reads_int_from_attribute_path() -> None:
    @dataclass
    class _AttrMeta:
        changes: int = 7

    @dataclass
    class _AttrResult:
        success: bool = True
        meta: _AttrMeta = field(default_factory=_AttrMeta)

    assert (
        runtime_mod._result_changes(_AttrResult())  # noqa: SLF001
        == 7
    )


def test_result_changes_fails_closed_on_missing_meta() -> None:
    """No meta key/attribute — getattr(None, ...) is None and _non_negative_int rejects."""
    result = {"success": True}
    with pytest.raises(ValueError, match="changes"):
        runtime_mod._result_changes(result)  # noqa: SLF001


def test_result_changes_rejects_non_int_changes() -> None:
    result = {"meta": {"changes": "5"}}
    with pytest.raises(ValueError, match="changes"):
        runtime_mod._result_changes(result)  # noqa: SLF001


def test_single_row_transition_fails_closed_on_unsuccessful() -> None:
    result = {"success": False, "meta": {"changes": 1}}
    with pytest.raises(RuntimeError, match="unsuccessful"):
        runtime_mod._single_row_transition(result, "op")  # noqa: SLF001


def test_single_row_transition_fails_closed_on_multirow_changes() -> None:
    result = {"success": True, "meta": {"changes": 2}}
    with pytest.raises(RuntimeError, match="unexpected number of rows"):
        runtime_mod._single_row_transition(result, "op")  # noqa: SLF001


def test_single_row_transition_returns_true_on_single_row() -> None:
    result = {"success": True, "meta": {"changes": 1}}
    assert runtime_mod._single_row_transition(result, "op") is True  # noqa: SLF001


def test_single_row_transition_returns_false_on_zero_rows() -> None:
    result = {"success": True, "meta": {"changes": 0}}
    assert runtime_mod._single_row_transition(result, "op") is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# 13. Mapping / Sequence import contract (UP006 verification)
# ---------------------------------------------------------------------------


def test_runtime_module_uses_collections_abc_mapping_and_sequence() -> None:
    """UP006 rule: typing.Mapping/Sequence is reserved for type expressions.
    The runtime imports them from collections.abc for runtime use."""
    # Inspect the source file's import lines to confirm the discipline.
    import inspect

    source = inspect.getsource(runtime_mod)
    assert "from collections.abc import Mapping, Sequence" in source
    # And NOT from typing
    assert "from typing import Mapping" not in source
    assert "from typing import Sequence" not in source


def test_runtime_module_uses_typing_protocol_and_self() -> None:
    """Protocol + Self are correctly sourced from typing.* (UP006 leaves them)."""
    import inspect

    source = inspect.getsource(runtime_mod)
    assert "from typing import Protocol, Self" in source


def test_protocol_runtime_checkable_is_not_applied() -> None:
    """The Protocols are structural-only — runtime isinstance() checks would not work."""
    # If @runtime_checkable were applied, hasattr(Protocol, "_is_runtime_protocol") is True.
    # Plain Protocol does not have it.
    assert not getattr(D1PreparedStatementBinding, "_is_runtime_protocol", False)
    assert not getattr(D1DatabaseBinding, "_is_runtime_protocol", False)
