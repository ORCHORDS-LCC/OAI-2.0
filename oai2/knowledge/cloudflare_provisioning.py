"""Explicit schema provisioning for the knowledge subsystem.

SEPARATION OF CONCERNS
----------------------
Provisioning and request serving are different activities with different risk,
and this module exists so they cannot be confused:

* :func:`provision_knowledge_schema` applies DDL. It is an operator/deployment
  step, run out of band. It is NOT called from any request path.
* :func:`verify_knowledge_schema` is read-only and is what a readiness check
  calls. The request path assumes the schema is present and fails clearly if it
  is not, rather than silently creating tables while serving traffic.

Why this was separated rather than cached
-----------------------------------------
The Worker request path used to call the factory with ``ensure_schema=True``.
That was ineffective as caching and wrong as behaviour. ``WorkerEntrypoint``
constructs a NEW instance per invocation, so the per-instance component memo
was rebuilt on every request and the DDL was re-attempted every request. Worse,
schema creation as an ordinary side effect of serving traffic means a request
can mutate schema, which is exactly the operation that should never be
implicit in a read or write path.

What may and may not be cached globally
---------------------------------------
This module holds no cached clients. Binding-derived objects must not be cached
at module or global scope, because Cloudflare can reuse an isolate across a
binding-only change and a globally cached client would then be stale against
different bindings. Every function here takes the database explicitly.

Nothing in this module has been run against a live D1 database. Applying the
new resource-lease table and the ``vectorize_id`` index to a live database is
a migration that has NOT been executed; see #261.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .gc_lease_d1 import (
    GC_LEASE_TABLE,
    gc_lease_schema_statements,
)
from .gc_lease_d1_runtime import D1DatabaseBinding
from .gc_resource_lease_d1 import (
    GC_RESOURCE_LEASE_TABLE,
    GC_RESOURCE_TYPE_R2_BLOB,
    GC_RESOURCE_TYPE_VECTOR,
    gc_resource_lease_schema_statements,
)
from .knowledge_d1 import (
    KNOWLEDGE_CORPUS_STATE_TABLE,
    KNOWLEDGE_INDEX_TABLE,
    knowledge_schema_statements,
)

#: Index the vector reclamation path depends on. Without it the vector
#: reference predicate inside the writer fence is a table scan on every write.
VECTORIZE_ID_INDEX = "idx_knowledge_index_vectorize_id"

#: The old #233 R2 lease table. Still created for compatibility: the writer SQL
#: consults BOTH lease tables, so dropping this one would silently unblock
#: writes during any sweep the #233 work owns.
LEGACY_R2_LEASE_TABLE = GC_LEASE_TABLE


#: The D1 capability provisioning needs. This is the SAME protocol the rest of
#: the knowledge runtime binds against, not a local near-copy: a second,
#: looser definition would accept a binding the real runtime cannot use, and
#: the mismatch would only surface as a mypy error at the call site.
D1SchemaBinding = D1DatabaseBinding


def provisioning_batches() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The ordered provisioning contract.

    Order is load-bearing and is asserted by test:

    1. knowledge base tables, so the lease reference predicates have something
       to resolve against;
    2. the legacy R2 lease table, which the writer SQL still consults;
    3. the typed resource lease table and the vectorize_id index, which the
       writer fence consults;
    4. only then is request serving expected to work.

    The resource lease cannot be created before the knowledge tables because its
    statements are written against ``knowledge_index`` columns.
    """
    base = tuple(knowledge_schema_statements())
    legacy = tuple(gc_lease_schema_statements())
    typed = tuple(gc_resource_lease_schema_statements())
    return (
        ("knowledge_base", base),
        ("legacy_r2_lease", legacy),
        ("typed_resource_lease", typed),
    )


def required_schema_objects() -> tuple[str, ...]:
    """Tables and indexes that must exist before request serving is correct."""
    return (
        KNOWLEDGE_INDEX_TABLE,
        KNOWLEDGE_CORPUS_STATE_TABLE,
        LEGACY_R2_LEASE_TABLE,
        GC_RESOURCE_LEASE_TABLE,
        VECTORIZE_ID_INDEX,
    )


async def provision_knowledge_schema(database: D1SchemaBinding) -> dict[str, int]:
    """Apply the knowledge schema in dependency order.

    Operator/deployment step. Not for a request path. Returns the number of
    statements applied per batch for evidence.
    """
    applied: dict[str, int] = {}
    for name, statements in provisioning_batches():
        if not statements:
            applied[name] = 0
            continue
        results = await database.batch([database.prepare(sql) for sql in statements])
        if len(results) != len(statements):
            raise RuntimeError(
                f"D1 knowledge provisioning batch {name!r} returned an "
                "unexpected result count"
            )
        applied[name] = len(statements)
    return applied


async def verify_knowledge_schema(database: D1SchemaBinding) -> None:
    """Read-only readiness check. Raises listing exactly what is missing.

    The request path does not call this on every request; a deployment
    readiness check calls it once, and a request that hits a missing object
    fails at the statement with a clear D1 error rather than being papered over
    by implicit DDL.
    """
    missing: list[str] = []
    for name, statement in _verify_schema_statements():
        result = await database.batch([database.prepare(statement)])
        present = _first_value(result[0] if result else None)
        if not _is_one(present):
            missing.append(name)
    if missing:
        raise RuntimeError(
            "knowledge schema is not provisioned; missing: " + ", ".join(missing)
        )


def _verify_schema_statements() -> tuple[tuple[str, str], ...]:
    # The column is aliased so the value can be read by NAME. D1 returns rows
    # as mappings, so an unaliased `COUNT(*)` would have to be indexed by its
    # driver-specific label, which is exactly the kind of assumption that
    # crashes a readiness check at the worst moment.
    return tuple(
        (
            name,
            "SELECT COUNT(*) AS present FROM sqlite_master "
            "WHERE type IN ('table','index') "
            f"AND name = '{name}'",
        )
        for name in required_schema_objects()
    )


def _is_one(value: object) -> bool:
    """Whether a COUNT result says "present".

    Compared without coercion, because a readiness check that raises on an
    unexpected driver value is worse than one that reports the object missing:
    absent-and-unknown both mean "not ready", and both must fail closed.
    """
    return isinstance(value, (int, bool)) and value == 1


def _first_value(result: object) -> object | None:
    """First column of the first row, for either a mapping or a sequence row.

    D1's batch results are mappings; a test double or a driver may hand back
    positional rows. Both are accepted so the readiness check is not coupled
    to one driver's row shape.
    """
    rows = result.get("results") if isinstance(result, Mapping) else getattr(
        result, "results", None
    )
    if rows is None:
        return None
    if hasattr(rows, "to_py"):
        rows = rows.to_py()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return None
    for row in rows:
        if hasattr(row, "to_py"):
            row = row.to_py()
        if isinstance(row, Mapping):
            return next(iter(row.values()), None)
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
            return row[0] if row else None
    return None


__all__ = [
    "D1SchemaBinding",
    "LEGACY_R2_LEASE_TABLE",
    "VECTORIZE_ID_INDEX",
    "GC_RESOURCE_TYPE_R2_BLOB",
    "GC_RESOURCE_TYPE_VECTOR",
    "provision_knowledge_schema",
    "provisioning_batches",
    "required_schema_objects",
    "verify_knowledge_schema",
]
