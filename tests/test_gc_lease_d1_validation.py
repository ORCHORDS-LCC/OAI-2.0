"""Outer-guard-rail validation tests for ``oai2.knowledge.gc_lease_d1``.

This module pins the public-safe D1 schema contract for GC deletion leases
(see ``oai2/knowledge/gc_lease_d1.py`` module docstring). The contract is a
wire format that the live Worker code is expected to apply through a bound
``D1Database`` using prepared/batched async calls; it intentionally contains
no account IDs, database IDs, bucket names, endpoints, or credentials.

What this slice pins (the surface that ``tests/test_gc_lease_d1.py`` does
NOT cover):

* ``GC_LEASE_SCHEMA_VERSION`` stability — a bump without coordinating with
  the live migrations would silently break schema rollout.
* ``GC_LEASE_TABLE`` literal — pinned as ``"knowledge_gc_delete_lease"``;
  the live migrations reference this literal.
* ``gc_lease_schema_statements()`` behaviour — tuple-of-strings return;
  CREATE TABLE + CREATE UNIQUE INDEX statements present; STRICT mode
  applied; empty-segment filter active; ≥2 statements; deterministic order.
* SQL-content pinning for the load-bearing substrings the live Worker
  code depends on (CHECK constraints, unique index, ON CONFLICT upsert
  clauses, ACQUIRE conditional gating, RECORD_FAILURE / FINALIZE /
  RELEASE token-guard + expiry + reference-recheck, cross-table
  guards via ``knowledge_corpus_state`` and ``knowledge_index``).
* ``__all__`` (13 names) + package-level re-export identity.
"""

from __future__ import annotations

import pytest

from oai2.knowledge.gc_lease_d1 import (
    GC_LEASE_ACQUIRE_SQL,
    GC_LEASE_FINALIZE_SQL,
    GC_LEASE_RECORD_FAILURE_SQL,
    GC_LEASE_REFERENCE_COUNT_SQL,
    GC_LEASE_RELEASE_SQL,
    GC_LEASE_SCHEMA_SQL,
    GC_LEASE_SCHEMA_VERSION,
    GC_LEASE_SELECT_SQL,
    GC_LEASE_TABLE,
    GC_LEASE_UPSERT_SQL,
    GC_LEASE_VALIDATE_SQL,
    GC_LEASE_WRITER_BLOCK_SQL,
    gc_lease_schema_statements,
)


def test_gc_lease_schema_version_is_pinned_int_one() -> None:
    """``GC_LEASE_SCHEMA_VERSION`` is pinned as int 1 — a bump without
    coordinating with the live migrations would silently corrupt the
    rollout, so the constant is pinned as a literal here."""
    assert isinstance(GC_LEASE_SCHEMA_VERSION, int)
    assert GC_LEASE_SCHEMA_VERSION == 1


def test_gc_lease_table_literal_is_pinned() -> None:
    """``GC_LEASE_TABLE`` is pinned as the string literal
    ``"knowledge_gc_delete_lease"`` — the live migrations reference this
    literal, so a refactor that renames the constant without coordinating
    with the migration chain would silently break rollout."""
    assert GC_LEASE_TABLE == "knowledge_gc_delete_lease"


def test_gc_lease_table_is_string_type() -> None:
    """``GC_LEASE_TABLE`` is a string — the ``f"CREATE TABLE {GC_LEASE_TABLE}"``
    f-string in the source module requires a string-coercible value, and
    downstream consumers that join the table name into raw SQL rely on
    it being a plain ``str``."""
    assert isinstance(GC_LEASE_TABLE, str)


def test_gc_lease_schema_statements_returns_tuple_of_strings() -> None:
    """``gc_lease_schema_statements()`` returns a tuple — callers iterate
    it for ordered application (the Worker applies schema statements in
    deterministic order during ``D1GcLeaseStore.ensure_schema()``)."""
    statements = gc_lease_schema_statements()
    assert isinstance(statements, tuple)
    assert statements
    for statement in statements:
        assert isinstance(statement, str)
        assert statement.strip()


def test_gc_lease_schema_statements_contains_create_table() -> None:
    """The schema statements contain a CREATE TABLE for the GC lease table
    — this is the wire-contract the live migrations apply; without it,
    upserts against the lease table would fail at runtime."""
    statements = "\n".join(gc_lease_schema_statements())
    assert "CREATE TABLE IF NOT EXISTS knowledge_gc_delete_lease" in statements


def test_gc_lease_schema_statements_contains_unique_index_on_token() -> None:
    """The schema statements contain a CREATE UNIQUE INDEX on the
    ``token`` column — the unique-index is the cross-row invariant that
    prevents two concurrent leases from sharing a token (the
    conditional-acquire path depends on it)."""
    statements = "\n".join(gc_lease_schema_statements())
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_gc_delete_lease_token" in statements
    assert "ON knowledge_gc_delete_lease(token)" in statements


def test_gc_lease_schema_statements_includes_strict_mode() -> None:
    """The CREATE TABLE statement uses ``STRICT`` mode — this is the
    SQLite STRICT table-type that enforces declared column types (a
    refactor that drops STRICT would silently broaden the column types
    and break the wire contract)."""
    statements = "\n".join(gc_lease_schema_statements())
    assert "STRICT" in statements


def test_gc_lease_schema_statements_filters_empty_segments() -> None:
    """``gc_lease_schema_statements()`` filters out empty segments — the
    ``str.split(";")`` followed by ``str.strip()`` produces empty
    strings from trailing semicolons; the helper must drop them, not
    pass them to D1 (an empty statement would be rejected by D1)."""
    statements = gc_lease_schema_statements()
    assert "" not in statements
    for statement in statements:
        assert statement
        assert not statement.endswith(";")


def test_gc_lease_schema_statements_returns_at_least_two_statements() -> None:
    """The schema is at minimum CREATE TABLE + CREATE UNIQUE INDEX —
    fewer than 2 statements means either the table or the index was
    dropped from the schema contract."""
    assert len(gc_lease_schema_statements()) >= 2


def test_gc_lease_schema_statements_returns_in_deterministic_order() -> None:
    """Calling ``gc_lease_schema_statements()`` twice returns the same
    tuple in the same order — the Worker relies on deterministic schema
    application so re-evaluating the schema does not churn the D1
    "schema already applied" check."""
    first = gc_lease_schema_statements()
    second = gc_lease_schema_statements()
    assert first == second


def test_gc_lease_schema_sql_contains_acquired_at_non_negative_check() -> None:
    """``GC_LEASE_SCHEMA_SQL`` pins ``CHECK(acquired_at >= 0)`` — the
    CHECK is the wire contract that prevents negative timestamps from
    poisoning the lease store; the runtime upsert/select path depends
    on non-negative values."""
    assert "CHECK(acquired_at >= 0)" in GC_LEASE_SCHEMA_SQL


def test_gc_lease_schema_sql_contains_expires_strictly_greater_than_acquired_check() -> None:
    """``GC_LEASE_SCHEMA_SQL`` pins ``CHECK(expires_at > acquired_at)`` —
    the strict-greater invariant means a lease whose expires_at equals
    its acquired_at is invalid (it would be expired on creation)."""
    assert "CHECK(expires_at > acquired_at)" in GC_LEASE_SCHEMA_SQL


def test_gc_lease_schema_sql_contains_state_in_set_check() -> None:
    """``GC_LEASE_SCHEMA_SQL`` pins a state-set CHECK — the five-state
    StrEnum ``GcDeleteLeaseState`` (active / delete_failed / deleted /
    released / replaced) is enforced at the D1 level so a typo or
    stale-string write cannot land in the lease table."""
    schema = GC_LEASE_SCHEMA_SQL
    assert "CHECK(" in schema
    assert "'active'" in schema
    assert "'delete_failed'" in schema
    assert "'deleted'" in schema
    assert "'released'" in schema
    assert "'replaced'" in schema


def test_gc_lease_schema_sql_contains_failure_count_non_negative_check() -> None:
    """``GC_LEASE_SCHEMA_SQL`` pins ``CHECK(failure_count >= 0)`` — the
    non-negative invariant on ``failure_count`` is what makes
    ``failure_count + 1`` safe (a refactor that dropped the CHECK would
    silently allow a corrupt negative count to propagate)."""
    assert "CHECK(failure_count >= 0)" in GC_LEASE_SCHEMA_SQL


def test_gc_lease_schema_sql_contains_authority_revision_non_negative_check() -> None:
    """``GC_LEASE_SCHEMA_SQL`` pins ``CHECK(authority_revision >= 0)`` —
    the non-negative invariant on the authority-revision matches the
    ``GcDeleteLeaseAuthority.revision = 0`` initial value (a negative
    revision would let a wrong-conditioned acquire SQL match)."""
    assert "CHECK(authority_revision >= 0)" in GC_LEASE_SCHEMA_SQL


def test_gc_lease_schema_sql_contains_updated_at_non_negative_check() -> None:
    """``GC_LEASE_SCHEMA_SQL`` pins ``CHECK(updated_at >= 0)`` — the
    non-negative invariant on ``updated_at`` matches the
    ``acquired_at >= 0`` invariant and prevents clock-skew negatives
    from poisoning the timestamp columns."""
    assert "CHECK(updated_at >= 0)" in GC_LEASE_SCHEMA_SQL


def test_gc_lease_schema_sql_contains_finalized_decision_or_null_in_set_check() -> None:
    """``GC_LEASE_SCHEMA_SQL`` pins the
    ``finalized_decision IS NULL OR finalized_decision IN (...)`` CHECK
    — the OR-NULL-in-set pattern allows null finalized_decision (no
    terminal decision yet) but constrains non-null values to the
    five-terminal-decision StrEnum (deleted / already_absent /
    already_finalized / retryable_failure_recorded / invalid_lease)."""
    schema = GC_LEASE_SCHEMA_SQL
    assert "finalized_decision IS NULL" in schema
    assert "finalized_decision IN" in schema
    assert "'deleted'" in schema
    assert "'already_absent'" in schema
    assert "'already_finalized'" in schema
    assert "'retryable_failure_recorded'" in schema
    assert "'invalid_lease'" in schema


def test_gc_lease_select_sql_selects_all_ten_columns() -> None:
    """``GC_LEASE_SELECT_SQL`` selects all 10 columns (object_key,
    token, owner, acquired_at, expires_at, state, failure_count,
    finalized_decision, authority_revision, updated_at) — a refactor
    that drops a column from the SELECT would silently truncate the
    ``GcDeleteLease`` reconstruction in the lease store."""
    select = GC_LEASE_SELECT_SQL
    expected_columns = (
        "object_key",
        "token",
        "owner",
        "acquired_at",
        "expires_at",
        "state",
        "failure_count",
        "finalized_decision",
        "authority_revision",
        "updated_at",
    )
    for column in expected_columns:
        assert column in select
    assert select.count("SELECT") == 1
    assert "FROM knowledge_gc_delete_lease" in select
    assert "WHERE object_key = ?1" in select


def test_gc_lease_reference_count_sql_counts_against_knowledge_index() -> None:
    """``GC_LEASE_REFERENCE_COUNT_SQL`` is pinned as a
    ``COUNT(*) FROM knowledge_index WHERE r2_blob_key = ?1`` query —
    the knowledge_index table is the authoritative reference ledger;
    a refactor that switched to a different table would let orphaned
    references escape the GC sweep."""
    assert "COUNT(*)" in GC_LEASE_REFERENCE_COUNT_SQL
    assert "FROM knowledge_index" in GC_LEASE_REFERENCE_COUNT_SQL
    assert "WHERE r2_blob_key = ?1" in GC_LEASE_REFERENCE_COUNT_SQL


def test_gc_lease_upsert_sql_uses_on_conflict_object_key() -> None:
    """``GC_LEASE_UPSERT_SQL`` uses ``ON CONFLICT(object_key) DO UPDATE``
    — the object_key is the primary key, and the upsert's conflict
    target must match it (a different conflict target would let two
    rows share an object_key)."""
    select = GC_LEASE_UPSERT_SQL
    assert "ON CONFLICT(object_key) DO UPDATE" in select
    assert "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)" in select
    assert "token = excluded.token" in select
    assert "owner = excluded.owner" in select
    assert "acquired_at = excluded.acquired_at" in select
    assert "expires_at = excluded.expires_at" in select
    assert "state = excluded.state" in select
    assert "failure_count = excluded.failure_count" in select
    assert "finalized_decision = excluded.finalized_decision" in select
    assert "authority_revision = excluded.authority_revision" in select
    assert "updated_at = excluded.updated_at" in select


def test_gc_lease_acquire_sql_gates_on_corpus_state_revision() -> None:
    """``GC_LEASE_ACQUIRE_SQL`` is gated on
    ``knowledge_corpus_state.singleton = 1 AND revision = ?6`` — the
    cross-document singleton+revision invariant matches
    ``KNOWLEDGE_CORPUS_REVISION_SQL`` and is what prevents a stale
    acquire from landing against a superseded corpus revision."""
    select = GC_LEASE_ACQUIRE_SQL
    assert "FROM knowledge_corpus_state" in select
    assert "WHERE singleton = 1" in select
    assert "revision = ?6" in select


def test_gc_lease_acquire_sql_blocks_referenced_objects() -> None:
    """``GC_LEASE_ACQUIRE_SQL`` blocks referenced objects — the
    ``NOT EXISTS (SELECT 1 FROM knowledge_index WHERE r2_blob_key = ?1)``
    gate is the cross-document reference check that prevents an
    acquire from succeeding when an authoritative reference still
    points at the object."""
    assert "NOT EXISTS" in GC_LEASE_ACQUIRE_SQL
    assert "FROM knowledge_index" in GC_LEASE_ACQUIRE_SQL
    assert "WHERE r2_blob_key = ?1" in GC_LEASE_ACQUIRE_SQL


def test_gc_lease_acquire_sql_allows_expired_takeover_in_active_or_delete_failed() -> None:
    """``GC_LEASE_ACQUIRE_SQL`` permits an expired-takeover only when
    the existing lease is in (active, delete_failed) — the takeover
    branch lives in the WHERE block and uses
    ``expires_at <= excluded.acquired_at`` to detect expiry (an
    equivalence that's load-bearing: a refactor that used ``<`` would let
    equal-acquired leases count as expired)."""
    select = GC_LEASE_ACQUIRE_SQL
    assert "state IN ('released', 'replaced')" in select
    assert "state IN ('active', 'delete_failed')" in select
    assert "expires_at <= excluded.acquired_at" in select


def test_gc_lease_acquire_sql_resets_to_active_with_zero_failure_count() -> None:
    """``GC_LEASE_ACQUIRE_SQL`` resets state to 'active' and
    failure_count to 0 on a takeover — the takeover is treated as a
    fresh lease, so a previous failure must not propagate into the new
    operator's history (a refactor that preserved failure_count would let
    a flaky previous operator's count poison the new operator)."""
    assert "state = 'active'" in GC_LEASE_ACQUIRE_SQL
    assert "failure_count = 0" in GC_LEASE_ACQUIRE_SQL
    assert "finalized_decision = NULL" in GC_LEASE_ACQUIRE_SQL


def test_gc_lease_writer_block_sql_returns_boolean_int() -> None:
    """``GC_LEASE_WRITER_BLOCK_SQL`` returns ``1`` (writer blocked) or
    ``0`` (not blocked) — the boolean-int-as-int encoding is what the
    runtime maps to a Python bool via ``non-boolean integer``
    rejection; a refactor that returned the column name directly
    would silently pass non-boolean scalars."""
    select = GC_LEASE_WRITER_BLOCK_SQL
    assert "THEN 1" in select
    assert "ELSE 0" in select
    assert "writer_blocked" in select
    assert "state IN ('active', 'delete_failed')" in select
    assert "expires_at > ?2" in select
    assert "FROM knowledge_gc_delete_lease" in select


def test_gc_lease_validate_sql_returns_boolean_int_and_rechecks_references() -> None:
    """``GC_LEASE_VALIDATE_SQL`` returns ``1`` (valid) or ``0`` (invalid)
    and re-checks authoritative references — the
    ``NOT EXISTS (SELECT 1 FROM knowledge_index WHERE r2_blob_key = ?1)``
    cross-check is what makes the validate path authoritative (a
    refactor that dropped the reference-recheck would let a
    newly-referenced object be deleted)."""
    select = GC_LEASE_VALIDATE_SQL
    assert "THEN 1" in select
    assert "ELSE 0" in select
    assert "lease_valid" in select
    assert "token = ?2" in select
    assert "state IN ('active', 'delete_failed')" in select
    assert "expires_at > ?3" in select
    assert "NOT EXISTS" in select
    assert "FROM knowledge_index" in select
    assert "WHERE r2_blob_key = ?1" in select


def test_gc_lease_record_failure_sql_increments_failure_count_and_sets_state() -> None:
    """``GC_LEASE_RECORD_FAILURE_SQL`` increments ``failure_count`` by 1
    and sets ``state`` to 'delete_failed' — the monotonic increment
    survives across multiple failures and is the load-bearing invariant
    for the retry-then-failover state machine."""
    select = GC_LEASE_RECORD_FAILURE_SQL
    assert "failure_count = failure_count + 1" in select
    assert "state = 'delete_failed'" in select
    assert "finalized_decision = 'retryable_failure_recorded'" in select
    assert "WHERE object_key = ?1" in select
    assert "AND token = ?2" in select
    assert "state IN ('active', 'delete_failed')" in select
    assert "expires_at > ?3" in select
    assert "NOT EXISTS" in select
    assert "FROM knowledge_index" in select
    assert "WHERE r2_blob_key = ?1" in select


def test_gc_lease_finalize_sql_constrains_terminal_decision() -> None:
    """``GC_LEASE_FINALIZE_SQL`` constrains the
    ``finalized_decision`` parameter to ``('deleted', 'already_absent')``
    — these are the two terminal decisions the finalize path emits;
    a refactor that broadened to all five would let ``invalid_lease``
    decisions be set as terminal, which would silently break the
    audit trail."""
    select = GC_LEASE_FINALIZE_SQL
    assert "state = 'deleted'" in select
    assert "finalized_decision = ?4" in select
    assert "?4 IN ('deleted', 'already_absent')" in select
    assert "WHERE object_key = ?1" in select
    assert "AND token = ?2" in select
    assert "state IN ('active', 'delete_failed')" in select
    assert "expires_at > ?3" in select
    assert "NOT EXISTS" in select
    assert "FROM knowledge_index" in select
    assert "WHERE r2_blob_key = ?1" in select


def test_gc_lease_release_sql_sets_released_with_null_finalized_decision() -> None:
    """``GC_LEASE_RELEASE_SQL`` sets ``state`` to 'released' and
    ``finalized_decision`` to NULL — the NULL reset matches the
    'released' state being non-terminal (the lease is gone but the
    object's lifecycle continues; a refactor that left
    'finalized_decision' populated would silently break the audit
    trail)."""
    select = GC_LEASE_RELEASE_SQL
    assert "state = 'released'" in select
    assert "finalized_decision = NULL" in select
    assert "WHERE object_key = ?1" in select
    assert "AND token = ?2" in select
    assert "state IN ('active', 'delete_failed')" in select
    assert "expires_at > ?3" in select


def test_gc_lease_schema_sql_is_non_empty_string() -> None:
    """``GC_LEASE_SCHEMA_SQL`` is a non-empty string — the Worker applies
    this via D1Database.prepare; an empty schema would silently apply
    no statements."""
    assert isinstance(GC_LEASE_SCHEMA_SQL, str)
    assert GC_LEASE_SCHEMA_SQL.strip()


def test_gc_lease_select_sql_targets_table_literal() -> None:
    """``GC_LEASE_SELECT_SQL`` targets the GC_LEASE_TABLE literal — the
    literal is the wire contract, not a Python f-string interpolation
    result; the source uses ``f"..."`` but the result must match the
    GC_LEASE_TABLE constant exactly."""
    assert GC_LEASE_TABLE in GC_LEASE_SELECT_SQL
    assert f"FROM {GC_LEASE_TABLE}" in GC_LEASE_SELECT_SQL


def test_gc_lease_writer_block_sql_targets_table_literal() -> None:
    """``GC_LEASE_WRITER_BLOCK_SQL`` targets the GC_LEASE_TABLE literal —
    same invariant as SELECT (the table name is the wire contract)."""
    assert GC_LEASE_TABLE in GC_LEASE_WRITER_BLOCK_SQL
    assert f"FROM {GC_LEASE_TABLE}" in GC_LEASE_WRITER_BLOCK_SQL


def test_gc_lease_record_failure_sql_targets_table_literal() -> None:
    """``GC_LEASE_RECORD_FAILURE_SQL`` targets the GC_LEASE_TABLE literal
    in its UPDATE clause."""
    assert GC_LEASE_TABLE in GC_LEASE_RECORD_FAILURE_SQL
    assert f"UPDATE {GC_LEASE_TABLE}" in GC_LEASE_RECORD_FAILURE_SQL


def test_gc_lease_finalize_sql_targets_table_literal() -> None:
    """``GC_LEASE_FINALIZE_SQL`` targets the GC_LEASE_TABLE literal in
    its UPDATE clause."""
    assert GC_LEASE_TABLE in GC_LEASE_FINALIZE_SQL
    assert f"UPDATE {GC_LEASE_TABLE}" in GC_LEASE_FINALIZE_SQL


def test_gc_lease_release_sql_targets_table_literal() -> None:
    """``GC_LEASE_RELEASE_SQL`` targets the GC_LEASE_TABLE literal in
    its UPDATE clause."""
    assert GC_LEASE_TABLE in GC_LEASE_RELEASE_SQL
    assert f"UPDATE {GC_LEASE_TABLE}" in GC_LEASE_RELEASE_SQL


def test_gc_lease_upsert_sql_targets_table_literal() -> None:
    """``GC_LEASE_UPSERT_SQL`` targets the GC_LEASE_TABLE literal in
    its INSERT INTO clause."""
    assert GC_LEASE_TABLE in GC_LEASE_UPSERT_SQL
    assert f"INSERT INTO {GC_LEASE_TABLE}" in GC_LEASE_UPSERT_SQL


def test_gc_lease_acquire_sql_targets_table_literal() -> None:
    """``GC_LEASE_ACQUIRE_SQL`` targets the GC_LEASE_TABLE literal in
    its INSERT INTO clause."""
    assert GC_LEASE_TABLE in GC_LEASE_ACQUIRE_SQL
    assert f"INSERT INTO {GC_LEASE_TABLE}" in GC_LEASE_ACQUIRE_SQL


@pytest.mark.parametrize(
    "sql_constant",
    [
        GC_LEASE_SCHEMA_SQL,
        GC_LEASE_SELECT_SQL,
        GC_LEASE_REFERENCE_COUNT_SQL,
        GC_LEASE_UPSERT_SQL,
        GC_LEASE_ACQUIRE_SQL,
        GC_LEASE_WRITER_BLOCK_SQL,
        GC_LEASE_VALIDATE_SQL,
        GC_LEASE_RECORD_FAILURE_SQL,
        GC_LEASE_FINALIZE_SQL,
        GC_LEASE_RELEASE_SQL,
    ],
)
def test_all_gc_lease_sql_constants_are_non_empty_strings(sql_constant: str) -> None:
    """Every public ``GC_LEASE_*_SQL`` constant is a non-empty ``str`` —
    a refactor that introduced an empty or non-string SQL constant
    would silently fail D1 preparation at runtime."""
    assert isinstance(sql_constant, str)
    assert sql_constant.strip()


@pytest.mark.parametrize(
    "sql_constant",
    [
        GC_LEASE_SELECT_SQL,
        GC_LEASE_REFERENCE_COUNT_SQL,
        GC_LEASE_UPSERT_SQL,
        GC_LEASE_ACQUIRE_SQL,
        GC_LEASE_WRITER_BLOCK_SQL,
        GC_LEASE_VALIDATE_SQL,
        GC_LEASE_RECORD_FAILURE_SQL,
        GC_LEASE_FINALIZE_SQL,
        GC_LEASE_RELEASE_SQL,
    ],
)
def test_all_single_statement_gc_lease_sql_constants_have_no_trailing_semicolon(
    sql_constant: str,
) -> None:
    """Every single-statement ``GC_LEASE_*_SQL`` constant has no trailing
    semicolon — D1 rejects trailing semicolons in single-statement
    prepare calls (a refactor that appended a ``;`` would break
    ``db.prepare(constant).bind(...).run()`` patterns).

    ``GC_LEASE_SCHEMA_SQL`` is the documented exception — it bundles the
    CREATE TABLE + CREATE UNIQUE INDEX into one literal with a trailing
    ``;`` separator and is consumed via ``gc_lease_schema_statements()``
    (which splits on ``;``), not via a single-statement ``prepare()``."""
    assert not sql_constant.rstrip().endswith(";")


def test_gc_lease_schema_sql_ends_with_trailing_semicolon() -> None:
    """``GC_LEASE_SCHEMA_SQL`` ends with a trailing semicolon — the
    trailing ``;`` is the statement-separator for the bundled
    CREATE TABLE + CREATE UNIQUE INDEX pair. ``gc_lease_schema_statements()``
    relies on this trailing separator to split the literal into
    application-order entries. A refactor that dropped the trailing ``;``
    would silently merge the two statements into one (D1 would reject
    the result)."""
    assert GC_LEASE_SCHEMA_SQL.rstrip().endswith(";")


def test_gc_lease_d1_module_all_exports_match_expected_set() -> None:
    """``__all__`` lists exactly the 13 public names — a refactor that
    silently dropped or renamed an export would break callers that
    rely on the ``from oai2.knowledge.gc_lease_d1 import X`` shape."""
    import oai2.knowledge.gc_lease_d1 as module

    expected = {
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
    }
    assert set(module.__all__) == expected
    assert len(module.__all__) == 13


def test_gc_lease_d1_module_all_includes_subpackage_consistent_order() -> None:
    """``__all__`` is the wire contract that the knowledge package
    relies on for ``from oai2.knowledge import *`` semantics — a
    refactor that silently dropped a name from ``__all__`` would
    break the package-level re-export in ``oai2/knowledge/__init__.py``."""
    import oai2.knowledge.gc_lease_d1 as module

    assert "gc_lease_schema_statements" in module.__all__
    assert "GC_LEASE_SCHEMA_VERSION" in module.__all__
    assert "GC_LEASE_TABLE" in module.__all__
    for sql_constant_name in (
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
    ):
        assert sql_constant_name in module.__all__


def test_gc_lease_d1_package_level_re_exports_match_source() -> None:
    """The 13 names are re-exported from ``oai2.knowledge`` and are the
    exact same objects as the source module — a refactor that silently
    removed a name from the package-level import statement in
    ``oai2/knowledge/__init__.py`` would break ``from oai2.knowledge
    import GC_LEASE_*`` callers."""
    import oai2.knowledge as package
    import oai2.knowledge.gc_lease_d1 as module

    for name in module.__all__:
        source_value = getattr(module, name)
        package_value = getattr(package, name)
        assert source_value is package_value, (
            f"package-level re-export of {name} diverged from source"
        )
