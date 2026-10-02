"""Outer-guard-rail boundary contracts for ``oai2.knowledge.gc_delete_d1_runtime``.

This file pins the public-safety boundary of the async D1/R2 destructive-delete
runtime that bridges the conservative sweep planner to Cloudflare's
asynchronous APIs. The companion module
``oai2/knowledge/gc_delete_d1_runtime.py`` defines the live-binding-facing
async function that mediates a single GC candidate's destruction.

These tests focus on the *outer guard rails*:

- ``D1DeleteOutcome`` StrEnum stability (4 values + per-value strings + value-lookup).
- ``D1DeleteResult`` shape (required-field set + frozen mutation rejected).
- ``delete_candidate_with_d1_lease`` argument validation: ``key`` empty/whitespace/unstripped/non-string rejected; ``owner`` empty/whitespace/unstripped rejected; explicit ``token`` empty/whitespace/unstripped rejected; default token auto-generated.
- ``delete_candidate_with_d1_lease`` decision matrix: each ``D1DeleteOutcome`` is produced exactly once by its documented sequence; each detail string is pinned; each ``D1GcLeaseStore`` method call sequence is pinned (acquire / validate / finalize / failure).
- ``blob_exists`` non-bool returns + raises for both the first and second existence checks are caught and routed to ``FAILED`` + ``record_delete_failure``.
- ``delete_blob`` raising is caught and routed to ``FAILED``.
- Lease store argument propagation: ``acquire_lease`` / ``record_delete_failure`` / ``finalize_delete`` kwargs match the documented schema; ``finalize_delete`` already_absent is a strict bool.
- ``__all__`` exports.

A refactor that swaps the ``.strip()`` non-empty check for an
``isinstance(key, str)`` only check, or that drops the second existence
verification, or that changes the default token format, or that silently
swallows a ``blob_exists`` exception, must trip one of these tests.
"""

from __future__ import annotations

import pytest

from oai2.knowledge import gc_delete_d1_runtime as gc_delete_d1_runtime_mod
from oai2.knowledge.gc_delete_d1_runtime import (
    D1DeleteOutcome,
    D1DeleteResult,
    delete_candidate_with_d1_lease,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLeaseStore:
    """Lightweight D1GcLeaseStore substitute for boundary tests.

    Records every call so each test can pin the exact (name, kwargs) tuple
    sequence. The boolean returns are configurable per instance.
    """

    def __init__(
        self,
        *,
        acquire_result: bool = True,
        valid_result: bool = True,
        finalize_result: bool = True,
        failure_result: bool = True,
    ) -> None:
        self.acquire_result = acquire_result
        self.valid_result = valid_result
        self.finalize_result = finalize_result
        self.failure_result = failure_result
        self.calls: list[tuple[str, object]] = []

    async def acquire_lease(self, **kwargs: object) -> bool:
        self.calls.append(("acquire", kwargs))
        return self.acquire_result

    async def lease_valid(self, key: str, token: str, *, now: float) -> bool:
        self.calls.append(("validate", (key, token, now)))
        return self.valid_result

    async def record_delete_failure(self, **kwargs: object) -> bool:
        self.calls.append(("failure", kwargs))
        return self.failure_result

    async def finalize_delete(self, **kwargs: object) -> bool:
        self.calls.append(("finalize", kwargs))
        return self.finalize_result


async def _exists_yes(key: str) -> bool:
    """Default blob_exists: returns True (object present)."""
    return True


async def _exists_no(key: str) -> bool:
    """Default blob_exists: returns False (object absent)."""
    return False


async def _delete_noop(key: str) -> None:
    """Default delete_blob: succeeds silently and removes the blob from any
    test fixture tracking set (caller's responsibility)."""
    return None


async def _delete_raises(key: str) -> None:
    """Default delete_blob on failure path: raises RuntimeError."""
    raise RuntimeError("simulated R2 outage")


# ---------------------------------------------------------------------------
# 1. Enum stability — D1DeleteOutcome (4 values)
# ---------------------------------------------------------------------------


def test_d1_delete_outcome_enum_has_four_distinct_members() -> None:
    """D1DeleteOutcome is closed; adding a value is a deliberate review event."""
    members = list(D1DeleteOutcome)
    assert len(members) == 4
    assert len({m.value for m in members}) == 4


@pytest.mark.parametrize(
    "member, expected",
    [
        (D1DeleteOutcome.LEASE_DENIED, "lease_denied"),
        (D1DeleteOutcome.ALREADY_ABSENT, "already_absent"),
        (D1DeleteOutcome.DELETED, "deleted"),
        (D1DeleteOutcome.FAILED, "failed"),
    ],
)
def test_d1_delete_outcome_string_values_are_stable(member: D1DeleteOutcome, expected: str) -> None:
    """The string forms are the wire contract — pinned for downstream consumers."""
    assert member.value == expected


def test_d1_delete_outcome_round_trips_from_value() -> None:
    """Each enum is constructible from its string value — used by snapshot
    deserialization."""
    for member in D1DeleteOutcome:
        assert D1DeleteOutcome(member.value) is member


# ---------------------------------------------------------------------------
# 2. D1DeleteResult shape + frozen mutation rejection
# ---------------------------------------------------------------------------


def test_d1_delete_result_required_field_set() -> None:
    """The four documented fields are required and preserved."""
    result = D1DeleteResult(
        key="oai2-blobs/a",
        outcome=D1DeleteOutcome.DELETED,
        token="gc-sweep:a:7",
        detail="R2 deletion confirmed and D1 lease finalized",
    )
    assert result.key == "oai2-blobs/a"
    assert result.outcome is D1DeleteOutcome.DELETED
    assert result.token == "gc-sweep:a:7"
    assert result.detail == "R2 deletion confirmed and D1 lease finalized"


def test_d1_delete_result_is_frozen() -> None:
    """D1DeleteResult is immutable after construction."""
    result = D1DeleteResult(
        key="k",
        outcome=D1DeleteOutcome.DELETED,
        token="t",
        detail="d",
    )
    with pytest.raises((AttributeError, Exception)):
        result.key = "tampered"  # type: ignore[misc]


def test_d1_delete_result_is_slotted() -> None:
    """D1DeleteResult uses slots — attribute injection outside the declared
    field set is rejected with AttributeError."""
    result = D1DeleteResult(
        key="k",
        outcome=D1DeleteOutcome.DELETED,
        token="t",
        detail="d",
    )
    with pytest.raises(AttributeError):
        result.injected = "value"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3. Argument validation — key / owner / token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", " leading", "trailing "],
    ids=["empty", "whitespace", "leading", "trailing"],
)
@pytest.mark.asyncio
async def test_key_validation_rejects_non_normalized_key(bad_key: str) -> None:
    """Empty / whitespace / unstripped keys raise — the lease store would
    silently accept any string without the runtime guard."""
    store = FakeLeaseStore()
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        await delete_candidate_with_d1_lease(
            key=bad_key,
            lease_store=store,  # type: ignore[arg-type]
            expected_revision=0,
            now=10.0,
            lease_ttl_seconds=30.0,
            blob_exists=_exists_yes,
            delete_blob=_delete_noop,
        )
    # acquire_lease must NOT have been called — validation runs first
    assert store.calls == []


@pytest.mark.parametrize(
    "bad_key",
    [None, 123, ["list"], {"d": 1}],
    ids=["none", "int", "list", "dict"],
)
@pytest.mark.asyncio
async def test_key_validation_rejects_non_string_key(bad_key: object) -> None:
    """Non-string keys raise — ``None`` is caught by the truthiness check;
    non-string non-None values raise ``AttributeError`` because the
    implementation calls ``key.strip()`` rather than an ``isinstance`` check.
    Note: ``bytes`` instances also implement ``.strip()``, so they pass
    validation — pinned here as a deliberate behaviour the guard does NOT
    reject bytes."""
    store = FakeLeaseStore()
    with pytest.raises((ValueError, AttributeError)):
        await delete_candidate_with_d1_lease(
            key=bad_key,  # type: ignore[arg-type]
            lease_store=store,  # type: ignore[arg-type]
            expected_revision=0,
            now=10.0,
            lease_ttl_seconds=30.0,
            blob_exists=_exists_yes,
            delete_blob=_delete_noop,
        )
    assert store.calls == []


@pytest.mark.parametrize(
    "bad_owner",
    ["", "   ", " leading", "trailing "],
    ids=["empty", "whitespace", "leading", "trailing"],
)
@pytest.mark.asyncio
async def test_owner_validation_rejects_non_normalized_owner(bad_owner: str) -> None:
    """Empty / whitespace / unstripped owner raises — same canonicalization
    rule as ``key``."""
    store = FakeLeaseStore()
    with pytest.raises(ValueError, match="owner must be a non-empty normalized string"):
        await delete_candidate_with_d1_lease(
            key="oai2-blobs/a",
            owner=bad_owner,
            lease_store=store,  # type: ignore[arg-type]
            expected_revision=0,
            now=10.0,
            lease_ttl_seconds=30.0,
            blob_exists=_exists_yes,
            delete_blob=_delete_noop,
        )
    assert store.calls == []


@pytest.mark.parametrize(
    "bad_owner",
    [None, 123, ["list"], {"d": 1}],
    ids=["none", "int", "list", "dict"],
)
@pytest.mark.asyncio
async def test_owner_validation_rejects_non_string_owner(bad_owner: object) -> None:
    """Non-string owner raises — ``None`` is caught by the truthiness check;
    non-string non-None values raise ``AttributeError`` because the
    implementation calls ``owner.strip()`` rather than an ``isinstance`` check.
    Note: ``bytes`` instances also implement ``.strip()`` and are NOT rejected
    — pinned here as a deliberate guard characteristic."""
    store = FakeLeaseStore()
    with pytest.raises((ValueError, AttributeError)):
        await delete_candidate_with_d1_lease(
            key="oai2-blobs/a",
            owner=bad_owner,  # type: ignore[arg-type]
            lease_store=store,  # type: ignore[arg-type]
            expected_revision=0,
            now=10.0,
            lease_ttl_seconds=30.0,
            blob_exists=_exists_yes,
            delete_blob=_delete_noop,
        )
    assert store.calls == []


@pytest.mark.parametrize(
    "bad_token",
    ["", "   ", " leading", "trailing "],
    ids=["empty", "whitespace", "leading", "trailing"],
)
@pytest.mark.asyncio
async def test_explicit_token_validation_rejects_non_normalized(bad_token: str) -> None:
    """An explicitly-passed empty / whitespace / unstripped token raises —
    the same canonicalization rule as ``key`` and ``owner``."""
    store = FakeLeaseStore()
    with pytest.raises(ValueError, match="token must be a non-empty normalized string"):
        await delete_candidate_with_d1_lease(
            key="oai2-blobs/a",
            token=bad_token,
            lease_store=store,  # type: ignore[arg-type]
            expected_revision=0,
            now=10.0,
            lease_ttl_seconds=30.0,
            blob_exists=_exists_yes,
            delete_blob=_delete_noop,
        )
    assert store.calls == []


@pytest.mark.parametrize(
    "bad_token",
    [123, True, 3.14, ["list"], {"d": 1}],
    ids=["int", "bool", "float", "list", "dict"],
)
@pytest.mark.asyncio
async def test_explicit_token_validation_rejects_non_string(bad_token: object) -> None:
    """Non-string explicit tokens raise — ``None`` falls through to the
    default-token path (this is the load-bearing ``if token is None`` arm).
    Non-string non-None values raise ``AttributeError`` because the
    implementation calls ``token.strip()`` rather than an ``isinstance`` check.
    Note: ``bytes`` instances also implement ``.strip()`` and are NOT rejected
    — pinned here as a deliberate guard characteristic."""
    store = FakeLeaseStore()
    with pytest.raises((ValueError, AttributeError)):
        await delete_candidate_with_d1_lease(
            key="oai2-blobs/a",
            token=bad_token,  # type: ignore[arg-type]
            lease_store=store,  # type: ignore[arg-type]
            expected_revision=0,
            now=10.0,
            lease_ttl_seconds=30.0,
            blob_exists=_exists_yes,
            delete_blob=_delete_noop,
        )
    assert store.calls == []


# ---------------------------------------------------------------------------
# 4. Default token auto-generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_none_auto_generates_with_key_and_revision() -> None:
    """When ``token=None``, the runtime synthesizes ``f"gc-sweep:{key}:{expected_revision}"``
    so the lease-store acquires a deterministic, recognizable lease identity."""
    store = FakeLeaseStore()
    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/x",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=99,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=_exists_no,  # already absent → fast finalize
        delete_blob=_delete_noop,
    )
    # Token surfaces in BOTH the D1DeleteResult and the acquire_lease call
    assert result.token == "gc-sweep:oai2-blobs/x:99"
    assert result.outcome is D1DeleteOutcome.ALREADY_ABSENT
    acquire_kwargs = store.calls[0][1]
    assert isinstance(acquire_kwargs, dict)
    assert acquire_kwargs["token"] == "gc-sweep:oai2-blobs/x:99"
    assert acquire_kwargs["object_key"] == "oai2-blobs/x"
    assert acquire_kwargs["authority_revision"] == 99


@pytest.mark.asyncio
async def test_default_owner_is_gc_sweep_passed_to_acquire_lease() -> None:
    """The default owner (``"gc-sweep"``) is forwarded to ``acquire_lease``
    so the lease-store attribution is consistent."""
    store = FakeLeaseStore()
    await delete_candidate_with_d1_lease(
        key="oai2-blobs/x",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=0,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=_exists_no,
        delete_blob=_delete_noop,
    )
    acquire_kwargs = store.calls[0][1]
    assert isinstance(acquire_kwargs, dict)
    assert acquire_kwargs["owner"] == "gc-sweep"


@pytest.mark.asyncio
async def test_explicit_owner_overrides_default() -> None:
    """A caller-provided ``owner`` overrides the default ``gc-sweep`` — the
    custom attribution propagates to ``acquire_lease``."""
    store = FakeLeaseStore()
    await delete_candidate_with_d1_lease(
        key="oai2-blobs/x",
        owner="worker-alpha",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=0,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=_exists_no,
        delete_blob=_delete_noop,
    )
    acquire_kwargs = store.calls[0][1]
    assert isinstance(acquire_kwargs, dict)
    assert acquire_kwargs["owner"] == "worker-alpha"


@pytest.mark.asyncio
async def test_explicit_token_overrides_default_token() -> None:
    """A caller-provided ``token`` overrides the auto-generated one — the
    custom token propagates to both the result and ``acquire_lease``."""
    store = FakeLeaseStore()
    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/x",
        token="custom-token-abc",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=_exists_no,
        delete_blob=_delete_noop,
    )
    assert result.token == "custom-token-abc"
    acquire_kwargs = store.calls[0][1]
    assert isinstance(acquire_kwargs, dict)
    assert acquire_kwargs["token"] == "custom-token-abc"


# ---------------------------------------------------------------------------
# 5. Acquire-lease argument propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_lease_kwargs_match_documented_schema() -> None:
    """The ``acquire_lease`` call surfaces every documented kwarg: object_key,
    token, owner, now, ttl_seconds, authority_revision — pinned so a refactor
    that drops one breaks here."""
    store = FakeLeaseStore()
    await delete_candidate_with_d1_lease(
        key="oai2-blobs/q",
        token="t1",
        owner="w1",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=42,
        now=17.5,
        lease_ttl_seconds=99.0,
        blob_exists=_exists_no,
        delete_blob=_delete_noop,
    )
    acquire_kwargs = store.calls[0][1]
    assert isinstance(acquire_kwargs, dict)
    assert acquire_kwargs["object_key"] == "oai2-blobs/q"
    assert acquire_kwargs["token"] == "t1"
    assert acquire_kwargs["owner"] == "w1"
    assert acquire_kwargs["now"] == 17.5
    assert acquire_kwargs["ttl_seconds"] == 99.0
    assert acquire_kwargs["authority_revision"] == 42


# ---------------------------------------------------------------------------
# 6. Decision matrix — each D1DeleteOutcome
# ---------------------------------------------------------------------------


# --- LEASE_DENIED (acquire) -------------------------------------------------


@pytest.mark.asyncio
async def test_lease_denied_when_acquire_returns_false() -> None:
    """``acquire_lease=False`` returns LEASE_DENIED with the documented
    detail message and DOES NOT call ``blob_exists`` / ``delete_blob`` /
    ``lease_valid`` / ``finalize_delete``."""
    store = FakeLeaseStore(acquire_result=False)
    touched: list[str] = []

    async def exists(key: str) -> bool:
        touched.append(f"exists:{key}")
        return True

    async def delete(key: str) -> None:
        touched.append(f"delete:{key}")

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )

    assert result.outcome is D1DeleteOutcome.LEASE_DENIED
    assert result.key == "oai2-blobs/a"
    assert result.detail == "D1 deletion lease acquisition denied"
    # Only acquire_lease called; touch surfaces never reach blob_exists/delete_blob
    assert [name for name, _ in store.calls] == ["acquire"]
    assert touched == []


# --- FAILED (exists check) --------------------------------------------------


@pytest.mark.asyncio
async def test_failed_when_first_blob_exists_returns_non_bool() -> None:
    """First ``blob_exists`` returning a non-bool raises TypeError, which the
    boundary catches, records as delete_failure, and returns FAILED with the
    ``"R2 existence check failed"`` detail message."""
    store = FakeLeaseStore()

    async def exists_int(key: str) -> int:
        return 1

    async def delete(key: str) -> None:
        raise AssertionError("delete must not be called")

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists_int,  # type: ignore[arg-type]
        delete_blob=delete,
    )

    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "R2 existence check failed"
    assert [name for name, _ in store.calls] == ["acquire", "failure"]


@pytest.mark.asyncio
async def test_failed_when_first_blob_exists_returns_none() -> None:
    """``blob_exists=None`` is a non-bool return and routes through the same
    TypeError → FAILED path."""
    store = FakeLeaseStore()

    async def exists_none(key: str) -> None:
        return None

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists_none,  # type: ignore[arg-type]
        delete_blob=_delete_noop,
    )
    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "R2 existence check failed"
    assert [name for name, _ in store.calls] == ["acquire", "failure"]


@pytest.mark.asyncio
async def test_failed_when_first_blob_exists_returns_list() -> None:
    """``blob_exists=[]`` is a non-bool return and routes through the same
    TypeError → FAILED path."""
    store = FakeLeaseStore()

    async def exists_list(key: str) -> list[str]:
        return []

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists_list,  # type: ignore[arg-type]
        delete_blob=_delete_noop,
    )
    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "R2 existence check failed"


@pytest.mark.asyncio
async def test_failed_when_first_blob_exists_raises() -> None:
    """A raised exception from ``blob_exists`` routes through the same
    record_delete_failure → FAILED path."""
    store = FakeLeaseStore()

    async def exists_raises(key: str) -> bool:
        raise RuntimeError("simulated R2 outage")

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists_raises,
        delete_blob=_delete_noop,
    )
    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "R2 existence check failed"
    assert [name for name, _ in store.calls] == ["acquire", "failure"]


@pytest.mark.asyncio
async def test_first_blob_exists_raises_records_failure_with_correct_kwargs() -> None:
    """The ``record_delete_failure`` call forwards object_key, token, now,
    authority_revision."""
    store = FakeLeaseStore()

    async def exists_raises(key: str) -> bool:
        raise RuntimeError("simulated R2 outage")

    await delete_candidate_with_d1_lease(
        key="oai2-blobs/q",
        token="custom-tok",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=11,
        now=13.5,
        lease_ttl_seconds=30.0,
        blob_exists=exists_raises,
        delete_blob=_delete_noop,
    )
    failure_kwargs = store.calls[-1][1]
    assert isinstance(failure_kwargs, dict)
    assert failure_kwargs["object_key"] == "oai2-blobs/q"
    assert failure_kwargs["token"] == "custom-tok"
    assert failure_kwargs["now"] == 13.5
    assert failure_kwargs["authority_revision"] == 11


# --- LEASE_DENIED (lease valid) ---------------------------------------------


@pytest.mark.asyncio
async def test_lease_denied_when_lease_revalidation_returns_false() -> None:
    """If ``lease_valid=False`` after the exists-check succeeds, returns
    LEASE_DENIED with ``"D1 deletion lease invalid before R2 operation"`` and
    does NOT call ``delete_blob`` / ``finalize_delete``."""
    store = FakeLeaseStore(valid_result=False)
    deleted: list[str] = []

    async def exists(key: str) -> bool:
        return True

    async def delete(key: str) -> None:
        deleted.append(key)

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )
    assert result.outcome is D1DeleteOutcome.LEASE_DENIED
    assert result.detail == "D1 deletion lease invalid before R2 operation"
    assert deleted == []
    assert [name for name, _ in store.calls] == ["acquire", "validate"]


# --- FAILED (already-absent finalize returns False) -------------------------


@pytest.mark.asyncio
async def test_failed_when_already_absent_finalize_returns_false() -> None:
    """If the object is already absent (``blob_exists=False``) and
    ``finalize_delete(already_absent=True)`` returns False, return FAILED
    with ``"D1 already-absent finalization was not applied"`` and emit the
    acquire/validate/finalize sequence."""
    store = FakeLeaseStore(finalize_result=False)
    blobs: set[str] = set()

    async def exists(key: str) -> bool:
        return key in blobs

    async def delete(key: str) -> None:
        blobs.remove(key)

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )
    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "D1 already-absent finalization was not applied"
    assert blobs == set()
    assert [name for name, _ in store.calls] == ["acquire", "validate", "finalize"]


# --- ALREADY_ABSENT (happy path for absent object) --------------------------


@pytest.mark.asyncio
async def test_already_absent_returns_already_absent_outcome() -> None:
    """Already-absent object + finalize succeeds returns ALREADY_ABSENT
    with the documented detail message. ``delete_blob`` is NOT called."""
    store = FakeLeaseStore()

    async def exists(key: str) -> bool:
        return False

    async def delete(key: str) -> None:
        raise AssertionError("delete must not be called for already-absent object")

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )
    assert result.outcome is D1DeleteOutcome.ALREADY_ABSENT
    assert result.detail == "R2 object was already absent and D1 lease was finalized"
    assert [name for name, _ in store.calls] == ["acquire", "validate", "finalize"]


@pytest.mark.asyncio
async def test_already_absent_finalize_passes_already_absent_true() -> None:
    """The ``finalize_delete`` call for the ALREADY_ABSENT outcome forwards
    ``already_absent=True``."""
    store = FakeLeaseStore()
    await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=_exists_no,
        delete_blob=_delete_noop,
    )
    finalize_kwargs = store.calls[-1][1]
    assert isinstance(finalize_kwargs, dict)
    assert finalize_kwargs["already_absent"] is True
    assert finalize_kwargs["object_key"] == "oai2-blobs/a"


# --- FAILED (delete raises) -------------------------------------------------


@pytest.mark.asyncio
async def test_failed_when_delete_blob_raises() -> None:
    """``delete_blob`` raising routes to ``record_delete_failure`` and
    FAILED with ``"R2 deletion failed"``. ``finalize_delete`` is NOT called."""
    store = FakeLeaseStore()
    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=_exists_yes,
        delete_blob=_delete_raises,
    )
    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "R2 deletion failed"
    assert [name for name, _ in store.calls] == ["acquire", "validate", "failure"]


@pytest.mark.asyncio
async def test_failed_when_second_blob_exists_returns_non_bool() -> None:
    """The second ``blob_exists`` returning a non-bool also routes to
    ``record_delete_failure`` and FAILED with ``"R2 deletion failed"``."""
    store = FakeLeaseStore()
    call_count = {"n": 0}

    async def exists_mixed(key: str) -> object:
        call_count["n"] += 1
        return True if call_count["n"] == 1 else "yes"

    async def delete(key: str) -> None:
        return None

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists_mixed,  # type: ignore[arg-type]
        delete_blob=delete,
    )
    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "R2 deletion failed"
    assert [name for name, _ in store.calls] == ["acquire", "validate", "failure"]


@pytest.mark.asyncio
async def test_failed_when_second_blob_exists_raises() -> None:
    """The second ``blob_exists`` raising routes to the same FAILED path."""
    store = FakeLeaseStore()
    call_count = {"n": 0}

    async def exists_raise2(key: str) -> bool:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return True
        raise RuntimeError("simulated R2 outage")

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists_raise2,
        delete_blob=_delete_noop,
    )
    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "R2 deletion failed"
    assert [name for name, _ in store.calls] == ["acquire", "validate", "failure"]


# --- FAILED (still exists after delete) -------------------------------------


@pytest.mark.asyncio
async def test_failed_when_blob_still_exists_after_delete() -> None:
    """If ``delete_blob`` succeeds but the second ``blob_exists`` returns
    True, return FAILED with ``"R2 object remained present after deletion"``."""
    store = FakeLeaseStore()

    async def delete(key: str) -> None:
        # No-op — the second blob_exists returns True
        return None

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=_exists_yes,
        delete_blob=delete,
    )
    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "R2 object remained present after deletion"
    assert [name for name, _ in store.calls] == ["acquire", "validate", "failure"]


# --- FAILED (post-delete finalize returns False) ----------------------------


@pytest.mark.asyncio
async def test_failed_when_post_delete_finalize_returns_false() -> None:
    """After a successful delete + second-exists=False, if
    ``finalize_delete(already_absent=False)`` returns False, return FAILED
    with ``"D1 delete finalization was not applied"``."""
    store = FakeLeaseStore(finalize_result=False)
    blobs = {"oai2-blobs/a"}

    async def exists(key: str) -> bool:
        return key in blobs

    async def delete(key: str) -> None:
        # Object is removed between the first and second existence checks
        blobs.remove(key)

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )
    assert result.outcome is D1DeleteOutcome.FAILED
    assert result.detail == "D1 delete finalization was not applied"
    assert blobs == set()
    # The function returns the boxed FAILED result immediately after finalize
    # returns False; no record_delete_failure is invoked on this branch.
    assert [name for name, _ in store.calls] == [
        "acquire",
        "validate",
        "finalize",
    ]


# --- DELETED (happy path) ---------------------------------------------------


@pytest.mark.asyncio
async def test_deleted_happy_path() -> None:
    """Acquire + delete (success) + second-exists=False + finalize succeeds
    returns DELETED with the documented detail message and the documented
    call sequence."""
    store = FakeLeaseStore()
    blobs = {"oai2-blobs/a"}
    events: list[str] = []

    async def exists(key: str) -> bool:
        events.append("exists")
        return key in blobs

    async def delete(key: str) -> None:
        events.append("delete")
        assert [name for name, _ in store.calls] == ["acquire", "validate"]
        blobs.remove(key)

    result = await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )

    assert result.outcome is D1DeleteOutcome.DELETED
    assert result.detail == "R2 deletion confirmed and D1 lease finalized"
    assert events == ["exists", "delete", "exists"]
    assert [name for name, _ in store.calls] == ["acquire", "validate", "finalize"]
    assert blobs == set()


@pytest.mark.asyncio
async def test_deleted_finalize_passes_already_absent_false() -> None:
    """The post-delete ``finalize_delete`` call forwards
    ``already_absent=False``."""
    store = FakeLeaseStore()
    blobs = {"oai2-blobs/a"}

    async def exists(key: str) -> bool:
        return key in blobs

    async def delete(key: str) -> None:
        blobs.remove(key)

    await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )
    finalize_kwargs = store.calls[-1][1]
    assert isinstance(finalize_kwargs, dict)
    assert finalize_kwargs["already_absent"] is False
    assert finalize_kwargs["object_key"] == "oai2-blobs/a"
    assert finalize_kwargs["now"] == 10.0
    assert finalize_kwargs["authority_revision"] == 7


@pytest.mark.asyncio
async def test_deleted_lease_valid_called_after_exists_before_delete() -> None:
    """The validate call sits BETWEEN ``acquire`` and the second
    ``blob_exists`` — the revalidation gate runs before the destructive R2
    call. The delete event timestamps allow the validate call to have happened
    BEFORE delete."""
    store = FakeLeaseStore()
    blobs = {"oai2-blobs/a"}

    async def exists(key: str) -> bool:
        return key in blobs

    async def delete(key: str) -> None:
        # At delete-time, only acquire + validate have been called
        assert [name for name, _ in store.calls] == ["acquire", "validate"]
        blobs.remove(key)

    await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )


@pytest.mark.asyncio
async def test_deleted_validate_receives_key_token_and_now() -> None:
    """The ``lease_valid`` call forwards (key, token, now) — pinned so a
    refactor that drops ``now`` breaks here."""
    store = FakeLeaseStore()
    await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        token="tok-7",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=7,
        now=22.5,
        lease_ttl_seconds=30.0,
        blob_exists=_exists_no,
        delete_blob=_delete_noop,
    )
    validate_args = store.calls[1][1]
    assert validate_args == ("oai2-blobs/a", "tok-7", 22.5)


# ---------------------------------------------------------------------------
# 7. record_delete_failure kwargs propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_delete_failure_receives_now_and_authority_revision() -> None:
    """Every ``record_delete_failure`` call forwards object_key, token, now,
    authority_revision — confirmed across both the exists-failure and
    delete-failure paths."""
    store = FakeLeaseStore()

    async def exists_raises(key: str) -> bool:
        raise RuntimeError("simulated R2 outage")

    await delete_candidate_with_d1_lease(
        key="oai2-blobs/k",
        token="tok-x",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=42,
        now=33.3,
        lease_ttl_seconds=30.0,
        blob_exists=exists_raises,
        delete_blob=_delete_noop,
    )
    failure_kwargs = store.calls[-1][1]
    assert isinstance(failure_kwargs, dict)
    assert set(failure_kwargs.keys()) >= {
        "object_key",
        "token",
        "now",
        "authority_revision",
    }
    assert failure_kwargs["object_key"] == "oai2-blobs/k"
    assert failure_kwargs["token"] == "tok-x"
    assert failure_kwargs["now"] == 33.3
    assert failure_kwargs["authority_revision"] == 42


# ---------------------------------------------------------------------------
# 8. already_absent bool routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_already_absent_true_for_already_absent_object() -> None:
    """If the object is absent at the first check, finalize is invoked with
    already_absent=True."""
    store = FakeLeaseStore()
    await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=0,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=_exists_no,
        delete_blob=_delete_noop,
    )
    finalize_kwargs = store.calls[-1][1]
    assert isinstance(finalize_kwargs, dict)
    assert finalize_kwargs["already_absent"] is True


@pytest.mark.asyncio
async def test_finalize_already_absent_false_for_successful_delete() -> None:
    """If the object is present, delete succeeds, and second check returns
    False, finalize is invoked with already_absent=False."""
    store = FakeLeaseStore()
    blobs = {"oai2-blobs/a"}

    async def exists(key: str) -> bool:
        return key in blobs

    async def delete(key: str) -> None:
        blobs.remove(key)

    await delete_candidate_with_d1_lease(
        key="oai2-blobs/a",
        lease_store=store,  # type: ignore[arg-type]
        expected_revision=0,
        now=10.0,
        lease_ttl_seconds=30.0,
        blob_exists=exists,
        delete_blob=delete,
    )
    finalize_kwargs = store.calls[-1][1]
    assert isinstance(finalize_kwargs, dict)
    assert finalize_kwargs["already_absent"] is False


# ---------------------------------------------------------------------------
# 9. __all__ exports + package-level identity
# ---------------------------------------------------------------------------


def test_module_all_lists_three_public_exports() -> None:
    """``__all__`` is the canonical public surface — adding a name is a
    deliberate API change and must be flagged here."""
    assert sorted(gc_delete_d1_runtime_mod.__all__) == sorted(
        ["D1DeleteOutcome", "D1DeleteResult", "delete_candidate_with_d1_lease"]
    )


def test_d1_delete_runtime_symbols_are_exported_from_knowledge_package() -> None:
    """The three public symbols are re-exported at the ``oai2.knowledge``
    package level — pinned against accidental re-export removal."""
    from oai2.knowledge import D1DeleteOutcome as ExportedOutcome
    from oai2.knowledge import D1DeleteResult as ExportedResult
    from oai2.knowledge import (
        delete_candidate_with_d1_lease as ExportedDeleteCandidate,
    )

    assert ExportedOutcome is D1DeleteOutcome
    assert ExportedResult is D1DeleteResult
    assert ExportedDeleteCandidate is delete_candidate_with_d1_lease
