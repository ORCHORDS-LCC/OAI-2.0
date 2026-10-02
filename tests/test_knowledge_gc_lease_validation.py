"""Outer-guard-rail boundary contracts for ``oai2.knowledge.gc_lease``.

This file pins the public-safety boundary of the deterministic
D1-authoritative GC deletion-lease authority model. The companion
module ``oai2/knowledge/gc_lease.py`` defines the state machine that a
live D1 transaction layer must provide before an external R2 deletion
can be safe.

These tests focus on the *outer guard rails*: enum stability, the
``GcDeleteLease.__post_init__`` boundary guards, ``GcDeleteLeaseResult``
shape, ``GcDeleteLeaseAuthority`` argument validation across every
public method, every ``StrEnum`` decision value, the
revision-increment discipline, and exhaustive error-path coverage for
``references_for`` / ``try_add_reference`` / ``remove_reference`` /
``acquire_delete_lease`` / ``validate_delete_lease`` /
``record_delete_failure`` / ``finalize_delete`` /
``release_delete_lease`` / ``restore_body``. They complement (rather
than duplicate) the happy-path and resumability tests in
``test_gc_lease.py``.

A refactor that swaps ``isinstance(value, int)`` for the wider
``(int, float)`` test — or that silently coerces non-string keys, or
that drops the ``expires_at > acquired_at`` invariant, or that bumps
revision on a rejected operation — must trip one of these tests.
"""

from __future__ import annotations

import math

import pytest

from oai2.knowledge import gc_lease as gc_lease_mod
from oai2.knowledge.gc_lease import (
    GcDeleteFinalizeDecision,
    GcDeleteLease,
    GcDeleteLeaseAuthority,
    GcDeleteLeaseDecision,
    GcDeleteLeaseResult,
    GcDeleteLeaseState,
    GcReferenceDecision,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _authority() -> GcDeleteLeaseAuthority:
    return GcDeleteLeaseAuthority()


def _acquired_authority(
    *,
    key: str = "oai2-blobs/a",
    token: str = "lease-1",
    owner: str = "worker-a",
    now: float = 10.0,
    ttl_seconds: float = 30.0,
) -> tuple[GcDeleteLeaseAuthority, int]:
    """Build an authority with one acquired lease; return (authority, revision)."""
    auth = _authority()
    result = auth.acquire_delete_lease(
        key=key,
        token=token,
        owner=owner,
        now=now,
        ttl_seconds=ttl_seconds,
        expected_revision=0,
    )
    assert result.decision is GcDeleteLeaseDecision.ACQUIRED
    return auth, result.revision


def _build_lease(
    *,
    key: str = "oai2-blobs/a",
    token: str = "t",
    owner: str = "o",
    acquired_at: float = 1.0,
    expires_at: float = 2.0,
    state: GcDeleteLeaseState = GcDeleteLeaseState.ACTIVE,
    failure_count: int = 0,
) -> GcDeleteLease:
    return GcDeleteLease(
        key=key,
        token=token,
        owner=owner,
        acquired_at=acquired_at,
        expires_at=expires_at,
        state=state,
        failure_count=failure_count,
    )


# ---------------------------------------------------------------------------
# 1. Enum stability — four StrEnums, value and member-count pinning
# ---------------------------------------------------------------------------


def test_lease_decision_enum_has_five_distinct_members() -> None:
    """GcDeleteLeaseDecision is closed; adding a value is a public-API change
    and must be flagged here so the reviewer's contract stays visible."""
    members = list(GcDeleteLeaseDecision)
    assert len(members) == 5
    assert len({m.value for m in members}) == 5


@pytest.mark.parametrize(
    "member, expected",
    [
        (GcDeleteLeaseDecision.ACQUIRED, "acquired"),
        (GcDeleteLeaseDecision.REFERENCED, "referenced"),
        (GcDeleteLeaseDecision.REVISION_CONFLICT, "revision_conflict"),
        (GcDeleteLeaseDecision.LEASE_HELD, "lease_held"),
        (GcDeleteLeaseDecision.BODY_DELETED, "body_deleted"),
    ],
)
def test_lease_decision_string_values_are_stable(
    member: GcDeleteLeaseDecision, expected: str
) -> None:
    """The string forms are persisted in D1 rows; do not rename."""
    assert member.value == expected


def test_reference_decision_enum_has_four_distinct_members() -> None:
    """GcReferenceDecision is closed."""
    members = list(GcReferenceDecision)
    assert len(members) == 4
    assert len({m.value for m in members}) == 4


@pytest.mark.parametrize(
    "member, expected",
    [
        (GcReferenceDecision.ADDED, "added"),
        (GcReferenceDecision.ALREADY_PRESENT, "already_present"),
        (GcReferenceDecision.BLOCKED_BY_DELETE_LEASE, "blocked_by_delete_lease"),
        (GcReferenceDecision.BODY_DELETED, "body_deleted"),
    ],
)
def test_reference_decision_string_values_are_stable(
    member: GcReferenceDecision, expected: str
) -> None:
    """GcReferenceDecision string forms — pinned for snapshot durability."""
    assert member.value == expected


def test_finalize_decision_enum_has_five_distinct_members() -> None:
    """GcDeleteFinalizeDecision is closed."""
    members = list(GcDeleteFinalizeDecision)
    assert len(members) == 5
    assert len({m.value for m in members}) == 5


@pytest.mark.parametrize(
    "member, expected",
    [
        (GcDeleteFinalizeDecision.DELETED, "deleted"),
        (GcDeleteFinalizeDecision.ALREADY_ABSENT, "already_absent"),
        (GcDeleteFinalizeDecision.ALREADY_FINALIZED, "already_finalized"),
        (
            GcDeleteFinalizeDecision.RETRYABLE_FAILURE_RECORDED,
            "retryable_failure_recorded",
        ),
        (GcDeleteFinalizeDecision.INVALID_LEASE, "invalid_lease"),
    ],
)
def test_finalize_decision_string_values_are_stable(
    member: GcDeleteFinalizeDecision, expected: str
) -> None:
    """GcDeleteFinalizeDecision string forms."""
    assert member.value == expected


def test_lease_state_enum_has_five_distinct_members() -> None:
    """GcDeleteLeaseState is closed — adding a state requires a coordinated
    D1 schema revision."""
    members = list(GcDeleteLeaseState)
    assert len(members) == 5
    assert len({m.value for m in members}) == 5


@pytest.mark.parametrize(
    "member, expected",
    [
        (GcDeleteLeaseState.ACTIVE, "active"),
        (GcDeleteLeaseState.DELETE_FAILED, "delete_failed"),
        (GcDeleteLeaseState.DELETED, "deleted"),
        (GcDeleteLeaseState.RELEASED, "released"),
        (GcDeleteLeaseState.REPLACED, "replaced"),
    ],
)
def test_lease_state_string_values_are_stable(member: GcDeleteLeaseState, expected: str) -> None:
    """GcDeleteLeaseState string forms — pinned for D1 column value."""
    assert member.value == expected


def test_all_four_enums_round_trip_from_value() -> None:
    """Each enum is constructible from its string value — used by snapshot
    deserialization."""
    for decision_member in GcDeleteLeaseDecision:
        assert GcDeleteLeaseDecision(decision_member.value) is decision_member
    for ref_member in GcReferenceDecision:
        assert GcReferenceDecision(ref_member.value) is ref_member
    for fin_member in GcDeleteFinalizeDecision:
        assert GcDeleteFinalizeDecision(fin_member.value) is fin_member
    for state_member in GcDeleteLeaseState:
        assert GcDeleteLeaseState(state_member.value) is state_member


# ---------------------------------------------------------------------------
# 2. GcDeleteLease __post_init__ guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", " leading", "trailing "],
    ids=["empty", "whitespace", "leading", "trailing"],
)
def test_lease_rejects_non_normalized_key(bad_key: str) -> None:
    """Empty / whitespace / unstripped keys are rejected — the lease is
    keyed on a canonical R2 object identifier."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        GcDeleteLease(
            key=bad_key,
            token="t",
            owner="o",
            acquired_at=1.0,
            expires_at=2.0,
        )


def test_lease_rejects_non_string_key() -> None:
    """A non-string key is a structural error — must fail loudly."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        GcDeleteLease(
            key=123,  # type: ignore[arg-type]
            token="t",
            owner="o",
            acquired_at=1.0,
            expires_at=2.0,
        )


@pytest.mark.parametrize(
    "bad_token", ["", "   ", " leading"], ids=["empty", "whitespace", "leading"]
)
def test_lease_rejects_non_normalized_token(bad_token: str) -> None:
    """The lease token is the writer-fence identifier — empty / whitespace
    would silently collapse multiple owners onto the same lease."""
    with pytest.raises(ValueError, match="token must be a non-empty normalized string"):
        GcDeleteLease(
            key="oai2-blobs/a",
            token=bad_token,
            owner="o",
            acquired_at=1.0,
            expires_at=2.0,
        )


@pytest.mark.parametrize(
    "bad_owner", ["", "   ", "trailing "], ids=["empty", "whitespace", "trailing"]
)
def test_lease_rejects_non_normalized_owner(bad_owner: str) -> None:
    """Owner is the GC-worker identifier — empty / whitespace would let
    unowned leases accumulate."""
    with pytest.raises(ValueError, match="owner must be a non-empty normalized string"):
        GcDeleteLease(
            key="oai2-blobs/a",
            token="t",
            owner=bad_owner,
            acquired_at=1.0,
            expires_at=2.0,
        )


@pytest.mark.parametrize(
    "bad_time",
    [math.nan, math.inf, -math.inf, -1.0, True],
    ids=["nan", "posinf", "neginf", "negative", "bool"],
)
def test_lease_rejects_invalid_acquired_at(bad_time: object) -> None:
    """``acquired_at`` must be a finite non-negative number — bool / NaN /
    inf / negative are silently accepted by Python's ``>``."""
    with pytest.raises(ValueError, match="acquired_at must be a finite non-negative number"):
        GcDeleteLease(
            key="oai2-blobs/a",
            token="t",
            owner="o",
            acquired_at=bad_time,  # type: ignore[arg-type]
            expires_at=2.0,
        )


@pytest.mark.parametrize(
    "bad_time",
    [math.nan, math.inf, -math.inf, -1.0, True],
    ids=["nan", "posinf", "neginf", "negative", "bool"],
)
def test_lease_rejects_invalid_expires_at(bad_time: object) -> None:
    """``expires_at`` must be a finite non-negative number — same arm as
    acquired_at."""
    with pytest.raises(ValueError, match="expires_at must be a finite non-negative number"):
        GcDeleteLease(
            key="oai2-blobs/a",
            token="t",
            owner="o",
            acquired_at=1.0,
            expires_at=bad_time,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("expires_at", [1.0, 0.5], ids=["equal-acquired", "before-acquired"])
def test_lease_rejects_expires_at_not_greater_than_acquired_at(expires_at: float) -> None:
    """``expires_at`` must be strictly greater than ``acquired_at`` — a
    zero-duration lease would never be valid for any operation."""
    with pytest.raises(ValueError, match="expires_at must be greater than acquired_at"):
        GcDeleteLease(
            key="oai2-blobs/a",
            token="t",
            owner="o",
            acquired_at=1.0,
            expires_at=expires_at,
        )


@pytest.mark.parametrize(
    "bad_count", [True, -1, 1.5, "3"], ids=["bool", "negative", "float", "string"]
)
def test_lease_rejects_invalid_failure_count(bad_count: object) -> None:
    """``failure_count`` must be a non-negative integer — bool is rejected
    because ``isinstance(True, int) is True``."""
    with pytest.raises(ValueError, match="failure_count must be a non-negative integer"):
        GcDeleteLease(
            key="oai2-blobs/a",
            token="t",
            owner="o",
            acquired_at=1.0,
            expires_at=2.0,
            failure_count=bad_count,  # type: ignore[arg-type]
        )


def test_lease_default_state_is_active_and_failure_count_zero() -> None:
    """A freshly constructed lease defaults to ACTIVE state with
    failure_count=0 — both must hold."""
    lease = _build_lease()
    assert lease.state is GcDeleteLeaseState.ACTIVE
    assert lease.failure_count == 0


def test_lease_is_frozen() -> None:
    """GcDeleteLease is immutable after construction — ``object.__setattr__``
    must raise ``FrozenInstanceError`` (or ``AttributeError``)."""
    lease = _build_lease()
    with pytest.raises((AttributeError, Exception)):
        lease.token = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. GcDeleteLeaseResult shape
# ---------------------------------------------------------------------------


def test_lease_result_required_field_set() -> None:
    """GcDeleteLeaseResult's required fields are ``decision`` + ``revision``;
    the rest default to safe sentinels."""
    result = GcDeleteLeaseResult(decision=GcDeleteLeaseDecision.ACQUIRED, revision=1)
    assert result.knowledge_ids == ()
    assert result.lease is None
    assert result.replaced_token is None


def test_lease_result_is_frozen() -> None:
    """GcDeleteLeaseResult is immutable after construction."""
    result = GcDeleteLeaseResult(decision=GcDeleteLeaseDecision.ACQUIRED, revision=1)
    with pytest.raises((AttributeError, Exception)):
        result.revision = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. Authority construction + revision property
# ---------------------------------------------------------------------------


def test_authority_starts_with_zero_revision() -> None:
    """A freshly constructed authority has revision=0 — the version of the
    initial empty state."""
    assert _authority().revision == 0


def test_authority_revision_is_read_only_property() -> None:
    """``revision`` is exposed as a read-only property — assignment should
    raise ``AttributeError`` (no setter)."""
    auth = _authority()
    with pytest.raises(AttributeError):
        auth.revision = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. references_for — empty / unknown / sorted-tuple
# ---------------------------------------------------------------------------


def test_references_for_unknown_key_returns_empty_tuple() -> None:
    """An unknown key has no references — returns an empty tuple (not None,
    not a list) so callers can iterate without a guard."""
    assert _authority().references_for("oai2-blobs/never") == ()


def test_references_for_returns_sorted_tuple() -> None:
    """``references_for`` sorts the underlying set — the contract is a
    deterministic ordered tuple."""
    auth = _authority()
    auth.try_add_reference("oai2-blobs/a", "ko-b")
    auth.try_add_reference("oai2-blobs/a", "ko-a")
    auth.try_add_reference("oai2-blobs/a", "ko-c")
    assert auth.references_for("oai2-blobs/a") == ("ko-a", "ko-b", "ko-c")


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", " leading", 123, None],
    ids=["empty", "whitespace", "leading", "non-string", "none"],
)
def test_references_for_rejects_invalid_key(bad_key: object) -> None:
    """Invalid keys raise — same canonicalization rule as ``try_add_reference``."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        _authority().references_for(bad_key)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. try_add_reference — validation + decision matrix + revision
# ---------------------------------------------------------------------------


def test_try_add_reference_unknown_object_returns_added() -> None:
    """First reference for an unknown object → ADDED + revision bump."""
    auth = _authority()
    assert auth.try_add_reference("oai2-blobs/a", "ko-1") is GcReferenceDecision.ADDED
    assert auth.revision == 1


def test_try_add_reference_already_present_returns_already_present_without_revision_bump() -> None:
    """Re-adding an existing knowledge_id returns ALREADY_PRESENT and does
    NOT bump revision — idempotent re-registration must not perturb
    downstream consumers."""
    auth = _authority()
    auth.try_add_reference("oai2-blobs/a", "ko-1")
    revision = auth.revision
    assert auth.try_add_reference("oai2-blobs/a", "ko-1") is GcReferenceDecision.ALREADY_PRESENT
    assert auth.revision == revision


def test_try_add_reference_for_deleted_object_returns_body_deleted() -> None:
    """A finalized delete seals the object — any subsequent reference add
    returns BODY_DELETED without bumping revision."""
    auth, _ = _acquired_authority()
    auth.finalize_delete(
        key="oai2-blobs/a",
        token="lease-1",
        now=11.0,
        already_absent=False,
    )
    revision = auth.revision
    assert auth.try_add_reference("oai2-blobs/a", "ko-new") is GcReferenceDecision.BODY_DELETED
    assert auth.revision == revision


def test_try_add_reference_blocked_by_active_lease() -> None:
    """An ACTIVE lease blocks writers — try_add_reference returns
    BLOCKED_BY_DELETE_LEASE without bumping revision."""
    auth, _ = _acquired_authority()
    revision = auth.revision
    assert (
        auth.try_add_reference("oai2-blobs/a", "ko-new")
        is GcReferenceDecision.BLOCKED_BY_DELETE_LEASE
    )
    assert auth.revision == revision


def test_try_add_reference_blocked_by_delete_failed_lease() -> None:
    """A DELETE_FAILED lease still blocks writers — the failure-recovery
    state must keep the writer path closed until the lease expires or is
    released."""
    auth, _ = _acquired_authority()
    auth.record_delete_failure(key="oai2-blobs/a", token="lease-1", now=11.0)
    assert (
        auth.try_add_reference("oai2-blobs/a", "ko-new")
        is GcReferenceDecision.BLOCKED_BY_DELETE_LEASE
    )


def test_try_add_reference_unblocked_after_lease_release() -> None:
    """A RELEASED lease no longer blocks writers — the lease has been
    intentionally surrendered."""
    auth, _ = _acquired_authority()
    auth.release_delete_lease("oai2-blobs/a", "lease-1")
    assert auth.try_add_reference("oai2-blobs/a", "ko-new") is GcReferenceDecision.ADDED


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", " leading", 123, None],
    ids=["empty", "whitespace", "leading", "non-string", "none"],
)
def test_try_add_reference_rejects_invalid_key(bad_key: object) -> None:
    """Invalid keys raise."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        _authority().try_add_reference(bad_key, "ko-1")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_kid",
    ["", "   ", "trailing ", 123, None],
    ids=["empty", "whitespace", "trailing", "non-string", "none"],
)
def test_try_add_reference_rejects_invalid_knowledge_id(bad_kid: object) -> None:
    """Invalid knowledge_ids raise."""
    with pytest.raises(ValueError, match="knowledge_id must be a non-empty normalized string"):
        _authority().try_add_reference("oai2-blobs/a", bad_kid)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. remove_reference — validation + RuntimeError on lease + revision
# ---------------------------------------------------------------------------


def test_remove_reference_unknown_returns_false_without_revision_bump() -> None:
    """Removing a knowledge_id that was never added returns False and does
    NOT bump revision."""
    auth = _authority()
    revision = auth.revision
    assert auth.remove_reference("oai2-blobs/a", "ko-missing") is False
    assert auth.revision == revision


def test_remove_reference_present_returns_true_with_revision_bump() -> None:
    """Removing an existing knowledge_id returns True and bumps revision."""
    auth = _authority()
    auth.try_add_reference("oai2-blobs/a", "ko-1")
    auth.try_add_reference("oai2-blobs/a", "ko-2")
    revision = auth.revision
    assert auth.remove_reference("oai2-blobs/a", "ko-1") is True
    assert auth.revision == revision + 1
    assert auth.references_for("oai2-blobs/a") == ("ko-2",)


def test_remove_reference_with_active_lease_raises_runtime_error() -> None:
    """An ACTIVE lease freezes the reference set — remove_reference must
    raise RuntimeError (not silently no-op, not silently succeed)."""
    auth, _ = _acquired_authority()
    with pytest.raises(
        RuntimeError,
        match="references cannot change while a delete lease is active",
    ):
        auth.remove_reference("oai2-blobs/a", "ko-1")


def test_remove_reference_with_delete_failed_lease_raises_runtime_error() -> None:
    """A DELETE_FAILED lease also freezes the reference set — the writer
    path stays blocked until the lease expires or is released."""
    auth, _ = _acquired_authority()
    auth.record_delete_failure(key="oai2-blobs/a", token="lease-1", now=11.0)
    with pytest.raises(
        RuntimeError,
        match="references cannot change while a delete lease is active",
    ):
        auth.remove_reference("oai2-blobs/a", "ko-1")


def test_remove_reference_after_lease_release_succeeds() -> None:
    """Once the lease is released, the reference set is mutable again —
    add then remove succeeds."""
    auth, _ = _acquired_authority()
    auth.release_delete_lease("oai2-blobs/a", "lease-1")
    # After release, try_add_reference proceeds (writer path open)
    assert auth.try_add_reference("oai2-blobs/a", "ko-1") is GcReferenceDecision.ADDED
    revision = auth.revision
    assert auth.remove_reference("oai2-blobs/a", "ko-1") is True
    assert auth.revision == revision + 1


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", " leading", 123, None],
    ids=["empty", "whitespace", "leading", "non-string", "none"],
)
def test_remove_reference_rejects_invalid_key(bad_key: object) -> None:
    """Invalid keys raise."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        _authority().remove_reference(bad_key, "ko-1")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_kid",
    ["", "   ", "trailing ", 123, None],
    ids=["empty", "whitespace", "trailing", "non-string", "none"],
)
def test_remove_reference_rejects_invalid_knowledge_id(bad_kid: object) -> None:
    """Invalid knowledge_ids raise."""
    with pytest.raises(ValueError, match="knowledge_id must be a non-empty normalized string"):
        _authority().remove_reference("oai2-blobs/a", bad_kid)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8. acquire_delete_lease argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", " leading", 123, None],
    ids=["empty", "whitespace", "leading", "non-string", "none"],
)
def test_acquire_delete_lease_rejects_invalid_key(bad_key: object) -> None:
    """Invalid keys raise — same canonicalization rule as the rest of the API."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        _authority().acquire_delete_lease(
            key=bad_key,  # type: ignore[arg-type]
            token="t",
            owner="o",
            now=10.0,
            ttl_seconds=30.0,
            expected_revision=0,
        )


@pytest.mark.parametrize(
    "bad_token",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_acquire_delete_lease_rejects_invalid_token(bad_token: object) -> None:
    """Invalid tokens raise — the lease token is the writer-fence identifier."""
    with pytest.raises(ValueError, match="token must be a non-empty normalized string"):
        _authority().acquire_delete_lease(
            key="oai2-blobs/a",
            token=bad_token,  # type: ignore[arg-type]
            owner="o",
            now=10.0,
            ttl_seconds=30.0,
            expected_revision=0,
        )


@pytest.mark.parametrize(
    "bad_owner",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_acquire_delete_lease_rejects_invalid_owner(bad_owner: object) -> None:
    """Invalid owners raise."""
    with pytest.raises(ValueError, match="owner must be a non-empty normalized string"):
        _authority().acquire_delete_lease(
            key="oai2-blobs/a",
            token="t",
            owner=bad_owner,  # type: ignore[arg-type]
            now=10.0,
            ttl_seconds=30.0,
            expected_revision=0,
        )


@pytest.mark.parametrize(
    "bad_now",
    [math.nan, math.inf, -math.inf, -1.0, True],
    ids=["nan", "posinf", "neginf", "negative", "bool"],
)
def test_acquire_delete_lease_rejects_invalid_now(bad_now: object) -> None:
    """``now`` must be a finite non-negative number."""
    with pytest.raises(ValueError, match="now must be a finite non-negative number"):
        _authority().acquire_delete_lease(
            key="oai2-blobs/a",
            token="t",
            owner="o",
            now=bad_now,  # type: ignore[arg-type]
            ttl_seconds=30.0,
            expected_revision=0,
        )


@pytest.mark.parametrize(
    "bad_ttl",
    [math.nan, math.inf, -math.inf, -1.0, 0.0, True],
    ids=["nan", "posinf", "neginf", "negative", "zero", "bool"],
)
def test_acquire_delete_lease_rejects_invalid_ttl_seconds(bad_ttl: object) -> None:
    """``ttl_seconds`` must be a finite POSITIVE number (zero duration would
    expire instantly — not a valid lease)."""
    with pytest.raises(
        ValueError,
        match=r"ttl_seconds must be (a finite non-negative number|positive)",
    ):
        _authority().acquire_delete_lease(
            key="oai2-blobs/a",
            token="t",
            owner="o",
            now=10.0,
            ttl_seconds=bad_ttl,  # type: ignore[arg-type]
            expected_revision=0,
        )


@pytest.mark.parametrize(
    "bad_revision",
    [-1, 1.5, "3", True],
    ids=["negative", "float", "string", "bool"],
)
def test_acquire_delete_lease_rejects_invalid_expected_revision(bad_revision: object) -> None:
    """``expected_revision`` must be a non-negative integer — bool is rejected."""
    with pytest.raises(ValueError, match="expected_revision must be a non-negative integer"):
        _authority().acquire_delete_lease(
            key="oai2-blobs/a",
            token="t",
            owner="o",
            now=10.0,
            ttl_seconds=30.0,
            expected_revision=bad_revision,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 9. acquire_delete_lease decision matrix
# ---------------------------------------------------------------------------


def test_acquire_returns_revision_conflict_on_mismatch() -> None:
    """A stale ``expected_revision`` returns REVISION_CONFLICT without
    bumping revision — the caller's precondition failed and the state is
    unchanged."""
    auth = _authority()
    result = auth.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="w",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=99,
    )
    assert result.decision is GcDeleteLeaseDecision.REVISION_CONFLICT
    assert result.revision == 0
    assert auth.revision == 0


def test_acquire_returns_referenced_with_knowledge_ids_tuple() -> None:
    """If references are present, acquire returns REFERENCED with the
    sorted knowledge_ids tuple — the planner needs the full reference
    set to defer the GC sweep."""
    auth = _authority()
    auth.try_add_reference("oai2-blobs/a", "ko-b")
    auth.try_add_reference("oai2-blobs/a", "ko-a")
    result = auth.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="w",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=auth.revision,
    )
    assert result.decision is GcDeleteLeaseDecision.REFERENCED
    assert result.knowledge_ids == ("ko-a", "ko-b")


def test_acquire_returns_lease_held_when_active_lease_not_expired() -> None:
    """If a current ACTIVE/DELETE_FAILED lease is in its window, acquire
    returns LEASE_HELD with the current lease pinned."""
    auth, _ = _acquired_authority()
    revision = auth.revision
    result = auth.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-2",
        owner="w",
        now=20.0,  # still within window (acquired at 10.0, ttl=30, expires at 40.0)
        ttl_seconds=30.0,
        expected_revision=revision,
    )
    assert result.decision is GcDeleteLeaseDecision.LEASE_HELD
    assert result.lease is not None
    assert result.lease.token == "lease-1"


def test_acquire_first_time_returns_acquired_with_no_replaced_token() -> None:
    """First acquire on an unreferenced object → ACQUIRED, replaced_token=None."""
    auth = _authority()
    result = auth.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="w",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=0,
    )
    assert result.decision is GcDeleteLeaseDecision.ACQUIRED
    assert result.replaced_token is None
    assert result.lease is not None
    assert result.lease.state is GcDeleteLeaseState.ACTIVE


def test_acquire_takeover_after_expiry_returns_acquired_with_replaced_token() -> None:
    """An expired lease is replaced — new ACQUIRED with the OLD token
    surfaced as ``replaced_token`` (writer-fence token, not the new one)."""
    auth, _ = _acquired_authority(now=10.0, ttl_seconds=5.0)
    result = auth.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-new",
        owner="w",
        now=20.0,  # past lease-1 expiry at 15.0
        ttl_seconds=30.0,
        expected_revision=auth.revision,
    )
    assert result.decision is GcDeleteLeaseDecision.ACQUIRED
    assert result.replaced_token == "lease-1"
    assert result.lease is not None
    assert result.lease.token == "lease-new"


def test_acquire_takeover_after_release_returns_acquired_with_no_replaced_token() -> None:
    """A RELEASED lease is no longer ACTIVE/DELETE_FAILED — takeover does
    not surface a replaced_token because there was no fencing event."""
    auth, _ = _acquired_authority()
    auth.release_delete_lease("oai2-blobs/a", "lease-1")
    revision = auth.revision
    result = auth.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-2",
        owner="w",
        now=20.0,
        ttl_seconds=30.0,
        expected_revision=revision,
    )
    assert result.decision is GcDeleteLeaseDecision.ACQUIRED
    assert result.replaced_token is None


def test_acquire_returns_body_deleted_for_finalized_object() -> None:
    """An acquire on a finalized (state.deleted=True) object returns
    BODY_DELETED without bumping revision."""
    auth, _ = _acquired_authority()
    auth.finalize_delete(
        key="oai2-blobs/a",
        token="lease-1",
        now=11.0,
        already_absent=False,
    )
    revision = auth.revision
    result = auth.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-2",
        owner="w",
        now=12.0,
        ttl_seconds=30.0,
        expected_revision=revision,
    )
    assert result.decision is GcDeleteLeaseDecision.BODY_DELETED
    assert auth.revision == revision


# ---------------------------------------------------------------------------
# 10. validate_delete_lease
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", " leading", 123, None],
    ids=["empty", "whitespace", "leading", "non-string", "none"],
)
def test_validate_delete_lease_rejects_invalid_key(bad_key: object) -> None:
    """Invalid keys raise."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        _authority().validate_delete_lease(bad_key, "t", now=10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_token",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_validate_delete_lease_rejects_invalid_token(bad_token: object) -> None:
    """Invalid tokens raise."""
    with pytest.raises(ValueError, match="token must be a non-empty normalized string"):
        _authority().validate_delete_lease("oai2-blobs/a", bad_token, now=10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_now",
    [math.nan, math.inf, -math.inf, -1.0, True],
    ids=["nan", "posinf", "neginf", "negative", "bool"],
)
def test_validate_delete_lease_rejects_invalid_now(bad_now: object) -> None:
    """``now`` must be a finite non-negative number."""
    with pytest.raises(ValueError, match="now must be a finite non-negative number"):
        _authority().validate_delete_lease(
            "oai2-blobs/a",
            "t",
            now=bad_now,  # type: ignore[arg-type]
        )


def test_validate_delete_lease_unknown_object_returns_false() -> None:
    """An unknown object has no lease to validate — returns False."""
    assert _authority().validate_delete_lease("oai2-blobs/a", "t", now=10.0) is False


def test_validate_delete_lease_finalized_object_returns_false() -> None:
    """A finalized object is sealed — even the original token cannot validate."""
    auth, _ = _acquired_authority()
    auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=11.0, already_absent=False)
    assert auth.validate_delete_lease("oai2-blobs/a", "lease-1", now=12.0) is False


def test_validate_delete_lease_referenced_object_returns_false() -> None:
    """An object with live references returns False — the lease was never
    held in the first place (REFERENCED decision would have blocked
    acquire), so there is nothing to validate."""
    auth = _authority()
    auth.try_add_reference("oai2-blobs/a", "ko-1")
    assert auth.validate_delete_lease("oai2-blobs/a", "lease-1", now=10.0) is False


def test_validate_delete_lease_wrong_token_returns_false() -> None:
    """A wrong token cannot validate — the writer-fence is exact-match."""
    auth, _ = _acquired_authority()
    assert auth.validate_delete_lease("oai2-blobs/a", "wrong", now=11.0) is False


def test_validate_delete_lease_expired_returns_false() -> None:
    """An expired lease (now >= expires_at) cannot validate — the takeover
    path applies."""
    auth, _ = _acquired_authority(now=10.0, ttl_seconds=5.0)
    assert auth.validate_delete_lease("oai2-blobs/a", "lease-1", now=20.0) is False


def test_validate_delete_lease_in_window_returns_true() -> None:
    """A lease with the right token in its window returns True."""
    auth, _ = _acquired_authority(now=10.0, ttl_seconds=30.0)
    assert auth.validate_delete_lease("oai2-blobs/a", "lease-1", now=20.0) is True


def test_validate_delete_lease_after_release_returns_false() -> None:
    """A RELEASED lease is not ACTIVE/DELETE_FAILED — validate returns False."""
    auth, _ = _acquired_authority()
    auth.release_delete_lease("oai2-blobs/a", "lease-1")
    assert auth.validate_delete_lease("oai2-blobs/a", "lease-1", now=11.0) is False


# ---------------------------------------------------------------------------
# 11. record_delete_failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_record_delete_failure_rejects_invalid_key(bad_key: object) -> None:
    """Invalid keys raise."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        _authority().record_delete_failure(
            key=bad_key,  # type: ignore[arg-type]
            token="t",
            now=10.0,
        )


@pytest.mark.parametrize(
    "bad_token",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_record_delete_failure_rejects_invalid_token(bad_token: object) -> None:
    """Invalid tokens raise."""
    with pytest.raises(ValueError, match="token must be a non-empty normalized string"):
        _authority().record_delete_failure(
            key="oai2-blobs/a",
            token=bad_token,  # type: ignore[arg-type]
            now=10.0,
        )


@pytest.mark.parametrize(
    "bad_now",
    [math.nan, math.inf, -math.inf, -1.0, True],
    ids=["nan", "posinf", "neginf", "negative", "bool"],
)
def test_record_delete_failure_rejects_invalid_now(bad_now: object) -> None:
    """``now`` must be a finite non-negative number."""
    with pytest.raises(ValueError, match="now must be a finite non-negative number"):
        _authority().record_delete_failure(
            key="oai2-blobs/a",
            token="t",
            now=bad_now,  # type: ignore[arg-type]
        )


def test_record_delete_failure_no_lease_returns_invalid_lease() -> None:
    """Without a valid lease, record_delete_failure returns INVALID_LEASE
    and does NOT bump revision."""
    auth = _authority()
    revision = auth.revision
    assert (
        auth.record_delete_failure(key="oai2-blobs/a", token="lease-1", now=10.0)
        is GcDeleteFinalizeDecision.INVALID_LEASE
    )
    assert auth.revision == revision


def test_record_delete_failure_wrong_token_returns_invalid_lease() -> None:
    """A wrong token cannot record a failure — the writer-fence is exact-match."""
    auth, _ = _acquired_authority()
    revision = auth.revision
    assert (
        auth.record_delete_failure(key="oai2-blobs/a", token="wrong", now=11.0)
        is GcDeleteFinalizeDecision.INVALID_LEASE
    )
    assert auth.revision == revision


def test_record_delete_failure_success_increments_failure_count() -> None:
    """Each successful record_delete_failure increments ``failure_count``
    monotonically."""
    auth, _ = _acquired_authority()
    auth.record_delete_failure(key="oai2-blobs/a", token="lease-1", now=11.0)
    # DELETE_FAILED still validates as a live lease
    assert auth.validate_delete_lease("oai2-blobs/a", "lease-1", now=11.0) is True
    # A second failure record is still allowed while in DELETE_FAILED state
    auth.record_delete_failure(key="oai2-blobs/a", token="lease-1", now=12.0)
    # Both failures succeeded — final state.lease.failure_count == 2
    assert auth._objects["oai2-blobs/a"].lease.failure_count == 2  # type: ignore[union-attr]
    assert auth._objects["oai2-blobs/a"].lease.state is GcDeleteLeaseState.DELETE_FAILED  # type: ignore[union-attr]


def test_record_delete_failure_success_returns_decision_and_bumps_revision() -> None:
    """A successful record_delete_failure returns RETRYABLE_FAILURE_RECORDED
    and bumps revision."""
    auth, _ = _acquired_authority()
    revision = auth.revision
    assert (
        auth.record_delete_failure(key="oai2-blobs/a", token="lease-1", now=11.0)
        is GcDeleteFinalizeDecision.RETRYABLE_FAILURE_RECORDED
    )
    assert auth.revision == revision + 1


# ---------------------------------------------------------------------------
# 12. finalize_delete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_finalize_delete_rejects_invalid_key(bad_key: object) -> None:
    """Invalid keys raise."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        _authority().finalize_delete(
            key=bad_key,  # type: ignore[arg-type]
            token="t",
            now=10.0,
            already_absent=False,
        )


@pytest.mark.parametrize(
    "bad_token",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_finalize_delete_rejects_invalid_token(bad_token: object) -> None:
    """Invalid tokens raise."""
    with pytest.raises(ValueError, match="token must be a non-empty normalized string"):
        _authority().finalize_delete(
            key="oai2-blobs/a",
            token=bad_token,  # type: ignore[arg-type]
            now=10.0,
            already_absent=False,
        )


@pytest.mark.parametrize(
    "bad_now",
    [math.nan, math.inf, -math.inf, -1.0, True],
    ids=["nan", "posinf", "neginf", "negative", "bool"],
)
def test_finalize_delete_rejects_invalid_now(bad_now: object) -> None:
    """``now`` must be a finite non-negative number."""
    with pytest.raises(ValueError, match="now must be a finite non-negative number"):
        _authority().finalize_delete(
            key="oai2-blobs/a",
            token="t",
            now=bad_now,  # type: ignore[arg-type]
            already_absent=False,
        )


@pytest.mark.parametrize(
    "bad_absent",
    [1, 0, "yes", None, [True], {"absent": True}],
    ids=["int-true", "int-false", "string", "none", "list", "dict"],
)
def test_finalize_delete_rejects_non_boolean_already_absent(bad_absent: object) -> None:
    """``already_absent`` must be a strict bool — int / str / None are rejected."""
    with pytest.raises(ValueError, match="already_absent must be a boolean"):
        _authority().finalize_delete(
            key="oai2-blobs/a",
            token="t",
            now=10.0,
            already_absent=bad_absent,  # type: ignore[arg-type]
        )


def test_finalize_delete_no_lease_returns_invalid_lease() -> None:
    """Without a valid lease, finalize returns INVALID_LEASE without
    bumping revision."""
    auth = _authority()
    revision = auth.revision
    assert (
        auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=10.0, already_absent=False)
        is GcDeleteFinalizeDecision.INVALID_LEASE
    )
    assert auth.revision == revision


def test_finalize_delete_wrong_token_returns_invalid_lease() -> None:
    """A wrong token cannot finalize."""
    auth, _ = _acquired_authority()
    revision = auth.revision
    assert (
        auth.finalize_delete(key="oai2-blobs/a", token="wrong", now=11.0, already_absent=False)
        is GcDeleteFinalizeDecision.INVALID_LEASE
    )
    assert auth.revision == revision


def test_finalize_delete_with_already_absent_false_returns_deleted() -> None:
    """Successful finalize with already_absent=False returns DELETED and
    bumps revision."""
    auth, _ = _acquired_authority()
    revision = auth.revision
    assert (
        auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=11.0, already_absent=False)
        is GcDeleteFinalizeDecision.DELETED
    )
    assert auth.revision == revision + 1
    assert auth._objects["oai2-blobs/a"].deleted is True
    assert auth._objects["oai2-blobs/a"].lease.state is GcDeleteLeaseState.DELETED  # type: ignore[union-attr]


def test_finalize_delete_with_already_absent_true_returns_already_absent() -> None:
    """Successful finalize with already_absent=True returns ALREADY_ABSENT
    and bumps revision."""
    auth, _ = _acquired_authority()
    revision = auth.revision
    assert (
        auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=11.0, already_absent=True)
        is GcDeleteFinalizeDecision.ALREADY_ABSENT
    )
    assert auth.revision == revision + 1
    assert auth._objects["oai2-blobs/a"].deleted is True


def test_finalize_delete_repeat_with_same_token_returns_already_finalized() -> None:
    """A second finalize with the same token returns ALREADY_FINALIZED
    without bumping revision — the operation is idempotent."""
    auth, _ = _acquired_authority()
    auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=11.0, already_absent=False)
    revision = auth.revision
    assert (
        auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=12.0, already_absent=False)
        is GcDeleteFinalizeDecision.ALREADY_FINALIZED
    )
    assert auth.revision == revision


# ---------------------------------------------------------------------------
# 13. release_delete_lease
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_release_delete_lease_rejects_invalid_key(bad_key: object) -> None:
    """Invalid keys raise."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        _authority().release_delete_lease(bad_key, "t")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_token",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_release_delete_lease_rejects_invalid_token(bad_token: object) -> None:
    """Invalid tokens raise."""
    with pytest.raises(ValueError, match="token must be a non-empty normalized string"):
        _authority().release_delete_lease("oai2-blobs/a", bad_token)  # type: ignore[arg-type]


def test_release_unknown_object_returns_false() -> None:
    """An unknown object has no lease to release — returns False."""
    assert _authority().release_delete_lease("oai2-blobs/a", "t") is False


def test_release_finalized_object_returns_false() -> None:
    """A finalized object is sealed — release returns False."""
    auth, _ = _acquired_authority()
    auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=11.0, already_absent=False)
    assert auth.release_delete_lease("oai2-blobs/a", "lease-1") is False


def test_release_wrong_token_returns_false() -> None:
    """A wrong token cannot release — the writer-fence is exact-match."""
    auth, _ = _acquired_authority()
    assert auth.release_delete_lease("oai2-blobs/a", "wrong") is False


def test_release_after_failure_returns_true() -> None:
    """A DELETE_FAILED lease can still be released — recovery from a
    transient delete failure is an authorized operation."""
    auth, _ = _acquired_authority()
    auth.record_delete_failure(key="oai2-blobs/a", token="lease-1", now=11.0)
    assert auth.release_delete_lease("oai2-blobs/a", "lease-1") is True
    assert auth._objects["oai2-blobs/a"].lease.state is GcDeleteLeaseState.RELEASED  # type: ignore[union-attr]


def test_release_active_lease_returns_true_and_marks_released() -> None:
    """An ACTIVE lease can be released — the lease state becomes RELEASED."""
    auth, _ = _acquired_authority()
    revision = auth.revision
    assert auth.release_delete_lease("oai2-blobs/a", "lease-1") is True
    assert auth.revision == revision + 1
    assert auth._objects["oai2-blobs/a"].lease.state is GcDeleteLeaseState.RELEASED  # type: ignore[union-attr]


def test_release_twice_returns_false_on_second_call() -> None:
    """A second release of the same RELEASED lease returns False — the
    release is not idempotent on its own (the acquire path handles that)."""
    auth, _ = _acquired_authority()
    assert auth.release_delete_lease("oai2-blobs/a", "lease-1") is True
    assert auth.release_delete_lease("oai2-blobs/a", "lease-1") is False


# ---------------------------------------------------------------------------
# 14. restore_body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    ["", "   ", "trailing ", 123],
    ids=["empty", "whitespace", "trailing", "non-string"],
)
def test_restore_body_rejects_invalid_key(bad_key: object) -> None:
    """Invalid keys raise."""
    with pytest.raises(ValueError, match="key must be a non-empty normalized string"):
        _authority().restore_body(bad_key, verified_present=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_present",
    [1, 0, "yes", None, [True]],
    ids=["int-true", "int-false", "string", "none", "list"],
)
def test_restore_body_rejects_non_boolean_verified_present(bad_present: object) -> None:
    """``verified_present`` must be a strict bool."""
    with pytest.raises(ValueError, match="verified_present must be a boolean"):
        _authority().restore_body("oai2-blobs/a", verified_present=bad_present)  # type: ignore[arg-type]


def test_restore_body_unknown_object_returns_false() -> None:
    """An unknown object is not deleted — restore returns False."""
    assert _authority().restore_body("oai2-blobs/a", verified_present=True) is False


def test_restore_body_with_verified_present_false_returns_false() -> None:
    """``verified_present=False`` always returns False — the explicit
    verification signal must be True to attempt restore."""
    auth, _ = _acquired_authority()
    auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=11.0, already_absent=False)
    assert auth.restore_body("oai2-blobs/a", verified_present=False) is False
    # Still deleted after the failed restore attempt
    assert auth._objects["oai2-blobs/a"].deleted is True


def test_restore_body_success_resets_state_and_bumps_revision() -> None:
    """A successful restore returns True, clears ``deleted``, clears the
    lease, and bumps revision."""
    auth, _ = _acquired_authority()
    auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=11.0, already_absent=False)
    revision = auth.revision
    assert auth.restore_body("oai2-blobs/a", verified_present=True) is True
    assert auth.revision == revision + 1
    assert auth._objects["oai2-blobs/a"].deleted is False
    assert auth._objects["oai2-blobs/a"].lease is None
    assert auth._objects["oai2-blobs/a"].finalized_token is None
    assert auth._objects["oai2-blobs/a"].finalized_decision is None


def test_restore_body_resets_writer_path() -> None:
    """After restore, ``try_add_reference`` works again — the writer path
    is unsealed."""
    auth, _ = _acquired_authority()
    auth.finalize_delete(key="oai2-blobs/a", token="lease-1", now=11.0, already_absent=False)
    assert auth.try_add_reference("oai2-blobs/a", "ko-new") is GcReferenceDecision.BODY_DELETED
    auth.restore_body("oai2-blobs/a", verified_present=True)
    assert auth.try_add_reference("oai2-blobs/a", "ko-new") is GcReferenceDecision.ADDED


# ---------------------------------------------------------------------------
# 15. __all__ exports
# ---------------------------------------------------------------------------


def test_module_all_lists_seven_public_exports() -> None:
    """``__all__`` is the canonical public surface — adding a name is a
    deliberate API change and must be flagged here."""
    assert sorted(gc_lease_mod.__all__) == sorted(
        [
            "GcDeleteLeaseDecision",
            "GcReferenceDecision",
            "GcDeleteFinalizeDecision",
            "GcDeleteLeaseState",
            "GcDeleteLease",
            "GcDeleteLeaseResult",
            "GcDeleteLeaseAuthority",
        ]
    )


def test_authority_is_in_knowledge_package_exports() -> None:
    """``GcDeleteLeaseAuthority`` is exposed at the ``oai2.knowledge``
    package level — confirmed by the existing slice-32 export test, but
    re-pinned here against accidental re-export removal."""
    from oai2.knowledge import GcDeleteLeaseAuthority as Exported

    assert Exported is GcDeleteLeaseAuthority
