"""D1-authoritative deletion-lease state contract for content-addressed bodies.

The implementation is deterministic and network-free. It models the state
transitions that a live D1 transaction layer must provide before an external
R2 deletion can be safe. It does not itself call D1 or R2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum


class GcDeleteLeaseDecision(StrEnum):
    ACQUIRED = "acquired"
    REFERENCED = "referenced"
    REVISION_CONFLICT = "revision_conflict"
    LEASE_HELD = "lease_held"
    BODY_DELETED = "body_deleted"


class GcReferenceDecision(StrEnum):
    ADDED = "added"
    ALREADY_PRESENT = "already_present"
    BLOCKED_BY_DELETE_LEASE = "blocked_by_delete_lease"
    BODY_DELETED = "body_deleted"


class GcDeleteFinalizeDecision(StrEnum):
    DELETED = "deleted"
    ALREADY_ABSENT = "already_absent"
    ALREADY_FINALIZED = "already_finalized"
    RETRYABLE_FAILURE_RECORDED = "retryable_failure_recorded"
    INVALID_LEASE = "invalid_lease"


class GcDeleteLeaseState(StrEnum):
    ACTIVE = "active"
    DELETE_FAILED = "delete_failed"
    DELETED = "deleted"
    RELEASED = "released"
    REPLACED = "replaced"


@dataclass(slots=True, frozen=True)
class GcDeleteLease:
    key: str
    token: str
    owner: str
    acquired_at: float
    expires_at: float
    state: GcDeleteLeaseState = GcDeleteLeaseState.ACTIVE
    failure_count: int = 0

    def __post_init__(self) -> None:
        _normalized(self.key, "key")
        _normalized(self.token, "token")
        _normalized(self.owner, "owner")
        _finite_non_negative(self.acquired_at, "acquired_at")
        _finite_non_negative(self.expires_at, "expires_at")
        if self.expires_at <= self.acquired_at:
            raise ValueError("expires_at must be greater than acquired_at")
        if (
            isinstance(self.failure_count, bool)
            or not isinstance(self.failure_count, int)
            or self.failure_count < 0
        ):
            raise ValueError("failure_count must be a non-negative integer")


@dataclass(slots=True, frozen=True)
class GcDeleteLeaseResult:
    decision: GcDeleteLeaseDecision
    revision: int
    lease: GcDeleteLease | None = None
    knowledge_ids: tuple[str, ...] = ()
    replaced_token: str | None = None


@dataclass(slots=True)
class _GcObjectState:
    references: set[str] = field(default_factory=set)
    lease: GcDeleteLease | None = None
    deleted: bool = False
    finalized_token: str | None = None
    finalized_decision: GcDeleteFinalizeDecision | None = None


class GcDeleteLeaseAuthority:
    """Deterministic model of the D1 transaction/lease authority.

    An active or failed lease blocks reference activation. Expiry permits a
    new GC owner to take over, but does not silently reopen the writer path.
    The old token is fenced by replacement.
    """

    def __init__(self) -> None:
        self._revision = 0
        self._objects: dict[str, _GcObjectState] = {}

    @property
    def revision(self) -> int:
        return self._revision

    def references_for(self, key: str) -> tuple[str, ...]:
        key = _normalized(key, "key")
        state = self._objects.get(key)
        if state is None:
            return ()
        return tuple(sorted(state.references))

    def try_add_reference(self, key: str, knowledge_id: str) -> GcReferenceDecision:
        key = _normalized(key, "key")
        knowledge_id = _normalized(knowledge_id, "knowledge_id")
        state = self._state(key)
        if state.deleted:
            return GcReferenceDecision.BODY_DELETED
        if self._lease_blocks_writer(state.lease):
            return GcReferenceDecision.BLOCKED_BY_DELETE_LEASE
        if knowledge_id in state.references:
            return GcReferenceDecision.ALREADY_PRESENT
        state.references.add(knowledge_id)
        self._revision += 1
        return GcReferenceDecision.ADDED

    def remove_reference(self, key: str, knowledge_id: str) -> bool:
        key = _normalized(key, "key")
        knowledge_id = _normalized(knowledge_id, "knowledge_id")
        state = self._state(key)
        if self._lease_blocks_writer(state.lease):
            raise RuntimeError("references cannot change while a delete lease is active")
        if knowledge_id not in state.references:
            return False
        state.references.remove(knowledge_id)
        self._revision += 1
        return True

    def acquire_delete_lease(
        self,
        *,
        key: str,
        token: str,
        owner: str,
        now: float,
        ttl_seconds: float,
        expected_revision: int,
    ) -> GcDeleteLeaseResult:
        key = _normalized(key, "key")
        token = _normalized(token, "token")
        owner = _normalized(owner, "owner")
        now = _finite_non_negative(now, "now")
        ttl_seconds = _finite_positive(ttl_seconds, "ttl_seconds")
        expected_revision = _non_negative_int(expected_revision, "expected_revision")
        state = self._state(key)

        if expected_revision != self._revision:
            return GcDeleteLeaseResult(
                decision=GcDeleteLeaseDecision.REVISION_CONFLICT,
                revision=self._revision,
            )
        if state.deleted:
            return GcDeleteLeaseResult(
                decision=GcDeleteLeaseDecision.BODY_DELETED,
                revision=self._revision,
            )
        if state.references:
            return GcDeleteLeaseResult(
                decision=GcDeleteLeaseDecision.REFERENCED,
                revision=self._revision,
                knowledge_ids=tuple(sorted(state.references)),
            )

        replaced_token: str | None = None
        current = state.lease
        if current is not None and current.state in {
            GcDeleteLeaseState.ACTIVE,
            GcDeleteLeaseState.DELETE_FAILED,
        }:
            if now < current.expires_at:
                return GcDeleteLeaseResult(
                    decision=GcDeleteLeaseDecision.LEASE_HELD,
                    revision=self._revision,
                    lease=current,
                )
            replaced_token = current.token
            state.lease = GcDeleteLease(
                key=current.key,
                token=current.token,
                owner=current.owner,
                acquired_at=current.acquired_at,
                expires_at=current.expires_at,
                state=GcDeleteLeaseState.REPLACED,
                failure_count=current.failure_count,
            )

        lease = GcDeleteLease(
            key=key,
            token=token,
            owner=owner,
            acquired_at=now,
            expires_at=now + ttl_seconds,
        )
        state.lease = lease
        state.finalized_token = None
        state.finalized_decision = None
        self._revision += 1
        return GcDeleteLeaseResult(
            decision=GcDeleteLeaseDecision.ACQUIRED,
            revision=self._revision,
            lease=lease,
            replaced_token=replaced_token,
        )

    def validate_delete_lease(self, key: str, token: str, *, now: float) -> bool:
        key = _normalized(key, "key")
        token = _normalized(token, "token")
        now = _finite_non_negative(now, "now")
        state = self._objects.get(key)
        if state is None or state.deleted or state.references:
            return False
        lease = state.lease
        return bool(
            lease is not None
            and lease.token == token
            and lease.state in {
                GcDeleteLeaseState.ACTIVE,
                GcDeleteLeaseState.DELETE_FAILED,
            }
            and now < lease.expires_at
        )

    def record_delete_failure(
        self,
        *,
        key: str,
        token: str,
        now: float,
    ) -> GcDeleteFinalizeDecision:
        key = _normalized(key, "key")
        token = _normalized(token, "token")
        now = _finite_non_negative(now, "now")
        state = self._state(key)
        if not self.validate_delete_lease(key, token, now=now):
            return GcDeleteFinalizeDecision.INVALID_LEASE
        assert state.lease is not None
        state.lease = GcDeleteLease(
            key=state.lease.key,
            token=state.lease.token,
            owner=state.lease.owner,
            acquired_at=state.lease.acquired_at,
            expires_at=state.lease.expires_at,
            state=GcDeleteLeaseState.DELETE_FAILED,
            failure_count=state.lease.failure_count + 1,
        )
        self._revision += 1
        return GcDeleteFinalizeDecision.RETRYABLE_FAILURE_RECORDED

    def finalize_delete(
        self,
        *,
        key: str,
        token: str,
        now: float,
        already_absent: bool,
    ) -> GcDeleteFinalizeDecision:
        key = _normalized(key, "key")
        token = _normalized(token, "token")
        now = _finite_non_negative(now, "now")
        if not isinstance(already_absent, bool):
            raise ValueError("already_absent must be a boolean")
        state = self._state(key)

        if state.deleted and state.finalized_token == token:
            return GcDeleteFinalizeDecision.ALREADY_FINALIZED
        if not self.validate_delete_lease(key, token, now=now):
            return GcDeleteFinalizeDecision.INVALID_LEASE

        decision = (
            GcDeleteFinalizeDecision.ALREADY_ABSENT
            if already_absent
            else GcDeleteFinalizeDecision.DELETED
        )
        assert state.lease is not None
        state.lease = GcDeleteLease(
            key=state.lease.key,
            token=state.lease.token,
            owner=state.lease.owner,
            acquired_at=state.lease.acquired_at,
            expires_at=state.lease.expires_at,
            state=GcDeleteLeaseState.DELETED,
            failure_count=state.lease.failure_count,
        )
        state.deleted = True
        state.finalized_token = token
        state.finalized_decision = decision
        self._revision += 1
        return decision

    def release_delete_lease(self, key: str, token: str) -> bool:
        key = _normalized(key, "key")
        token = _normalized(token, "token")
        state = self._objects.get(key)
        if state is None or state.deleted or state.lease is None:
            return False
        if state.lease.token != token or state.lease.state not in {
            GcDeleteLeaseState.ACTIVE,
            GcDeleteLeaseState.DELETE_FAILED,
        }:
            return False
        state.lease = GcDeleteLease(
            key=state.lease.key,
            token=state.lease.token,
            owner=state.lease.owner,
            acquired_at=state.lease.acquired_at,
            expires_at=state.lease.expires_at,
            state=GcDeleteLeaseState.RELEASED,
            failure_count=state.lease.failure_count,
        )
        self._revision += 1
        return True

    def restore_body(self, key: str, *, verified_present: bool) -> bool:
        key = _normalized(key, "key")
        if not isinstance(verified_present, bool):
            raise ValueError("verified_present must be a boolean")
        state = self._state(key)
        if not state.deleted or not verified_present:
            return False
        state.deleted = False
        state.lease = None
        state.finalized_token = None
        state.finalized_decision = None
        self._revision += 1
        return True

    def _state(self, key: str) -> _GcObjectState:
        return self._objects.setdefault(key, _GcObjectState())

    @staticmethod
    def _lease_blocks_writer(lease: GcDeleteLease | None) -> bool:
        return lease is not None and lease.state in {
            GcDeleteLeaseState.ACTIVE,
            GcDeleteLeaseState.DELETE_FAILED,
        }


def _normalized(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


def _finite_non_negative(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _finite_positive(value: object, name: str) -> float:
    result = _finite_non_negative(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "GcDeleteLeaseDecision",
    "GcReferenceDecision",
    "GcDeleteFinalizeDecision",
    "GcDeleteLeaseState",
    "GcDeleteLease",
    "GcDeleteLeaseResult",
    "GcDeleteLeaseAuthority",
]
