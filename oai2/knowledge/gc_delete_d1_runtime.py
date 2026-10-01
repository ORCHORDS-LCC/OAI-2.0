"""Async D1/R2 destructive-delete boundary for one GC candidate.

This module is the live-binding-facing bridge between the conservative sweep
planner and Cloudflare's asynchronous D1/R2 APIs. It does not decide whether a
candidate is eligible for destruction; callers must first satisfy the existing
sweep grace, authorization, recovery-readiness, and reference-recheck gates.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from .gc_lease_d1_runtime import D1GcLeaseStore


class D1DeleteOutcome(StrEnum):
    LEASE_DENIED = "lease_denied"
    ALREADY_ABSENT = "already_absent"
    DELETED = "deleted"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class D1DeleteResult:
    key: str
    outcome: D1DeleteOutcome
    token: str
    detail: str


async def delete_candidate_with_d1_lease(
    *,
    key: str,
    lease_store: D1GcLeaseStore,
    expected_revision: int,
    now: float,
    lease_ttl_seconds: float,
    blob_exists: Callable[[str], Awaitable[bool]],
    delete_blob: Callable[[str], Awaitable[None]],
    token: str | None = None,
    owner: str = "gc-sweep",
) -> D1DeleteResult:
    """Delete one candidate only while an authoritative D1 lease remains valid."""
    if not key or key != key.strip():
        raise ValueError("key must be a non-empty normalized string")
    if not owner or owner != owner.strip():
        raise ValueError("owner must be a non-empty normalized string")
    if token is None:
        token = f"gc-sweep:{key}:{expected_revision}"
    if not token or token != token.strip():
        raise ValueError("token must be a non-empty normalized string")

    acquired = await lease_store.acquire_lease(
        object_key=key,
        token=token,
        owner=owner,
        now=now,
        ttl_seconds=lease_ttl_seconds,
        authority_revision=expected_revision,
    )
    if not acquired:
        return D1DeleteResult(
            key=key,
            outcome=D1DeleteOutcome.LEASE_DENIED,
            token=token,
            detail="D1 deletion lease acquisition denied",
        )

    try:
        exists_before = await blob_exists(key)
        if not isinstance(exists_before, bool):
            raise TypeError("blob_exists returned a non-boolean value")
    except Exception:
        await lease_store.record_delete_failure(
            object_key=key,
            token=token,
            now=now,
            authority_revision=expected_revision,
        )
        return D1DeleteResult(
            key=key,
            outcome=D1DeleteOutcome.FAILED,
            token=token,
            detail="R2 existence check failed",
        )

    valid = await lease_store.lease_valid(key, token, now=now)
    if not valid:
        return D1DeleteResult(
            key=key,
            outcome=D1DeleteOutcome.LEASE_DENIED,
            token=token,
            detail="D1 deletion lease invalid before R2 operation",
        )

    if not exists_before:
        finalized = await lease_store.finalize_delete(
            object_key=key,
            token=token,
            now=now,
            already_absent=True,
            authority_revision=expected_revision,
        )
        if not finalized:
            return D1DeleteResult(
                key=key,
                outcome=D1DeleteOutcome.FAILED,
                token=token,
                detail="D1 already-absent finalization was not applied",
            )
        return D1DeleteResult(
            key=key,
            outcome=D1DeleteOutcome.ALREADY_ABSENT,
            token=token,
            detail="R2 object was already absent and D1 lease was finalized",
        )

    try:
        await delete_blob(key)
        still_exists = await blob_exists(key)
        if not isinstance(still_exists, bool):
            raise TypeError("blob_exists returned a non-boolean value")
    except Exception:
        await lease_store.record_delete_failure(
            object_key=key,
            token=token,
            now=now,
            authority_revision=expected_revision,
        )
        return D1DeleteResult(
            key=key,
            outcome=D1DeleteOutcome.FAILED,
            token=token,
            detail="R2 deletion failed",
        )

    if still_exists:
        await lease_store.record_delete_failure(
            object_key=key,
            token=token,
            now=now,
            authority_revision=expected_revision,
        )
        return D1DeleteResult(
            key=key,
            outcome=D1DeleteOutcome.FAILED,
            token=token,
            detail="R2 object remained present after deletion",
        )

    finalized = await lease_store.finalize_delete(
        object_key=key,
        token=token,
        now=now,
        already_absent=False,
        authority_revision=expected_revision,
    )
    if not finalized:
        return D1DeleteResult(
            key=key,
            outcome=D1DeleteOutcome.FAILED,
            token=token,
            detail="D1 delete finalization was not applied",
        )

    return D1DeleteResult(
        key=key,
        outcome=D1DeleteOutcome.DELETED,
        token=token,
        detail="R2 deletion confirmed and D1 lease finalized",
    )


__all__ = [
    "D1DeleteOutcome",
    "D1DeleteResult",
    "delete_candidate_with_d1_lease",
]
