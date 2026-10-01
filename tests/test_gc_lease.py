"""Tests for D1-authoritative GC deletion lease semantics."""

from __future__ import annotations

from oai2.knowledge.gc_lease import (
    GcDeleteFinalizeDecision,
    GcDeleteLeaseAuthority,
    GcDeleteLeaseDecision,
    GcReferenceDecision,
)


def test_existing_reference_blocks_delete_lease() -> None:
    authority = GcDeleteLeaseAuthority()
    assert authority.try_add_reference("oai2-blobs/a", "ko-1") is GcReferenceDecision.ADDED
    revision = authority.revision

    result = authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="worker-a",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=revision,
    )

    assert result.decision is GcDeleteLeaseDecision.REFERENCED
    assert result.knowledge_ids == ("ko-1",)


def test_revision_conflict_blocks_claim() -> None:
    authority = GcDeleteLeaseAuthority()

    result = authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="worker-a",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=99,
    )

    assert result.decision is GcDeleteLeaseDecision.REVISION_CONFLICT


def test_active_lease_blocks_concurrent_reference() -> None:
    authority = GcDeleteLeaseAuthority()
    result = authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="worker-a",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=0,
    )
    assert result.decision is GcDeleteLeaseDecision.ACQUIRED

    assert (
        authority.try_add_reference("oai2-blobs/a", "ko-race")
        is GcReferenceDecision.BLOCKED_BY_DELETE_LEASE
    )
    assert authority.references_for("oai2-blobs/a") == ()


def test_expired_lease_can_be_taken_over_and_fences_old_token() -> None:
    authority = GcDeleteLeaseAuthority()
    first = authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-old",
        owner="worker-a",
        now=10.0,
        ttl_seconds=5.0,
        expected_revision=0,
    )
    assert first.decision is GcDeleteLeaseDecision.ACQUIRED

    takeover = authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-new",
        owner="worker-b",
        now=16.0,
        ttl_seconds=10.0,
        expected_revision=authority.revision,
    )
    assert takeover.decision is GcDeleteLeaseDecision.ACQUIRED
    assert takeover.replaced_token == "lease-old"
    assert authority.validate_delete_lease("oai2-blobs/a", "lease-old", now=16.0) is False
    assert authority.validate_delete_lease("oai2-blobs/a", "lease-new", now=16.0) is True


def test_delete_failure_keeps_claim_recoverable_and_blocks_writer() -> None:
    authority = GcDeleteLeaseAuthority()
    authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="worker-a",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=0,
    )

    failed = authority.record_delete_failure(
        key="oai2-blobs/a", token="lease-1", now=11.0
    )
    assert failed is GcDeleteFinalizeDecision.RETRYABLE_FAILURE_RECORDED
    assert (
        authority.try_add_reference("oai2-blobs/a", "ko-new")
        is GcReferenceDecision.BLOCKED_BY_DELETE_LEASE
    )
    assert authority.validate_delete_lease("oai2-blobs/a", "lease-1", now=12.0)


def test_delete_finalize_is_idempotent_and_deleted_body_blocks_reference() -> None:
    authority = GcDeleteLeaseAuthority()
    authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="worker-a",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=0,
    )

    first = authority.finalize_delete(
        key="oai2-blobs/a",
        token="lease-1",
        now=11.0,
        already_absent=False,
    )
    second = authority.finalize_delete(
        key="oai2-blobs/a",
        token="lease-1",
        now=12.0,
        already_absent=False,
    )

    assert first is GcDeleteFinalizeDecision.DELETED
    assert second is GcDeleteFinalizeDecision.ALREADY_FINALIZED
    assert authority.try_add_reference("oai2-blobs/a", "ko-new") is GcReferenceDecision.BODY_DELETED


def test_release_allows_reference_without_marking_body_deleted() -> None:
    authority = GcDeleteLeaseAuthority()
    authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="worker-a",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=0,
    )

    assert authority.release_delete_lease("oai2-blobs/a", "lease-1") is True
    assert authority.try_add_reference("oai2-blobs/a", "ko-new") is GcReferenceDecision.ADDED


def test_restore_deleted_body_requires_explicit_operation() -> None:
    authority = GcDeleteLeaseAuthority()
    authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="worker-a",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=0,
    )
    authority.finalize_delete(
        key="oai2-blobs/a",
        token="lease-1",
        now=11.0,
        already_absent=True,
    )

    assert authority.try_add_reference("oai2-blobs/a", "ko-new") is GcReferenceDecision.BODY_DELETED
    assert authority.restore_body("oai2-blobs/a", verified_present=False) is False
    assert authority.try_add_reference("oai2-blobs/a", "ko-new") is GcReferenceDecision.BODY_DELETED
    assert authority.restore_body("oai2-blobs/a", verified_present=True) is True
    assert authority.try_add_reference("oai2-blobs/a", "ko-new") is GcReferenceDecision.ADDED


def test_wrong_token_cannot_finalize_or_release() -> None:
    authority = GcDeleteLeaseAuthority()
    authority.acquire_delete_lease(
        key="oai2-blobs/a",
        token="lease-1",
        owner="worker-a",
        now=10.0,
        ttl_seconds=30.0,
        expected_revision=0,
    )

    assert (
        authority.finalize_delete(
            key="oai2-blobs/a",
            token="wrong",
            now=11.0,
            already_absent=False,
        )
        is GcDeleteFinalizeDecision.INVALID_LEASE
    )
    assert authority.release_delete_lease("oai2-blobs/a", "wrong") is False
    assert authority.validate_delete_lease("oai2-blobs/a", "lease-1", now=11.0)
