"""Tests for conservative, resumable R2 orphan sweep behavior."""

from __future__ import annotations

import pytest

from oai2.knowledge.gc import (
    GcDryRunReport,
    GcObjectDisposition,
    GcObjectRecord,
)
from oai2.knowledge.gc_lease import (
    GcDeleteLeaseAuthority,
    GcDeleteLeaseDecision,
    GcReferenceDecision,
)
from oai2.knowledge.sweep import (
    GcSweepDisposition,
    GcSweepState,
)


def _report(*keys: str, observed_at: float = 10.0) -> GcDryRunReport:
    records = tuple(
        GcObjectRecord(
            key=key,
            disposition=GcObjectDisposition.UNREFERENCED_CANDIDATE,
            size_bytes=5,
            uploaded_at=1.0,
            age_seconds=observed_at - 1.0,
        )
        for key in keys
    )
    return GcDryRunReport(
        observed_at=observed_at,
        pages_processed=1,
        records=records,
        referenced_present_count=0,
        referenced_missing_count=0,
        unreferenced_candidate_count=len(records),
        inventory_object_count=len(records),
        inventory_bytes_known=5 * len(records),
        unreferenced_candidate_bytes_known=5 * len(records),
        unknown_size_object_count=0,
        oldest_unreferenced_candidate_age_seconds=observed_at - 1.0,
    )


def test_grace_period_defers_without_authoritative_or_delete_calls() -> None:
    state = GcSweepState.from_report(_report("oai2-blobs/a", observed_at=10.0))

    result = state.process_batch(
        now=15.0,
        grace_seconds=10.0,
        reference_lookup=lambda _key: (_ for _ in ()).throw(AssertionError()),
        blob_exists=lambda _key: (_ for _ in ()).throw(AssertionError()),
        delete_blob=lambda _key: (_ for _ in ()).throw(AssertionError()),
    )

    assert result.records[0].disposition is GcSweepDisposition.DEFERRED_GRACE
    assert result.complete is True


def test_new_authoritative_reference_prevents_delete() -> None:
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/shared"))

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: ("ko-new",),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )

    assert result.records[0].disposition is GcSweepDisposition.RE_REFERENCED
    assert result.records[0].knowledge_ids == ("ko-new",)
    assert deleted == []


def test_dry_run_and_missing_authorization_never_delete() -> None:
    deleted: list[str] = []
    report = _report("oai2-blobs/a")

    dry = GcSweepState.from_report(report)
    dry_result = dry.process_batch(
        now=100.0,
        grace_seconds=0.0,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )
    assert dry_result.records[0].disposition is GcSweepDisposition.DRY_RUN

    blocked = GcSweepState.from_report(report)
    blocked_result = blocked.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=False,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )
    assert blocked_result.records[0].disposition is GcSweepDisposition.UNAPPROVED
    assert deleted == []


def test_recovery_precondition_blocks_destructive_delete() -> None:
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=False,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )

    assert result.records[0].disposition is GcSweepDisposition.RECOVERY_BLOCKED
    assert deleted == []


def test_confirmed_orphan_delete_is_idempotent_across_passes() -> None:
    blobs = {"oai2-blobs/a"}
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    first = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
    )
    assert first.records[0].disposition is GcSweepDisposition.DELETED
    assert blobs == set()

    state.start_next_pass()
    second = state.process_batch(
        now=101.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
    )
    assert second.records[0].disposition is GcSweepDisposition.ALREADY_ABSENT


def test_partial_failure_resumes_at_failed_candidate() -> None:
    blobs = {"oai2-blobs/a", "oai2-blobs/b"}
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))
    fail_b = True

    def delete(key: str) -> None:
        nonlocal fail_b
        if key == "oai2-blobs/b" and fail_b:
            fail_b = False
            raise RuntimeError("dependency outage")
        blobs.remove(key)

    first = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        max_items=2,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=delete,
    )

    assert [record.disposition for record in first.records] == [
        GcSweepDisposition.DELETED,
        GcSweepDisposition.FAILED,
    ]
    assert first.next_cursor == 1
    assert state.cursor == 1
    assert blobs == {"oai2-blobs/b"}

    resumed = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        max_items=2,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=delete,
    )
    assert resumed.records[-1].disposition is GcSweepDisposition.DELETED
    assert resumed.complete is True
    assert blobs == set()


def test_snapshot_round_trip_preserves_cursor_and_first_seen() -> None:
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))
    result = state.process_batch(
        now=20.0,
        grace_seconds=100.0,
        max_items=1,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=lambda _key: None,
    )
    assert result.next_cursor == 1

    restored = GcSweepState.from_snapshot(state.to_snapshot())
    assert restored.cursor == 1
    assert restored.candidates[0].first_seen_at == 10.0


def test_snapshot_rejects_cursor_tampering() -> None:
    state = GcSweepState.from_report(_report("oai2-blobs/a", "oai2-blobs/b"))
    snapshot = state.to_snapshot()
    snapshot["cursor"] = 1

    with pytest.raises(ValueError, match="snapshot fingerprint is invalid"):
        GcSweepState.from_snapshot(snapshot)


def test_reference_added_after_initial_lookup_prevents_delete() -> None:
    deleted: list[str] = []
    references: tuple[str, ...] = ()
    lookups = 0
    state = GcSweepState.from_report(_report("oai2-blobs/race"))

    def reference_lookup(_key: str) -> tuple[str, ...]:
        nonlocal lookups
        lookups += 1
        return references

    def blob_exists(_key: str) -> bool:
        nonlocal references
        references = ("ko-race",)
        return True

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=reference_lookup,
        blob_exists=blob_exists,
        delete_blob=deleted.append,
    )

    assert lookups == 2
    assert result.records[0].disposition is GcSweepDisposition.RE_REFERENCED
    assert result.records[0].knowledge_ids == ("ko-race",)
    assert deleted == []


def test_rereferenced_candidate_requires_new_dry_run_before_future_delete() -> None:
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/shared"))

    first = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: ("ko-live",),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )
    assert first.records[0].disposition is GcSweepDisposition.RE_REFERENCED

    state.start_next_pass()
    second = state.process_batch(
        now=200.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: True,
        delete_blob=deleted.append,
    )

    assert second.complete is True
    assert second.records == ()
    assert deleted == []


def test_non_boolean_blob_existence_result_fails_closed() -> None:
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda _key: "yes",  # type: ignore[return-value]
        delete_blob=deleted.append,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert state.cursor == 0
    assert deleted == []


def test_sweep_without_lease_authority_unchanged() -> None:
    """When ``lease_authority`` is not provided the destructive path is
    unchanged: no D1 lease is queried, delete_blob runs, and the candidate
    is recorded as DELETED.
    """
    blobs = {"oai2-blobs/a"}
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
    )

    assert result.records[0].disposition is GcSweepDisposition.DELETED
    assert blobs == set()


def test_sweep_lease_authority_acquires_before_and_finalizes_after_delete() -> None:
    """With ``lease_authority`` provided the destructive path acquires a
    D1-authoritative lease immediately before delete_blob and finalizes the
    lease after a confirmed absence.
    """
    blobs = {"oai2-blobs/a"}
    state = GcSweepState.from_report(_report("oai2-blobs/a"))
    authority = GcDeleteLeaseAuthority()

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
        lease_authority=authority,
    )

    assert result.records[0].disposition is GcSweepDisposition.DELETED
    assert blobs == set()
    # Lease was finalized: a subsequent reference attempt is blocked as a
    # deleted body (proves finalize_delete ran successfully).
    assert (
        authority.try_add_reference("oai2-blobs/a", "ko-after")
        is GcReferenceDecision.BODY_DELETED
    )


def test_sweep_lease_authority_writer_reference_during_delete_path_denies() -> None:
    """If the lease-authority state changes between the second reference
    lookup and the lease acquire (simulated here by a writer adding a
    reference inside blob_exists), the sweep emits LEASE_DENIED and does
    not call delete_blob.
    """
    deleted: list[str] = []
    state = GcSweepState.from_report(_report("oai2-blobs/race"))
    authority = GcDeleteLeaseAuthority()

    def reference_lookup(_key: str) -> tuple[str, ...]:
        return ()

    def blob_exists(_key: str) -> bool:
        # Race: a writer adds a reference via the lease authority between
        # the second sweep reference lookup and the lease acquire.
        authority.try_add_reference("oai2-blobs/race", "ko-late")
        return True

    def delete_blob(key: str) -> None:
        deleted.append(key)

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=reference_lookup,
        blob_exists=blob_exists,
        delete_blob=delete_blob,
        lease_authority=authority,
    )

    assert result.records[0].disposition is GcSweepDisposition.LEASE_DENIED
    assert deleted == []
    # No lease is held for this key after a denial.
    assert (
        authority.try_add_reference("oai2-blobs/race", "ko-other")
        is GcReferenceDecision.ADDED
    )


def test_sweep_lease_authority_records_failure_when_delete_fails() -> None:
    """When ``lease_authority`` is provided and delete_blob raises, the sweep
    records a retryable failure on the lease (DELETE_FAILED state) so the
    next attempt can recover the claim and writers stay excluded.
    """
    blobs = {"oai2-blobs/a"}
    state = GcSweepState.from_report(_report("oai2-blobs/a"))
    authority = GcDeleteLeaseAuthority()

    def delete_blob(key: str) -> None:
        raise RuntimeError("simulated R2 outage")

    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=delete_blob,
        lease_authority=authority,
    )

    assert result.records[0].disposition is GcSweepDisposition.FAILED
    assert state.cursor == 0
    # The lease stays held and continues to block writers (recoverable).
    assert (
        authority.try_add_reference("oai2-blobs/a", "ko-recover")
        is GcReferenceDecision.BLOCKED_BY_DELETE_LEASE
    )



def test_sweep_lease_authority_revalidates_before_delete() -> None:
    """An invalidated lease must fail closed without deleting the body."""
    blobs = {"oai2-blobs/a"}

    class InvalidatingAuthority(GcDeleteLeaseAuthority):
        def validate_delete_lease(self, key: str, token: str, *, now: float) -> bool:
            return False

    state = GcSweepState.from_report(_report("oai2-blobs/a"))
    result = state.process_batch(
        now=100.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda key: key in blobs,
        delete_blob=blobs.remove,
        lease_authority=InvalidatingAuthority(),
    )

    assert result.records[0].disposition is GcSweepDisposition.LEASE_DENIED
    assert result.complete is False
    assert state.cursor == 0
    assert blobs == {"oai2-blobs/a"}


def test_sweep_lease_ttl_must_be_positive() -> None:
    state = GcSweepState.from_report(_report("oai2-blobs/a"))

    with pytest.raises(ValueError, match="lease_ttl_seconds must be greater than zero"):
        state.process_batch(
            now=100.0,
            grace_seconds=0.0,
            destructive=True,
            authorized=True,
            recovery_ready=True,
            reference_lookup=lambda _key: (),
            blob_exists=lambda _key: True,
            delete_blob=lambda _key: None,
            lease_authority=GcDeleteLeaseAuthority(),
            lease_ttl_seconds=0.0,
        )



def test_sweep_expired_takeover_fences_stale_token() -> None:
    """A takeover must not reuse an expired worker's lease token."""
    key = "oai2-blobs/takeover"
    blobs = {key}
    state = GcSweepState.from_report(_report(key))
    authority = GcDeleteLeaseAuthority()
    stale_token = f"gc-sweep:{key}"

    first = authority.acquire_delete_lease(
        key=key,
        token=stale_token,
        owner="stale-worker",
        now=10.0,
        ttl_seconds=5.0,
        expected_revision=0,
    )
    assert first.decision is GcDeleteLeaseDecision.ACQUIRED

    stale_token_valid_during_delete: list[bool] = []

    def delete_blob(candidate_key: str) -> None:
        stale_token_valid_during_delete.append(
            authority.validate_delete_lease(
                candidate_key, stale_token, now=20.0
            )
        )
        blobs.remove(candidate_key)

    result = state.process_batch(
        now=20.0,
        grace_seconds=0.0,
        destructive=True,
        authorized=True,
        recovery_ready=True,
        reference_lookup=lambda _key: (),
        blob_exists=lambda candidate_key: candidate_key in blobs,
        delete_blob=delete_blob,
        lease_authority=authority,
    )

    assert result.records[0].disposition is GcSweepDisposition.DELETED
    assert stale_token_valid_during_delete == [False]
    assert blobs == set()


# ---------------------------------------------------------------------------
# REQ-GC-021 / AC-GC-023: the grace window must be satisfiable across runs.
#
# The gap: `from_report` stamped `first_seen_at = report.observed_at` on every
# candidate, every time. A scheduled sweep re-runs its dry-run each cycle, so
# the clock restarted every cycle, `age` was always ~0, and every candidate was
# `deferred_grace` forever. REQ-GC-021 held vacuously and AC-GC-023 -- "a
# confirmed old orphan is deleted exactly once" -- was unreachable.
# ---------------------------------------------------------------------------

GRACE = 900.0


def _sweep(
    state: GcSweepState,
    *,
    now: float,
    referenced: set[str] | None = None,
    present: set[str] | None = None,
    grace_seconds: float = GRACE,
    destructive: bool = True,
    authorized: bool = True,
    recovery_ready: bool = True,
):
    """Run one batch against an in-memory store, recording real deletes."""
    store = set(present) if present is not None else set()
    removed: list[str] = []
    refs = referenced if referenced is not None else set()

    def _exists(key: str) -> bool:
        return key in store

    def _delete(key: str) -> None:
        removed.append(key)
        store.discard(key)

    result = state.process_batch(
        now=now,
        grace_seconds=grace_seconds,
        reference_lookup=lambda key: sorted(refs),
        blob_exists=_exists,
        delete_blob=_delete,
        destructive=destructive,
        authorized=authorized,
        recovery_ready=recovery_ready,
    )
    return result, removed


class TestGraceCarriesAcrossRuns:
    def test_repeated_dry_run_eventually_deletes_a_confirmed_orphan(self) -> None:
        """The whole point: a window that never opens deletes nothing, forever.

        This is the scheduled-sweep shape -- a fresh dry-run per cycle, with
        the previous state carried forward. Every cycle deletes nothing until
        the window opens, and then it deletes exactly once (AC-GC-023).
        """
        prior = None
        dispositions: list[str] = []
        deleted: list[str] = []
        for now in (1_000.0, 1_500.0, 2_000.0):
            state = GcSweepState.from_report(
                _report("oai2-blobs/a", observed_at=now), prior=prior
            )
            result, removed = _sweep(state, now=now, present={"oai2-blobs/a"})
            dispositions.extend(r.disposition.value for r in result.records)
            deleted.extend(removed)
            prior = state
        assert dispositions == ["deferred_grace", "deferred_grace", "deleted"]
        assert deleted == ["oai2-blobs/a"]

    def test_first_seen_is_carried_not_restamped(self) -> None:
        first = GcSweepState.from_report(_report("k", observed_at=1_000.0))
        second = GcSweepState.from_report(
            _report("k", observed_at=1_500.0), prior=first
        )
        assert first.candidates[0].first_seen_at == 1_000.0
        assert second.candidates[0].first_seen_at == 1_000.0

    def test_a_re_appearing_reference_discards_the_accumulated_age(self) -> None:
        """History is discarded, not paused.

        The row that came and went may have been a different one, so the old
        age says nothing about this new period of unreferencedness.
        """
        first = GcSweepState.from_report(_report("k", observed_at=1_000.0))
        first.retired_keys.add("k")
        again = GcSweepState.from_report(
            _report("k", observed_at=1_500.0), prior=first
        )
        assert again.candidates[0].first_seen_at == 1_500.0

    def test_a_backwards_clock_cannot_manufacture_age(self) -> None:
        """Clamping fails safe: the candidate defers a little longer."""
        first = GcSweepState.from_report(_report("k", observed_at=9_000.0))
        second = GcSweepState.from_report(
            _report("k", observed_at=1_000.0), prior=first
        )
        assert second.candidates[0].first_seen_at == 1_000.0

    def test_a_new_key_gets_a_fresh_clock(self) -> None:
        first = GcSweepState.from_report(_report("k", observed_at=1_000.0))
        second = GcSweepState.from_report(
            _report("k", "k2", observed_at=1_500.0), prior=first
        )
        by_key = {c.key: c.first_seen_at for c in second.candidates}
        assert by_key == {"k": 1_000.0, "k2": 1_500.0}

    def test_without_prior_the_window_still_opens_on_the_same_state(self) -> None:
        """The long-lived-state path keeps working; this is additive."""
        state = GcSweepState.from_report(_report("k", observed_at=10.0))
        result, _ = _sweep(state, now=10.0, present={"k"})
        assert [r.disposition.value for r in result.records] == ["deferred_grace"]
        state.start_next_pass()
        result, removed = _sweep(state, now=10.0 + 901.0, present={"k"})
        assert [r.disposition.value for r in result.records] == ["deleted"]
        assert removed == ["k"]


class TestGraceCarriesNegativeControls:
    """Mutate the carry-forward at the source level; the guards must fail.

    The behaviours under test are the control flow inside ``from_report``, so
    the mutation has to be made in the text and the module reloaded. An
    attribute patch cannot reach a comprehension body.
    """

    @staticmethod
    def _mutant(tmp_path, mutate):  # type: ignore[no-untyped-def]
        import importlib.util
        import pathlib
        import sys

        import oai2.knowledge.sweep as sweep_mod

        path = pathlib.Path(sweep_mod.__file__)
        source = path.read_text()
        mutated = mutate(source)
        assert mutated != source, "mutation did not apply -- control is vacuous"
        target = tmp_path / "mutant_sweep.py"
        target.write_text(mutated)
        name = "oai2.knowledge._mutant_sweep"
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

    def test_control_retired_key_still_carries_its_age(self, tmp_path) -> None:
        """Dropping the retired skip lets a re-referenced key keep its age."""
        module = self._mutant(
            tmp_path,
            lambda s: s.replace(
                "                if candidate.key in prior.retired_keys:\n"
                "                    continue\n",
                "",
                1,
            ),
        )
        first = module.GcSweepState.from_report(_report("k", observed_at=1_000.0))
        first.retired_keys.add("k")
        again = module.GcSweepState.from_report(
            _report("k", observed_at=1_500.0), prior=first
        )
        assert again.candidates[0].first_seen_at == 1_000.0, "mutation did not bite"
        with pytest.raises(AssertionError):
            assert again.candidates[0].first_seen_at == 1_500.0

    def test_control_backwards_clock_clamp_removed(self, tmp_path) -> None:
        """Without the clamp a backwards clock carries a future first_seen."""
        module = self._mutant(
            tmp_path,
            lambda s: s.replace(
                "                carried[candidate.key] = min(\n"
                "                    candidate.first_seen_at, report.observed_at\n"
                "                )\n",
                "                carried[candidate.key] = candidate.first_seen_at\n",
                1,
            ),
        )
        first = module.GcSweepState.from_report(_report("k", observed_at=9_000.0))
        again = module.GcSweepState.from_report(
            _report("k", observed_at=1_000.0), prior=first
        )
        assert again.candidates[0].first_seen_at == 9_000.0, "mutation did not bite"
        with pytest.raises(AssertionError):
            assert again.candidates[0].first_seen_at == 1_000.0

    def test_control_prior_ignored_entirely(self, tmp_path) -> None:
        """The original defect: a fresh clock from the report, every cycle."""
        module = self._mutant(
            tmp_path,
            lambda s: s.replace(
                "                first_seen_at=carried.get(record.key, report.observed_at),",
                "                first_seen_at=report.observed_at,",
                1,
            ),
        )
        first = module.GcSweepState.from_report(_report("k", observed_at=1_000.0))
        again = module.GcSweepState.from_report(
            _report("k", observed_at=1_500.0), prior=first
        )
        assert again.candidates[0].first_seen_at == 1_500.0, "mutation did not bite"
        with pytest.raises(AssertionError):
            assert again.candidates[0].first_seen_at == 1_000.0
