"""Runtime evidence-path adapter tests for WI-TRUTH-001."""

from __future__ import annotations

from oai2.core import EvidenceId
from oai2.verification import (
    ClaimEvidencePolicy,
    Evidence,
    EvidenceBinding,
    EvidenceClass,
    PolicyDecision,
    assess_repository_evidence,
    assess_tool_runtime_evidence,
    assess_web_evidence,
)


def _policy() -> ClaimEvidencePolicy:
    return ClaimEvidencePolicy(current_external_max_age_seconds=60.0)


def _evidence(
    evidence_id: str,
    cls: EvidenceClass,
    *,
    observed_at: float = 0.0,
) -> Evidence:
    return Evidence(
        id=EvidenceId(evidence_id),
        cls=cls,
        summary=f"evidence {evidence_id}",
        observed_at=observed_at,
    )


def test_repository_path_invalidates_old_source_state() -> None:
    binding = EvidenceBinding(
        evidence=_evidence("repo", EvidenceClass.REPO_SOURCE),
        state_version="sha-a",
    )
    current = assess_repository_evidence(
        _policy(),
        [binding],
        current_state_version="sha-a",
    )
    assert current.decision is PolicyDecision.SUPPORTED

    stale = assess_repository_evidence(
        _policy(),
        [binding],
        current_state_version="sha-b",
    )
    assert stale.decision is PolicyDecision.NEEDS_EVIDENCE
    assert stale.rejected[0].reason == "state_version_mismatch"


def test_tool_runtime_path_preserves_conflict() -> None:
    assessment = assess_tool_runtime_evidence(
        _policy(),
        [
            EvidenceBinding(
                evidence=_evidence("tool-ok", EvidenceClass.RUNTIME_OBS),
                supports=True,
                state_version="run-1",
            ),
            EvidenceBinding(
                evidence=_evidence("tool-bad", EvidenceClass.RUNTIME_OBS),
                supports=False,
                state_version="run-1",
            ),
        ],
        current_state_version="run-1",
    )
    assert assessment.decision is PolicyDecision.CONFLICTING


def test_web_current_path_enforces_freshness() -> None:
    binding = EvidenceBinding(
        evidence=_evidence(
            "web-current",
            EvidenceClass.EXTERNAL,
            observed_at=100.0,
        )
    )
    fresh = assess_web_evidence(
        _policy(),
        [binding],
        current=True,
        now=150.0,
    )
    assert fresh.decision is PolicyDecision.SUPPORTED

    stale = assess_web_evidence(
        _policy(),
        [binding],
        current=True,
        now=161.0,
    )
    assert stale.decision is PolicyDecision.STALE


def test_web_stable_path_does_not_require_freshness_clock() -> None:
    binding = EvidenceBinding(
        evidence=_evidence("web-stable", EvidenceClass.EXTERNAL)
    )
    assessment = assess_web_evidence(
        _policy(),
        [binding],
        current=False,
    )
    assert assessment.decision is PolicyDecision.SUPPORTED


def test_path_adapters_export_from_verification_package() -> None:
    from oai2.verification.paths import assess_repository_evidence as SourceRepository
    from oai2.verification.paths import assess_tool_runtime_evidence as SourceTool
    from oai2.verification.paths import assess_web_evidence as SourceWeb

    assert assess_repository_evidence is SourceRepository
    assert assess_tool_runtime_evidence is SourceTool
    assert assess_web_evidence is SourceWeb
