"""Runtime evidence-path adapter tests for WI-TRUTH-001."""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
    ("path_kind", "evidence_class"),
    [
        ("repository", EvidenceClass.EXTERNAL),
        ("tool", EvidenceClass.REPO_SOURCE),
        ("web", EvidenceClass.RUNTIME_OBS),
    ],
)
def test_path_boundary_rejects_wrong_evidence_class(
    path_kind: str,
    evidence_class: EvidenceClass,
) -> None:
    binding = EvidenceBinding(
        evidence=_evidence("wrong", evidence_class, observed_at=100.0),
        state_version="state",
    )
    policy = _policy()

    with pytest.raises(ValueError, match="requires one of"):
        if path_kind == "repository":
            assess_repository_evidence(
                policy,
                [binding],
                current_state_version="state",
            )
        elif path_kind == "tool":
            assess_tool_runtime_evidence(
                policy,
                [binding],
                current_state_version="state",
            )
        else:
            assess_web_evidence(
                policy,
                [binding],
                current=True,
                now=120.0,
            )


def test_repository_path_preserves_refutation_state() -> None:
    assessment = assess_repository_evidence(
        _policy(),
        [
            EvidenceBinding(
                evidence=_evidence("refute", EvidenceClass.DETERMINISTIC),
                supports=False,
                state_version="sha-a",
            )
        ],
        current_state_version="sha-a",
    )
    assert assessment.decision is PolicyDecision.REFUTED


def test_verification_module_exports_from_verification_package() -> None:
    from oai2.verification import EVIDENCE_POLICY_VERSION as ExportedPolicyVersion
    from oai2.verification import ClaimClass as ExportedClaimClass
    from oai2.verification import ClaimEvidencePolicy as ExportedClaimEvidencePolicy
    from oai2.verification import ClaimState as ExportedClaimState
    from oai2.verification import ClaimStatus as ExportedClaimStatus
    from oai2.verification import Evidence as ExportedEvidence
    from oai2.verification import EvidenceAssessment as ExportedEvidenceAssessment
    from oai2.verification import EvidenceBinding as ExportedEvidenceBinding
    from oai2.verification import EvidenceClass as ExportedEvidenceClass
    from oai2.verification import EvidenceGraph as ExportedEvidenceGraph
    from oai2.verification import EvidenceNode as ExportedEvidenceNode
    from oai2.verification import EvidenceRejection as ExportedEvidenceRejection
    from oai2.verification import EvidenceRequirement as ExportedEvidenceRequirement
    from oai2.verification import EvidenceStatus as ExportedEvidenceStatus
    from oai2.verification import PolicyDecision as ExportedPolicyDecision
    from oai2.verification import SupportNeed as ExportedSupportNeed
    from oai2.verification import VerificationContext as ExportedVerificationContext
    from oai2.verification import (
        assess_repository_evidence as ExportedRepositoryAdapter,
    )
    from oai2.verification import (
        assess_tool_runtime_evidence as ExportedToolRuntimeAdapter,
    )
    from oai2.verification import assess_web_evidence as ExportedWebAdapter
    from oai2.verification.evidence import Evidence as SourceEvidence
    from oai2.verification.evidence import EvidenceClass as SourceEvidenceClass
    from oai2.verification.evidence import EvidenceGraph as SourceEvidenceGraph
    from oai2.verification.evidence import EvidenceNode as SourceEvidenceNode
    from oai2.verification.evidence import EvidenceStatus as SourceEvidenceStatus
    from oai2.verification.paths import (
        assess_repository_evidence as SourceRepositoryAdapter,
    )
    from oai2.verification.paths import (
        assess_tool_runtime_evidence as SourceToolRuntimeAdapter,
    )
    from oai2.verification.paths import assess_web_evidence as SourceWebAdapter
    from oai2.verification.policy import EVIDENCE_POLICY_VERSION as SourcePolicyVersion
    from oai2.verification.policy import ClaimClass as SourceClaimClass
    from oai2.verification.policy import ClaimEvidencePolicy as SourceClaimEvidencePolicy
    from oai2.verification.policy import (
        EvidenceAssessment as SourceEvidenceAssessment,
    )
    from oai2.verification.policy import EvidenceBinding as SourceEvidenceBinding
    from oai2.verification.policy import EvidenceRejection as SourceEvidenceRejection
    from oai2.verification.policy import (
        EvidenceRequirement as SourceEvidenceRequirement,
    )
    from oai2.verification.policy import PolicyDecision as SourcePolicyDecision
    from oai2.verification.state import ClaimState as SourceClaimState
    from oai2.verification.state import ClaimStatus as SourceClaimStatus
    from oai2.verification.state import SupportNeed as SourceSupportNeed
    from oai2.verification.state import VerificationContext as SourceVerificationContext

    assert ExportedClaimClass is SourceClaimClass
    assert ExportedClaimEvidencePolicy is SourceClaimEvidencePolicy
    assert ExportedClaimState is SourceClaimState
    assert ExportedClaimStatus is SourceClaimStatus
    assert ExportedPolicyVersion is SourcePolicyVersion
    assert ExportedEvidence is SourceEvidence
    assert ExportedEvidenceAssessment is SourceEvidenceAssessment
    assert ExportedEvidenceBinding is SourceEvidenceBinding
    assert ExportedEvidenceClass is SourceEvidenceClass
    assert ExportedEvidenceGraph is SourceEvidenceGraph
    assert ExportedEvidenceNode is SourceEvidenceNode
    assert ExportedEvidenceRejection is SourceEvidenceRejection
    assert ExportedEvidenceRequirement is SourceEvidenceRequirement
    assert ExportedEvidenceStatus is SourceEvidenceStatus
    assert ExportedPolicyDecision is SourcePolicyDecision
    assert ExportedSupportNeed is SourceSupportNeed
    assert ExportedVerificationContext is SourceVerificationContext
    assert ExportedRepositoryAdapter is SourceRepositoryAdapter
    assert ExportedToolRuntimeAdapter is SourceToolRuntimeAdapter
    assert ExportedWebAdapter is SourceWebAdapter
