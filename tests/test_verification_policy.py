from __future__ import annotations

import pytest

from oai2.core import EvidenceId
from oai2.verification.evidence import Evidence, EvidenceClass
from oai2.verification.policy import (
    EVIDENCE_POLICY_VERSION,
    ClaimClass,
    ClaimEvidencePolicy,
    EvidenceBinding,
    PolicyDecision,
)


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


def _policy() -> ClaimEvidencePolicy:
    return ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)


def test_claim_taxonomy_contains_all_required_classes() -> None:
    assert {item.value for item in ClaimClass} == {
        "repository_state",
        "tool_runtime_observation",
        "external_current_fact",
        "external_stable_fact",
        "inference",
        "assumption",
        "plan",
        "target",
        "preference",
        "hypothetical",
    }


def test_repository_state_requires_matching_state_version() -> None:
    policy = _policy()
    evidence = _evidence("repo-1", EvidenceClass.REPO_SOURCE)

    supported = policy.assess(
        ClaimClass.REPOSITORY_STATE,
        [EvidenceBinding(evidence=evidence, state_version="sha-a")],
        current_state_version="sha-a",
    )
    assert supported.decision is PolicyDecision.SUPPORTED

    invalidated = policy.assess(
        ClaimClass.REPOSITORY_STATE,
        [EvidenceBinding(evidence=evidence, state_version="sha-a")],
        current_state_version="sha-b",
    )
    assert invalidated.decision is PolicyDecision.NEEDS_EVIDENCE
    assert invalidated.rejected[0].reason == "state_version_mismatch"


def test_current_external_fact_expires_by_configured_freshness_window() -> None:
    policy = _policy()
    evidence = _evidence("web-1", EvidenceClass.EXTERNAL, observed_at=1000.0)

    fresh = policy.assess(
        ClaimClass.EXTERNAL_CURRENT_FACT,
        [EvidenceBinding(evidence=evidence)],
        now=4500.0,
    )
    assert fresh.decision is PolicyDecision.SUPPORTED

    stale = policy.assess(
        ClaimClass.EXTERNAL_CURRENT_FACT,
        [EvidenceBinding(evidence=evidence)],
        now=4601.0,
    )
    assert stale.decision is PolicyDecision.STALE
    assert stale.rejected[0].reason == "evidence_stale"


def test_current_external_fact_requires_observation_time() -> None:
    assessment = _policy().assess(
        ClaimClass.EXTERNAL_CURRENT_FACT,
        [EvidenceBinding(evidence=_evidence("web-1", EvidenceClass.EXTERNAL))],
        now=2000.0,
    )
    assert assessment.decision is PolicyDecision.STALE
    assert assessment.rejected[0].reason == "observed_at_missing"


def test_wrong_evidence_class_does_not_leak_support_between_claims() -> None:
    assessment = _policy().assess(
        ClaimClass.REPOSITORY_STATE,
        [
            EvidenceBinding(
                evidence=_evidence("external-1", EvidenceClass.EXTERNAL),
                state_version="sha-a",
            )
        ],
        current_state_version="sha-a",
    )
    assert assessment.decision is PolicyDecision.NEEDS_EVIDENCE
    assert assessment.rejected[0].reason == "evidence_class_not_accepted"


def test_support_and_refutation_produce_policy_conflict() -> None:
    assessment = _policy().assess(
        ClaimClass.TOOL_RUNTIME_OBSERVATION,
        [
            EvidenceBinding(
                evidence=_evidence("runtime-support", EvidenceClass.RUNTIME_OBS),
                supports=True,
                state_version="run-v1",
            ),
            EvidenceBinding(
                evidence=_evidence("runtime-refute", EvidenceClass.RUNTIME_OBS),
                supports=False,
                state_version="run-v1",
            ),
        ],
        current_state_version="run-v1",
    )
    assert assessment.decision is PolicyDecision.CONFLICTING


def test_refuting_evidence_produces_explicit_refuted_state() -> None:
    assessment = _policy().assess(
        ClaimClass.EXTERNAL_STABLE_FACT,
        [
            EvidenceBinding(
                evidence=_evidence("source-refute", EvidenceClass.EXTERNAL),
                supports=False,
            )
        ],
    )
    assert assessment.decision is PolicyDecision.REFUTED


@pytest.mark.parametrize(
    "claim_class",
    [
        ClaimClass.ASSUMPTION,
        ClaimClass.PLAN,
        ClaimClass.TARGET,
        ClaimClass.PREFERENCE,
        ClaimClass.HYPOTHETICAL,
    ],
)
def test_non_evidence_claim_classes_are_not_forced_through_external_evidence(
    claim_class: ClaimClass,
) -> None:
    assessment = _policy().assess(claim_class, [])
    assert assessment.decision is PolicyDecision.NOT_REQUIRED


def test_policy_version_is_traceable_in_every_assessment() -> None:
    policy = _policy()
    assessment = policy.assess(ClaimClass.HYPOTHETICAL, [])
    assert policy.version == EVIDENCE_POLICY_VERSION == "1"
    assert assessment.policy_version == "1"


def test_state_and_freshness_inputs_fail_closed_when_required() -> None:
    policy = _policy()

    with pytest.raises(ValueError, match="current_state_version"):
        policy.assess(ClaimClass.REPOSITORY_STATE, [])

    with pytest.raises(ValueError, match="now is required"):
        policy.assess(ClaimClass.EXTERNAL_CURRENT_FACT, [])
