from __future__ import annotations

import pytest

from oai2.core import EvidenceId
from oai2.verification.adapters import (
    RepositoryEvidencePath,
    ToolRuntimeEvidencePath,
    WebEvidencePath,
)
from oai2.verification.evidence import Evidence, EvidenceClass
from oai2.verification.policy import ClaimEvidencePolicy, PolicyDecision


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
    return ClaimEvidencePolicy(current_external_max_age_seconds=300.0)


def test_repository_path_supports_current_sha_and_invalidates_old_sha() -> None:
    path = RepositoryEvidencePath(_policy())
    evidence = _evidence("repo-1", EvidenceClass.REPO_SOURCE)

    current = path.assess(
        evidence,
        state_version="sha-1",
        current_state_version="sha-1",
        content_hash="abc123",
    )
    assert current.decision is PolicyDecision.SUPPORTED

    stale = path.assess(
        evidence,
        state_version="sha-1",
        current_state_version="sha-2",
        content_hash="abc123",
    )
    assert stale.decision is PolicyDecision.NEEDS_EVIDENCE
    assert stale.rejected[0].reason == "state_version_mismatch"


def test_tool_runtime_path_invalidates_prior_run_state() -> None:
    path = ToolRuntimeEvidencePath(_policy())
    evidence = _evidence("run-1", EvidenceClass.RUNTIME_OBS)

    assert path.assess(
        evidence,
        state_version="run-v1",
        current_state_version="run-v1",
    ).decision is PolicyDecision.SUPPORTED

    assessment = path.assess(
        evidence,
        state_version="run-v1",
        current_state_version="run-v2",
    )
    assert assessment.decision is PolicyDecision.NEEDS_EVIDENCE


def test_web_current_path_enforces_freshness_but_stable_path_does_not() -> None:
    path = WebEvidencePath(_policy())
    evidence = _evidence("web-1", EvidenceClass.EXTERNAL, observed_at=100.0)

    assert path.assess_current(
        evidence,
        now=350.0,
    ).decision is PolicyDecision.SUPPORTED
    assert path.assess_current(
        evidence,
        now=401.0,
    ).decision is PolicyDecision.STALE

    # Stable facts still require external evidence, but not the current-fact age
    # window.
    assert path.assess_stable(evidence).decision is PolicyDecision.SUPPORTED


@pytest.mark.parametrize(
    ("path_kind", "evidence_class"),
    [
        ("repository", EvidenceClass.EXTERNAL),
        ("tool", EvidenceClass.REPO_SOURCE),
        ("web", EvidenceClass.RUNTIME_OBS),
    ],
)
def test_paths_reject_wrong_evidence_class(
    path_kind: str,
    evidence_class: EvidenceClass,
) -> None:
    evidence = _evidence("wrong", evidence_class, observed_at=100.0)
    policy = _policy()

    with pytest.raises(ValueError, match="requires one of"):
        if path_kind == "repository":
            RepositoryEvidencePath(policy).assess(
                evidence,
                state_version="sha",
                current_state_version="sha",
            )
        elif path_kind == "tool":
            ToolRuntimeEvidencePath(policy).assess(
                evidence,
                state_version="run",
                current_state_version="run",
            )
        else:
            WebEvidencePath(policy).assess_current(evidence, now=150.0)


def test_paths_preserve_refutation_semantics() -> None:
    repo = RepositoryEvidencePath(_policy())
    assessment = repo.assess(
        _evidence("refute", EvidenceClass.DETERMINISTIC),
        state_version="sha",
        current_state_version="sha",
        supports=False,
    )
    assert assessment.decision is PolicyDecision.REFUTED
