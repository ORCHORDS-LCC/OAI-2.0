"""Runtime-facing claim-evidence path adapters.

These helpers make the versioned ClaimEvidencePolicy consumable by repository,
tool/runtime, and web/external evidence paths without duplicating claim-class
or freshness/state-version rules at each call site.
"""

from __future__ import annotations

from collections.abc import Sequence

from .policy import (
    ClaimClass,
    ClaimEvidencePolicy,
    EvidenceAssessment,
    EvidenceBinding,
)


def assess_repository_evidence(
    policy: ClaimEvidencePolicy,
    bindings: Sequence[EvidenceBinding],
    *,
    current_state_version: str,
) -> EvidenceAssessment:
    """Assess repository/source evidence against the current repository state."""
    return policy.assess(
        ClaimClass.REPOSITORY_STATE,
        tuple(bindings),
        current_state_version=current_state_version,
    )


def assess_tool_runtime_evidence(
    policy: ClaimEvidencePolicy,
    bindings: Sequence[EvidenceBinding],
    *,
    current_state_version: str,
) -> EvidenceAssessment:
    """Assess tool/runtime observations against the current execution state."""
    return policy.assess(
        ClaimClass.TOOL_RUNTIME_OBSERVATION,
        tuple(bindings),
        current_state_version=current_state_version,
    )


def assess_web_evidence(
    policy: ClaimEvidencePolicy,
    bindings: Sequence[EvidenceBinding],
    *,
    current: bool,
    now: float | None = None,
) -> EvidenceAssessment:
    """Assess external/web evidence as current or stable."""
    claim_class = (
        ClaimClass.EXTERNAL_CURRENT_FACT
        if current
        else ClaimClass.EXTERNAL_STABLE_FACT
    )
    if current:
        return policy.assess(claim_class, tuple(bindings), now=now)
    return policy.assess(claim_class, tuple(bindings))


__all__ = [
    "assess_repository_evidence",
    "assess_tool_runtime_evidence",
    "assess_web_evidence",
]
