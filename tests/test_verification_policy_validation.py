"""Outer guard-rail tests for ``oai2.verification.policy``.

WI-TRUTH-001 requires evidence requirements to be explicit rather than inferred
from nearby claims. The tests verify that ``oai2.verification.policy`` exposes
the versioned claim taxonomy, evidence policy, freshness rules, and
invalidation rules without leaking deployment-specific surface.

These are deliberately *outer guard rails*: they pin the public-safety boundary
of the policy module so that refactors of the policy evaluator surface as
deliberate contract changes (an added ``PolicyDecision`` member, a new
validation rule on ``EvidenceBinding``, a renamed ``ClaimClass`` wire string).
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from oai2.core import EvidenceId
from oai2.verification import policy as policy_module
from oai2.verification.evidence import Evidence, EvidenceClass
from oai2.verification.policy import (
    EVIDENCE_POLICY_VERSION,
    ClaimClass,
    ClaimEvidencePolicy,
    EvidenceAssessment,
    EvidenceBinding,
    EvidenceRejection,
    EvidenceRequirement,
    PolicyDecision,
)

# ---------------------------------------------------------------------------
# 1. Module docstring + import discipline
# ---------------------------------------------------------------------------


def test_policy_module_docstring_is_pinned_to_wi_truth_001() -> None:
    """The module docstring must reference WI-TRUTH-001 explicitly."""
    docstring = policy_module.__doc__ or ""
    assert "WI-TRUTH-001" in docstring, f"docstring must mention WI-TRUTH-001; got: {docstring!r}"
    assert "evidence" in docstring.lower()


def test_policy_module_has_no_hardcoded_credentials() -> None:
    """The module source must not contain credentials, tokens, or secrets."""
    source = inspect.getsource(policy_module)
    for forbidden in (
        "AKIA",  # AWS access key prefix
        "sk-",  # OpenAI / many providers
        "ghp_",  # GitHub PAT
        "password=",
        "secret=",
        "BEGIN PRIVATE KEY",
    ):
        assert forbidden not in source, (
            f"forbidden credential marker `{forbidden!r}` found in policy.py source"
        )


def test_policy_module_does_not_import_cloud_runtime_modules() -> None:
    """The policy module must remain cloud-neutral (no Worker, Cloudflare, etc.)."""
    source = inspect.getsource(policy_module)
    for forbidden in (
        "import oai2.workers",
        "from oai2.workers",
        "import oai2.cloudflare",
        "from oai2.cloudflare",
        "import oai2.runtime",
        "from oai2.runtime",
        "import boto3",
        "import urllib3",
        "import httpx",
        "import requests",
    ):
        assert forbidden not in source, (
            f"cloud-runtime import `{forbidden!r}` leaked into policy.py"
        )


def test_policy_module_uses_no_typing_collections_at_runtime() -> None:
    """UP006-style: ``typing.Mapping`` / ``typing.Sequence`` must not be runtime-use."""
    source = inspect.getsource(policy_module)
    assert "typing.Mapping" not in source, (
        "typing.Mapping leaked into policy.py — use collections.abc.Mapping"
    )
    assert "typing.Sequence" not in source, (
        "typing.Sequence leaked into policy.py — use collections.abc.Sequence"
    )
    assert "typing.Iterable" not in source, (
        "typing.Iterable leaked into policy.py — use collections.abc.Iterable"
    )


# ---------------------------------------------------------------------------
# 2. __all__ completeness + package-level re-export
# ---------------------------------------------------------------------------


def test_policy_module_all_is_exactly_eight_names() -> None:
    """``__all__`` must list exactly the 8 documented public names."""
    assert set(policy_module.__all__) == {
        "EVIDENCE_POLICY_VERSION",
        "ClaimClass",
        "PolicyDecision",
        "EvidenceRequirement",
        "EvidenceBinding",
        "EvidenceRejection",
        "EvidenceAssessment",
        "ClaimEvidencePolicy",
    }


def test_policy_module_all_names_are_importable() -> None:
    """Each name in ``__all__`` must resolve as a module attribute."""
    for name in policy_module.__all__:
        assert hasattr(policy_module, name), f"`{name}` declared in __all__ but missing from module"


def test_policy_symbols_are_re_exported_at_package_level() -> None:
    """The package-level ``oai2.verification`` namespace must re-export each policy symbol."""
    from oai2.verification import (
        EVIDENCE_POLICY_VERSION as PackageVersion,
    )
    from oai2.verification import (  # noqa: PLC0415 - lazy import is the test
        ClaimClass as PackageClaimClass,
    )
    from oai2.verification import (
        ClaimEvidencePolicy as PackagePolicy,
    )
    from oai2.verification import (
        EvidenceAssessment as PackageAssessment,
    )
    from oai2.verification import (
        EvidenceBinding as PackageBinding,
    )
    from oai2.verification import (
        EvidenceRejection as PackageRejection,
    )
    from oai2.verification import (
        EvidenceRequirement as PackageRequirement,
    )
    from oai2.verification import (
        PolicyDecision as PackageDecision,
    )

    assert PackageClaimClass is ClaimClass
    assert PackagePolicy is ClaimEvidencePolicy
    assert PackageVersion is EVIDENCE_POLICY_VERSION
    assert PackageAssessment is EvidenceAssessment
    assert PackageBinding is EvidenceBinding
    assert PackageRejection is EvidenceRejection
    assert PackageRequirement is EvidenceRequirement
    assert PackageDecision is PolicyDecision


# ---------------------------------------------------------------------------
# 3. EVIDENCE_POLICY_VERSION literal pinning
# ---------------------------------------------------------------------------


def test_evidence_policy_version_literal_is_pinned() -> None:
    """``EVIDENCE_POLICY_VERSION`` must be the pinned string literal ``"1"``."""
    assert EVIDENCE_POLICY_VERSION == "1"
    assert isinstance(EVIDENCE_POLICY_VERSION, str)


# ---------------------------------------------------------------------------
# 4. ClaimClass StrEnum
# ---------------------------------------------------------------------------


_CLAIM_CLASS_WIRE_STRINGS = {
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


def test_claim_class_enum_value_set_is_pinned() -> None:
    """``ClaimClass`` must have exactly 10 members with the documented wire strings."""
    assert {c.value for c in ClaimClass} == _CLAIM_CLASS_WIRE_STRINGS


def test_claim_class_member_count_is_ten() -> None:
    """``ClaimClass`` must contain exactly 10 members (no truncation / no extras)."""
    assert len(list(ClaimClass)) == 10


def test_claim_class_subclasses_strenum() -> None:
    """``ClaimClass`` must subclass ``str`` (so ``op.value`` round-trips as a string)."""
    from enum import StrEnum  # noqa: PLC0415 - StrEnum is the load-bearing superclass

    assert issubclass(ClaimClass, StrEnum)


def test_claim_class_value_round_trips() -> None:
    """``ClaimClass(<wire string>)`` must produce the same enum member."""
    for wire_string in _CLAIM_CLASS_WIRE_STRINGS:
        assert ClaimClass(wire_string).value == wire_string


# ---------------------------------------------------------------------------
# 5. PolicyDecision StrEnum
# ---------------------------------------------------------------------------


_POLICY_DECISION_WIRE_STRINGS = {
    "not_required",
    "needs_evidence",
    "supported",
    "refuted",
    "stale",
    "conflicting",
}


def test_policy_decision_enum_value_set_is_pinned() -> None:
    """``PolicyDecision`` must have exactly 6 members with the documented wire strings."""
    assert {d.value for d in PolicyDecision} == _POLICY_DECISION_WIRE_STRINGS


def test_policy_decision_member_count_is_six() -> None:
    """``PolicyDecision`` must contain exactly 6 members."""
    assert len(list(PolicyDecision)) == 6


def test_policy_decision_subclasses_strenum() -> None:
    """``PolicyDecision`` must subclass ``str``."""
    from enum import StrEnum  # noqa: PLC0415

    assert issubclass(PolicyDecision, StrEnum)


def test_policy_decision_value_round_trips() -> None:
    """``PolicyDecision(<wire string>)`` must produce the same enum member."""
    for wire_string in _POLICY_DECISION_WIRE_STRINGS:
        assert PolicyDecision(wire_string).value == wire_string


# ---------------------------------------------------------------------------
# 6. EvidenceRequirement dataclass shape + validation
# ---------------------------------------------------------------------------


def test_evidence_requirement_field_set_is_pinned() -> None:
    """``EvidenceRequirement`` must have exactly 4 fields with documented defaults."""
    fields = {f.name for f in EvidenceRequirement.__dataclass_fields__.values()}
    assert fields == {"accepted_classes", "min_support", "state_scoped", "freshness_required"}


def test_evidence_requirement_is_slots_and_frozen() -> None:
    """``EvidenceRequirement`` must be ``slots=True`` + ``frozen=True``."""
    params = EvidenceRequirement.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_evidence_requirement_rejects_attribute_mutation() -> None:
    """``EvidenceRequirement`` must reject attribute writes (frozen contract)."""
    requirement = EvidenceRequirement(accepted_classes=(EvidenceClass.EXTERNAL,))
    with pytest.raises((AttributeError, Exception)):
        requirement.min_support = 2  # type: ignore[misc]


@pytest.mark.parametrize("bad_min_support", [True, False, 1.0, "1", None, [], {}, object()])
def test_evidence_requirement_rejects_non_integer_min_support(
    bad_min_support: object,
) -> None:
    """``min_support`` must be a non-negative ``int`` (bool rejected via the load-bearing guard)."""
    with pytest.raises(ValueError, match="min_support"):
        EvidenceRequirement(
            accepted_classes=(EvidenceClass.EXTERNAL,),
            min_support=bad_min_support,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_min_support", [-1, -2, -100])
def test_evidence_requirement_rejects_negative_min_support(
    bad_min_support: int,
) -> None:
    """``min_support`` must be non-negative; negative values raise ValueError."""
    with pytest.raises(ValueError, match="min_support"):
        EvidenceRequirement(
            accepted_classes=(EvidenceClass.EXTERNAL,),
            min_support=bad_min_support,
        )


def test_evidence_requirement_default_min_support_is_one() -> None:
    """The default ``min_support`` must be ``1`` (one supporting evidence required)."""
    requirement = EvidenceRequirement(accepted_classes=(EvidenceClass.EXTERNAL,))
    assert requirement.min_support == 1


def test_evidence_requirement_default_state_scoped_is_false() -> None:
    """The default ``state_scoped`` must be ``False``."""
    requirement = EvidenceRequirement(accepted_classes=(EvidenceClass.EXTERNAL,))
    assert requirement.state_scoped is False


def test_evidence_requirement_default_freshness_required_is_false() -> None:
    """The default ``freshness_required`` must be ``False``."""
    requirement = EvidenceRequirement(accepted_classes=(EvidenceClass.EXTERNAL,))
    assert requirement.freshness_required is False


def test_evidence_requirement_evidence_required_true_when_min_support_positive() -> None:
    """``evidence_required`` must be True when ``min_support > 0``."""
    requirement = EvidenceRequirement(accepted_classes=(EvidenceClass.EXTERNAL,), min_support=2)
    assert requirement.evidence_required is True


def test_evidence_requirement_evidence_required_false_when_min_support_zero() -> None:
    """``evidence_required`` must be False when ``min_support == 0`` (ASSUMPTION-style)."""
    requirement = EvidenceRequirement(accepted_classes=(), min_support=0)
    assert requirement.evidence_required is False


def test_evidence_requirement_evidence_required_true_when_min_support_is_one() -> None:
    """``evidence_required`` must be True at the boundary ``min_support == 1``."""
    requirement = EvidenceRequirement(accepted_classes=(EvidenceClass.EXTERNAL,))
    assert requirement.evidence_required is True


# ---------------------------------------------------------------------------
# 7. EvidenceBinding dataclass shape + validation
# ---------------------------------------------------------------------------


def test_evidence_binding_field_set_is_pinned() -> None:
    """``EvidenceBinding`` must have exactly 4 fields with documented defaults."""
    fields = {f.name for f in EvidenceBinding.__dataclass_fields__.values()}
    assert fields == {"evidence", "supports", "state_version", "content_hash"}


def test_evidence_binding_is_slots_and_frozen() -> None:
    """``EvidenceBinding`` must be ``slots=True`` + ``frozen=True``."""
    params = EvidenceBinding.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_evidence_binding_rejects_attribute_mutation() -> None:
    """``EvidenceBinding`` must reject attribute writes (frozen contract)."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    binding = EvidenceBinding(evidence=evidence)
    with pytest.raises((AttributeError, Exception)):
        binding.supports = False  # type: ignore[misc]


@pytest.mark.parametrize("bad_supports", [0, 1, "true", None, [], {}, 1.0, object()])
def test_evidence_binding_rejects_non_boolean_supports(bad_supports: object) -> None:
    """``supports`` must be a strict ``bool`` (the ``isinstance(x, bool)`` guard is load-bearing)."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    with pytest.raises(ValueError, match="supports must be a boolean"):
        EvidenceBinding(evidence=evidence, supports=bad_supports)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_value",
    ["", "   ", "\n", "\t", "  \t\n  "],
)
def test_evidence_binding_rejects_empty_or_whitespace_state_version(bad_value: str) -> None:
    """``state_version`` must be non-empty / non-whitespace when provided."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    with pytest.raises(ValueError, match="state_version"):
        EvidenceBinding(evidence=evidence, state_version=bad_value)


@pytest.mark.parametrize(
    "bad_value",
    [0, 1, True, False, 1.0, [], {}, b"bytes", object()],
)
def test_evidence_binding_rejects_non_string_state_version(bad_value: object) -> None:
    """``state_version`` must be a normalized string when provided (None is the default)."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    with pytest.raises(ValueError, match="state_version"):
        EvidenceBinding(evidence=evidence, state_version=bad_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_value",
    ["", "   ", "\n", "\t", "  \t\n  "],
)
def test_evidence_binding_rejects_empty_or_whitespace_content_hash(bad_value: str) -> None:
    """``content_hash`` must be non-empty / non-whitespace when provided."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    with pytest.raises(ValueError, match="content_hash"):
        EvidenceBinding(evidence=evidence, content_hash=bad_value)


@pytest.mark.parametrize(
    "bad_value",
    [0, 1, True, False, 1.0, [], {}, b"bytes", object()],
)
def test_evidence_binding_rejects_non_string_content_hash(bad_value: object) -> None:
    """``content_hash`` must be a normalized string when provided (None is the default)."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    with pytest.raises(ValueError, match="content_hash"):
        EvidenceBinding(evidence=evidence, content_hash=bad_value)  # type: ignore[arg-type]


def test_evidence_binding_accepts_already_normalized_strings() -> None:
    """``EvidenceBinding`` must accept strings that are already normalized (no whitespace padding)."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    binding = EvidenceBinding(
        evidence=evidence,
        state_version="sha-abc",
        content_hash="hash-xyz",
    )
    assert binding.state_version == "sha-abc"
    assert binding.content_hash == "hash-xyz"


def test_evidence_binding_rejects_padded_state_version_as_unnormalized() -> None:
    """Padded strings are NOT normalized — the validator rejects them (not silently normalizes)."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    with pytest.raises(ValueError, match="state_version must be a non-empty normalized string"):
        EvidenceBinding(evidence=evidence, state_version="  sha-abc  ")


def test_evidence_binding_rejects_padded_content_hash_as_unnormalized() -> None:
    """Padded content_hash strings are rejected as unnormalized."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    with pytest.raises(ValueError, match="content_hash must be a non-empty normalized string"):
        EvidenceBinding(evidence=evidence, content_hash="  hash-xyz  ")


def test_evidence_binding_default_supports_is_true() -> None:
    """The default ``supports`` must be ``True`` (supporting evidence is the default)."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    binding = EvidenceBinding(evidence=evidence)
    assert binding.supports is True


def test_evidence_binding_default_state_version_and_content_hash_are_none() -> None:
    """The defaults for ``state_version`` and ``content_hash`` must be ``None``."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    binding = EvidenceBinding(evidence=evidence)
    assert binding.state_version is None
    assert binding.content_hash is None


def test_evidence_binding_padded_state_version_is_rejected_not_silently_normalized() -> None:
    """Padded ``state_version`` strings are rejected in ``__post_init__`` (fail-closed)."""
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    with pytest.raises(ValueError, match="state_version"):
        EvidenceBinding(evidence=evidence, state_version="  sha-a  ")


# ---------------------------------------------------------------------------
# 8. EvidenceRejection dataclass shape
# ---------------------------------------------------------------------------


def test_evidence_rejection_field_set_is_pinned() -> None:
    """``EvidenceRejection`` must have exactly 2 fields."""
    fields = {f.name for f in EvidenceRejection.__dataclass_fields__.values()}
    assert fields == {"evidence_id", "reason"}


def test_evidence_rejection_is_slots_and_frozen() -> None:
    """``EvidenceRejection`` must be ``slots=True`` + ``frozen=True``."""
    params = EvidenceRejection.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_evidence_rejection_rejects_attribute_mutation() -> None:
    """``EvidenceRejection`` must reject attribute writes (frozen contract)."""
    rejection = EvidenceRejection(evidence_id="e-1", reason="x")
    with pytest.raises((AttributeError, Exception)):
        rejection.reason = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 9. EvidenceAssessment dataclass shape
# ---------------------------------------------------------------------------


def test_evidence_assessment_field_set_is_pinned() -> None:
    """``EvidenceAssessment`` must have exactly 6 fields with documented defaults."""
    fields = {f.name for f in EvidenceAssessment.__dataclass_fields__.values()}
    assert fields == {
        "policy_version",
        "claim_class",
        "decision",
        "accepted_support_ids",
        "accepted_refute_ids",
        "rejected",
    }


def test_evidence_assessment_is_slots_and_frozen() -> None:
    """``EvidenceAssessment`` must be ``slots=True`` + ``frozen=True``."""
    params = EvidenceAssessment.__dataclass_params__  # type: ignore[attr-defined]
    assert params.slots is True
    assert params.frozen is True


def test_evidence_assessment_rejects_attribute_mutation() -> None:
    """``EvidenceAssessment`` must reject attribute writes (frozen contract)."""
    assessment = EvidenceAssessment(
        policy_version="1",
        claim_class=ClaimClass.HYPOTHETICAL,
        decision=PolicyDecision.NOT_REQUIRED,
    )
    with pytest.raises((AttributeError, Exception)):
        assessment.decision = PolicyDecision.SUPPORTED  # type: ignore[misc]


def test_evidence_assessment_default_accepted_ids_are_empty_tuples() -> None:
    """The defaults for both ``accepted_*_ids`` must be the empty tuple."""
    assessment = EvidenceAssessment(
        policy_version="1",
        claim_class=ClaimClass.HYPOTHETICAL,
        decision=PolicyDecision.NOT_REQUIRED,
    )
    assert assessment.accepted_support_ids == ()
    assert assessment.accepted_refute_ids == ()
    assert assessment.rejected == ()


# ---------------------------------------------------------------------------
# 10. ClaimEvidencePolicy construction + version
# ---------------------------------------------------------------------------


def test_policy_constructor_rejects_non_positive_max_age() -> None:
    """``current_external_max_age_seconds`` must be positive and finite."""
    with pytest.raises(ValueError, match="current_external_max_age_seconds"):
        ClaimEvidencePolicy(current_external_max_age_seconds=0.0)


@pytest.mark.parametrize(
    "bad_value",
    [None, True, False, "1", [], {}, -1, -1.0, -1e9, float("nan"), float("inf"), -float("inf")],
)
def test_policy_constructor_rejects_invalid_max_age(bad_value: object) -> None:
    """``current_external_max_age_seconds`` must be a positive finite number (bool excluded)."""
    with pytest.raises(ValueError, match="current_external_max_age_seconds"):
        ClaimEvidencePolicy(current_external_max_age_seconds=bad_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("good_value", [0.0001, 1.0, 60.0, 3600.0, 86400.0, 1e9, 1.5e9])
def test_policy_constructor_accepts_positive_finite_max_age(good_value: float) -> None:
    """Any positive finite number must be accepted as ``current_external_max_age_seconds``."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=good_value)
    assert policy.current_external_max_age_seconds == good_value


def test_policy_version_property_returns_pinned_string() -> None:
    """``policy.version`` must equal ``EVIDENCE_POLICY_VERSION``."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    assert policy.version == "1"
    assert policy.version == EVIDENCE_POLICY_VERSION


# ---------------------------------------------------------------------------
# 11. ClaimEvidencePolicy.rule_for
# ---------------------------------------------------------------------------


def test_policy_rule_for_returns_documented_requirement_for_repository_state() -> None:
    """``rule_for(REPOSITORY_STATE)`` must return a state-scoped requirement."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    requirement = policy.rule_for(ClaimClass.REPOSITORY_STATE)
    assert isinstance(requirement, EvidenceRequirement)
    assert requirement.state_scoped is True
    assert EvidenceClass.REPO_SOURCE in requirement.accepted_classes
    assert EvidenceClass.CHANGE_HISTORY in requirement.accepted_classes
    assert EvidenceClass.DETERMINISTIC in requirement.accepted_classes


def test_policy_rule_for_returns_documented_requirement_for_tool_runtime() -> None:
    """``rule_for(TOOL_RUNTIME_OBSERVATION)`` must accept runtime + deterministic only."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    requirement = policy.rule_for(ClaimClass.TOOL_RUNTIME_OBSERVATION)
    assert requirement.state_scoped is True
    assert requirement.accepted_classes == (
        EvidenceClass.RUNTIME_OBS,
        EvidenceClass.DETERMINISTIC,
    )


def test_policy_rule_for_returns_freshness_requirement_for_external_current_fact() -> None:
    """``rule_for(EXTERNAL_CURRENT_FACT)`` must set ``freshness_required=True``."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    requirement = policy.rule_for(ClaimClass.EXTERNAL_CURRENT_FACT)
    assert requirement.freshness_required is True
    assert requirement.state_scoped is False
    assert requirement.accepted_classes == (EvidenceClass.EXTERNAL,)


def test_policy_rule_for_returns_stable_fact_requirement() -> None:
    """``rule_for(EXTERNAL_STABLE_FACT)`` must accept EXTERNAL only (no freshness / no state)."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    requirement = policy.rule_for(ClaimClass.EXTERNAL_STABLE_FACT)
    assert requirement.freshness_required is False
    assert requirement.state_scoped is False
    assert requirement.accepted_classes == (EvidenceClass.EXTERNAL,)


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
def test_policy_rule_for_non_evidence_classes_have_min_support_zero(
    claim_class: ClaimClass,
) -> None:
    """ASSUMPTION / PLAN / TARGET / PREFERENCE / HYPOTHETICAL rules must have ``min_support=0``."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    requirement = policy.rule_for(claim_class)
    assert requirement.min_support == 0
    assert requirement.evidence_required is False


# ---------------------------------------------------------------------------
# 12. ClaimEvidencePolicy.assess — NOT_REQUIRED for non-evidence classes
# ---------------------------------------------------------------------------


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
def test_policy_assess_returns_not_required_for_non_evidence_classes(
    claim_class: ClaimClass,
) -> None:
    """ASSUMPTION / PLAN / TARGET / PREFERENCE / HYPOTHETICAL assess to NOT_REQUIRED."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    assessment = policy.assess(claim_class, [])
    assert assessment.decision is PolicyDecision.NOT_REQUIRED
    assert assessment.policy_version == "1"
    assert assessment.claim_class is claim_class
    assert assessment.accepted_support_ids == ()
    assert assessment.accepted_refute_ids == ()
    assert assessment.rejected == ()


def test_policy_assess_non_evidence_class_returns_not_required_even_with_bindings() -> None:
    """Non-evidence classes must NOT_REQUIRED regardless of bindings provided."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    bindings = (EvidenceBinding(evidence=evidence),)
    assessment = policy.assess(ClaimClass.PLAN, bindings)
    assert assessment.decision is PolicyDecision.NOT_REQUIRED


# ---------------------------------------------------------------------------
# 13. ClaimEvidencePolicy.assess — state-scoped claim evaluation
# ---------------------------------------------------------------------------


def test_policy_assess_repository_state_supported_when_state_versions_match() -> None:
    """Matching state_version produces SUPPORTED for REPOSITORY_STATE."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("repo-1", EvidenceClass.REPO_SOURCE)
    assessment = policy.assess(
        ClaimClass.REPOSITORY_STATE,
        [EvidenceBinding(evidence=evidence, state_version="sha-a")],
        current_state_version="sha-a",
    )
    assert assessment.decision is PolicyDecision.SUPPORTED
    assert assessment.accepted_support_ids == ("repo-1",)
    assert assessment.rejected == ()


def test_policy_assess_repository_state_rejects_missing_state_version_in_binding() -> None:
    """REPOSITORY_STATE with binding.state_version=None must reject as state_version_missing."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("repo-1", EvidenceClass.REPO_SOURCE)
    assessment = policy.assess(
        ClaimClass.REPOSITORY_STATE,
        [EvidenceBinding(evidence=evidence, state_version=None)],
        current_state_version="sha-a",
    )
    assert assessment.decision is PolicyDecision.NEEDS_EVIDENCE
    assert assessment.rejected[0].reason == "state_version_missing"
    assert assessment.rejected[0].evidence_id == "repo-1"


def test_policy_assess_repository_state_rejects_mismatching_state_version() -> None:
    """Mismatching state_version produces state_version_mismatch rejection."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("repo-1", EvidenceClass.REPO_SOURCE)
    assessment = policy.assess(
        ClaimClass.REPOSITORY_STATE,
        [EvidenceBinding(evidence=evidence, state_version="sha-a")],
        current_state_version="sha-b",
    )
    assert assessment.decision is PolicyDecision.NEEDS_EVIDENCE
    assert assessment.rejected[0].reason == "state_version_mismatch"


def test_policy_assess_repository_state_requires_current_state_version() -> None:
    """REPOSITORY_STATE without current_state_version raises ValueError."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    with pytest.raises(ValueError, match="current_state_version"):
        policy.assess(ClaimClass.REPOSITORY_STATE, [])


def test_policy_assess_state_scoped_rejects_empty_or_non_string_current_state_version() -> None:
    """Empty / whitespace / non-string current_state_version raises ValueError."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("repo-1", EvidenceClass.REPO_SOURCE)
    bindings = (EvidenceBinding(evidence=evidence, state_version="sha-a"),)

    with pytest.raises(ValueError, match="current_state_version"):
        policy.assess(ClaimClass.REPOSITORY_STATE, bindings, current_state_version="")

    with pytest.raises(ValueError, match="current_state_version"):
        policy.assess(ClaimClass.REPOSITORY_STATE, bindings, current_state_version="   ")

    with pytest.raises(ValueError, match="current_state_version"):
        policy.assess(
            ClaimClass.REPOSITORY_STATE,
            bindings,
            current_state_version=123,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 14. ClaimEvidencePolicy.assess — freshness-scoped claim evaluation
# ---------------------------------------------------------------------------


def test_policy_assess_external_current_fact_supported_within_window() -> None:
    """EXTERNAL_CURRENT_FACT with observed_at within window produces SUPPORTED."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-1", EvidenceClass.EXTERNAL, observed_at=1000.0)
    assessment = policy.assess(
        ClaimClass.EXTERNAL_CURRENT_FACT,
        [EvidenceBinding(evidence=evidence)],
        now=4500.0,
    )
    assert assessment.decision is PolicyDecision.SUPPORTED
    assert assessment.accepted_support_ids == ("web-1",)


def test_policy_assess_external_current_fact_stale_outside_window() -> None:
    """EXTERNAL_CURRENT_FACT past max_age produces STALE."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-1", EvidenceClass.EXTERNAL, observed_at=1000.0)
    assessment = policy.assess(
        ClaimClass.EXTERNAL_CURRENT_FACT,
        [EvidenceBinding(evidence=evidence)],
        now=4601.0,
    )
    assert assessment.decision is PolicyDecision.STALE
    assert assessment.rejected[0].reason == "evidence_stale"


def test_policy_assess_external_current_fact_rejects_future_observation() -> None:
    """EXTERNAL_CURRENT_FACT with observed_at in the future produces STALE."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-1", EvidenceClass.EXTERNAL, observed_at=2000.0)
    assessment = policy.assess(
        ClaimClass.EXTERNAL_CURRENT_FACT,
        [EvidenceBinding(evidence=evidence)],
        now=1000.0,
    )
    assert assessment.decision is PolicyDecision.STALE
    assert assessment.rejected[0].reason == "observed_at_in_future"


def test_policy_assess_external_current_fact_rejects_missing_observed_at() -> None:
    """EXTERNAL_CURRENT_FACT with observed_at=0.0 produces observed_at_missing."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-1", EvidenceClass.EXTERNAL, observed_at=0.0)
    assessment = policy.assess(
        ClaimClass.EXTERNAL_CURRENT_FACT,
        [EvidenceBinding(evidence=evidence)],
        now=2000.0,
    )
    assert assessment.decision is PolicyDecision.STALE
    assert assessment.rejected[0].reason == "observed_at_missing"


def test_policy_assess_external_current_fact_rejects_non_finite_observed_at() -> None:
    """EXTERNAL_CURRENT_FACT with NaN / inf observed_at is rejected as missing."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-1", EvidenceClass.EXTERNAL, observed_at=float("nan"))
    assessment = policy.assess(
        ClaimClass.EXTERNAL_CURRENT_FACT,
        [EvidenceBinding(evidence=evidence)],
        now=2000.0,
    )
    assert assessment.decision is PolicyDecision.STALE
    assert assessment.rejected[0].reason == "observed_at_missing"


def test_policy_assess_external_current_fact_requires_now() -> None:
    """EXTERNAL_CURRENT_FACT without ``now`` raises ValueError."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    with pytest.raises(ValueError, match="now is required"):
        policy.assess(ClaimClass.EXTERNAL_CURRENT_FACT, [])


@pytest.mark.parametrize("bad_now", [-1.0, -1e9, float("-inf"), float("nan"), "1"])
def test_policy_assess_external_current_fact_rejects_invalid_now(bad_now: object) -> None:
    """``now`` must be a non-negative finite number (bool excluded by the load-bearing guard)."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    with pytest.raises(ValueError, match="now"):
        policy.assess(ClaimClass.EXTERNAL_CURRENT_FACT, [], now=bad_now)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 15. ClaimEvidencePolicy.assess — evidence class filtering
# ---------------------------------------------------------------------------


def test_policy_assess_rejects_disallowed_evidence_class() -> None:
    """An evidence class not in ``accepted_classes`` produces evidence_class_not_accepted."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    assessment = policy.assess(
        ClaimClass.REPOSITORY_STATE,
        [EvidenceBinding(evidence=evidence, state_version="sha-a")],
        current_state_version="sha-a",
    )
    assert assessment.decision is PolicyDecision.NEEDS_EVIDENCE
    assert assessment.rejected[0].reason == "evidence_class_not_accepted"
    assert assessment.rejected[0].evidence_id == "e-1"


def test_policy_assess_external_stable_fact_supported_by_external_evidence() -> None:
    """EXTERNAL_STABLE_FACT accepts EXTERNAL evidence and produces SUPPORTED."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-1", EvidenceClass.EXTERNAL)
    assessment = policy.assess(
        ClaimClass.EXTERNAL_STABLE_FACT,
        [EvidenceBinding(evidence=evidence)],
    )
    assert assessment.decision is PolicyDecision.SUPPORTED
    assert assessment.accepted_support_ids == ("web-1",)


# ---------------------------------------------------------------------------
# 16. ClaimEvidencePolicy.assess — conflict / refutation matrix
# ---------------------------------------------------------------------------


def test_policy_assess_conflict_when_support_and_refute_are_both_present() -> None:
    """A mix of supporting and refuting evidence produces CONFLICTING."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    support = _make_evidence("e-support", EvidenceClass.RUNTIME_OBS)
    refute = _make_evidence("e-refute", EvidenceClass.RUNTIME_OBS)
    assessment = policy.assess(
        ClaimClass.TOOL_RUNTIME_OBSERVATION,
        [
            EvidenceBinding(evidence=support, supports=True, state_version="run-v1"),
            EvidenceBinding(evidence=refute, supports=False, state_version="run-v1"),
        ],
        current_state_version="run-v1",
    )
    assert assessment.decision is PolicyDecision.CONFLICTING
    assert assessment.accepted_support_ids == ("e-support",)
    assert assessment.accepted_refute_ids == ("e-refute",)


def test_policy_assess_refuted_when_only_refuting_evidence_present() -> None:
    """Refuting-only evidence produces REFUTED."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    refute = _make_evidence("e-refute", EvidenceClass.EXTERNAL)
    assessment = policy.assess(
        ClaimClass.EXTERNAL_STABLE_FACT,
        [EvidenceBinding(evidence=refute, supports=False)],
    )
    assert assessment.decision is PolicyDecision.REFUTED
    assert assessment.accepted_refute_ids == ("e-refute",)
    assert assessment.accepted_support_ids == ()


def test_policy_assess_needs_evidence_when_no_bindings() -> None:
    """An evidence-required claim with empty bindings produces NEEDS_EVIDENCE."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    assessment = policy.assess(
        ClaimClass.EXTERNAL_STABLE_FACT,
        [],
    )
    assert assessment.decision is PolicyDecision.NEEDS_EVIDENCE


def test_policy_assess_needs_evidence_when_all_bindings_rejected() -> None:
    """When every binding is rejected, the decision is NEEDS_EVIDENCE (not STALE)."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    # REPOSITORY_STATE accepts REPO_SOURCE / CHANGE_HISTORY / DETERMINISTIC — EXTERNAL is rejected
    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    assessment = policy.assess(
        ClaimClass.REPOSITORY_STATE,
        [EvidenceBinding(evidence=evidence, state_version="sha-a")],
        current_state_version="sha-a",
    )
    assert assessment.decision is PolicyDecision.NEEDS_EVIDENCE
    assert assessment.rejected[0].reason == "evidence_class_not_accepted"


def test_policy_assess_evidence_id_serializes_to_string() -> None:
    """``str(evidence.id)`` must be the wire form of the evidence id in assessments."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-abc", EvidenceClass.EXTERNAL)
    assessment = policy.assess(
        ClaimClass.EXTERNAL_STABLE_FACT,
        [EvidenceBinding(evidence=evidence, supports=False)],
    )
    assert "web-abc" in assessment.accepted_refute_ids


# ---------------------------------------------------------------------------
# 17. assess() — policy version propagation
# ---------------------------------------------------------------------------


def test_policy_assessment_carries_policy_version_one() -> None:
    """Every ``EvidenceAssessment`` returned by ``assess`` must carry ``policy_version == "1"``."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)

    assessment = policy.assess(ClaimClass.HYPOTHETICAL, [])
    assert assessment.policy_version == "1"

    evidence = _make_evidence("e-1", EvidenceClass.EXTERNAL)
    assessment = policy.assess(
        ClaimClass.EXTERNAL_STABLE_FACT,
        [EvidenceBinding(evidence=evidence)],
    )
    assert assessment.policy_version == "1"


def test_policy_assessment_claim_class_field_round_trips() -> None:
    """The ``claim_class`` field on the assessment must equal the input claim."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    assessment = policy.assess(ClaimClass.HYPOTHETICAL, [])
    assert assessment.claim_class is ClaimClass.HYPOTHETICAL


# ---------------------------------------------------------------------------
# 18. assess() — tuple OR list bindings
# ---------------------------------------------------------------------------


def test_policy_assess_accepts_tuple_bindings() -> None:
    """``assess`` must accept a ``tuple[EvidenceBinding, ...]`` (the canonical container)."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-1", EvidenceClass.EXTERNAL)
    bindings = (EvidenceBinding(evidence=evidence),)
    assessment = policy.assess(ClaimClass.EXTERNAL_STABLE_FACT, bindings)
    assert assessment.decision is PolicyDecision.SUPPORTED


def test_policy_assess_accepts_list_bindings() -> None:
    """``assess`` must accept a ``list[EvidenceBinding]`` (the alternate container)."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-1", EvidenceClass.EXTERNAL)
    bindings = [EvidenceBinding(evidence=evidence)]
    assessment = policy.assess(ClaimClass.EXTERNAL_STABLE_FACT, bindings)
    assert assessment.decision is PolicyDecision.SUPPORTED


def test_policy_assess_keyword_only_arguments_after_bindings() -> None:
    """The post-bindings arguments to ``assess`` must be keyword-only.

    ``self`` / ``claim_class`` / ``bindings`` are positional-or-keyword (or
    positional for ``self``); ``now`` and ``current_state_version`` MUST be
    keyword-only because of the ``*`` separator in the source.
    """
    sig = inspect.signature(ClaimEvidencePolicy.assess)
    expected_keyword_only = {"now", "current_state_version"}
    for name, parameter in sig.parameters.items():
        if name in expected_keyword_only:
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
                f"`{name}` must be KEYWORD_ONLY; got {parameter.kind!r}"
            )


# ---------------------------------------------------------------------------
# 19. assess() — keywords (now / current_state_version) are keyword-only
# ---------------------------------------------------------------------------


def test_policy_assess_now_keyword_only() -> None:
    """``assess`` must reject positional ``now=``."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("web-1", EvidenceClass.EXTERNAL)
    with pytest.raises(TypeError):
        policy.assess(  # type: ignore[call-arg]
            ClaimClass.EXTERNAL_CURRENT_FACT,
            [EvidenceBinding(evidence=evidence)],
            2000.0,
        )


def test_policy_assess_current_state_version_keyword_only() -> None:
    """``assess`` must reject positional ``current_state_version=``."""
    policy = ClaimEvidencePolicy(current_external_max_age_seconds=3600.0)
    evidence = _make_evidence("repo-1", EvidenceClass.REPO_SOURCE)
    with pytest.raises(TypeError):
        policy.assess(  # type: ignore[call-arg]
            ClaimClass.REPOSITORY_STATE,
            [EvidenceBinding(evidence=evidence, state_version="sha-a")],
            "sha-a",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 20. helpers
# ---------------------------------------------------------------------------


def _make_evidence(evidence_id: str, cls: EvidenceClass, *, observed_at: float = 0.0) -> Evidence:
    """Build an ``Evidence`` instance for parametrized / falsified scenarios."""
    return Evidence(
        id=EvidenceId(evidence_id),
        cls=cls,
        summary=f"evidence {evidence_id}",
        observed_at=observed_at,
    )


# Note: keep ``Any`` import so the test file is recognized as type-hinted.
_ = Any
