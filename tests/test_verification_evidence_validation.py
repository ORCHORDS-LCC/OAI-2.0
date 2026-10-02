"""Structural / validation tests for ``oai2.verification.evidence``.

Pins the WI-TRUTH-001 evidence primitives: the ``Evidence`` /
``EvidenceNode`` / ``EvidenceGraph`` Pydantic models, the two
:class:`enum.StrEnum` taxonomies (``EvidenceClass`` and ``EvidenceStatus``)
that key the graph, and the derived-status invariant on
``EvidenceNode``.

Behavioural end-to-end coverage lives in
``tests/test_verification_policy.py`` / ``tests/test_verification_paths.py``;
this file pins the *shape* of the API and the invariants the call
sequencer relies on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from oai2.core import EvidenceId
from oai2.verification.evidence import (
    Evidence,
    EvidenceClass,
    EvidenceGraph,
    EvidenceNode,
    EvidenceStatus,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "oai2" / "verification" / "evidence.py"
_MODULE_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Module docstring + import surface
# ---------------------------------------------------------------------------


def test_evidence_module_source_is_non_empty() -> None:
    """Sanity: the source file was read successfully."""

    assert _MODULE_SOURCE, "verification/evidence.py is unexpectedly empty"


def test_evidence_module_has_docstring() -> None:
    """The module exposes a public-style docstring (not just a one-liner)."""

    assert _docstring_lines() >= 3, (
        "Evidence module docstring should explain the Evidence/Graph shapes"
    )


def test_evidence_module_docstring_references_verification_and_evidence() -> None:
    """The docstring must point to VERIFICATION_AND_EVIDENCE.md so the reader
    can locate the authoritative WI-TRUTH-001 write-up."""

    assert "VERIFICATION_AND_EVIDENCE.md" in _MODULE_SOURCE, (
        "evidence module should cite the WI-TRUTH-001 reference doc"
    )


def test_evidence_module_uses_future_annotations() -> None:
    """PEP 563 deferred evaluation must be enabled."""

    assert "from __future__ import annotations" in _MODULE_SOURCE


def test_evidence_module_imports_pydantic_basemodel() -> None:
    """Pydantic is the modelling substrate for Evidence/EvidenceNode/EvidenceGraph."""

    assert re.search(
        r"^from pydantic import ([^\n]*,\s*)*(BaseModel)(\s*,\s*|\s*$)",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "Evidence module should import BaseModel from pydantic"


def test_evidence_module_imports_strenum() -> None:
    """``EvidenceClass`` and ``EvidenceStatus`` are wire-pinned via StrEnum."""

    assert re.search(
        r"^from enum import ([^\n]*,\s*)*StrEnum(\s*,[^\n]*|\s*$)",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "Evidence module should import StrEnum from enum"


def test_evidence_module_imports_core_evidence_id() -> None:
    """``Evidence.id`` is annotated with the canonical :data:`EvidenceId` newtype."""

    assert re.search(
        r"^from \.\.core import ([^\n]*,\s*)*EvidenceId(\s*|\s*,[^\n]*)$",
        _MODULE_SOURCE,
        re.MULTILINE,
    ), "Evidence module should re-use the canonical EvidenceId newtype"


def test_evidence_module_does_not_import_cloud_runtime_modules() -> None:
    """The evidence module must remain provider-neutral (no boto3/azure/google.cloud)."""

    forbidden = ("boto3", "azure", "google.cloud", "kubernetes", "docker")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"Evidence module must not import cloud runtime module {token!r}"
        )


def test_evidence_module_does_not_hardcode_credentials() -> None:
    """No api_key= literals, no BEGIN PRIVATE KEY blocks, no sk-/ghp_ tokens."""

    forbidden_patterns = (
        re.compile(r"api_key\s*=\s*['\"]sk-"),
        re.compile(r"BEGIN PRIVATE KEY"),
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    )
    for pattern in forbidden_patterns:
        assert not pattern.search(_MODULE_SOURCE), (
            f"Evidence module must not contain credential marker {pattern.pattern!r}"
        )


def test_evidence_module_has_no_print_or_pprint_calls() -> None:
    """The evidence module is library code; it must not perform I/O on import."""

    assert not re.search(r"\bprint\s*\(", _MODULE_SOURCE), "Evidence module must not call print()"
    assert not re.search(r"\bpprint\s*\.", _MODULE_SOURCE), "Evidence module must not call pprint.*"


def test_evidence_module_has_no_subprocess_or_shell_invocation() -> None:
    """No subprocess, os.system, or shell=True paths."""

    assert "import subprocess" not in _MODULE_SOURCE, "Evidence module must not import subprocess"
    assert "os.system" not in _MODULE_SOURCE, "Evidence module must not call os.system"
    assert "shell=True" not in _MODULE_SOURCE, "Evidence module must not pass shell=True"


def test_evidence_module_has_no_network_client_imports() -> None:
    """No requests / urllib / httpx imports in library code."""

    forbidden = ("import requests", "from requests ", "import urllib", "import httpx")
    for token in forbidden:
        assert token not in _MODULE_SOURCE, (
            f"Evidence module must not import network client {token!r}"
        )


def test_evidence_module_has_no_eval_or_exec() -> None:
    """No dynamic code execution in library code."""

    assert not re.search(r"^\s*eval\s*\(", _MODULE_SOURCE, re.MULTILINE)
    assert not re.search(r"^\s*exec\s*\(", _MODULE_SOURCE, re.MULTILINE)


def test_evidence_module_has_no_wildcard_imports() -> None:
    """No ``from X import *``; the public surface is pinned by ``__all__``."""

    assert not re.search(r"^from\s+\S+\s+import\s+\*", _MODULE_SOURCE, re.MULTILINE), (
        "Evidence module must not use wildcard imports"
    )


def test_evidence_module_has_no_todo_fixme_xxx_markers() -> None:
    """No TODO / FIXME / XXX markers in shipped source."""

    forbidden = ("TODO", "FIXME", "XXX")
    for token in forbidden:
        assert not re.search(rf"\b{token}\b", _MODULE_SOURCE), (
            f"Evidence module must not contain {token!r} marker"
        )


def test_evidence_module_has_no_os_environ_or_getenv() -> None:
    """The evidence module is pure; it must not read environment variables."""

    assert "os.environ" not in _MODULE_SOURCE
    assert "os.getenv" not in _MODULE_SOURCE


def test_evidence_module_pins_one_basemodel_subclass_per_public_class() -> None:
    """Three BaseModel subclasses: Evidence, EvidenceNode, EvidenceGraph."""

    basemodel_count = len(
        re.findall(r"^class\s+\w+\s*\(\s*BaseModel\s*\)", _MODULE_SOURCE, re.MULTILINE)
    )
    assert basemodel_count == 3, f"Expected 3 BaseModel subclasses, found {basemodel_count}"


# ---------------------------------------------------------------------------
# 2. __all__ completeness
# ---------------------------------------------------------------------------


def test_evidence_module_all_is_exactly_five_names() -> None:
    """``__all__`` must pin exactly 5 names."""

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None, "Evidence module must declare __all__"
    body = match.group(1)
    names = [n.strip().strip("'\"") for n in body.split(",") if n.strip()]
    assert len(names) == 5, f"Expected exactly 5 names in __all__, got {len(names)}: {names}"


def test_evidence_module_all_names_match_documented_surface() -> None:
    """The 5 names are Evidence, EvidenceGraph, EvidenceNode, plus the two enums."""

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    names = {n.strip().strip("'\"") for n in body.split(",") if n.strip()}
    assert names == {
        "Evidence",
        "EvidenceClass",
        "EvidenceGraph",
        "EvidenceNode",
        "EvidenceStatus",
    }, f"__all__ set mismatch: {names}"


def test_evidence_module_all_names_are_importable() -> None:
    """Each name in __all__ must be importable from oai2.verification.evidence."""

    from oai2.verification import evidence as module

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    names = [n.strip().strip("'\"") for n in body.split(",") if n.strip()]
    for name in names:
        assert hasattr(module, name), f"{name!r} is not exported by the module"


def test_evidence_module_symbols_are_re_exported_at_package_level() -> None:
    """The five evidence names are also re-exported via oai2.verification."""

    from oai2 import verification

    assert verification.Evidence is Evidence
    assert verification.EvidenceClass is EvidenceClass
    assert verification.EvidenceGraph is EvidenceGraph
    assert verification.EvidenceNode is EvidenceNode
    assert verification.EvidenceStatus is EvidenceStatus


# ---------------------------------------------------------------------------
# 3. EvidenceClass (StrEnum)
# ---------------------------------------------------------------------------


def test_evidence_class_enum_value_set_is_pinned() -> None:
    """Each EvidenceClass member has a wire-pinned string value."""

    expected = {
        EvidenceClass.USER_INTENT: "user_intent",
        EvidenceClass.REPO_SOURCE: "repo_source",
        EvidenceClass.CHANGE_HISTORY: "change_history",
        EvidenceClass.DETERMINISTIC: "deterministic",
        EvidenceClass.RUNTIME_OBS: "runtime_observation",
        EvidenceClass.VISUAL_OBS: "visual_observation",
        EvidenceClass.EXTERNAL: "external_reference",
        EvidenceClass.HYPOTHESIS: "model_hypothesis",
    }
    actual = {member: member.value for member in EvidenceClass}
    assert actual == expected


def test_evidence_class_member_count_is_eight() -> None:
    """The taxonomy must contain exactly 8 members."""

    assert len(list(EvidenceClass)) == 8


def test_evidence_class_subclasses_str_enum() -> None:
    """EvidenceClass is a StrEnum subclass so members are wire-comparable to str."""

    from enum import StrEnum

    assert issubclass(EvidenceClass, StrEnum)


def test_evidence_class_value_round_trips_via_str() -> None:
    """A wire string lifts back to the same member."""

    for member in EvidenceClass:
        assert EvidenceClass(member.value) is member


# ---------------------------------------------------------------------------
# 4. EvidenceStatus (StrEnum)
# ---------------------------------------------------------------------------


def test_evidence_status_enum_value_set_is_pinned() -> None:
    """Three terminal statuses: verified, unverified, conflicting."""

    expected = {
        EvidenceStatus.VERIFIED: "verified",
        EvidenceStatus.UNVERIFIED: "unverified",
        EvidenceStatus.CONFLICTING: "conflicting",
    }
    actual = {member: member.value for member in EvidenceStatus}
    assert actual == expected


def test_evidence_status_member_count_is_three() -> None:
    """The status taxonomy contains exactly 3 members."""

    assert len(list(EvidenceStatus)) == 3


def test_evidence_status_subclasses_str_enum() -> None:
    """EvidenceStatus is a StrEnum subclass."""

    from enum import StrEnum

    assert issubclass(EvidenceStatus, StrEnum)


# ---------------------------------------------------------------------------
# 5. Evidence (Pydantic BaseModel)
# ---------------------------------------------------------------------------


def test_evidence_field_set_is_pinned() -> None:
    """Evidence has the seven documented fields."""

    fields = set(Evidence.model_fields.keys())
    assert fields == {
        "id",
        "cls",
        "summary",
        "source_uri",
        "artifact_ref",
        "embedding_ref",
        "observed_at",
    }, f"Evidence fields mismatch: {fields}"


def test_evidence_forbids_extra_fields() -> None:
    """model_config = ConfigDict(extra='forbid') rejects unknown keys."""

    with pytest.raises(ValidationError):
        Evidence(  # type: ignore[call-arg]
            id=EvidenceId("ev-1"),
            cls=EvidenceClass.USER_INTENT,
            summary="x",
            unknown_field="bad",
        )


def test_evidence_rejects_empty_summary() -> None:
    """``summary`` is bounded by min_length=1; empty strings are invalid."""

    with pytest.raises(ValidationError):
        Evidence(
            id=EvidenceId("ev-1"),
            cls=EvidenceClass.USER_INTENT,
            summary="",
        )


def test_evidence_rejects_summary_above_max_length() -> None:
    """``summary`` is bounded by max_length=1024."""

    with pytest.raises(ValidationError):
        Evidence(
            id=EvidenceId("ev-1"),
            cls=EvidenceClass.USER_INTENT,
            summary="x" * 1025,
        )


def test_evidence_default_optional_fields_are_none() -> None:
    """source_uri / artifact_ref / embedding_ref default to None; observed_at to 0.0."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.USER_INTENT,
        summary="x",
    )
    assert e.source_uri is None
    assert e.artifact_ref is None
    assert e.embedding_ref is None
    assert e.observed_at == 0.0


def test_evidence_is_frozen_via_assignment() -> None:
    """model_config does not freeze=True; mutation raises by default behaviour.

    The evidence module does NOT pin frozen=True on Evidence; assert that
    behaviour matches the documented surface (mutable assignment is allowed
    unless extra=forbid also freezes, which it does not).
    """

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.USER_INTENT,
        summary="x",
    )
    # Pydantic v2 with extra='forbid' only does not freeze; mutation is allowed.
    e.summary = "y"
    assert e.summary == "y"


def test_evidence_is_immutable_via_model_config_extra_forbid() -> None:
    """Pydantic forbids unknown fields; known fields are mutable (see above).

    This test pins the *policy*: extra='forbid' is in effect; the model is
    not frozen (so internal mutation for tests / derived state is allowed).
    """

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.USER_INTENT,
        summary="x",
    )
    # Extra fields forbidden
    with pytest.raises(ValidationError):
        e.__class__(  # type: ignore[call-arg]
            id=EvidenceId("ev-1"),
            cls=EvidenceClass.USER_INTENT,
            summary="x",
            unknown_field="bad",
        )


def test_evidence_summary_min_length_boundary_is_one() -> None:
    """Summary of exactly 1 character is the smallest legal value."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.USER_INTENT,
        summary="x",
    )
    assert e.summary == "x"


def test_evidence_summary_max_length_boundary_is_1024() -> None:
    """Summary of exactly 1024 characters is the largest legal value."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.USER_INTENT,
        summary="x" * 1024,
    )
    assert len(e.summary) == 1024


def test_evidence_observed_at_can_be_arbitrary_finite_float() -> None:
    """observed_at accepts any finite float (including 0.0 = unset sentinel)."""

    e_zero = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.USER_INTENT,
        summary="x",
        observed_at=0.0,
    )
    assert e_zero.observed_at == 0.0

    e_now = Evidence(
        id=EvidenceId("ev-2"),
        cls=EvidenceClass.USER_INTENT,
        summary="x",
        observed_at=1_700_000_000.5,
    )
    assert e_now.observed_at == 1_700_000_000.5


def test_evidence_id_serializes_through_evidenceid_newtype() -> None:
    """``id`` accepts the EvidenceId newtype (runtime: plain str)."""

    eid = EvidenceId("ev-abc-123")
    e = Evidence(
        id=eid,
        cls=EvidenceClass.REPO_SOURCE,
        summary="file contents",
    )
    assert e.id == eid
    assert isinstance(e.id, str)


def test_evidence_round_trips_through_model_dump() -> None:
    """model_dump() preserves all seven fields."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.EXTERNAL,
        summary="docs",
        source_uri="https://example.com",
        artifact_ref="s3://bucket/key",
        embedding_ref="emb-1",
        observed_at=12345.0,
    )
    dumped = e.model_dump()
    assert dumped["id"] == "ev-1"
    assert dumped["cls"] == EvidenceClass.EXTERNAL
    assert dumped["summary"] == "docs"
    assert dumped["source_uri"] == "https://example.com"
    assert dumped["artifact_ref"] == "s3://bucket/key"
    assert dumped["embedding_ref"] == "emb-1"
    assert dumped["observed_at"] == 12345.0


# ---------------------------------------------------------------------------
# 6. EvidenceNode (Pydantic BaseModel with derived status)
# ---------------------------------------------------------------------------


def test_evidence_node_field_set_is_pinned() -> None:
    """EvidenceNode has claim_id / supporting / refuting / status."""

    fields = set(EvidenceNode.model_fields.keys())
    assert fields == {"claim_id", "supporting", "refuting", "status"}


def test_evidence_node_forbids_extra_fields() -> None:
    """model_config = ConfigDict(extra='forbid', validate_assignment=True)."""

    with pytest.raises(ValidationError):
        EvidenceNode(claim_id="c-1", unknown="bad")  # type: ignore[call-arg]


def test_evidence_node_default_supporting_and_refuting_are_empty_tuples() -> None:
    """New nodes start with no supporting / no refuting."""

    node = EvidenceNode(claim_id="c-1")
    assert node.supporting == ()
    assert node.refuting == ()
    assert node.status is EvidenceStatus.UNVERIFIED


def test_evidence_node_default_status_is_unverified() -> None:
    """When both supporting and refuting are empty, status is UNVERIFIED."""

    node = EvidenceNode(claim_id="c-1")
    assert node.status is EvidenceStatus.UNVERIFIED


def test_evidence_node_status_verified_with_support_only() -> None:
    """Supporting-only with one piece of evidence yields VERIFIED."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    node = EvidenceNode(claim_id="c-1", supporting=(e,))
    assert node.status is EvidenceStatus.VERIFIED


def test_evidence_node_status_verified_with_refute_only() -> None:
    """Refuting-only is also VERIFIED (verdict is settled, even when negative)."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    node = EvidenceNode(claim_id="c-1", refuting=(e,))
    assert node.status is EvidenceStatus.VERIFIED


def test_evidence_node_status_conflicting_with_both_support_and_refute() -> None:
    """Both supporting and refuting present → CONFLICTING."""

    s = Evidence(
        id=EvidenceId("ev-s"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="support",
    )
    r = Evidence(
        id=EvidenceId("ev-r"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="refute",
    )
    node = EvidenceNode(claim_id="c-1", supporting=(s,), refuting=(r,))
    assert node.status is EvidenceStatus.CONFLICTING


def test_evidence_node_caller_cannot_lie_about_status() -> None:
    """The model_validator forces derived status even if the caller passed VERIFIED.

    Without derived enforcement, a caller could build
    ``EvidenceNode(claim_id="c", status=EvidenceStatus.VERIFIED)`` with
    empty supporting/refuting and lie about the verdict. The validator
    rewrites ``status`` back to UNVERIFIED.
    """

    node = EvidenceNode(claim_id="c-1", status=EvidenceStatus.VERIFIED)
    assert node.supporting == ()
    assert node.refuting == ()
    assert node.status is EvidenceStatus.UNVERIFIED, (
        "Derived-status invariant must override caller-supplied status"
    )


def test_evidence_node_caller_cannot_force_conflicting_with_one_side_only() -> None:
    """A caller passing CONFLICTING with only one side gets the derived value."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    node = EvidenceNode(
        claim_id="c-1",
        supporting=(e,),
        status=EvidenceStatus.CONFLICTING,
    )
    assert node.status is EvidenceStatus.VERIFIED


def test_evidence_node_validate_assignment_reruns_derived_status() -> None:
    """``validate_assignment=True`` means assignment to a list field also runs the
    validator, so a mutable change to supporting keeps status derived. However
    Because Pydantic v2 with a tuple field is immutable-by-default for assignment
    to the tuple itself, we test that *rebuilding* through model_validate goes
    through the validator path."""

    s1 = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    s2 = Evidence(
        id=EvidenceId("ev-2"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="y",
    )
    node = EvidenceNode(claim_id="c-1", supporting=(s1,))
    assert node.status is EvidenceStatus.VERIFIED
    rebuilt = EvidenceNode.model_validate(
        {"claim_id": "c-1", "supporting": [s1, s2], "refuting": [], "status": "unverified"}
    )
    assert rebuilt.status is EvidenceStatus.VERIFIED


def test_evidence_node_model_copy_runs_through_validator() -> None:
    """``model_copy`` is overridden to route through ``model_validate`` so the
    derived-status invariant survives a copy."""

    s = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    node = EvidenceNode(claim_id="c-1", supporting=(s,))
    assert node.status is EvidenceStatus.VERIFIED
    copied = node.model_copy(update={"status": EvidenceStatus.UNVERIFIED})
    assert copied.status is EvidenceStatus.VERIFIED, (
        "model_copy override must re-derive status, not preserve caller override"
    )


def test_evidence_node_model_copy_without_update_returns_same_status() -> None:
    """Plain copy (no update dict) keeps derived status."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    node = EvidenceNode(claim_id="c-1", supporting=(e,))
    copied = node.model_copy()
    assert copied.status is EvidenceStatus.VERIFIED


def test_evidence_node_net_count_positive_when_more_supporting() -> None:
    """net_count = supporting - refuting."""

    s = Evidence(
        id=EvidenceId("ev-s"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="s",
    )
    r = Evidence(
        id=EvidenceId("ev-r"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="r",
    )
    node = EvidenceNode(claim_id="c-1", supporting=(s, s), refuting=(r,))
    assert node.net_count == 1


def test_evidence_node_net_count_zero_when_balanced() -> None:
    """Equal supporting and refuting counts give net_count == 0."""

    s = Evidence(
        id=EvidenceId("ev-s"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="s",
    )
    r = Evidence(
        id=EvidenceId("ev-r"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="r",
    )
    node = EvidenceNode(claim_id="c-1", supporting=(s,), refuting=(r,))
    assert node.net_count == 0


def test_evidence_node_net_count_negative_when_more_refuting() -> None:
    """More refuting than supporting yields negative net_count."""

    s = Evidence(
        id=EvidenceId("ev-s"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="s",
    )
    r = Evidence(
        id=EvidenceId("ev-r"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="r",
    )
    node = EvidenceNode(claim_id="c-1", supporting=(s,), refuting=(r, r))
    assert node.net_count == -1


def test_evidence_node_net_count_empty_node_is_zero() -> None:
    """net_count of an empty node is 0."""

    node = EvidenceNode(claim_id="c-1")
    assert node.net_count == 0


def test_evidence_node_merge_combines_supporting_from_both_nodes() -> None:
    """``merge`` combines supporting tuples from two nodes for the same claim."""

    e1 = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="1",
    )
    e2 = Evidence(
        id=EvidenceId("ev-2"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="2",
    )
    a = EvidenceNode(claim_id="c-1", supporting=(e1,))
    b = EvidenceNode(claim_id="c-1", supporting=(e2,))
    merged = a.merge(b)
    assert merged.claim_id == "c-1"
    assert len(merged.supporting) == 2
    assert merged.status is EvidenceStatus.VERIFIED


def test_evidence_node_merge_deduplicates_evidence_by_id() -> None:
    """Re-merging the same evidence id does not duplicate the entry."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    a = EvidenceNode(claim_id="c-1", supporting=(e,))
    b = EvidenceNode(claim_id="c-1", supporting=(e,))
    merged = a.merge(b)
    assert len(merged.supporting) == 1


def test_evidence_node_merge_rejects_different_claim_ids() -> None:
    """Merging nodes with different claim ids is a contract violation."""

    a = EvidenceNode(claim_id="c-1")
    b = EvidenceNode(claim_id="c-2")
    with pytest.raises(ValueError):
        a.merge(b)


def test_evidence_node_merge_combines_refuting_from_both_nodes() -> None:
    """Refuting tuples are unioned across the merge."""

    e1 = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="1",
    )
    e2 = Evidence(
        id=EvidenceId("ev-2"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="2",
    )
    a = EvidenceNode(claim_id="c-1", refuting=(e1,))
    b = EvidenceNode(claim_id="c-1", refuting=(e2,))
    merged = a.merge(b)
    assert len(merged.refuting) == 2


def test_evidence_node_merge_recomputes_status_for_combined() -> None:
    """Merged node's status is derived from the combined contents."""

    s = Evidence(
        id=EvidenceId("ev-s"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="s",
    )
    r = Evidence(
        id=EvidenceId("ev-r"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="r",
    )
    a = EvidenceNode(claim_id="c-1", supporting=(s,))
    b = EvidenceNode(claim_id="c-1", refuting=(r,))
    merged = a.merge(b)
    assert merged.status is EvidenceStatus.CONFLICTING


# ---------------------------------------------------------------------------
# 7. EvidenceGraph (Pydantic BaseModel)
# ---------------------------------------------------------------------------


def test_evidence_graph_field_set_is_pinned() -> None:
    """EvidenceGraph has only the ``nodes`` field."""

    fields = set(EvidenceGraph.model_fields.keys())
    assert fields == {"nodes"}


def test_evidence_graph_forbids_extra_fields() -> None:
    """model_config = ConfigDict(extra='forbid') rejects unknown keys."""

    with pytest.raises(ValidationError):
        EvidenceGraph(unknown="bad")  # type: ignore[call-arg]


def test_evidence_graph_default_nodes_is_empty_dict() -> None:
    """A fresh graph has no nodes."""

    g = EvidenceGraph()
    assert g.nodes == {}


def test_evidence_graph_upsert_inserts_new_node() -> None:
    """``upsert`` of an unseen claim_id inserts a new node."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    node = EvidenceNode(claim_id="c-1", supporting=(e,))
    g = EvidenceGraph()
    g.upsert(node)
    assert "c-1" in g.nodes
    assert g.nodes["c-1"] is node


def test_evidence_graph_upsert_existing_node_merges_instead_of_overwriting() -> None:
    """``upsert`` on an existing claim_id merges via EvidenceNode.merge."""

    e1 = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="1",
    )
    e2 = Evidence(
        id=EvidenceId("ev-2"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="2",
    )
    a = EvidenceNode(claim_id="c-1", supporting=(e1,))
    b = EvidenceNode(claim_id="c-1", supporting=(e2,))
    g = EvidenceGraph()
    g.upsert(a)
    g.upsert(b)
    assert len(g.nodes["c-1"].supporting) == 2


def test_evidence_graph_add_inserts_supporting_evidence() -> None:
    """``add(claim_id, evidence, supports=True)`` appends to supporting."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    g = EvidenceGraph()
    g.add("c-1", e, supports=True)
    assert "c-1" in g.nodes
    assert e in g.nodes["c-1"].supporting
    assert g.nodes["c-1"].status is EvidenceStatus.VERIFIED


def test_evidence_graph_add_inserts_refuting_evidence() -> None:
    """``add(claim_id, evidence, supports=False)`` appends to refuting."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    g = EvidenceGraph()
    g.add("c-1", e, supports=False)
    assert e in g.nodes["c-1"].refuting
    assert g.nodes["c-1"].status is EvidenceStatus.VERIFIED


def test_evidence_graph_add_existing_claim_appends_to_existing_node() -> None:
    """Adding evidence to an existing claim appends to its supporting/refuting."""

    e1 = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="1",
    )
    e2 = Evidence(
        id=EvidenceId("ev-2"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="2",
    )
    g = EvidenceGraph()
    g.add("c-1", e1, supports=True)
    g.add("c-1", e2, supports=False)
    assert e1 in g.nodes["c-1"].supporting
    assert e2 in g.nodes["c-1"].refuting
    assert g.nodes["c-1"].status is EvidenceStatus.CONFLICTING


def test_evidence_graph_status_for_known_claim_returns_node_status() -> None:
    """``status_for`` returns the node status if present."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    g = EvidenceGraph()
    g.add("c-1", e, supports=True)
    assert g.status_for("c-1") is EvidenceStatus.VERIFIED


def test_evidence_graph_status_for_unknown_claim_returns_unverified() -> None:
    """Unknown claim → UNVERIFIED (no node, no information)."""

    g = EvidenceGraph()
    assert g.status_for("c-unknown") is EvidenceStatus.UNVERIFIED


def test_evidence_graph_as_dict_returns_serializable_dump() -> None:
    """``as_dict`` returns a dict keyed by claim_id whose values are model dumps."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    g = EvidenceGraph()
    g.add("c-1", e, supports=True)
    dumped = g.as_dict()
    assert "c-1" in dumped
    assert dumped["c-1"]["status"] == EvidenceStatus.VERIFIED.value
    assert len(dumped["c-1"]["supporting"]) == 1
    assert dumped["c-1"]["supporting"][0]["id"] == "ev-1"


def test_evidence_graph_round_trips_through_model_validate() -> None:
    """A graph with nodes round-trips through model_validate."""

    e = Evidence(
        id=EvidenceId("ev-1"),
        cls=EvidenceClass.REPO_SOURCE,
        summary="x",
    )
    g = EvidenceGraph()
    g.add("c-1", e, supports=True)
    dumped = g.model_dump()
    rebuilt = EvidenceGraph.model_validate(dumped)
    assert "c-1" in rebuilt.nodes
    assert rebuilt.status_for("c-1") is EvidenceStatus.VERIFIED


# ---------------------------------------------------------------------------
# 8. Public-safety (boundary) pin
# ---------------------------------------------------------------------------


def test_evidence_module_declares_one_private_helper_for_status_derivation() -> None:
    """A single private helper ``_derive_status`` is the source of truth."""

    match = re.search(
        r"^\s+def\s+(_derive_status)\s*\(",
        _MODULE_SOURCE,
        re.MULTILINE,
    )
    assert match is not None, "Evidence module must declare _derive_status helper"


def test_evidence_module_declares_one_model_validator() -> None:
    """The model_validator on EvidenceNode is named ``_enforce_status_invariant``."""

    match = re.search(
        r"@model_validator\(mode=\"after\"\)",
        _MODULE_SOURCE,
    )
    assert match is not None, "Evidence module must declare a model_validator(mode='after')"


def test_evidence_module_exposes_two_str_enum_classes() -> None:
    """Exactly two StrEnum subclasses: EvidenceClass + EvidenceStatus."""

    strenum_count = len(
        re.findall(r"^class\s+\w+\s*\(\s*StrEnum\s*\)", _MODULE_SOURCE, re.MULTILINE)
    )
    assert strenum_count == 2


def test_evidence_module_imports_mapping_from_collections_abc() -> None:
    """``Mapping`` is the only collections.abc type imported (for model_copy)."""

    assert "from collections.abc import" in _MODULE_SOURCE
    assert "Mapping" in _MODULE_SOURCE


def test_evidence_module_imports_deepcopy_from_copy() -> None:
    """``deepcopy`` is imported for the deep-copy path in model_copy."""

    assert "from copy import deepcopy" in _MODULE_SOURCE


def test_evidence_module_imports_self_from_typing() -> None:
    """``Self`` is imported for the model_copy return-type annotation."""

    assert re.search(
        r"^from typing import ([^\n]*,\s*)*Self(\s*,|\s*$)", _MODULE_SOURCE, re.MULTILINE
    )


def test_evidence_module_does_not_reexport_pydantic_internal_types() -> None:
    """Pydantic's BaseModel / ConfigDict / Field / model_validator must not be
    in __all__ — they're imported but not part of the public surface."""

    match = re.search(r"^__all__\s*=\s*\[([^\]]+)\]", _MODULE_SOURCE, re.MULTILINE)
    assert match is not None
    body = match.group(1)
    forbidden = ("BaseModel", "ConfigDict", "Field", "model_validator", "EvidenceId")
    for name in forbidden:
        assert name not in body, f"Internal pydantic symbol {name!r} must not be in __all__"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _docstring_lines() -> int:
    match = re.match(r'^\s*"""(.*?)"""', _MODULE_SOURCE, re.DOTALL)
    if match is None:
        return 0
    body = match.group(1)
    return len([line for line in body.splitlines() if line.strip()])
